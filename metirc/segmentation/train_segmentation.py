#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent))

from dataset import JointTransform, SegmentationDataset, resolve_roots
from metrics import MetricTracker, compute_dice_iou
from models import build_segformer, build_unet


def train_for_steps(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    num_classes: int,
    global_step: int,
    stop_step: int,
    max_steps: int,
    pbar: tqdm,
) -> Dict[str, float]:
    model.train()
    tracker = MetricTracker()
    total_loss = 0.0
    loader_iter = iter(loader)
    while global_step < stop_step and global_step < max_steps:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        if hasattr(outputs, "logits"):
            logits = outputs.logits
        elif isinstance(outputs, dict) and "logits" in outputs:
            logits = outputs["logits"]
        else:
            logits = outputs
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = torch.nn.functional.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        loss = loss_fn(logits, masks)
        loss.backward()
        optimizer.step()
        global_step += 1
        pbar.update(1)

        dice, iou = compute_dice_iou(logits, masks, num_classes=num_classes)
        tracker.update(dice, iou, images.size(0))
        total_loss += float(loss) * images.size(0)

    metrics = tracker.compute()
    metrics["loss"] = total_loss / max(tracker.count, 1)
    metrics["global_step"] = global_step
    return metrics


def eval_one_epoch(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device, num_classes: int) -> Dict[str, float]:
    model.eval()
    tracker = MetricTracker()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc="val", leave=False):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            outputs = model(images)
            if hasattr(outputs, "logits"):
                logits = outputs.logits
            elif isinstance(outputs, dict) and "logits" in outputs:
                logits = outputs["logits"]
            else:
                logits = outputs
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = torch.nn.functional.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            loss = loss_fn(logits, masks)

            dice, iou = compute_dice_iou(logits, masks, num_classes=num_classes)
            tracker.update(dice, iou, images.size(0))
            total_loss += float(loss) * images.size(0)

    metrics = tracker.compute()
    metrics["loss"] = total_loss / max(tracker.count, 1)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train UNet or SegFormer on HAM10000 segmentation")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--data-layout", choices=["ham10000", "mixed"], default="ham10000")
    parser.add_argument("--model", choices=["unet", "segformer"], default="unet")
    parser.add_argument("--max-steps", type=int, required=True, help="Stop after this many training steps")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--save-dir", type=Path, default=Path("/root/autodl-tmp/output/segmentation/outputs"))
    parser.add_argument("--val-interval", type=int, default=3000, help="Validate every N steps")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    image_train, mask_train = resolve_roots(args.dataset_root, "train", args.data_layout)
    image_val, mask_val = resolve_roots(args.dataset_root, "val", args.data_layout)

    train_dataset = SegmentationDataset(image_train, mask_train, JointTransform(args.image_size, is_train=True))
    val_dataset = SegmentationDataset(image_val, mask_val, JointTransform(args.image_size, is_train=False))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    num_classes = 2
    if args.model == "unet":
        model = build_unet(num_classes=num_classes)
    else:
        model = build_segformer(num_classes=num_classes, pretrained=args.pretrained)

    device = torch.device(args.device)
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    best_dice = 0.0
    history = []

    global_step = 0
    step_run = 0
    val_interval = max(1, args.val_interval)
    pbar = tqdm(total=args.max_steps, initial=0, desc="train_steps")
    try:
        while global_step < args.max_steps:
            step_run += 1
            stop_step = min(args.max_steps, global_step + val_interval)
            train_metrics = train_for_steps(
                model,
                train_loader,
                optimizer,
                loss_fn,
                device,
                num_classes,
                global_step,
                stop_step,
                args.max_steps,
                pbar,
            )
            global_step = train_metrics.get("global_step", global_step)
            val_metrics = eval_one_epoch(model, val_loader, loss_fn, device, num_classes)

            step_metrics = {
                "step_run": step_run,
                "step_stop": stop_step,
                "train": train_metrics,
                "val": val_metrics,
            }
            history.append(step_metrics)

            print(
                f"StepRun {step_run}: steps={global_step} train loss={train_metrics['loss']:.4f} mDice={train_metrics['mDice']:.4f} mIoU={train_metrics['mIoU']:.4f} | "
                f"val loss={val_metrics['loss']:.4f} mDice={val_metrics['mDice']:.4f} mIoU={val_metrics['mIoU']:.4f}"
            )

            if val_metrics["mDice"] > best_dice:
                best_dice = val_metrics["mDice"]
                ckpt_path = args.save_dir / f"{args.model}_best.pt"
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), ckpt_path)

            if global_step >= args.max_steps:
                print(f"Reached max steps: {args.max_steps}. Stopping training.")
                break
    finally:
        pbar.close()

    metrics_path = args.save_dir / f"{args.model}_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
