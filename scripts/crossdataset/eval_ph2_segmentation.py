#!/usr/bin/env python3
import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

repo_root = Path(__file__).resolve().parents[2]
sys.path.append(str(repo_root / "metirc" / "segmentation"))

from dataset import JointTransform, SegmentationDataset  # noqa: E402
from metrics import MetricTracker, compute_dice_iou  # noqa: E402
from models import build_segformer, build_unet  # noqa: E402


@dataclass
class SegResult:
    dataset: str
    model: str
    checkpoint: str
    samples: int
    mDice: float
    mIoU: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate segmentation models on PH2.")
    parser.add_argument(
        "--output-root",
        type=str,
        default="/root/autodl-tmp/output/segmentation/outputs",
    )
    parser.add_argument(
        "--ph2-root",
        type=str,
        default="/root/autodl-tmp/data/PH2/input",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/root/autodl-tmp/output/crossdataset",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on samples.")
    return parser.parse_args()


def discover_models(output_root: Path) -> list[tuple[str, str, Path]]:
    items: list[tuple[str, str, Path]] = []
    for path in sorted(output_root.iterdir()):
        if not path.is_dir():
            continue
        if path.name.startswith("unet_"):
            model_type = "unet"
        elif path.name.startswith("segformer_"):
            model_type = "segformer"
        else:
            continue
        dataset_name = path.name[len(model_type) + 1 :]
        if dataset_name.endswith("_512"):
            dataset_name = dataset_name[: -len("_512")]
        checkpoint = path / f"{model_type}_best.pt"
        if checkpoint.exists():
            items.append((dataset_name, model_type, checkpoint))
    return items


def build_dataset(ph2_root: Path, image_size: int, limit: int | None) -> SegmentationDataset:
    image_root = ph2_root / "img_class"
    mask_root = ph2_root / "seg_class"
    transform = JointTransform(image_size, is_train=False)
    dataset = SegmentationDataset(image_root, mask_root, transform=transform)
    if limit is None or len(dataset) <= limit:
        return dataset
    dataset.samples = dataset.samples[:limit]
    return dataset


def load_checkpoint(model: torch.nn.Module, ckpt_path: Path) -> None:
    state_dict = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict, strict=False)


def eval_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = 2,
) -> tuple[float, float]:
    model.eval()
    tracker = MetricTracker()
    with torch.no_grad():
        for batch in tqdm(loader, desc="eval", leave=False):
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
                logits = torch.nn.functional.interpolate(
                    logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
                )
            dice, iou = compute_dice_iou(logits, masks, num_classes=num_classes)
            tracker.update(dice, iou, images.size(0))
    metrics = tracker.compute()
    return metrics["mDice"], metrics["mIoU"]


def write_results(results: Iterable[SegResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "ph2_segmentation.csv"
    md_path = output_dir / "ph2_segmentation.md"

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["dataset", "model", "checkpoint", "samples", "mDice", "mIoU"])
        for row in results:
            writer.writerow(
                [
                    row.dataset,
                    row.model,
                    row.checkpoint,
                    row.samples,
                    f"{row.mDice:.6f}",
                    f"{row.mIoU:.6f}",
                ]
            )

    with md_path.open("w", encoding="utf-8") as md_file:
        md_file.write("| dataset | model | checkpoint | samples | mDice | mIoU |\n")
        md_file.write("| --- | --- | --- | --- | --- | --- |\n")
        for row in results:
            md_file.write(
                f"| {row.dataset} | {row.model} | {row.checkpoint} | {row.samples} | {row.mDice:.6f} | {row.mIoU:.6f} |\n"
            )

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    ph2_root = Path(args.ph2_root).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not output_root.exists():
        raise SystemExit(f"Output root not found: {output_root}")
    if not ph2_root.exists():
        raise SystemExit(f"PH2 root not found: {ph2_root}")

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)

    dataset = build_dataset(ph2_root, args.image_size, args.limit)
    if len(dataset) == 0:
        raise SystemExit("No PH2 samples found for segmentation evaluation.")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model_entries = discover_models(output_root)
    if not model_entries:
        raise SystemExit("No segmentation checkpoints found under output root.")

    results: list[SegResult] = []
    for dataset_name, model_type, ckpt_path in model_entries:
        print(f"Evaluating {model_type}/{dataset_name} on PH2...")
        if model_type == "unet":
            model = build_unet(num_classes=2)
        else:
            model = build_segformer(num_classes=2, pretrained=False)
        load_checkpoint(model, ckpt_path)
        model.to(device)
        mDice, mIoU = eval_model(model, loader, device)
        results.append(
            SegResult(
                dataset=dataset_name,
                model=model_type,
                checkpoint=str(ckpt_path),
                samples=len(dataset),
                mDice=mDice,
                mIoU=mIoU,
            )
        )
        print(f"mDice={mDice:.4f} mIoU={mIoU:.4f}")

    write_results(results, output_dir)


if __name__ == "__main__":
    main()
