#!/usr/bin/env python3
import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


@dataclass
class EvalResult:
    model: str
    arch: str
    checkpoint: str
    total: int
    correct: int
    accuracy: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate HAM10000 classifiers on ISIC2017.")
    parser.add_argument(
        "--checkpoint-root",
        type=str,
        default="/root/autodl-tmp/output/classifier/checkpoints/ham10000_mix",
    )
    parser.add_argument(
        "--isic-root",
        type=str,
        default="/root/autodl-tmp/data/ISIC2017/input/img_class",
    )
    parser.add_argument(
        "--ham-root",
        type=str,
        default="/root/autodl-tmp/data/HAM10000/input/train/HAM10000_img_class",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/root/autodl-tmp/output/crossdataset",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on images per class.")
    return parser.parse_args()


def list_classes(root: Path) -> list[str]:
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def list_images(folder: Path) -> Iterable[Path]:
    for item in folder.iterdir():
        if item.is_file() and item.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            yield item


class MappedImageDataset(Dataset):
    def __init__(
        self,
        root: Path,
        class_to_idx: dict[str, int],
        transform: transforms.Compose,
        limit: int | None = None,
    ) -> None:
        self.samples: list[tuple[Path, int]] = []
        self.transform = transform
        for cls_name in sorted(class_to_idx.keys()):
            class_dir = root / cls_name
            if not class_dir.is_dir():
                continue
            images = list(list_images(class_dir))
            if limit is not None:
                images = images[:limit]
            self.samples.extend([(img, class_to_idx[cls_name]) for img in images])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, target = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), target


def load_models_module(repo_root: Path):
    sys.path.insert(0, str(repo_root))
    import torchvision.models as models  # noqa: WPS433
    import models.imagenet as customized_models  # noqa: WPS433

    for name in customized_models.__dict__:
        if name.islower() and not name.startswith("__") and callable(customized_models.__dict__[name]):
            models.__dict__[name] = customized_models.__dict__[name]
    return models


def adapt_final_layer(model: torch.nn.Module, num_classes: int) -> None:
    try:
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, num_classes)
        return
    except AttributeError:
        pass

    try:
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = torch.nn.Linear(num_ftrs, num_classes)
        return
    except (AttributeError, IndexError, TypeError):
        pass

    try:
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(num_ftrs, num_classes)
        return
    except (AttributeError, IndexError, TypeError):
        pass

    try:
        num_ftrs = model.heads.head.in_features
        model.heads.head = torch.nn.Linear(num_ftrs, num_classes)
        return
    except (AttributeError, IndexError, TypeError):
        pass

    try:
        num_ftrs = model.classifier.in_features
        model.classifier = torch.nn.Linear(num_ftrs, num_classes)
        return
    except AttributeError as exc:
        raise RuntimeError("Unsupported model head for adaptation") from exc


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not state_dict:
        raise RuntimeError(f"Empty checkpoint: {checkpoint_path}")

    model_state = model.state_dict()
    state_keys = list(state_dict.keys())
    model_keys = list(model_state.keys())

    if state_keys and model_keys:
        if state_keys[0].startswith("module.") and not model_keys[0].startswith("module."):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        elif not state_keys[0].startswith("module.") and model_keys[0].startswith("module."):
            state_dict = {f"module.{k}": v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[int, int]:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.numel()
    return correct, total


def discover_checkpoints(root: Path) -> list[tuple[str, str, Path]]:
    checkpoints = []
    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue
        for arch_dir in sorted(model_dir.iterdir()):
            if not arch_dir.is_dir():
                continue
            checkpoint = arch_dir / "model_best.pth.tar"
            if checkpoint.exists():
                checkpoints.append((model_dir.name, arch_dir.name, checkpoint))
    return checkpoints


def write_results(results: list[EvalResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "isic2017_accuracy.csv"
    md_path = output_dir / "isic2017_accuracy.md"

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["model", "arch", "checkpoint", "total", "correct", "accuracy"])
        for row in results:
            writer.writerow([
                row.model,
                row.arch,
                row.checkpoint,
                row.total,
                row.correct,
                f"{row.accuracy:.4f}",
            ])

    with md_path.open("w", encoding="utf-8") as md_file:
        md_file.write("| model | arch | checkpoint | total | correct | accuracy |\n")
        md_file.write("| --- | --- | --- | --- | --- | --- |\n")
        for row in results:
            md_file.write(
                f"| {row.model} | {row.arch} | {row.checkpoint} | {row.total} | {row.correct} | {row.accuracy:.4f} |\n"
            )

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


def main() -> None:
    args = parse_args()
    checkpoint_root = Path(args.checkpoint_root).resolve()
    isic_root = Path(args.isic_root).resolve()
    ham_root = Path(args.ham_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    repo_root = Path("/root/autodl-tmp/metirc/pytorch-classification-extended-master").resolve()

    if not checkpoint_root.exists():
        raise SystemExit(f"Checkpoint root not found: {checkpoint_root}")
    if not isic_root.exists():
        raise SystemExit(f"ISIC root not found: {isic_root}")
    if not ham_root.exists():
        raise SystemExit(f"HAM root not found: {ham_root}")

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)

    ham_classes = list_classes(ham_root)
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(ham_classes)}
    if not class_to_idx:
        raise SystemExit("No HAM10000 classes discovered.")

    isic_classes = sorted([p.name for p in isic_root.iterdir() if p.is_dir()])
    missing = [cls for cls in isic_classes if cls not in class_to_idx]
    if missing:
        print(f"Warning: ISIC classes missing in HAM10000 mapping: {missing}")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = MappedImageDataset(isic_root, class_to_idx, transform, limit=args.limit)
    if len(dataset) == 0:
        raise SystemExit("No images found for ISIC2017 dataset.")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    models = load_models_module(repo_root)
    checkpoints = discover_checkpoints(checkpoint_root)
    if not checkpoints:
        raise SystemExit("No checkpoints found under ham10000_mix.")

    results: list[EvalResult] = []
    for model_name, arch, ckpt_path in checkpoints:
        print(f"Evaluating {model_name}/{arch} on ISIC2017...")
        model = models.__dict__[arch]()
        adapt_final_layer(model, num_classes=len(ham_classes))
        load_checkpoint(model, ckpt_path)
        model = model.to(device)
        if device.type == "cuda" and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

        correct, total = evaluate(model, dataloader, device)
        accuracy = 100.0 * correct / total if total else 0.0
        results.append(
            EvalResult(
                model=model_name,
                arch=arch,
                checkpoint=str(ckpt_path),
                total=total,
                correct=correct,
                accuracy=accuracy,
            )
        )
        print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")

    write_results(results, output_dir)


if __name__ == "__main__":
    main()
