#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import subprocess


def run_training(command: list[str], metrics_path: Path) -> dict:
    subprocess.run(command, check=True)
    with metrics_path.open("r", encoding="utf-8") as f:
        history = json.load(f)
    best = max(history, key=lambda x: x["val"]["mDice"])
    return {
        "best_mDice": best["val"]["mDice"],
        "best_mIoU": best["val"]["mIoU"],
        "epoch": best["epoch"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run segmentation experiments on real and mixed datasets")
    parser.add_argument("--real-root", required=True, type=Path)
    parser.add_argument("--mixed-root", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    args = parser.parse_args()

    results = []
    for model in ["unet", "segformer"]:
        for dataset_name, dataset_root, layout in [
            ("real", args.real_root, "ham10000"),
            ("mixed", args.mixed_root, "mixed"),
        ]:
            save_dir = Path("/root/autodl-tmp/output/segmentation/outputs") / f"{model}_{dataset_name}"
            save_dir.mkdir(parents=True, exist_ok=True)
            command = [
                "python",
                "metirc/segmentation/train_segmentation.py",
                "--dataset-root",
                str(dataset_root),
                "--data-layout",
                layout,
                "--model",
                model,
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--image-size",
                str(args.image_size),
                "--save-dir",
                str(save_dir),
            ]
            metrics_path = save_dir / f"{model}_metrics.json"
            metrics = run_training(command, metrics_path)
            results.append({
                "model": model,
                "dataset": dataset_name,
                **metrics,
            })

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "dataset", "best_mDice", "best_mIoU", "epoch"])
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
