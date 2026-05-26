#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import subprocess
from typing import List


def read_best_metrics(metrics_path: Path) -> dict:
    if not metrics_path.exists():
        return {"status": "missing_metrics"}
    with metrics_path.open("r", encoding="utf-8") as f:
        history = json.load(f)
    if not history:
        return {"status": "empty_metrics"}
    best = max(history, key=lambda x: x["val"]["mDice"])
    return {
        "status": "ok",
        "best_mDice": best["val"]["mDice"],
        "best_mIoU": best["val"]["mIoU"],
        "step_run": best.get("step_run"),
        "step_stop": best.get("step_stop"),
    }


def build_command(
    dataset_root: Path,
    layout: str,
    model: str,
    batch_size: int,
    image_size: int,
    save_dir: Path,
    max_steps: int,
    val_interval: int,
    device: str,
) -> List[str]:
    return [
        "python",
        "metirc/segmentation/train_segmentation.py",
        "--dataset-root",
        str(dataset_root),
        "--data-layout",
        layout,
        "--model",
        model,
        "--batch-size",
        str(batch_size),
        "--image-size",
        str(image_size),
        "--save-dir",
        str(save_dir),
        "--max-steps",
        str(max_steps),
        "--val-interval",
        str(val_interval),
        "--device",
        device,
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run segmentation on generated mixed datasets")
    parser.add_argument("--real-root", type=Path, default=Path("/root/autodl-tmp/data/HAM10000/input"))
    parser.add_argument("--mixed-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("/root/autodl-tmp/output/segmentation/outputs"))
    parser.add_argument("--models", type=str, default=None, help="Comma-separated dataset names to run")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=15000)
    parser.add_argument("--val-interval", type=int, default=3000)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    mixed_root = args.mixed_root.resolve()
    if not mixed_root.exists():
        raise FileNotFoundError(f"Mixed root not found: {mixed_root}")

    if args.models:
        dataset_names = [name.strip() for name in args.models.split(",") if name.strip()]
    else:
        dataset_names = sorted([p.name for p in mixed_root.iterdir() if p.is_dir()])

    results = []
    for dataset_name in dataset_names:
        dataset_root = mixed_root / dataset_name
        if not dataset_root.exists():
            continue
        for model in ["unet", "segformer"]:
            save_dir = args.output_dir / f"{model}_{dataset_name}_512"
            save_dir.mkdir(parents=True, exist_ok=True)
            command = build_command(
                dataset_root,
                "mixed",
                model,
                args.batch_size,
                args.image_size,
                save_dir,
                args.max_steps,
                args.val_interval,
                args.device,
            )
            subprocess.run(command, check=True)
            metrics_path = save_dir / f"{model}_metrics.json"
            metrics = read_best_metrics(metrics_path)
            results.append({
                "dataset": dataset_name,
                "model": model,
                **metrics,
            })

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
