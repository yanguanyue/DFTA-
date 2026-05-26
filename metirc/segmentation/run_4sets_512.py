#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import subprocess


def run_training(command: list[str], metrics_path: Path, log_path: Path) -> dict:
    if metrics_path.exists():
        return {"status": "skipped", "reason": "metrics_exists"}

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=log_file,
            start_new_session=True,
        )
    return {"status": "started", "pid": process.pid, "log": str(log_path)}


def build_command(
    dataset_root: Path,
    layout: str,
    model: str,
    batch_size: int,
    image_size: int,
    save_dir: Path,
    max_steps: int,
    val_interval: int,
) -> list[str]:
    command = [
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
    ]
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description="Train on real + Siamese + Siamese-update + flow at 512x512")
    parser.add_argument("--real-root", required=True, type=Path)
    parser.add_argument("--mixed-root", required=True, type=Path, help="Root containing mixed datasets subfolders")
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--max-steps", type=int, default=15000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--val-interval", type=int, default=3000)
    args = parser.parse_args()

    datasets = [
        ("real", args.real_root, "ham10000"),
        ("Siamese", args.mixed_root / "Siamese", "mixed"),
        ("flow", args.mixed_root / "flow", "mixed"),
        ("Controlnet", args.mixed_root / "flow", "mixed"),
        ("T2I-Adapter", args.mixed_root / "flow", "mixed"),
        ("DFMGAN", args.mixed_root / "flow", "mixed"),
    ]

    results = []
    for model in ["unet", "segformer"]:
        for dataset_name, dataset_root, layout in datasets:
            save_dir = Path("/root/autodl-tmp/output/segmentation/outputs") / f"{model}_{dataset_name}_512"
            save_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = save_dir / f"{model}_metrics.json"
            log_path = save_dir / "log.txt"
            command = build_command(
                dataset_root,
                layout,
                model,
                args.batch_size,
                args.image_size,
                save_dir,
                args.max_steps,
                args.val_interval,
            )
            metrics = run_training(command, metrics_path, log_path)
            results.append({
                "model": model,
                "dataset": dataset_name,
                **metrics,
            })

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
