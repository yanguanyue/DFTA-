#!/usr/bin/env python3
import argparse
import os
import random
import shutil
from collections import Counter
from pathlib import Path


def list_images(folder: Path):
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]


def list_images_recursive(folder: Path):
    return [
        p
        for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ]


def ensure_empty_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def symlink(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src, dst)


def main():
    parser = argparse.ArgumentParser(description="Mix synthetic images into training set")
    parser.add_argument("--real-train", required=True, type=Path)
    parser.add_argument("--real-val", required=True, type=Path)
    parser.add_argument("--synthetic-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--majority-cap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)

    classes = sorted([p.name for p in args.real_train.iterdir() if p.is_dir()])
    counts = {}
    for cls in classes:
        counts[cls] = len(list_images(args.real_train / cls))

    # Majority defined as counts greater than median
    sorted_counts = sorted(counts.values())
    median = sorted_counts[len(sorted_counts) // 2]
    majority_classes = {c for c, n in counts.items() if n > median}
    minority_classes = set(classes) - majority_classes

    print("Class counts:", counts)
    print("Median:", median)
    print("Majority classes:", sorted(majority_classes))
    print("Minority classes:", sorted(minority_classes))

    if args.dry_run:
        return

    # Prepare output dirs
    train_out = args.output_root / "train"
    val_out = args.output_root / "val"
    ensure_empty_dir(args.output_root)
    train_out.mkdir(parents=True, exist_ok=True)
    val_out.mkdir(parents=True, exist_ok=True)

    # Link validation set (real only)
    for cls in classes:
        src_dir = args.real_val / cls
        dst_dir = val_out / cls
        dst_dir.mkdir(parents=True, exist_ok=True)
        for img in list_images(src_dir):
            symlink(img, dst_dir / img.name)

    # Link real training data + synthetic
    for cls in classes:
        cls_train_out = train_out / cls
        cls_train_out.mkdir(parents=True, exist_ok=True)

        # Real images
        for img in list_images(args.real_train / cls):
            symlink(img, cls_train_out / img.name)

        # Synthetic images
        syn_dir = args.synthetic_root / cls
        syn_images = list_images_recursive(syn_dir) if syn_dir.exists() else []

        if cls in majority_classes:
            k = min(args.majority_cap, len(syn_images))
            syn_selected = random.sample(syn_images, k) if k > 0 else []
        else:
            syn_selected = syn_images

        for img in syn_selected:
            symlink(img, cls_train_out / f"syn_{img.name}")

    print(f"Mixed dataset created at {args.output_root}")


if __name__ == "__main__":
    main()
