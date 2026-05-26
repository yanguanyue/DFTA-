#!/usr/bin/env python3
import argparse
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(folder: Path) -> List[Path]:
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


def ensure_empty_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src, dst)


def build_real_index(image_dir: Path, mask_dir: Path) -> List[Tuple[Path, Path]]:
    image_index: Dict[str, Path] = {img.stem: img for img in list_images(image_dir)}
    pairs: List[Tuple[Path, Path]] = []
    for mask in list_images(mask_dir):
        stem = mask.stem
        if stem.endswith("_segmentation"):
            stem = stem.replace("_segmentation", "")
        image = image_index.get(stem)
        if image is not None:
            pairs.append((image, mask))
    return pairs


def _resolve_syn_dirs(syn_root: Path) -> Tuple[Path | None, Path | None]:
    candidates = [
        (syn_root / "images", syn_root / "masks"),
        (syn_root / "image", syn_root / "mask"),
    ]
    for images_dir, masks_dir in candidates:
        if images_dir.exists() and masks_dir.exists():
            return images_dir, masks_dir
    return None, None


def build_synthetic_pairs(syn_root: Path) -> List[Tuple[Path, Path]]:
    images_dir, masks_dir = _resolve_syn_dirs(syn_root)
    if images_dir is None or masks_dir is None:
        return []
    mask_index = {p.name: p for p in list_images(masks_dir)}
    pairs = []
    for img in list_images(images_dir):
        mask = mask_index.get(img.name)
        if mask is not None:
            pairs.append((img, mask))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Mix synthetic segmentation images into training set")
    parser.add_argument("--real-root", required=True, type=Path, help="/data/HAM10000/input")
    parser.add_argument("--synthetic-root", required=True, type=Path, help="/data/output/Siamese")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--minority-cap", type=int, default=1500)
    parser.add_argument("--majority-cap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)

    real_train_images = args.real_root / "train" / "HAM10000_img_class"
    real_train_masks = args.real_root / "train" / "HAM10000_seg_class"
    real_val_images = args.real_root / "val" / "HAM10000_img_class"
    real_val_masks = args.real_root / "val" / "HAM10000_seg_class"

    classes = sorted([p.name for p in real_train_images.iterdir() if p.is_dir()])

    real_counts = {}
    for cls in classes:
        real_pairs = build_real_index(real_train_images / cls, real_train_masks / cls)
        real_counts[cls] = len(real_pairs)

    sorted_counts = sorted(real_counts.values())
    median = sorted_counts[len(sorted_counts) // 2]
    majority_classes = {c for c, n in real_counts.items() if n > median}
    minority_classes = set(classes) - majority_classes

    print("Real class counts:", real_counts)
    print("Median:", median)
    print("Majority classes:", sorted(majority_classes))
    print("Minority classes:", sorted(minority_classes))

    if args.dry_run:
        return

    ensure_empty_dir(args.output_root)
    train_images_out = args.output_root / "train" / "images"
    train_masks_out = args.output_root / "train" / "masks"
    val_images_out = args.output_root / "val" / "images"
    val_masks_out = args.output_root / "val" / "masks"

    # Link validation set (real only)
    for cls in classes:
        val_img_dir = real_val_images / cls
        val_mask_dir = real_val_masks / cls
        pairs = build_real_index(val_img_dir, val_mask_dir)
        for img, mask in pairs:
            symlink(img, val_images_out / cls / img.name)
            symlink(mask, val_masks_out / cls / mask.name)

    # Link real training data + synthetic
    for cls in classes:
        real_pairs = build_real_index(real_train_images / cls, real_train_masks / cls)
        for img, mask in real_pairs:
            symlink(img, train_images_out / cls / img.name)
            symlink(mask, train_masks_out / cls / mask.name)

        syn_pairs = build_synthetic_pairs(args.synthetic_root / cls)
        if cls in majority_classes:
            cap = args.majority_cap
        else:
            cap = args.minority_cap
        if cap > 0 and len(syn_pairs) > 0:
            syn_selected = random.sample(syn_pairs, min(cap, len(syn_pairs)))
        else:
            syn_selected = []
        for img, mask in syn_selected:
            symlink(img, train_images_out / cls / f"syn_{img.name}")
            symlink(mask, train_masks_out / cls / f"syn_{mask.name}")

    print(f"Mixed dataset created at {args.output_root}")


if __name__ == "__main__":
    main()
