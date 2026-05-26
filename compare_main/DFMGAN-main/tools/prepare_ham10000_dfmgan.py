import json
import os
from pathlib import Path
import sys

IMG_ROOT = Path("/root/autodl-tmp/data/HAM10000/input/train/HAM10000_img_class")
MASK_ROOT = Path("/root/autodl-tmp/data/HAM10000/input/train/HAM10000_seg_class")
LABELED_IMG_ROOT = Path("/root/autodl-tmp/data/HAM10000/input/train/HAM10000_img_class_labeled")
FILTERED_IMG_ROOT = Path("/root/autodl-tmp/data/HAM10000/input/train/HAM10000_img_class_masked")
FLAT_MASK_ROOT = Path("/root/autodl-tmp/data/HAM10000/input/train/HAM10000_seg_class_flat")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_mask(mask_dir: Path, stem: str) -> Path | None:
    candidates = [
        mask_dir / f"{stem}_segmentation.png",
        mask_dir / f"{stem}_segmentation.jpg",
        mask_dir / f"{stem}_segmentation.jpeg",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def ensure_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.symlink(src, dst)
    except FileExistsError:
        return
    except OSError:
        # Fallback to hard link or copy if symlink not supported
        try:
            os.link(src, dst)
        except OSError:
            import shutil
            shutil.copy2(src, dst)


def main() -> int:
    if not IMG_ROOT.exists() or not MASK_ROOT.exists():
        print("Image or mask root not found.")
        return 1

    classes = sorted([p.name for p in IMG_ROOT.iterdir() if p.is_dir()])
    if not classes:
        print("No class folders found.")
        return 1

    label_map = {name: idx for idx, name in enumerate(classes)}
    labels = []
    labeled_labels = []
    kept = 0
    missing = 0

    for class_name in classes:
        img_dir = IMG_ROOT / class_name
        mask_dir = MASK_ROOT / class_name
        if not img_dir.exists():
            continue

        for img_path in sorted(img_dir.rglob("*")):
            if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTS:
                continue

            stem = img_path.stem
            rel_img_path = Path(class_name) / img_path.name

            # Labeled images for base training (image-only)
            labeled_dst = LABELED_IMG_ROOT / rel_img_path
            ensure_symlink(img_path, labeled_dst)
            labeled_labels.append([rel_img_path.as_posix(), label_map[class_name]])

            mask_path = find_mask(mask_dir, stem) if mask_dir.exists() else None
            if mask_path is None:
                missing += 1
                continue

            dst_img = FILTERED_IMG_ROOT / rel_img_path
            ensure_symlink(img_path, dst_img)

            # create flat mask symlink with expected naming: *_mask.<img_ext>
            mask_dst_name = f"{stem}_mask{img_path.suffix.lower()}"
            dst_mask = FLAT_MASK_ROOT / mask_dst_name
            ensure_symlink(mask_path, dst_mask)

            labels.append([rel_img_path.as_posix(), label_map[class_name]])
            kept += 1

    FILTERED_IMG_ROOT.mkdir(parents=True, exist_ok=True)
    dataset_json = FILTERED_IMG_ROOT / "dataset.json"
    with dataset_json.open("w", encoding="utf-8") as f:
        json.dump({"labels": labels}, f)

    LABELED_IMG_ROOT.mkdir(parents=True, exist_ok=True)
    labeled_json = LABELED_IMG_ROOT / "dataset.json"
    with labeled_json.open("w", encoding="utf-8") as f:
        json.dump({"labels": labeled_labels}, f)

    print(f"Classes: {len(classes)}")
    print(f"Kept images with masks: {kept}")
    print(f"Missing masks skipped: {missing}")
    print(f"Labeled image root: {LABELED_IMG_ROOT}")
    print(f"Filtered image root: {FILTERED_IMG_ROOT}")
    print(f"Flat mask root: {FLAT_MASK_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
