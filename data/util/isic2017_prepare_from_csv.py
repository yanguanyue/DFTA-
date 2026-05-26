import argparse
import csv
from pathlib import Path

from PIL import Image

PROMPTS = {
    "mel": "An image of a skin area with melanoma.",
    "bkl": "An image of a skin area with benign keratosis-like lesions.",
    "nv": "An image of a skin area with melanocytic nevi.",
}


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def resolve_image_path(image_dir: Path, image_id: str) -> Path:
    for suffix in [".jpg", ".jpeg", ".png"]:
        candidate = image_dir / f"{image_id}{suffix}"
        if candidate.exists():
            return candidate
    return image_dir / f"{image_id}.jpg"


def resolve_seg_path(seg_dir: Path | None, image_id: str) -> Path | None:
    if not seg_dir:
        return None
    candidate = seg_dir / f"{image_id}_segmentation.png"
    return candidate if candidate.exists() else None


def get_class_from_labels(melanoma: float, seborrheic_keratosis: float) -> str | None:
    if melanoma == 1 and seborrheic_keratosis == 0:
        return "mel"
    if melanoma == 0 and seborrheic_keratosis == 1:
        return "bkl"
    if melanoma == 0 and seborrheic_keratosis == 0:
        return "nv"
    return None


def main():
    parser = argparse.ArgumentParser(description="Prepare ISIC2017 dataset from ground-truth CSV")
    parser.add_argument("--image_dir", required=True, help="Directory with ISIC2017 training images")
    parser.add_argument("--seg_dir", default="", help="Directory with ISIC2017 training segmentations")
    parser.add_argument("--groundtruth_csv", required=True, help="ISIC2017 training ground truth CSV")
    parser.add_argument("--output", default="data/ISIC2017/input", help="Output dataset root")
    parser.add_argument("--split", default="all", help="Dataset split name (train/val/test/all)")
    parser.add_argument(
        "--append_metadata",
        action="store_true",
        help="Append to existing metadata.csv instead of overwriting",
    )
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    seg_dir = Path(args.seg_dir) if args.seg_dir else None
    csv_path = Path(args.groundtruth_csv)
    output_root = Path(args.output)

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if seg_dir and not seg_dir.exists():
        print(f"[WARN] Segmentation directory not found: {seg_dir}")
        seg_dir = None
    if not csv_path.exists():
        raise FileNotFoundError(f"Ground truth CSV not found: {csv_path}")

    split_name = args.split or "all"
    metadata_rows = []
    missing_images = 0
    missing_segs = 0

    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or []
        if "image_id" not in fieldnames:
            raise ValueError("Ground truth CSV must include 'image_id' column")

        for row in reader:
            image_id = row.get("image_id")
            if not image_id:
                continue
            try:
                melanoma = float(row.get("melanoma", 0))
            except ValueError:
                melanoma = 0.0
            try:
                seborrheic = float(row.get("seborrheic_keratosis", 0))
            except ValueError:
                seborrheic = 0.0

            class_key = get_class_from_labels(melanoma, seborrheic)
            if not class_key:
                continue

            src_path = resolve_image_path(image_dir, image_id)
            if not src_path.exists():
                missing_images += 1
                if missing_images <= 10:
                    print(f"[WARN] Image not found: {src_path}")
                continue

            if split_name in {"train", "val", "test"}:
                img_out_dir = output_root / split_name / "HuggingFace" / class_key / "ISIC2017_img_class"
                seg_out_dir = output_root / split_name / "HuggingFace" / class_key / "ISIC2017_seg_class"
            else:
                img_out_dir = output_root / "img_class" / class_key
                seg_out_dir = output_root / "seg_class" / class_key
            ensure_dir(img_out_dir)
            ensure_dir(seg_out_dir)

            out_path = img_out_dir / f"{image_id}.jpg"

            with Image.open(src_path) as img:
                img = img.convert("RGB")
                img.save(out_path, format="JPEG", quality=95)

            seg_path = resolve_seg_path(seg_dir, image_id)
            seg_out_path = ""
            if seg_path:
                seg_out_path = seg_out_dir / f"{image_id}_segmentation.png"
                with Image.open(seg_path) as msk:
                    msk = msk.convert("L")
                    msk.save(seg_out_path)
            elif seg_dir:
                missing_segs += 1
                if missing_segs <= 10:
                    print(f"[WARN] Segmentation not found: {seg_dir}/{image_id}_segmentation.png")

            metadata_rows.append(
                [
                    str(out_path),
                    str(seg_out_path) if seg_out_path else "",
                    class_key,
                    PROMPTS[class_key],
                    split_name,
                ]
            )

    metadata_path = output_root / "metadata.csv"
    ensure_dir(metadata_path.parent)
    write_header = True
    if args.append_metadata and metadata_path.exists():
        write_header = False
    write_mode = "a" if args.append_metadata else "w"
    with metadata_path.open(write_mode, newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        if write_header:
            writer.writerow(["img_path", "seg_path", "class", "prompt", "dataset_split"])
        writer.writerows(metadata_rows)

    print(f"Saved metadata to {metadata_path}")
    if missing_images:
        print(f"[WARN] Missing images: {missing_images}")
    if missing_segs:
        print(f"[WARN] Missing segmentations: {missing_segs}")


if __name__ == "__main__":
    main()
