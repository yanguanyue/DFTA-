import argparse
import csv
import re
from pathlib import Path

from PIL import Image

PROMPTS = {
    "mel": "An image of a skin area with melanoma.",
    "nv": "An image of a skin area with melanocytic nevi.",
}


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def resolve_image_path(image_dir: Path, image_id: str) -> Path:
    for suffix in [".jpg", ".jpeg", ".png", ".bmp"]:
        candidate = image_dir / f"{image_id}{suffix}"
        if candidate.exists():
            return candidate
    return image_dir / f"{image_id}.jpg"


def resolve_seg_path(seg_dir: Path | None, image_id: str) -> Path | None:
    if not seg_dir:
        return None
    candidate = seg_dir / f"{image_id}_segmentation.png"
    return candidate if candidate.exists() else None


def parse_ph2_dataset(file_path: Path):
    data = []
    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        next(f, None)
        for line in f:
            if not line.strip() or "||" not in line:
                continue
            parts = [part.strip() for part in line.split("||")]
            image_id = ""
            if len(parts) > 1:
                image_id = parts[1].strip()
            elif parts:
                image_id = parts[0].strip()

            image_id = re.sub(r"(IMD)(\d+)", r"\1_\2", image_id)

            diagnosis = None
            hist_diag = parts[2].strip() if len(parts) > 2 else ""
            clinical_diag = parts[3].strip() if len(parts) > 3 else ""

            if hist_diag.isdigit():
                diagnosis = int(hist_diag)
            elif clinical_diag.isdigit():
                diagnosis = int(clinical_diag)

            if image_id and diagnosis is not None:
                data.append({"image_id": image_id, "diagnosis": diagnosis})
    return data


def get_class_from_diagnosis(diagnosis: int) -> str | None:
    if diagnosis in {0, 1}:
        return "nv"
    if diagnosis == 2:
        return "mel"
    return None


def main():
    parser = argparse.ArgumentParser(description="Prepare PH2 dataset from PH2_dataset.txt")
    parser.add_argument("--image_dir", required=True, help="Directory with PH2 images")
    parser.add_argument("--seg_dir", default="", help="Directory with PH2 segmentations")
    parser.add_argument("--groundtruth_txt", required=True, help="PH2_dataset.txt path")
    parser.add_argument("--output", default="data/PH2/input", help="Output dataset root")
    parser.add_argument("--split", default="all", help="Dataset split name (train/val/test/all)")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    seg_dir = Path(args.seg_dir) if args.seg_dir else None
    txt_path = Path(args.groundtruth_txt)
    output_root = Path(args.output)

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if seg_dir and not seg_dir.exists():
        print(f"[WARN] Segmentation directory not found: {seg_dir}")
        seg_dir = None
    if not txt_path.exists():
        raise FileNotFoundError(f"PH2_dataset.txt not found: {txt_path}")

    split_name = args.split or "all"
    metadata_rows = []
    missing_images = 0
    missing_segs = 0

    for row in parse_ph2_dataset(txt_path):
        image_id = row["image_id"]
        diagnosis = row["diagnosis"]
        class_key = get_class_from_diagnosis(diagnosis)
        if not class_key:
            continue

        src_path = resolve_image_path(image_dir, image_id)
        if not src_path.exists():
            missing_images += 1
            if missing_images <= 10:
                print(f"[WARN] Image not found: {src_path}")
            continue

        if split_name in {"train", "val", "test"}:
            img_out_dir = output_root / split_name / "HuggingFace" / class_key / "PH2_img_class"
            seg_out_dir = output_root / split_name / "HuggingFace" / class_key / "PH2_seg_class"
        else:
            img_out_dir = output_root / "img_class" / class_key
            seg_out_dir = output_root / "seg_class" / class_key
        ensure_dir(img_out_dir)
        ensure_dir(seg_out_dir)

        out_path = img_out_dir / f"{image_id}.jpg"
        with Image.open(src_path) as img:
            img = img.convert("RGB")
            img.save(out_path, format="JPEG", quality=95)

        seg_out_path = ""
        seg_path = resolve_seg_path(seg_dir, image_id)
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
    with metadata_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["img_path", "seg_path", "class", "prompt", "dataset_split"])
        writer.writerows(metadata_rows)

    print(f"Saved metadata to {metadata_path}")
    if missing_images:
        print(f"[WARN] Missing images: {missing_images}")
    if missing_segs:
        print(f"[WARN] Missing segmentations: {missing_segs}")


if __name__ == "__main__":
    main()
