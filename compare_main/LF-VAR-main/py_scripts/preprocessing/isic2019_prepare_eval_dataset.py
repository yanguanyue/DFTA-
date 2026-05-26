import argparse
import csv
import os
import random
from pathlib import Path

from PIL import Image

CLASS_MAP = {
    "squamous cell carcinoma": "akiec",
    "basal cell carcinoma": "bcc",
    "pigmented benign keratosis": "bkl",
    "dermatofibroma": "df",
    "melanoma": "mel",
    "nevus": "nv",
    "vascular lesion": "vasc",
}

PROMPTS = {
    "akiec": "An image of a skin area with actinic keratoses or intraepithelial carcinoma.",
    "bcc": "An image of a skin area with basal cell carcinoma.",
    "bkl": "An image of a skin area with benign keratosis-like lesions.",
    "df": "An image of a skin area with dermatofibroma.",
    "mel": "An image of a skin area with melanoma.",
    "nv": "An image of a skin area with melanocytic nevi.",
    "vasc": "An image of a skin area with a vascular lesion.",
}


def iter_images(folder):
    for path in folder.iterdir():
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            yield path


def split_files(files, val_ratio, test_ratio, seed):
    rng = random.Random(seed)
    files = list(files)
    rng.shuffle(files)
    total = len(files)
    val_count = int(total * val_ratio)
    test_count = int(total * test_ratio)
    val_files = files[:val_count]
    test_files = files[val_count : val_count + test_count]
    train_files = files[val_count + test_count :]
    return train_files, val_files, test_files


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Prepare ISIC2019 evaluation dataset for main/Derm-T2IM")
    parser.add_argument("--root", default="data/local/ISIC2019", help="ISIC2019 root directory")
    parser.add_argument("--output", default="data/local/ISIC2019/input", help="Output dataset root")
    parser.add_argument("--img_dir_name", default="ISIC2019_img_class")
    parser.add_argument("--seg_dir_name", default="ISIC2019_seg_class")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--test_ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.root)
    src_img_root = root / "new_ei"
    src_mask_root = root / "mask"
    output_root = Path(args.output)

    if not src_img_root.exists():
        raise FileNotFoundError(f"Image root not found: {src_img_root}")
    if not src_mask_root.exists():
        raise FileNotFoundError(f"Mask root not found: {src_mask_root}")

    splits = {
        "train": {},
        "val": {},
        "test": {},
    }

    for src_folder, class_key in CLASS_MAP.items():
        class_folder = src_img_root / src_folder
        if not class_folder.exists():
            print(f"[WARN] class folder not found: {class_folder}")
            continue
        files = list(iter_images(class_folder))
        train_files, val_files, test_files = split_files(files, args.val_ratio, args.test_ratio, args.seed)
        splits["train"][class_key] = train_files
        splits["val"][class_key] = val_files
        splits["test"][class_key] = test_files

    metadata_rows = []

    for split_name, class_files in splits.items():
        if split_name == "test" and args.test_ratio == 0:
            continue
        for class_key, files in class_files.items():
            img_out_dir = output_root / split_name / "HuggingFace" / class_key / args.img_dir_name
            seg_out_dir = output_root / split_name / "HuggingFace" / class_key / args.seg_dir_name
            ensure_dir(img_out_dir)
            ensure_dir(seg_out_dir)

            for img_path in files:
                base_name = img_path.stem
                mask_path = src_mask_root / f"{base_name}.png"
                if not mask_path.exists():
                    print(f"[WARN] mask not found for {img_path.name}")
                    continue

                img_out_path = img_out_dir / f"{base_name}.jpg"
                seg_out_path = seg_out_dir / f"{base_name}_segmentation.png"

                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    img.save(img_out_path, format="JPEG", quality=95)

                with Image.open(mask_path) as msk:
                    msk = msk.convert("L")
                    msk.save(seg_out_path)

                metadata_rows.append(
                    [
                        str(img_out_path),
                        str(seg_out_path),
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


if __name__ == "__main__":
    main()
