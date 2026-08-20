#!/usr/bin/env python3
import csv
import json
import os
import shutil
from pathlib import Path

MODELS = [
    "ArSDM", "Controlnet", "DFMGAN", "Derm-T2IM", "DreamBooth",
    "LesionGen", "Siamese", "T2I-Adapter", "flow",
]
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
SOURCE = Path("/root/autodl-tmp/output/generate")
OUT = Path("/root/autodl-tmp/output/generate_evaluation")
NORMALIZED = OUT / "normalized"


def image_files(folder):
    if folder is None or not folder.is_dir():
        return []
    return sorted(
        (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTS),
        key=lambda p: p.name,
    )


def choose_folder(class_root, names, allow_direct=False):
    for name in names:
        candidate = class_root / name
        if image_files(candidate):
            return candidate
    if allow_direct and image_files(class_root):
        return class_root
    return None


def link(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(source, target)


if NORMALIZED.exists():
    shutil.rmtree(NORMALIZED)
NORMALIZED.mkdir(parents=True)
rows = []

for model in MODELS:
    model_root = SOURCE / model
    if not model_root.is_dir():
        raise FileNotFoundError(model_root)
    for cls in CLASSES:
        class_root = model_root / cls
        image_dir = choose_folder(class_root, ["images", "image"], allow_direct=True)
        mask_dir = choose_folder(class_root, ["masks", "mask"])
        images = image_files(image_dir)
        masks = image_files(mask_dir)
        paired = min(len(images), len(masks)) if masks else 0
        image_out = NORMALIZED / model / cls / "images"
        mask_out = NORMALIZED / model / cls / "masks"
        for index, source in enumerate(images):
            link(source, image_out / f"sample_{index:06d}.png")
        for index, source in enumerate(masks[:paired]):
            link(source, mask_out / f"sample_{index:06d}.png")
        rows.append({
            "model": model,
            "class": cls,
            "source_images": len(images),
            "source_masks": len(masks),
            "paired": paired,
            "normalized_images": len(images),
            "normalized_masks": paired,
            "status": "ok" if images and (not masks or paired == len(images)) else "warning",
        })

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "input_report.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
with (OUT / "input_report.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps(rows, indent=2))
