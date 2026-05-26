#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def collect_images(image_dir: Path) -> list[Path]:
    images = []
    for path in image_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            images.append(path)
    return sorted(images)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create metadata.jsonl for HF imagefolder datasets.")
    parser.add_argument("--image_dir", required=True, help="Directory containing images.")
    parser.add_argument("--caption", required=True, help="Caption text to use for every image.")
    parser.add_argument("--output", default=None, help="Optional output path for metadata.jsonl.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing metadata.jsonl.")
    args = parser.parse_args()

    image_dir = Path(args.image_dir).expanduser().resolve()
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    output_path = Path(args.output).expanduser().resolve() if args.output else image_dir / "metadata.jsonl"
    if output_path.exists() and not args.overwrite:
        print(f"metadata.jsonl already exists: {output_path}")
        return

    images = collect_images(image_dir)
    if not images:
        raise RuntimeError(f"No images found under {image_dir}")

    with output_path.open("w", encoding="utf-8") as f:
        for image_path in images:
            rel_path = image_path.relative_to(image_dir)
            record = {"file_name": str(rel_path), "text": args.caption}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(images)} captions to {output_path}")


if __name__ == "__main__":
    main()
