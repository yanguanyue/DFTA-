import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from torch_fidelity import calculate_metrics


SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")


def list_images(directory):
    paths = []
    for ext in SUPPORTED_EXTS:
        paths.extend(Path(directory).glob(f"*{ext}"))
        paths.extend(Path(directory).glob(f"*{ext.upper()}"))
    return sorted(set(paths))


def find_real_image(real_dir, base_name):
    for ext in SUPPORTED_EXTS:
        candidate = Path(real_dir) / f"{base_name}{ext}"
        if candidate.exists():
            return candidate
        candidate_upper = Path(real_dir) / f"{base_name}{ext.upper()}"
        if candidate_upper.exists():
            return candidate_upper
    return None


def load_mask(mask_path, size):
    mask = Image.open(mask_path).convert("L")
    if mask.size != size:
        mask = mask.resize(size, Image.NEAREST)
    mask_array = np.array(mask)
    return mask_array > 0


def apply_mask(image, mask_array):
    image_array = np.array(image)
    if image_array.ndim == 2:
        image_array = np.repeat(image_array[:, :, None], 3, axis=2)
    masked = image_array.copy()
    masked[~mask_array] = 0
    return Image.fromarray(masked)


def infer_base_name(stem):
    if "_" not in stem:
        return stem
    base, suffix = stem.rsplit("_", 1)
    if suffix.isdigit():
        return base
    return stem


def build_pairs_by_name(gen_paths, real_img_dir, real_mask_dir):
    pairs = []
    missing = 0
    for gen_path in gen_paths:
        base_name = infer_base_name(gen_path.stem)
        real_img = find_real_image(real_img_dir, base_name)
        mask_path = Path(real_mask_dir) / f"{base_name}_segmentation.png"
        if real_img is None or not mask_path.exists():
            missing += 1
            continue
        pairs.append((gen_path, real_img, mask_path))
    return pairs, missing


def build_pairs_by_index(gen_paths, real_img_dir, real_mask_dir):
    real_images = list_images(real_img_dir)
    if not real_images:
        raise ValueError(f"No real images found in {real_img_dir}")

    real_pairs = []
    for img_path in real_images:
        base_name = img_path.stem
        mask_path = Path(real_mask_dir) / f"{base_name}_segmentation.png"
        if mask_path.exists():
            real_pairs.append((img_path, mask_path))

    if not real_pairs:
        raise ValueError(f"No matching masks found in {real_mask_dir}")

    min_count = min(len(gen_paths), len(real_pairs))
    pairs = []
    for idx in range(min_count):
        gen_path = gen_paths[idx]
        real_img, mask_path = real_pairs[idx]
        pairs.append((gen_path, real_img, mask_path))

    return pairs, len(gen_paths) - min_count


def prepare_masked_sets(pairs, gen_out_dir, real_out_dir):
    gen_out_dir.mkdir(parents=True, exist_ok=True)
    real_out_dir.mkdir(parents=True, exist_ok=True)

    for idx, (gen_path, real_path, mask_path) in enumerate(pairs):
        gen_img = Image.open(gen_path).convert("RGB")
        real_img = Image.open(real_path).convert("RGB")

        mask_array = load_mask(mask_path, real_img.size)
        if gen_img.size != real_img.size:
            gen_img = gen_img.resize(real_img.size, Image.LANCZOS)

        masked_real = apply_mask(real_img, mask_array)
        masked_gen = apply_mask(gen_img, mask_array)

        masked_real.save(real_out_dir / f"{idx:05d}.png")
        masked_gen.save(gen_out_dir / f"{idx:05d}.png")


def main():
    parser = argparse.ArgumentParser(description="Compute ROI FID/KID/IS using lesion masks")
    parser.add_argument("--gen_dir", required=True, help="Generated images directory")
    parser.add_argument("--real_img_dir", required=True, help="Real images directory")
    parser.add_argument("--real_mask_dir", required=True, help="Real masks directory")
    parser.add_argument("--output", required=True, help="Output metrics json path")
    parser.add_argument("--cache_dir", default=None, help="Cache dir for masked images")
    parser.add_argument("--mode", choices=["name", "index"], default="name")
    parser.add_argument("--gpu", action="store_true", default=True)

    args = parser.parse_args()

    gen_dir = Path(args.gen_dir)
    real_img_dir = Path(args.real_img_dir)
    real_mask_dir = Path(args.real_mask_dir)

    gen_paths = list_images(gen_dir)
    if not gen_paths:
        raise ValueError(f"No generated images found in {gen_dir}")

    if args.mode == "name":
        pairs, missing = build_pairs_by_name(gen_paths, real_img_dir, real_mask_dir)
    else:
        pairs, missing = build_pairs_by_index(gen_paths, real_img_dir, real_mask_dir)

    if not pairs:
        raise ValueError("No valid image-mask pairs found for ROI evaluation.")

    cache_root = Path(args.cache_dir) if args.cache_dir else Path(args.output).parent / "roi_cache"
    gen_cache = cache_root / "generated"
    real_cache = cache_root / "real"

    prepare_masked_sets(pairs, gen_cache, real_cache)

    metrics = calculate_metrics(
        input1=str(gen_cache),
        input2=str(real_cache),
        cuda=args.gpu,
        fid=True,
        kid=True,
        isc=True,
    )

    report = {
        "gen_dir": str(gen_dir),
        "real_img_dir": str(real_img_dir),
        "real_mask_dir": str(real_mask_dir),
        "mode": args.mode,
        "total_generated": len(gen_paths),
        "paired": len(pairs),
        "missing": int(missing),
        "metrics": metrics,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()