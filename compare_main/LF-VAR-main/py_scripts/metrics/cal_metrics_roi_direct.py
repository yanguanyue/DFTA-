import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import lpips
from PIL import Image
from torchvision import models, transforms

SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
IMAGENET_MEAN_RGB = (124, 116, 104)


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


def infer_base_name(stem):
    if "_" not in stem:
        return stem
    base, suffix = stem.rsplit("_", 1)
    if suffix.isdigit():
        return base
    return stem


def load_mask(mask_path, size):
    mask = Image.open(mask_path).convert("L")
    if mask.size != size:
        mask = mask.resize(size, Image.NEAREST)
    mask_array = np.array(mask)
    return mask_array > 0


def apply_mask(image, mask_array, fill_color=IMAGENET_MEAN_RGB):
    image_array = np.array(image)
    if image_array.ndim == 2:
        image_array = np.repeat(image_array[:, :, None], 3, axis=2)
    masked = image_array.copy()
    masked[~mask_array] = np.array(fill_color, dtype=masked.dtype)
    return Image.fromarray(masked)


def crop_by_mask(image, mask_array):
    ys, xs = np.where(mask_array)
    if len(xs) == 0 or len(ys) == 0:
        return image, mask_array
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    cropped_image = image.crop((x_min, y_min, x_max + 1, y_max + 1))
    cropped_mask = mask_array[y_min : y_max + 1, x_min : x_max + 1]
    return cropped_image, cropped_mask


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


def get_inception(device):
    model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
    model.eval().to(device)
    return model


def get_preprocess():
    return transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def compute_features(pairs, device, batch_size=32, use_mask=True, crop=True, fill_color=IMAGENET_MEAN_RGB):
    model = get_inception(device)
    preprocess = get_preprocess()

    features_gen = []
    features_real = []
    logits_gen = []

    # hook to capture avgpool features (2048-d)
    pool_features = []

    def hook_fn(_, __, output):
        pool_features.append(output)

    handle = model.avgpool.register_forward_hook(hook_fn)

    for i in range(0, len(pairs), batch_size):
        batch_pairs = pairs[i : i + batch_size]
        gen_imgs = []
        real_imgs = []
        for gen_path, real_path, mask_path in batch_pairs:
            gen_img = Image.open(gen_path).convert("RGB")
            real_img = Image.open(real_path).convert("RGB")
            if gen_img.size != real_img.size:
                gen_img = gen_img.resize(real_img.size, Image.LANCZOS)

            if use_mask and mask_path is not None:
                mask_array = load_mask(mask_path, real_img.size)
                if crop:
                    original_mask = mask_array
                    gen_img, mask_array = crop_by_mask(gen_img, original_mask)
                    real_img, _ = crop_by_mask(real_img, original_mask)
                gen_img = apply_mask(gen_img, mask_array, fill_color=fill_color)
                real_img = apply_mask(real_img, mask_array, fill_color=fill_color)

            gen_imgs.append(preprocess(gen_img))
            real_imgs.append(preprocess(real_img))

        gen_batch = torch.stack(gen_imgs).to(device)
        real_batch = torch.stack(real_imgs).to(device)

        pool_features.clear()
        with torch.no_grad():
            logits = model(gen_batch)
        gen_pool = pool_features[0]

        pool_features.clear()
        with torch.no_grad():
            _ = model(real_batch)
        real_pool = pool_features[0]

        logits_gen.append(logits.cpu())
    features_gen.append(gen_pool.squeeze(-1).squeeze(-1).cpu())
    features_real.append(real_pool.squeeze(-1).squeeze(-1).cpu())

    handle.remove()

    features_gen = torch.cat(features_gen).numpy()
    features_real = torch.cat(features_real).numpy()
    logits_gen = torch.cat(logits_gen).numpy()

    return features_gen, features_real, logits_gen


def calculate_kid(feats1, feats2, subset_size=1000, subsets=50, gamma=None):
    n1 = feats1.shape[0]
    n2 = feats2.shape[0]
    subset_size = min(subset_size, n1, n2)
    if subset_size < 2:
        return float("nan"), float("nan")

    rng = np.random.default_rng(1234)
    values = []
    for _ in range(subsets):
        inds1 = rng.choice(n1, subset_size, replace=False)
        inds2 = rng.choice(n2, subset_size, replace=False)
        x = feats1[inds1]
        y = feats2[inds2]
        # polynomial kernel
        if gamma is None:
            gamma = 1.0 / x.shape[1]
        k_xx = (gamma * x.dot(x.T) + 1) ** 3
        k_yy = (gamma * y.dot(y.T) + 1) ** 3
        k_xy = (gamma * x.dot(y.T) + 1) ** 3
        m = subset_size
        value = (
            (np.sum(k_xx) - np.trace(k_xx)) / (m * (m - 1))
            + (np.sum(k_yy) - np.trace(k_yy)) / (m * (m - 1))
            - 2 * np.mean(k_xy)
        )
        values.append(value)
    return float(np.mean(values)), float(np.std(values))


def calculate_is(logits, splits=10):
    probs = F.softmax(torch.from_numpy(logits), dim=1).numpy()
    split_scores = []
    n = probs.shape[0]
    splits = min(splits, n)
    for i in range(splits):
        part = probs[i * n // splits : (i + 1) * n // splits]
        py = np.mean(part, axis=0, keepdims=True)
        kl = part * (np.log(part + 1e-8) - np.log(py + 1e-8))
        kl_sum = np.sum(kl, axis=1)
        split_scores.append(np.exp(np.mean(kl_sum)))
    return float(np.mean(split_scores)), float(np.std(split_scores))


def get_lpips_model(device):
    model = lpips.LPIPS(net="alex")
    model.eval().to(device)
    return model


def get_lpips_preprocess():
    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def compute_lpips(pairs, device, batch_size=16, use_mask=True, crop=True, fill_color=IMAGENET_MEAN_RGB):
    model = get_lpips_model(device)
    preprocess = get_lpips_preprocess()

    values = []
    for i in range(0, len(pairs), batch_size):
        batch_pairs = pairs[i : i + batch_size]
        gen_imgs = []
        real_imgs = []
        for gen_path, real_path, mask_path in batch_pairs:
            gen_img = Image.open(gen_path).convert("RGB")
            real_img = Image.open(real_path).convert("RGB")
            if gen_img.size != real_img.size:
                gen_img = gen_img.resize(real_img.size, Image.LANCZOS)

            if use_mask and mask_path is not None:
                mask_array = load_mask(mask_path, real_img.size)
                if crop:
                    original_mask = mask_array
                    gen_img, mask_array = crop_by_mask(gen_img, original_mask)
                    real_img, _ = crop_by_mask(real_img, original_mask)
                gen_img = apply_mask(gen_img, mask_array, fill_color=fill_color)
                real_img = apply_mask(real_img, mask_array, fill_color=fill_color)

            gen_imgs.append(preprocess(gen_img))
            real_imgs.append(preprocess(real_img))

        gen_batch = torch.stack(gen_imgs).to(device)
        real_batch = torch.stack(real_imgs).to(device)

        with torch.no_grad():
            lpips_values = model(gen_batch, real_batch)
        values_batch = lpips_values.detach().cpu().numpy().reshape(-1)
        values.extend(values_batch.tolist())

    values = np.array(values, dtype=np.float32)
    return float(np.mean(values)), float(np.std(values))


def main():
    parser = argparse.ArgumentParser(description="Compute ROI KID/IS/LPIPS without saving masked images")
    parser.add_argument("--gen_dir", required=True, help="Generated images directory")
    parser.add_argument("--real_img_dir", required=True, help="Real images directory")
    parser.add_argument("--real_mask_dir", default=None, help="Real masks directory")
    parser.add_argument("--output", required=True, help="Output metrics json path")
    parser.add_argument("--mode", choices=["name", "index"], default="name")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gpu", action="store_true", default=True)
    parser.add_argument("--use_mask", action="store_true", default=True, help="Use real masks for ROI evaluation")
    parser.add_argument("--no_mask", action="store_true", default=False, help="Disable mask usage and evaluate full images")
    parser.add_argument("--no_crop", action="store_true", default=False, help="Disable mask-based cropping")
    parser.add_argument("--fill_color", type=int, nargs=3, default=list(IMAGENET_MEAN_RGB), help="Fill color outside mask (RGB)")

    args = parser.parse_args()

    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"

    gen_dir = Path(args.gen_dir)
    real_img_dir = Path(args.real_img_dir)
    real_mask_dir = Path(args.real_mask_dir) if args.real_mask_dir else None

    gen_paths = list_images(gen_dir)
    if not gen_paths:
        raise ValueError(f"No generated images found in {gen_dir}")
    if args.limit:
        gen_paths = gen_paths[: args.limit]

    use_mask = args.use_mask and not args.no_mask
    crop = not args.no_crop
    fill_color = tuple(args.fill_color)

    if use_mask:
        if real_mask_dir is None:
            raise ValueError("real_mask_dir is required when using masks")
        if args.mode == "name":
            pairs, missing = build_pairs_by_name(gen_paths, real_img_dir, real_mask_dir)
        else:
            pairs, missing = build_pairs_by_index(gen_paths, real_img_dir, real_mask_dir)
    else:
        real_images = list_images(real_img_dir)
        if not real_images:
            raise ValueError(f"No real images found in {real_img_dir}")
        min_count = min(len(gen_paths), len(real_images))
        pairs = [(gen_paths[idx], real_images[idx], None) for idx in range(min_count)]
        missing = len(gen_paths) - min_count

    if not pairs:
        raise ValueError("No valid image-mask pairs found for ROI evaluation.")

    feats_gen, feats_real, logits_gen = compute_features(
        pairs,
        device,
        batch_size=args.batch_size,
        use_mask=use_mask,
        crop=crop,
        fill_color=fill_color,
    )

    kid_mean, kid_std = calculate_kid(feats_gen, feats_real)
    is_mean, is_std = calculate_is(logits_gen)
    lpips_mean, lpips_std = compute_lpips(
        pairs,
        device,
        batch_size=args.batch_size,
        use_mask=use_mask,
        crop=crop,
        fill_color=fill_color,
    )

    report = {
        "gen_dir": str(gen_dir),
        "real_img_dir": str(real_img_dir),
        "real_mask_dir": str(real_mask_dir),
        "mode": args.mode,
    "use_mask": bool(use_mask),
    "crop": bool(crop),
    "fill_color": list(fill_color),
        "total_generated": len(gen_paths),
        "limit": args.limit,
        "paired": len(pairs),
        "missing": int(missing),
        "metrics": {
            "inception_score_mean": is_mean,
            "inception_score_std": is_std,
            "kernel_inception_distance_mean": kid_mean,
            "kernel_inception_distance_std": kid_std,
            "lpips_mean": lpips_mean,
            "lpips_std": lpips_std,
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
