import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import models

SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
IMAGENET_MEAN_RGB = (124, 116, 104)


def list_images(directory):
    paths = []
    for ext in SUPPORTED_EXTS:
        paths.extend(Path(directory).glob(f"*{ext}"))
        paths.extend(Path(directory).glob(f"*{ext.upper()}"))
    return sorted(set(paths))


 

def get_inception(device):
    model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
    model.eval().to(device)
    return model


def pil_to_tensor_no_numpy(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    width, height = image.size
    data = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
    tensor = data.view(height, width, 3).permute(2, 0, 1).float() / 255.0
    return tensor


def get_preprocess():
    def _prep(image):
        image = image.resize((299, 299), Image.BILINEAR)
        tensor = pil_to_tensor_no_numpy(image)
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        return (tensor - mean) / std
    return _prep


def compute_features(
    image_paths,
    device,
    batch_size=32,
    force_size=None,
    model=None,
    preprocess=None,
):
    if model is None:
        model = get_inception(device)
    if preprocess is None:
        preprocess = get_preprocess()

    features = []

    pool_features = []

    def hook_fn(_, __, output):
        pool_features.append(output)

    handle = model.avgpool.register_forward_hook(hook_fn)

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        images = []
        for image_path in batch_paths:
            image = Image.open(image_path).convert("RGB")
            if force_size:
                image = image.resize((force_size, force_size), Image.LANCZOS)
            images.append(preprocess(image))

        batch = torch.stack(images).to(device)

        pool_features.clear()
        with torch.no_grad():
            _ = model(batch)
        pool = pool_features[0]
        features.append(pool.squeeze(-1).squeeze(-1).cpu())

    handle.remove()

    if not features:
        return torch.empty((0, 0))

    return torch.cat(features)


def calculate_kid(feats_gen, feats_real, gamma=None, kernel_batch_size=1024):
    """Full unbiased polynomial-kernel MMD^2 using unequal sample counts."""
    n_gen = feats_gen.shape[0]
    n_real = feats_real.shape[0]
    if n_gen < 2 or n_real < 2:
        return float("nan")
    gen = feats_gen.to(dtype=torch.float64, device="cpu")
    real = feats_real.to(dtype=torch.float64, device="cpu")
    if gamma is None:
        gamma = 1.0 / gen.shape[1]

    def kernel_sum(left, right, exclude_diagonal=False):
        total = torch.zeros((), dtype=torch.float64)
        same = left.data_ptr() == right.data_ptr() and left.shape == right.shape
        for start in range(0, left.shape[0], kernel_batch_size):
            chunk = left[start : start + kernel_batch_size]
            values = (gamma * (chunk @ right.T) + 1.0) ** 3
            total += values.sum()
            if exclude_diagonal and same:
                rows = torch.arange(values.shape[0])
                cols = torch.arange(start, start + values.shape[0])
                total -= values[rows, cols].sum()
        return total

    k_gen = kernel_sum(gen, gen, exclude_diagonal=True) / (n_gen * (n_gen - 1))
    k_real = kernel_sum(real, real, exclude_diagonal=True) / (n_real * (n_real - 1))
    k_cross = kernel_sum(gen, real) / (n_gen * n_real)
    return float((k_gen + k_real - 2.0 * k_cross).item())


def _kth_nn_distance(features: torch.Tensor, k: int, batch_size: int) -> torch.Tensor:
    n = features.shape[0]
    if n <= k:
        raise ValueError(f"Need at least {k + 1} samples for D&C, got {n}")
    distances_k = []
    for i in range(0, n, batch_size):
        chunk = features[i : i + batch_size]
        dist = torch.cdist(chunk, features, p=2)
        vals, _ = torch.topk(dist, k + 1, largest=False)
        distances_k.append(vals[:, -1])
    return torch.cat(distances_k, dim=0)


def calculate_density_coverage(
    feats_gen: torch.Tensor,
    feats_real: torch.Tensor,
    k: int = 5,
    batch_size: int = 1024,
    device: str | torch.device = "cpu",
) -> tuple[float, float]:
    if feats_gen.numel() == 0 or feats_real.numel() == 0:
        return float("nan"), float("nan")

    real = feats_real.to(device)
    fake = feats_gen.to(device)

    real_kth = _kth_nn_distance(real, k=k, batch_size=batch_size)

    counts = torch.zeros(fake.shape[0], device=device)
    coverage_hits = torch.zeros(real.shape[0], dtype=torch.bool, device=device)

    for i in range(0, real.shape[0], batch_size):
        real_chunk = real[i : i + batch_size]
        real_kth_chunk = real_kth[i : i + batch_size]
        dist = torch.cdist(real_chunk, fake, p=2)
        within = dist < real_kth_chunk[:, None]
        counts += within.sum(dim=0)
        min_dist, _ = dist.min(dim=1)
        coverage_hits[i : i + batch_size] = min_dist < real_kth_chunk

    density = (counts / k).mean().item()
    coverage = coverage_hits.float().mean().item()
    return float(density), float(coverage)


def resolve_real_dirs(real_root, class_key, split=None):
    candidates = []
    if split:
        candidates.append(Path(real_root) / split)
    else:
        candidates.extend([Path(real_root) / "train", Path(real_root) / "val", Path(real_root)])

    for base in candidates:
        img_dir = base / "HAM10000_img_class" / class_key
        if img_dir.exists():
            return img_dir
    return None


def build_output_paths(output_dir: Path, model_name: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"metrics_{model_name}.json"
    xlsx_path = output_dir / f"metrics_{model_name}.xlsx"
    return json_path, xlsx_path


def write_excel(report: dict, xlsx_path: Path):
    try:
        from openpyxl import Workbook
    except Exception as exc:
        raise RuntimeError("openpyxl is required to write Excel files") from exc

    def _to_scalar(value):
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, np.ndarray):
            if value.size == 1:
                return value.item()
            return ",".join([str(v) for v in value.tolist()])
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.detach().cpu().item()
            return ",".join([str(v) for v in value.detach().cpu().tolist()])
        return value

    headers = ["class", "kid_mean", "density_inception", "coverage_inception"]
    wb = Workbook()
    ws = wb.active
    ws.append(headers)

    for class_name, info in report.get("classes", {}).items():
        if not isinstance(info, dict) or "metrics" not in info:
            continue
        metrics = info.get("metrics", {})
        ws.append([
            class_name,
            _to_scalar(metrics.get("kid_mean")),
            _to_scalar(metrics.get("density_inception")),
            _to_scalar(metrics.get("coverage_inception")),
        ])

    summary = report.get("summary", {})
    ws.append([
        "summary",
        _to_scalar(summary.get("kid_mean")),
        _to_scalar(summary.get("density_inception")),
        _to_scalar(summary.get("coverage_inception")),
    ])

    wb.save(xlsx_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute KID and density/coverage (Inception)")
    parser.add_argument("--gen_root", required=True, help="Root directory with generated class folders")
    parser.add_argument("--real_root", default="/root/autodl-tmp/data/HAM10000/input", help="Root of HAM10000 input data")
    parser.add_argument("--real_split", default="val", help="Optional split name under real_root (train/val)")
    parser.add_argument("--output_dir", default="/root/autodl-tmp/output/metric", help="Directory for output json/xlsx")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--class_list", type=str, default=None, help="Comma-separated class folder names to evaluate")
    parser.add_argument("--force_size", type=int, default=None, help="Force resize both real/gen to a square size before masking")
    parser.add_argument("--gpu", action="store_true", default=True)
    parser.add_argument("--dc_k", type=int, default=5, help="k for Density & Coverage")
    parser.add_argument("--dc_batch_size", type=int, default=1024, help="Batch size for D&C distance computation")
    args = parser.parse_args()

    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"

    gen_root = Path(args.gen_root)
    if not gen_root.exists():
        raise ValueError(f"Generated root does not exist: {gen_root}")

    class_filter = None
    if args.class_list:
        class_filter = {item.strip() for item in args.class_list.split(",") if item.strip()}

    model_name = gen_root.name
    json_path, xlsx_path = build_output_paths(Path(args.output_dir), model_name)

    class_reports = {}
    inception_model = get_inception(device)
    inception_preprocess = get_preprocess()
    for class_dir in sorted([p for p in gen_root.iterdir() if p.is_dir()]):
        class_name = class_dir.name
        if class_filter and class_name not in class_filter:
            continue
        class_key = class_name

        candidates = [class_dir / "images", class_dir / "image", class_dir]
        gen_dir = None
        for cand in candidates:
            if cand.exists() and cand.is_dir():
                imgs = list_images(cand)
                if imgs:
                    gen_dir = cand
                    break
        if gen_dir is None:
            for cand in candidates:
                if cand.exists() and cand.is_dir():
                    gen_dir = cand
                    break
        if gen_dir is None:
            continue

        gen_paths = list_images(gen_dir)
        if not gen_paths:
            continue
        if args.limit:
            gen_paths = gen_paths[: args.limit]

        real_img_dir = resolve_real_dirs(args.real_root, class_key, args.real_split)
        if real_img_dir is None:
            class_reports[class_name] = {"error": "missing_real_data"}
            continue

        real_paths = list_images(real_img_dir)
        if not real_paths:
            class_reports[class_name] = {"error": "no_real_images"}
            continue

        feats_gen = compute_features(
            gen_paths,
            device,
            batch_size=args.batch_size,
            force_size=args.force_size,
            model=inception_model,
            preprocess=inception_preprocess,
        )
        feats_real = compute_features(
            real_paths,
            device,
            batch_size=args.batch_size,
            force_size=args.force_size,
            model=inception_model,
            preprocess=inception_preprocess,
        )

        kid_mean = calculate_kid(feats_gen, feats_real)
        try:
            density, coverage = calculate_density_coverage(
                feats_gen,
                feats_real,
                k=args.dc_k,
                batch_size=args.dc_batch_size,
                device=device,
            )
        except ValueError as exc:
            class_reports[class_name] = {"error": str(exc)}
            continue

        class_reports[class_name] = {
            "gen_dir": str(gen_dir),
            "real_img_dir": str(real_img_dir),
            "mode": "full",
            "total_generated": len(gen_paths),
            "num_generated": len(gen_paths),
            "num_real": len(real_paths),
            "num_generated_used": int(feats_gen.shape[0]),
            "num_real_used": int(feats_real.shape[0]),
            "limit": args.limit,
            "paired": False,
            "kid_method": "full_unbiased_polynomial_mmd",
            "kid_kernel_degree": 3,
            "density_coverage_k": args.dc_k,
            "metrics": {
                "kid_mean": kid_mean,
                "density_inception": density,
                "coverage_inception": coverage,
            },
        }

    summary = {}
    for key in ["kid_mean", "density_inception", "coverage_inception"]:
        values = []
        for report in class_reports.values():
            if not isinstance(report, dict) or "metrics" not in report:
                continue
            value = report["metrics"].get(key)
            if value is None:
                continue
            if isinstance(value, float) and (value != value):
                continue
            values.append(value)
        if values:
            summary[key] = float(np.mean(values, dtype=np.float64))
        else:
            summary[key] = float("nan")

    report = {
        "gen_root": str(gen_root),
        "real_root": str(Path(args.real_root)),
        "real_split": args.real_split,
    "mode": "full",
        "classes": class_reports,
        "summary": summary,
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    write_excel(report, xlsx_path)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
