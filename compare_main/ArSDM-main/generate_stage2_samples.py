import argparse
import itertools
import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader

from utils import instantiate_from_config

DEFAULT_CONFIGS = [
    "configs/HAM10000_akiec.yaml",
    "configs/HAM10000_bcc.yaml",
    "configs/HAM10000_bkl_lora_512.yaml",
    "configs/HAM10000_df_lora_512.yaml",
    "configs/HAM10000_mel_lora_512.yaml",
    "configs/HAM10000_nv_lora_512.yaml",
    "configs/HAM10000_vasc_lora_512.yaml",
]


def find_latest_run(log_root: Path, exp_name: str) -> Path | None:
    exp_dir = log_root / exp_name
    if not exp_dir.exists():
        return None
    runs = []
    for p in exp_dir.glob("*"):
        if not p.is_dir():
            continue
        if p.name.startswith("."):
            continue
        ckpt_dir = p / "checkpoints"
        if ckpt_dir.exists():
            runs.append(p)
    runs = sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0] if runs else None


def select_best_ckpt(run_dir: Path) -> Path | None:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None
    ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not ckpts:
        return None
    non_last = [p for p in ckpts if p.name != "last.ckpt"]
    return non_last[0] if non_last else ckpts[0]


def infer_class_name(config: OmegaConf, cfg_path: Path) -> str:
    try:
        images_dir = config.data.params.train.params.images_dir
        return Path(images_dir).name
    except Exception:
        name = cfg_path.stem
        if name.startswith("HAM10000_"):
            name = name[len("HAM10000_") :]
        name = name.replace("_lora_512", "").replace("_512", "")
        return name


def load_model(config: OmegaConf, ckpt_path: Path, device: torch.device):
    print(f"Loading model from {ckpt_path}")
    model = instantiate_from_config(config.model)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"State loaded with {len(missing)} missing and {len(unexpected)} unexpected keys")
    model.to(device)
    model.eval()
    return model


def prepare_dataloader(config: OmegaConf, batch_size: int) -> DataLoader:
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    dataset = data.datasets["train"]
    return DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=True)


def save_batch(images, masks, out_image_dir: Path, out_mask_dir: Path, start_idx: int):
    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    for i in range(images.shape[0]):
        img = images[i]
        mask = masks[i]

        img = (img + 1.0) / 2.0
        img = img.permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype(np.uint8)

        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]
        mask = mask.cpu().numpy()
        mask = (mask * 255).astype(np.uint8)

        img_path = out_image_dir / f"img_{start_idx + i:06}.png"
        mask_path = out_mask_dir / f"mask_{start_idx + i:06}.png"
        Image.fromarray(img).save(img_path)
        Image.fromarray(mask).save(mask_path)


def generate_for_config(
    cfg_path: Path,
    output_root: Path,
    num_images: int,
    batch_size: int,
    ddim_steps: int,
    device: torch.device,
    log_root_override: Path | None = None,
):
    config = OmegaConf.load(cfg_path)
    ckpt_path = config.get("model", {}).get("params", {}).get("ckpt_path")
    if ckpt_path and not Path(ckpt_path).exists():
        config.model.params.ckpt_path = None
    exp = config.get("exp", {})
    exp_name = exp.get("exp_name")
    if log_root_override is not None:
        log_root = log_root_override
    else:
        log_root = Path(exp.get("logdir", "/root/autodl-tmp/model/ArSDM_exps"))

    if not exp_name:
        raise ValueError(f"exp.exp_name not found in {cfg_path}")

    latest_run = find_latest_run(log_root, exp_name)
    if latest_run is None:
        raise FileNotFoundError(
            f"No valid run directory (with checkpoints) found for {exp_name} in {log_root}"
        )

    ckpt_path = select_best_ckpt(latest_run)
    if ckpt_path is None or not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint found in {latest_run}/checkpoints")

    class_name = infer_class_name(config, cfg_path)
    out_image_dir = output_root / class_name / "image"
    out_mask_dir = output_root / class_name / "mask"

    model = load_model(config, ckpt_path, device)
    loader = prepare_dataloader(config, batch_size)

    print(
        f"Generating {num_images} samples for {class_name} using {ckpt_path.name} (batch_size={batch_size}, ddim_steps={ddim_steps})"
    )

    # If resume is enabled, detect how many images already exist and continue
    produced = 0
    try:
        existing_imgs = sorted(out_image_dir.glob("img_*.png")) if out_image_dir.exists() else []
        existing_masks = sorted(out_mask_dir.glob("mask_*.png")) if out_mask_dir.exists() else []
        produced = max(len(existing_imgs), len(existing_masks))
        if produced >= num_images:
            print(f"Skipping {class_name}: already have {produced} images (>= {num_images})")
            return
        if produced > 0:
            print(f"Resuming {class_name}: found {produced} existing images, will continue to {num_images}")
    except Exception:
        produced = 0
    iterator = itertools.cycle(loader)
    with torch.no_grad():
        with model.ema_scope():
            while produced < num_images:
                batch = next(iterator)
                current_bs = min(batch_size, num_images - produced)
                images = model.log_images(
                    batch,
                    N=current_bs,
                    split="train",
                    ddim_steps=ddim_steps,
                )
                samples = images["samples"]
                masks = images["conditioning"]
                save_batch(samples, masks, out_image_dir, out_mask_dir, produced)
                produced += current_bs

    print(f"Done {class_name}: {produced} images saved to {output_root / class_name}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate samples for stage2 classes")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=DEFAULT_CONFIGS,
        help="Stage2 config files to use",
    )
    parser.add_argument(
        "--output_root",
        default="/root/autodl-tmp/output/generate/ArSDM",
        help="Output root directory for generated images",
    )
    parser.add_argument("--num_images", type=int, default=1500)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--ddim_steps", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--log_root",
        default=None,
        help="Override exp.logdir root when locating checkpoints",
    )
    parser.add_argument("--resume", type=lambda x: bool(int(x)), default=1, help="Resume generation if output exists (1/0)")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    output_root = Path(args.output_root)
    log_root_override = Path(args.log_root) if args.log_root else None
    for cfg in args.configs:
        cfg_path = Path(cfg)
        if not cfg_path.is_absolute():
            cfg_path = Path("/root/autodl-tmp/ArSDM-main") / cfg_path
        generate_for_config(
            cfg_path,
            output_root,
            num_images=args.num_images,
            batch_size=args.batch_size,
            ddim_steps=args.ddim_steps,
            device=device,
            log_root_override=log_root_override,
        )


if __name__ == "__main__":
    main()
