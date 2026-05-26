#!/usr/bin/env python
"""Generate HAM10000 long-tail filled panels using training inference logic.

This script reuses the ControlLDM inference pipeline from
`DFTA-main/tutorial_inference.py`, while keeping the per-class
counting logic from the original generate_tail implementation.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Tuple

import albumentations
import numpy as np
from PIL import Image

import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from cldm.model import create_model, load_state_dict


HF_TO_CATEGORY = {
    "akiec": "akiec",
    "bcc": "bcc",
    "bkl": "bkl",
    "df": "df",
    "mel": "mel",
    "nv": "nv",
    "vasc": "vasc",
}

CLASS_NAMES = list(HF_TO_CATEGORY.values())


def get_model(ckpt_path: str, device: str) -> torch.nn.Module:
    config_path = ROOT_DIR / "config" / "flow_matching.yaml"
    model = create_model(str(config_path)).cpu()
    model.load_state_dict(load_state_dict(ckpt_path, location="cpu"), strict=False)
    model.to(device)
    if hasattr(model, "cond_stage_model"):
        cond_stage = model.cond_stage_model
        if hasattr(cond_stage, "transformer"):
            cond_stage.transformer.to(device)
        if hasattr(cond_stage, "device"):
            cond_stage.device = device
    model.eval()
    return model


def log_local(save_dir: str, images: Dict[str, torch.Tensor], batch_idx: int) -> None:
    samples_root = os.path.join(save_dir, "images")
    mask_root = os.path.join(save_dir, "masks")

    for key, value in images.items():
        if not isinstance(value, torch.Tensor):
            continue
        value = value.detach().cpu()
        if key.startswith("samples_cfg_scale_") or key == "samples":
            value = torch.clamp((value + 1.0) / 2.0, 0.0, 1.0)
            for idx in range(value.shape[0]):
                filename = f"b-{batch_idx:06}_idx-{idx}.png"
                path = os.path.join(samples_root, filename)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                torchvision.utils.save_image(value[idx], path)
        elif key in {"control_mask", "control"}:
            value = torch.clamp(value, 0.0, 1.0)
            for idx in range(value.shape[0]):
                filename = f"b-{batch_idx:06}_idx-{idx}.png"
                path = os.path.join(mask_root, filename)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                torchvision.utils.save_image(value[idx], path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate HAM10000 yuan-style panels from HuggingFace validation data")
    parser.add_argument(
        "--hf_root",
        type=str,
        default="data/HAM10000/input/val/HuggingFace",
        help="Root of HuggingFace HAM10000 validation data",
    )
    parser.add_argument("--output_root", type=str, default="data/generated_yuan_tail_fill_v4")
    parser.add_argument("--target_per_class", type=int, default=1500)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/root/autodl-tmp/model/flow_matching/lightning_logs/version_4/checkpoints/last.ckpt",
    )
    parser.add_argument("--checkpoint_root", type=str, default=".")
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default=None,
        help="Optional fixed checkpoint filename (used when checkpoint_path is not set)",
    )
    parser.add_argument("--max_items", type=int, default=None, help="Limit dataset items for debugging")
    parser.add_argument("--dry_run", action="store_true", help="Only report counts, do not generate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class_shards", type=int, default=1, help="Total number of parallel shards for classes")
    parser.add_argument("--class_rank", type=int, default=0, help="Shard index for this process (0-based)")
    parser.add_argument("--class_list", type=str, default=None, help="Comma-separated class keys (akiec,bcc,...) to process")
    parser.add_argument("--class_file", type=str, default=None, help="Path to a text file with one class key per line")
    return parser.parse_args()


def find_latest_checkpoint(ckpt_dir: Path, fallback_name: str | None = None) -> Path:
    if fallback_name:
        candidate = ckpt_dir / fallback_name
        if candidate.exists():
            return candidate

    pattern = re.compile(r"checkpoint_epoch_(\d+)\.pth")
    candidates: List[Tuple[int, Path]] = []
    for path in ckpt_dir.glob("checkpoint_epoch_*.pth"):
        match = pattern.search(path.name)
        if match:
            candidates.append((int(match.group(1)), path))

    if not candidates:
        raise FileNotFoundError(f"No checkpoint files found in {ckpt_dir}")

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def count_existing_generated(output_root: Path, category: str) -> int:
    category_dir = output_root / category / "images"
    if not category_dir.exists():
        return 0
    return len(list(category_dir.glob("b-*.png")))


def generate_for_class(
    category: str,
    model: torch.nn.Module,
    dataset: "PromptHAMDataset",
    output_root: Path,
    target_total: int,
    steps: int,
    device: str,
    max_items: int | None = None,
) -> None:
    existing_generated = count_existing_generated(output_root, category)
    remaining = max(0, target_total - existing_generated)
    if remaining == 0:
        print(f"[Skip] {category}: generated files already meet target ({target_total})")
        return

    print(
        f"[Generate] {category}: context_samples={len(dataset)}, "
        f"target={target_total}, existing_generated={existing_generated}, remaining={remaining}"
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    generated = 0
    pbar = tqdm(total=remaining, desc=f"{category}", unit="img")
    with torch.inference_mode():
        ema_ctx = model.ema_scope() if hasattr(model, "ema_scope") else nullcontext()
        with ema_ctx:
            while generated < remaining:
                for idx, batch in enumerate(dataloader):
                    if max_items is not None and idx >= max_items:
                        break
                    if generated >= remaining:
                        break

                    batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
                    images = model.log_images(
                        batch,
                        N=4,
                        ddim_steps=steps,
                        ddim_eta=0.0,
                    )

                    for key in images:
                        if isinstance(images[key], torch.Tensor):
                            images[key] = images[key].detach().cpu()
                            images[key] = torch.clamp(images[key], -1.0, 1.0)

                    global_step = existing_generated + generated
                    log_local(str(output_root / category), images, global_step)
                    generated += 1
                    pbar.update(1)

                    if generated % 25 == 0 or generated == remaining:
                        pbar.set_postfix_str(f"{generated}/{remaining}")
    pbar.close()


class PromptHAMDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        prompt_path: Path,
        hf_class: str,
        category_name: str,
        image_size: int = 512,
    ) -> None:
        super().__init__()
        self.prompt_path = prompt_path
        self.hf_class = hf_class
        self.category_name = category_name
        self.image_size = image_size

        self.samples: List[Dict[str, str]] = []
        if self.prompt_path.exists():
            with self.prompt_path.open("r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    source_path = Path(item["source"])
                    if self._class_from_path(source_path) != hf_class:
                        continue
                    self.samples.append(
                        {
                            "source": item["source"],
                            "target": item["target"],
                            "prompt": item["prompt"],
                        }
                    )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        sample = self.samples[idx]
        source_filename = sample["source"]
        target_filename = sample["target"]
        prompt = sample["prompt"]

        source = Image.open(source_filename).convert("L")
        source_array = np.array(source)
        threshold = 127
        binary_array = np.where(source_array > threshold, 255, 0).astype(np.uint8)
        binary_image = Image.fromarray(binary_array)
        source = binary_image.convert("RGB")

        target = Image.open(target_filename).convert("RGB")

        source = np.array(source).astype(np.uint8)
        target = np.array(target).astype(np.uint8)

        preprocess = self.transform()(image=target, mask=source)
        source, target = preprocess["mask"], preprocess["image"]

        source = source.astype(np.float32) / 255.0
        target = target.astype(np.float32) / 127.5 - 1.0

        return dict(
            jpg=torch.tensor(target, dtype=torch.float32).contiguous(),
            txt=prompt,
            hint=torch.tensor(source, dtype=torch.float32).contiguous(),
        )

    def _class_from_path(self, source_path: Path) -> str | None:
        parts = source_path.parts
        if "HAM10000_seg_class" in parts:
            idx = parts.index("HAM10000_seg_class")
            if idx + 1 < len(parts):
                return parts[idx + 1]
        return None

    def transform(self, size: int | None = None):
        size = size or self.image_size
        transforms = albumentations.Compose(
            [
                albumentations.Resize(height=size, width=size),
            ]
        )
        return transforms


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_root = Path(args.output_root)
    prompt_path = ROOT_DIR / "data" / "prompt.json"
    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else None

    if checkpoint_path is None:
        raise ValueError("--checkpoint_path is required for tutorial inference generation.")

    model = None
    if not args.dry_run:
        model = get_model(str(checkpoint_path), args.device)

    if args.class_shards < 1:
        raise ValueError("--class_shards must be >= 1")
    if not (0 <= args.class_rank < args.class_shards):
        raise ValueError("--class_rank must be in [0, class_shards)")

    class_items = list(HF_TO_CATEGORY.items())
    if args.class_list:
        wanted = {item.strip() for item in args.class_list.split(',') if item.strip()}
        class_items = [item for item in class_items if item[0] in wanted]
    elif args.class_file:
        wanted = set()
        with Path(args.class_file).expanduser().open('r', encoding='utf-8') as f:
            for line in f:
                name = line.strip()
                if name:
                    wanted.add(name)
        class_items = [item for item in class_items if item[0] in wanted]
    else:
        class_items = [item for idx, item in enumerate(class_items) if idx % args.class_shards == args.class_rank]

    for hf_class, category in class_items:
        dataset = PromptHAMDataset(
            prompt_path=prompt_path,
            hf_class=hf_class,
            category_name=category,
            image_size=args.image_size,
        )

        if len(dataset) == 0:
            print(f"[Skip] {category}: HuggingFace dataset empty for {hf_class}")
            continue

        existing_generated = count_existing_generated(output_root, category)

        print(
            f"[Info] {category}: context_samples={len(dataset)}, "
            f"target={args.target_per_class}, existing_generated={existing_generated}"
        )

        if args.dry_run:
            continue

        generate_for_class(
            category=category,
            model=model,
            dataset=dataset,
            output_root=output_root,
            target_total=args.target_per_class,
            steps=args.steps,
            device=args.device,
            max_items=args.max_items,
        )


if __name__ == "__main__":
    main()
