#!/usr/bin/env python3
"""
Ablation Study Inference Script (Optimized: 2 Trained Models + 3 Generation Modes)

Train only 2 ablation models:
- Model A: Single-Flow (No Image-Flow, No OSEA)
- Model B: Dual-Flow with Trajectory Alignment (No OSEA)

Mode 3 uses the existing full DFTA model from checkpoint/flow

Generation Modes:
Mode 1: Single-Flow + CSFS Stochastic Sampling (original mode 2)
Mode 2: Dual-Flow (No Aug) + CSFS Stochastic Sampling (original mode 4)
Mode 3: Dual-Flow (Full/Pretrained) + Deterministic Sampling (original mode 5)
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from PIL import Image
from tqdm import tqdm
import random
from torch.utils.data import DataLoader, Dataset

from share import *  # noqa: F401,F403
from cldm.model import create_model, load_state_dict


MODE_CONFIGS = {
    "1": {
        "name": "single_flow_csfs",
        "description": "Single-Flow + CSFS Stochastic Sampling",
        "config": "Ablation/config_single_flow.yaml",
        "checkpoint_dir": "01_single_flow",
        "stochastic": True,
        "noise_scale": 0.1,
    },
    "2": {
        "name": "dual_flow_no_aug_csfs",
        "description": "Dual-Flow (No Aug) + CSFS Stochastic Sampling",
        "config": "Ablation/config_dual_flow_no_aug.yaml",
        "checkpoint_dir": "02_dual_flow_no_aug",
        "stochastic": True,
        "noise_scale": 0.1,
    },
    "3": {
        "name": "dual_flow_full_det",
        "description": "Dual-Flow (Full/Pretrained) + Deterministic Sampling",
        "config": "config/flow_matching.yaml",
        "checkpoint_dir": None,
        "use_pretrained_flow": True,
        "stochastic": False,
        "noise_scale": 0.0,
    },
}


class PromptDataset(Dataset):
    def __init__(
        self,
        prompt_json: str,
        size: int = 384,
        class_key: str | None = None,
        max_items: int | None = None,
        shuffle: bool = False,
        seed: int = 0,
    ) -> None:
        self.size = size
        self.data = []
        with open(prompt_json, "rt") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    if class_key:
                        source_path = Path(item.get("source", ""))
                        if self._class_from_path(source_path) != class_key:
                            continue
                    self.data.append(item)

        if shuffle:
            random.Random(seed).shuffle(self.data)
        if max_items is not None:
            self.data = self.data[:max_items]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        source_filename = item["source"]
        target_filename = item["target"]
        prompt = item["prompt"]

        source = Image.open(source_filename).convert("L")
        source_array = np.array(source)
        binary_array = np.where(source_array > 127, 255, 0).astype(np.uint8)
        source = Image.fromarray(binary_array).convert("RGB")

        target = Image.open(target_filename).convert("RGB")

        source = source.resize((self.size, self.size), resample=Image.NEAREST)
        target = target.resize((self.size, self.size), resample=Image.BILINEAR)

        source = np.array(source).astype(np.float32) / 255.0
        target = np.array(target).astype(np.float32) / 127.5 - 1.0

        source = torch.tensor(source, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)

        return {"jpg": target, "txt": prompt, "hint": source}

    def _class_from_path(self, source_path: Path) -> str | None:
        parts = source_path.parts
        if "HAM10000_seg_class" in parts:
            idx = parts.index("HAM10000_seg_class")
            if idx + 1 < len(parts):
                return parts[idx + 1]
        return None


def build_model(config_path: str, ckpt_path: str, device: str) -> torch.nn.Module:
    model = create_model(config_path).cpu()
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


def save_images(save_dir: str, images: dict[str, torch.Tensor], batch_idx: int) -> None:
    samples_root = os.path.join(save_dir, "images")
    mask_root = os.path.join(save_dir, "masks")

    for key, tensor in images.items():
        for idx, image in enumerate(tensor):
            if key == "samples_cfg_scale_9.00_mask":
                filename = f"b-{batch_idx:06}_idx-{idx}.png"
                path = os.path.join(samples_root, filename)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                if isinstance(image, torch.Tensor):
                    save_tensor = (image.detach().cpu() + 1.0) / 2.0
                    torchvision.utils.save_image(save_tensor, path)
                    continue
                if isinstance(image, np.ndarray):
                    if image.dtype == object:
                        image = np.array(image.tolist(), dtype=np.float32)
                else:
                    image = np.array(image, dtype=np.float32)
                if isinstance(image, np.ndarray) and image.dtype == object:
                    image = np.array(image.tolist(), dtype=np.float32)
                if image.ndim == 3 and image.shape[0] in (1, 3):
                    image = np.transpose(image, (1, 2, 0))
                if image.ndim == 3 and image.shape[-1] == 1:
                    image = image.squeeze(-1)
                image = (image + 1.0) / 2.0
                image = (image * 255).astype(np.uint8)
                Image.fromarray(image).save(path)

            if key == "control_mask":
                if isinstance(image, torch.Tensor):
                    mask_tensor = image.detach().cpu()
                    filename = f"b-{batch_idx:06}_idx-{idx}.png"
                    path = os.path.join(mask_root, filename)
                    os.makedirs(os.path.split(path)[0], exist_ok=True)
                    torchvision.utils.save_image(mask_tensor, path)
                    continue
                else:
                    image = np.asarray(image)
                if image.dtype == object:
                    if image.size == 1 and isinstance(image.flat[0], torch.Tensor):
                        image = image.flat[0].detach().cpu().numpy()
                    else:
                        image = np.array(image.tolist(), dtype=np.float32)
                if image.ndim == 3 and image.shape[-1] == 3:
                    image = image[..., 0]
                if image.ndim == 3 and image.shape[-1] == 1:
                    image = image.squeeze(-1)
                mask = (image * 255).astype(np.uint8)
                if mask.dtype == object:
                    continue
                filename = f"b-{batch_idx:06}_idx-{idx}.png"
                path = os.path.join(mask_root, filename)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                Image.fromarray(mask).convert("1").save(path)


def iter_batches(dataloader: DataLoader, max_samples: int, repeat: bool):
    batch_idx = 0
    while batch_idx < max_samples:
        for batch in dataloader:
            yield batch_idx, batch
            batch_idx += 1
            if batch_idx >= max_samples:
                return
        if not repeat:
            return


def run_generation(
    model: torch.nn.Module,
    dataloader: DataLoader,
    output_dir: str,
    max_samples: int,
    ddim_steps: int,
    stochastic: bool,
    noise_scale: float,
    generator: torch.Generator,
    repeat: bool,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        with model.ema_scope():
            pbar = tqdm(
                iter_batches(dataloader, max_samples, repeat),
                total=max_samples,
                desc=f"Generating",
                unit="batch",
                dynamic_ncols=True,
            )
            for idx, batch in pbar:
                images = model.log_images(
                    batch,
                    N=dataloader.batch_size,
                    ddim_steps=ddim_steps,
                    ddim_eta=0.0,
                    stochastic=stochastic,
                    noise_scale=(noise_scale if stochastic else 0.0),
                    generator=generator,
                )
                for key, value in images.items():
                    if isinstance(value, torch.Tensor):
                        images[key] = torch.clamp(value.detach().cpu(), -1.0, 1.0)
                save_images(output_dir, images, idx)
                pbar.set_postfix({"saved": idx + 1, "total": max_samples})


def find_latest_checkpoint(checkpoint_dir: str) -> str | None:
    """Find the latest .ckpt file in directory."""
    if not os.path.exists(checkpoint_dir):
        return None
    ckpt_files = []
    for root, dirs, files in os.walk(checkpoint_dir):
        for file in files:
            if file.endswith(".ckpt"):
                ckpt_files.append(os.path.join(root, file))
    ckpt_files.sort(key=os.path.getmtime, reverse=True)
    return ckpt_files[0] if ckpt_files else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablation study inference script (2 models + 3 modes)")
    parser.add_argument("--mode", type=str, required=True, choices=["1", "2", "3", "all"],
                        help="Generation mode (1-3 or 'all' for all modes)")
    parser.add_argument("--ablation-ckpt-root", type=str,
                        default="/root/autodl-tmp/checkpoint/ablation",
                        help="Root directory containing ablation checkpoints (for modes 1-2)")
    parser.add_argument("--flow-ckpt-root", type=str,
                        default="/root/autodl-tmp/checkpoint/flow",
                        help="Root directory containing full DFTA model (for mode 3)")
    parser.add_argument("--output-root", type=str,
                        default="/root/autodl-tmp/output/ablation",
                        help="Output root directory")
    parser.add_argument("--main-dir", type=str,
                        default=None,
                        help="Main directory (auto-detected if not specified)")
    parser.add_argument("--prompt-json", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--num-per-class", type=int, default=None)
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--class-list", type=str, default=None)
    parser.add_argument("--class-file", type=str, default=None)
    parser.add_argument("--repeat-dataloader", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    main_dir = args.main_dir if args.main_dir else str(repo_root)

    os.chdir(main_dir)
    os.environ.setdefault("TMPDIR", "/root/autodl-tmp")

    if args.prompt_json is None:
        args.prompt_json = str(Path(main_dir) / "data" / "prompt.json")

    pl.seed_everything(args.seed, workers=True)

    modes_to_run = []
    if args.mode == "all":
        modes_to_run = list(MODE_CONFIGS.keys())
    else:
        modes_to_run = [args.mode]

    print(f"\n{'🎯'*30}")
    print(f"   Ablation Study: {len(modes_to_run)} mode(s) to run")
    print(f"{'🎯'*30}")

    mode_pbar = tqdm(modes_to_run, desc="Modes", unit="mode", dynamic_ncols=True)

    for mode_id in mode_pbar:
        mode_config = MODE_CONFIGS[mode_id]
        mode_pbar.set_description(f"Mode {mode_id}: {mode_config['name']}")

        print(f"\n{'='*60}")
        print(f"Running Mode {mode_id}: {mode_config['description']}")
        print(f"{'='*60}")

        config_path = str(Path(main_dir) / mode_config["config"])

        if mode_config.get("use_pretrained_flow", False):
            checkpoint_dir = args.flow_ckpt_root
            print(f"[Mode {mode_id}] Using pretrained full DFTA model from: {checkpoint_dir}")
        else:
            checkpoint_dir = os.path.join(args.ablation_ckpt_root, mode_config["checkpoint_dir"])
            print(f"Checkpoint dir: {checkpoint_dir}")

        latest_ckpt = find_latest_checkpoint(checkpoint_dir)

        if not latest_ckpt:
            print(f"[Skip] No checkpoint found for mode {mode_id} in {checkpoint_dir}")
            continue

        print(f"Using checkpoint: {latest_ckpt}")

        model = build_model(config_path, latest_ckpt, args.device)

        if "cuda" in args.device:
            generator = torch.Generator(device=args.device).manual_seed(args.seed)
        else:
            generator = torch.Generator().manual_seed(args.seed)

        class_keys = None
        if args.class_list:
            class_keys = [item.strip() for item in args.class_list.split(",") if item.strip()]
        elif args.class_file:
            class_keys = []
            with Path(args.class_file).expanduser().open("r", encoding="utf-8") as f:
                for line in f:
                    name = line.strip()
                    if name:
                        class_keys.append(name)

        output_mode_root = Path(args.output_root) / mode_config["name"]

        if class_keys:
            max_samples = args.num_per_class if args.num_per_class is not None else args.max_samples
            repeat = args.repeat_dataloader or (args.num_per_class is not None)

            for idx, class_key in enumerate(class_keys):
                dataset = PromptDataset(
                    args.prompt_json,
                    size=args.image_size,
                    class_key=class_key,
                    shuffle=True,
                    seed=args.seed + idx,
                )
                if len(dataset) == 0:
                    print(f"[Skip] class={class_key} has no prompt items")
                    continue

                dataloader = DataLoader(
                    dataset,
                    num_workers=args.num_workers,
                    batch_size=args.batch_size,
                    shuffle=True,
                )

                class_output_dir = str(output_mode_root / class_key)
                run_generation(
                    model=model,
                    dataloader=dataloader,
                    output_dir=class_output_dir,
                    max_samples=max_samples,
                    ddim_steps=args.ddim_steps,
                    stochastic=mode_config["stochastic"],
                    noise_scale=mode_config["noise_scale"],
                    generator=generator,
                    repeat=repeat,
                )
        else:
            dataset = PromptDataset(args.prompt_json, size=args.image_size)
            dataloader = DataLoader(
                dataset,
                num_workers=args.num_workers,
                batch_size=args.batch_size,
                shuffle=True,
            )

            run_generation(
                model=model,
                dataloader=dataloader,
                output_dir=str(output_mode_root),
                max_samples=args.max_samples,
                ddim_steps=args.ddim_steps,
                stochastic=mode_config["stochastic"],
                noise_scale=mode_config["noise_scale"],
                generator=generator,
                repeat=args.repeat_dataloader,
            )

        print(f"✓ Mode {mode_id} completed! Output saved to: {output_mode_root}")

    print(f"\n{'='*60}")
    print("All requested modes completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
