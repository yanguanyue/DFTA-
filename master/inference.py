import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from PIL import Image
import random
from torch.utils.data import DataLoader, Dataset

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from share import *  # noqa: F401,F403
from cldm.model import create_model, load_state_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference for DFTA (Dual-Flow Trajectory Alignment).")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/root/autodl-tmp/model/flow_matching/lightning_logs/version_0/checkpoints/last.ckpt",
    )
    parser.add_argument(
        "--config",
        type=str,
    default=str(Path(__file__).resolve().parent / "config" / "flow_matching.yaml"),
    )
    parser.add_argument("--output-dir", type=str, default="./generated_results/flow_matching")
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Optional root dir for class subfolders when using --class-list/--class-file",
    )
    parser.add_argument(
        "--prompt-json",
        type=str,
        default=str(Path(__file__).resolve().parent / "data" / "prompt.json"),
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--num-per-class", type=int, default=None)
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--ddim-eta", type=float, default=0.0)
    parser.add_argument("--stochastic", action="store_true", default=False,
                        help="Enable CSFS (Conditional Stochastic Flow Sampling)")
    parser.add_argument("--noise-scale", type=float, default=0.1,
                        help="CSFS noise scale (only used when --stochastic is set)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--class-list", type=str, default=None)
    parser.add_argument("--class-file", type=str, default=None)
    parser.add_argument("--repeat-dataloader", action="store_true", default=False)
    return parser.parse_args()


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
            if key.startswith("samples_cfg_scale_") or key == "samples":
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

            if key in {"control_mask", "control"}:
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


def iter_batches(dataloader: DataLoader, max_samples: int, repeat: bool) -> int:
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
    ddim_eta: float,
    stochastic: bool,
    noise_scale: float,
    generator: torch.Generator,
    repeat: bool,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        with model.ema_scope():
            for idx, batch in iter_batches(dataloader, max_samples, repeat):
                images = model.log_images(
                    batch,
                    N=dataloader.batch_size,
                    ddim_steps=ddim_steps,
                    ddim_eta=ddim_eta,
                    stochastic=stochastic,
                    noise_scale=(noise_scale if stochastic else 0.0),
                    generator=generator,
                )
                for key, value in images.items():
                    if isinstance(value, torch.Tensor):
                        images[key] = torch.clamp(value.detach().cpu(), -1.0, 1.0)
                save_images(output_dir, images, idx)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    os.chdir(repo_root)
    os.environ.setdefault("TMPDIR", "/root/autodl-tmp")

    pl.seed_everything(args.seed, workers=True)

    model = build_model(args.config, args.ckpt, args.device)

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

    if class_keys:
        output_root = Path(args.output_root) if args.output_root else Path(args.output_dir)
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
            class_output_dir = str(output_root / class_key)
            run_generation(
                model=model,
                dataloader=dataloader,
                output_dir=class_output_dir,
                max_samples=max_samples,
                ddim_steps=args.ddim_steps,
                ddim_eta=args.ddim_eta,
                stochastic=args.stochastic,
                noise_scale=args.noise_scale,
                generator=generator,
                repeat=repeat,
            )
    else:
        dataset = PromptDataset(args.prompt_json, size=args.image_size)
        dataloader = DataLoader(
            dataset,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            shuffle=False,
        )
        max_samples = args.max_samples
        run_generation(
            model=model,
            dataloader=dataloader,
            output_dir=args.output_dir,
            max_samples=max_samples,
            ddim_steps=args.ddim_steps,
            ddim_eta=args.ddim_eta,
            stochastic=args.stochastic,
            noise_scale=args.noise_scale,
            generator=generator,
            repeat=args.repeat_dataloader,
        )


if __name__ == "__main__":
    main()
