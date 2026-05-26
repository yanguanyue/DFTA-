import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from share import *  # noqa: F401,F403
from cldm.model import create_model, load_state_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Siamese-Diffusion inference for 7-class generation.")
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default="/root/autodl-tmp/checkpoint/compare_models/Siamese/merged_pytorch_model.pth",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent / "models" / "cldm_v15.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/root/autodl-tmp/output/generate2/Siamese",
    )
    parser.add_argument("--prompt-json", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--gpu-ids", type=str, default="0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-per-class", type=int, default=1500)
    parser.add_argument(
        "--classes",
        type=str,
        default="akiec,bcc,bkl,df,mel,nv,vasc",
    )
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--ddim-eta", type=float, default=0.0)
    parser.add_argument("--stochastic", action="store_true", default=False)
    parser.add_argument("--noise-scale", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=512)
    return parser.parse_args()


class PromptDataset(Dataset):
    def __init__(self, items: list[dict], size: int = 384) -> None:
        self.items = items
        self.size = size

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
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

        return {
            "jpg": torch.tensor(target, dtype=torch.float32),
            "txt": prompt,
            "hint": torch.tensor(source, dtype=torch.float32),
        }


def resolve_prompt_json(prompt_json: str | None) -> str:
    if prompt_json:
        return prompt_json
    repo_root = Path(__file__).resolve().parent
    local_prompt = repo_root / "data" / "prompt.json"
    if local_prompt.exists():
        return str(local_prompt)
    fallback_prompt = Path("/root/autodl-tmp/main/Siamese-Diffusion-main/data/prompt.json")
    if fallback_prompt.exists():
        return str(fallback_prompt)
    raise FileNotFoundError("prompt.json not found; please pass --prompt-json")


def load_prompt_items(prompt_json: str) -> list[dict]:
    items: list[dict] = []
    with open(prompt_json, "rt") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def infer_class(item: dict) -> str:
    target_path = Path(item["target"])
    return target_path.parent.name


def build_class_items(
    items: list[dict],
    class_name: str,
    num_per_class: int,
    rng: random.Random,
) -> list[dict]:
    class_items = [item for item in items if infer_class(item) == class_name]
    if not class_items:
        raise ValueError(f"No samples found for class {class_name}")
    rng.shuffle(class_items)
    if num_per_class <= len(class_items):
        return class_items[:num_per_class]
    repeats = num_per_class // len(class_items)
    remainder = num_per_class % len(class_items)
    return class_items * repeats + class_items[:remainder]


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


def log_local(save_dir: str, images: dict, batch_idx: int) -> None:
    samples_root = os.path.join(save_dir, "images")
    mask_root = os.path.join(save_dir, "masks")

    for key, batch in images.items():
        for idx, image in enumerate(batch):
            if key == "samples_cfg_scale_9.00_mask":
                if isinstance(image, torch.Tensor):
                    image = image.detach().cpu()
                    image = (image + 1.0) / 2.0
                    image = image.permute(1, 2, 0).numpy()
                else:
                    image = np.array(image)
                image = (image * 255).astype(np.uint8)
                filename = f"b-{batch_idx:06}_idx-{idx}.png"
                path = os.path.join(samples_root, filename)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                Image.fromarray(image).save(path)

            if key == "control_mask":
                if isinstance(image, torch.Tensor):
                    image = image.detach().cpu().permute(1, 2, 0).numpy()
                else:
                    image = np.array(image)
                if image.ndim == 3 and image.shape[-1] == 1:
                    image = image.squeeze(-1)
                elif image.ndim == 3 and image.shape[-1] == 3:
                    image = image[..., 0]
                mask = (image * 255).astype(np.uint8)
                filename = f"b-{batch_idx:06}_idx-{idx}.png"
                path = os.path.join(mask_root, filename)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                Image.fromarray(mask).convert("1").save(path)


def main() -> None:
    args = parse_args()

    if args.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    pl.seed_everything(args.seed, workers=True)

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    prompt_json = resolve_prompt_json(args.prompt_json)
    items = load_prompt_items(prompt_json)
    rng = random.Random(args.seed)

    model = build_model(args.config, str(ckpt_path), args.device)

    os.makedirs(args.output_dir, exist_ok=True)

    class_list = [cls.strip() for cls in args.classes.split(",") if cls.strip()]

    with torch.no_grad():
        with model.ema_scope():
            for class_name in class_list:
                class_items = build_class_items(items, class_name, args.num_per_class, rng)
                dataset = PromptDataset(class_items, size=args.image_size)
                dataloader = DataLoader(
                    dataset,
                    num_workers=args.num_workers,
                    batch_size=args.batch_size,
                    shuffle=False,
                )
                class_dir = os.path.join(args.output_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)

                for idx, batch in enumerate(dataloader):
                    images = model.log_images(
                        batch,
                        N=args.batch_size,
                        ddim_steps=args.ddim_steps,
                        ddim_eta=args.ddim_eta,
                        stochastic=args.stochastic,
                        noise_scale=(args.noise_scale if args.stochastic else 0.0),
                    )
                    for key, value in images.items():
                        if isinstance(value, torch.Tensor):
                            images[key] = torch.clamp(value.detach().cpu(), -1.0, 1.0)
                    log_local(class_dir, images, idx)
                    if (idx + 1) * args.batch_size >= args.num_per_class:
                        break


if __name__ == "__main__":
    main()

