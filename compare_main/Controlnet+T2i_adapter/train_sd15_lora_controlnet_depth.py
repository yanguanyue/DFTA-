#!/usr/bin/env python
# coding=utf-8
"""Train SD1.5 LoRA with ControlNet Depth conditioning (mask as control).

- Images + masks from CSV columns img_path/seg_path.
- Text from llava_prompt.
- ControlNet model is local path: sd-controlnet-depth.
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import List

import pandas as pd
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.training_utils import cast_training_params
from diffusers.utils import convert_state_dict_to_diffusers
from transformers import CLIPTextModel, CLIPTokenizer


logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="Train SD1.5 LoRA with ControlNet Depth conditioning.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to local SD1.5 diffusers model.",
    )
    parser.add_argument(
        "--controlnet_path",
        type=str,
        required=True,
        help="Path to local ControlNet depth model.",
    )
    parser.add_argument(
        "--csv_paths",
        type=str,
        required=True,
        help="Comma-separated CSVs that include img_path, seg_path, llava_prompt columns.",
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default="/root/autodl-tmp",
        help="Root path used to resolve relative img_path/seg_path from CSVs.",
    )
    parser.add_argument("--output_dir", type=str, default="/root/autodl-tmp/checkpoint/compare_models/ControlNet")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_train_steps", type=int, default=1000)
    parser.add_argument("--checkpointing_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=1.0)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--random_flip", action="store_true")
    parser.add_argument("--center_crop", action="store_true")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument(
        "--sample_output_dir",
        type=str,
        default=None,
        help="Directory to save a sample image/mask/text for inspection.",
    )
    return parser.parse_args()


def resolve_path(path_str: str, image_root: str) -> Path:
    path = Path(path_str)
    if path.is_absolute() and path.exists():
        return path
    if path.exists():
        return path

    candidate = Path(image_root) / path_str
    if candidate.exists():
        return candidate

    if path_str.startswith("data/local/"):
        candidate = Path(image_root) / path_str.replace("data/local/", "data/")
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Cannot resolve path: {path_str}")


class HamLlavaDataset(Dataset):
    def __init__(
        self,
        csv_paths: List[str],
        image_root: str,
        resolution: int,
        random_flip: bool,
        center_crop: bool,
    ):
        dfs = [pd.read_csv(p) for p in csv_paths]
        self.df = pd.concat(dfs, ignore_index=True)
        self.image_root = image_root

        if "llava_prompt" not in self.df.columns:
            raise ValueError("CSV missing required column: llava_prompt")
        if "img_path" not in self.df.columns or "seg_path" not in self.df.columns:
            raise ValueError("CSV missing required columns: img_path/seg_path")

        if center_crop:
            crop = transforms.CenterCrop(resolution)
        else:
            crop = transforms.RandomCrop(resolution)

        self.image_transform = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                crop,
                transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.NEAREST),
                crop,
                transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = resolve_path(row["img_path"], self.image_root)
        seg_path = resolve_path(row["seg_path"], self.image_root)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(seg_path).convert("RGB")

        pixel_values = self.image_transform(image)
        conditioning_image = self.mask_transform(mask)

        prompt = row.get("llava_prompt")
        if not isinstance(prompt, str) or len(prompt.strip()) == 0:
            prompt = row.get("prompt", "")
        return {
            "pixel_values": pixel_values,
            "conditioning_image": conditioning_image,
            "prompt": prompt,
        }


def collate_fn(examples):
    pixel_values = torch.stack([e["pixel_values"] for e in examples])
    conditioning_image = torch.stack([e["conditioning_image"] for e in examples])
    prompts = [e["prompt"] for e in examples]
    return {"pixel_values": pixel_values, "conditioning_image": conditioning_image, "prompts": prompts}


def save_sample_batch(batch, output_dir: str):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image = batch["pixel_values"][0].detach().cpu()
    mask = batch["conditioning_image"][0].detach().cpu()
    prompt = batch["prompts"][0]

    image = (image * 0.5 + 0.5).clamp(0, 1)
    mask = mask.clamp(0, 1)

    image_pil = transforms.ToPILImage()(image)
    mask_pil = transforms.ToPILImage()(mask)

    image_pil.save(output_path / "sample_image.png")
    mask_pil.save(output_path / "sample_mask.png")
    with open(output_path / "sample_prompt.txt", "w", encoding="utf-8") as f:
        f.write(prompt)


def main():
    args = parse_args()

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=os.path.join(args.output_dir, "logs"))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)

    csv_paths = [p.strip() for p in args.csv_paths.split(",") if p.strip()]
    dataset = HamLlavaDataset(
        csv_paths=csv_paths,
        image_root=args.image_root,
        resolution=args.resolution,
        random_flip=args.random_flip,
        center_crop=args.center_crop,
    )
    if args.max_train_samples is not None:
        dataset = torch.utils.data.Subset(dataset, range(min(args.max_train_samples, len(dataset))))

    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    if accelerator.is_main_process and args.sample_output_dir:
        first_batch = next(iter(dataloader))
        save_sample_batch(first_batch, args.sample_output_dir)

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    controlnet = ControlNetModel.from_pretrained(args.controlnet_path)

    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.requires_grad_(False)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    controlnet.to(accelerator.device, dtype=weight_dtype)

    unet_lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(unet_lora_config)
    if args.mixed_precision == "fp16":
        cast_training_params(unet, dtype=torch.float32)

    lora_layers = filter(lambda p: p.requires_grad, unet.parameters())
    optimizer = torch.optim.AdamW(lora_layers, lr=args.learning_rate)

    unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)

    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.max_train_steps
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    global_step = 0
    for epoch in range(num_train_epochs):
        unet.train()
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                conditioning_image = batch["conditioning_image"].to(dtype=weight_dtype)

                prompts = batch["prompts"]
                tokenized = tokenizer(
                    prompts,
                    padding="max_length",
                    truncation=True,
                    max_length=tokenizer.model_max_length,
                    return_tensors="pt",
                )
                input_ids = tokenized.input_ids.to(accelerator.device)
                encoder_hidden_states = text_encoder(input_ids)[0]

                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=latents.device,
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                down_residuals, mid_residual = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states,
                    controlnet_cond=conditioning_image,
                    conditioning_scale=args.controlnet_conditioning_scale,
                    return_dict=False,
                )

                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states,
                    down_block_additional_residuals=[state.clone() for state in down_residuals],
                    mid_block_additional_residual=mid_residual,
                    return_dict=False,
                )[0]

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                else:
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                accelerator.log({"train_loss": loss.detach().item()}, step=global_step)

                if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)

                if global_step >= max_train_steps:
                    break

        if global_step >= max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unet.to(torch.float32)
        unwrapped_unet = accelerator.unwrap_model(unet)
        unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_unet))
        StableDiffusionPipeline.save_lora_weights(
            save_directory=args.output_dir,
            unet_lora_layers=unet_lora_state_dict,
            safe_serialization=True,
        )

        with open(os.path.join(args.output_dir, "train_args.json"), "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2, ensure_ascii=False)

    accelerator.end_training()


if __name__ == "__main__":
    main()
