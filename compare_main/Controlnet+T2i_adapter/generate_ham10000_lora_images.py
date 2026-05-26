#!/usr/bin/env python
# coding=utf-8
"""Generate images with LoRA (T2I-Adapter and ControlNet Depth) for HAM10000.

- Prompts are sampled from all llava_prompt per class (train/val/test CSVs).
- Masks are sampled only from validation CSV.
- Each class generates N images (default 1500).
- Two output roots (t2i adapter vs controlnet depth), each with image/ and mask/ subfolders.
"""

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from safetensors.torch import load_file
from PIL import Image
from tqdm import tqdm

from diffusers import (
    ControlNetModel,
    StableDiffusionAdapterPipeline,
    StableDiffusionControlNetPipeline,
    T2IAdapter,
    UNet2DConditionModel,
)
from peft import LoraConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate HAM10000 images with LoRA + controls.")
    parser.add_argument(
        "--sd_model",
        type=str,
        default="/root/autodl-tmp/model/AI-ModelScope/stable-diffusion-v1-5",
        help="Local SD1.5 diffusers model path.",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="/root/autodl-tmp/model/AI-ModelScope/t2iadapter_zoedepth_sd15v1",
        help="Local T2I-Adapter model path.",
    )
    parser.add_argument(
        "--controlnet_path",
        type=str,
        default="/root/autodl-tmp/model/AI-ModelScope/sd-controlnet-depth",
        help="Local ControlNet depth model path.",
    )
    parser.add_argument(
        "--lora_t2i_dir",
        type=str,
        default="/root/autodl-tmp/checkpoint/compare_models/T2i_adapter",
        help="LoRA weights directory for T2I-Adapter model.",
    )
    parser.add_argument(
        "--lora_controlnet_dir",
        type=str,
        default="/root/autodl-tmp/checkpoint/compare_models/ControlNet",
        help="LoRA weights directory for ControlNet model.",
    )
    parser.add_argument(
        "--csv_train",
        type=str,
        default="/root/autodl-tmp/data/metadata_train_llava.csv",
    )
    parser.add_argument(
        "--csv_val",
        type=str,
        default="/root/autodl-tmp/data/metadata_val_llava.csv",
    )
    parser.add_argument(
        "--csv_test",
        type=str,
        default="/root/autodl-tmp/data/metadata_test_llava.csv",
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default="/root/autodl-tmp",
        help="Root path for resolving CSV relative paths.",
    )
    parser.add_argument(
        "--output_t2i",
        type=str,
        default="/root/autodl-tmp/output/generate/T2i_adapter",
    )
    parser.add_argument(
        "--output_controlnet",
        type=str,
        default="/root/autodl-tmp/output/generate/ControlNet",
    )
    parser.add_argument("--num_per_class", type=int, default=1500)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--adapter_conditioning_scale", type=float, default=1.0)
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_classes", type=int, default=None)
    parser.add_argument(
        "--output_layout",
        type=str,
        default="class_first",
        choices=["image_first", "class_first"],
        help="Output directory layout: image_first=image/class, class_first=class/image.",
    )
    parser.add_argument("--run_t2i", action="store_true", help="Generate images with T2I-Adapter pipeline.")
    parser.add_argument("--run_controlnet", action="store_true", help="Generate images with ControlNet pipeline.")
    parser.add_argument(
        "--prompts_from_val_only",
        action="store_true",
        help="Use only validation CSV prompts (and masks) per class.",
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


def build_prompt_index(df: pd.DataFrame) -> Dict[str, List[str]]:
    prompts_by_class: Dict[str, List[str]] = {}
    for _, row in df.iterrows():
        label = row["class"]
        prompt = row.get("llava_prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            prompt = row.get("prompt", "")
        prompts_by_class.setdefault(label, []).append(prompt)
    return prompts_by_class


def build_mask_index(df: pd.DataFrame) -> Dict[str, List[str]]:
    masks_by_class: Dict[str, List[str]] = {}
    for _, row in df.iterrows():
        label = row["class"]
        masks_by_class.setdefault(label, []).append(row["seg_path"])
    return masks_by_class


def prepare_output_dirs(root: Path, classes: List[str], layout: str) -> Dict[str, Dict[str, Path]]:
    meta_root = root / "meta"
    meta_root.mkdir(parents=True, exist_ok=True)

    if layout == "class_first":
        image_root = root
        mask_root = root
        for cls in classes:
            (image_root / cls / "image").mkdir(parents=True, exist_ok=True)
            (mask_root / cls / "mask").mkdir(parents=True, exist_ok=True)
        return {"image": image_root, "mask": mask_root, "meta": meta_root}

    image_root = root / "image"
    mask_root = root / "mask"
    for cls in classes:
        (image_root / cls).mkdir(parents=True, exist_ok=True)
        (mask_root / cls).mkdir(parents=True, exist_ok=True)
    return {"image": image_root, "mask": mask_root, "meta": meta_root}


def save_metadata(meta_path: Path, record: dict):
    with meta_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_lora_into_unet(unet: UNet2DConditionModel, lora_path: str) -> None:
    lora_state = load_file(lora_path)
    mapped = {}
    for key, value in lora_state.items():
        if not key.startswith("unet."):
            continue
        new_key = key[len("unet.") :]
        if ".lora.down." in new_key:
            new_key = new_key.replace(".lora.down.", ".lora_A.default.")
        elif ".lora.up." in new_key:
            new_key = new_key.replace(".lora.up.", ".lora_B.default.")
        else:
            continue
        mapped[new_key] = value

    unet.load_state_dict(mapped, strict=False)
    if hasattr(unet, "set_adapter"):
        try:
            unet.set_adapter("default")
        except Exception:
            pass


def generate_for_model(
    pipe,
    classes: List[str],
    prompts_by_class: Dict[str, List[str]],
    masks_by_class: Dict[str, List[str]],
    output_root: Path,
    image_root: str,
    num_per_class: int,
    seed: int,
    num_inference_steps: int,
    guidance_scale: float,
    conditioning_scale: float,
    model_tag: str,
    output_layout: str,
):
    output_dirs = prepare_output_dirs(output_root, classes, output_layout)
    meta_path = output_dirs["meta"] / f"metadata_{model_tag}.jsonl"

    rng = random.Random(seed)
    device = pipe.device

    for cls in classes:
        prompts = prompts_by_class[cls]
        masks = masks_by_class[cls]
        progress = tqdm(range(num_per_class), desc=f"{model_tag}:{cls}", unit="img")
        total_time = 0.0

        for idx in progress:
            prompt = rng.choice(prompts)
            mask_path_str = rng.choice(masks)
            mask_path = resolve_path(mask_path_str, image_root)
            mask_image = Image.open(mask_path).convert("RGB")

            generator = torch.Generator(device=device)
            generator.manual_seed(seed + idx)

            start_time = time.perf_counter()
            if model_tag == "t2i_adapter":
                result = pipe(
                    prompt,
                    image=mask_image,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    adapter_conditioning_scale=conditioning_scale,
                    generator=generator,
                )
            else:
                result = pipe(
                    prompt,
                    image=mask_image,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    controlnet_conditioning_scale=conditioning_scale,
                    generator=generator,
                )
            elapsed = time.perf_counter() - start_time
            total_time += elapsed
            avg_time = total_time / (idx + 1)

            image = result.images[0]
            filename = f"{cls}_{idx:05d}.png"
            if output_layout == "class_first":
                image_out = output_dirs["image"] / cls / "image" / filename
                mask_out = output_dirs["mask"] / cls / "mask" / filename
            else:
                image_out = output_dirs["image"] / cls / filename
                mask_out = output_dirs["mask"] / cls / filename

            image.save(image_out)
            mask_image.save(mask_out)

            save_metadata(
                meta_path,
                {
                    "class": cls,
                    "prompt": prompt,
                    "image": str(image_out),
                    "mask": str(mask_out),
                    "source_mask": str(mask_path),
                    "elapsed_sec": round(elapsed, 4),
                    "avg_sec": round(avg_time, 4),
                },
            )

            progress.set_postfix({"sec": f"{elapsed:.2f}", "avg": f"{avg_time:.2f}"})


def main():
    args = parse_args()

    val_df = pd.read_csv(args.csv_val)
    if args.prompts_from_val_only:
        prompt_df = val_df
    else:
        train_df = pd.read_csv(args.csv_train)
        test_df = pd.read_csv(args.csv_test)
        prompt_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    prompts_by_class = build_prompt_index(prompt_df)
    masks_by_class = build_mask_index(val_df)

    classes = sorted(masks_by_class.keys())
    if args.max_classes is not None:
        classes = classes[: args.max_classes]

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_t2i = args.run_t2i
    run_controlnet = args.run_controlnet
    if not run_t2i and not run_controlnet:
        run_controlnet = True

    t2i_pipe = None
    controlnet_pipe = None

    if run_t2i:
        adapter = T2IAdapter.from_pretrained(args.adapter_path, torch_dtype=torch_dtype)
        t2i_unet = UNet2DConditionModel.from_pretrained(args.sd_model, subfolder="unet", torch_dtype=torch_dtype)
        t2i_unet.add_adapter(
            LoraConfig(
                r=8,
                lora_alpha=8,
                init_lora_weights="gaussian",
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            )
        )
        load_lora_into_unet(t2i_unet, os.path.join(args.lora_t2i_dir, "pytorch_lora_weights.safetensors"))
        t2i_pipe = StableDiffusionAdapterPipeline.from_pretrained(
            args.sd_model,
            adapter=adapter,
            unet=t2i_unet,
            torch_dtype=torch_dtype,
            safety_checker=None,
            feature_extractor=None,
        )
        t2i_pipe.to(device)

    if run_controlnet:
        controlnet = ControlNetModel.from_pretrained(args.controlnet_path, torch_dtype=torch_dtype)
        cn_unet = UNet2DConditionModel.from_pretrained(args.sd_model, subfolder="unet", torch_dtype=torch_dtype)
        cn_unet.add_adapter(
            LoraConfig(
                r=8,
                lora_alpha=8,
                init_lora_weights="gaussian",
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            )
        )
        load_lora_into_unet(cn_unet, os.path.join(args.lora_controlnet_dir, "pytorch_lora_weights.safetensors"))
        controlnet_pipe = StableDiffusionControlNetPipeline.from_pretrained(
            args.sd_model,
            controlnet=controlnet,
            unet=cn_unet,
            torch_dtype=torch_dtype,
            safety_checker=None,
            feature_extractor=None,
        )
        controlnet_pipe.to(device)

    output_t2i = Path(args.output_t2i)
    output_controlnet = Path(args.output_controlnet)
    output_t2i.mkdir(parents=True, exist_ok=True)
    output_controlnet.mkdir(parents=True, exist_ok=True)

    if run_t2i:
        generate_for_model(
            t2i_pipe,
            classes=classes,
            prompts_by_class=prompts_by_class,
            masks_by_class=masks_by_class,
            output_root=output_t2i,
            image_root=args.image_root,
            num_per_class=args.num_per_class,
            seed=args.seed,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            conditioning_scale=args.adapter_conditioning_scale,
            model_tag="t2i_adapter",
            output_layout=args.output_layout,
        )

    if run_controlnet:
        generate_for_model(
            controlnet_pipe,
            classes=classes,
            prompts_by_class=prompts_by_class,
            masks_by_class=masks_by_class,
            output_root=output_controlnet,
            image_root=args.image_root,
            num_per_class=args.num_per_class,
            seed=args.seed,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            conditioning_scale=args.controlnet_conditioning_scale,
            model_tag="controlnet_depth",
            output_layout=args.output_layout,
        )


if __name__ == "__main__":
    main()
