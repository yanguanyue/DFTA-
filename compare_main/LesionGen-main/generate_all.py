import torch
import os
import random
import argparse
import sys
from pathlib import Path
from typing import Dict, Optional
from PIL import Image


def _ensure_diffusers_pipeline():
    try:
        from diffusers import StableDiffusionPipeline
        return StableDiffusionPipeline
    except ImportError:
        local_diffusers = Path(__file__).resolve().parent / "external" / "diffusers" / "src"
        if local_diffusers.exists():
            sys.path.insert(0, str(local_diffusers))
            from diffusers import StableDiffusionPipeline
            return StableDiffusionPipeline
        raise


def _resolve_device() -> str:
    preferred = os.getenv("DIFFUSERS_DEVICE")
    if preferred:
        return preferred
    return "cuda" if torch.cuda.is_available() else "cpu"


def _parse_bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y"}

DEFAULT_CLASS_PROMPTS: Dict[str, str] = {
    "akiec": "Actinic keratoses and intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}

DEFAULT_CLASS_ORDER = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def _load_pipeline(model_path: str, base_model_path: Optional[str]) -> "StableDiffusionPipeline":
    base_model = base_model_path or os.getenv("DIFFUSERS_BASE_MODEL", "CompVis/stable-diffusion-v1-4")
    device = _resolve_device()
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    local_files_only = _parse_bool_env("DIFFUSERS_LOCAL_FILES_ONLY", False)
    StableDiffusionPipeline = _ensure_diffusers_pipeline()
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=dtype,
        local_files_only=local_files_only,
    )
    pipe.unet.load_attn_procs(model_path)
    pipe = pipe.to(device)
    pipe.safety_checker = None
    pipe.set_progress_bar_config(disable=True)
    if device == "cpu":
        pipe.enable_attention_slicing()
    return pipe


def generate_images(
    condition: str,
    mode: str,
    output_dir: str,
    num_images: int,
    model_path: str = "Lora/weights/checkpoint-15000",
    base_model_path: Optional[str] = None,
    output_subdir: Optional[str] = None,
    seed: Optional[int] = None,
    output_layout: str = "nested",
):
    pipe = _load_pipeline(model_path, base_model_path)

    def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
        if image_tensor.ndim == 4:
            image_tensor = image_tensor[0]
        image_tensor = image_tensor.detach().cpu().clamp(0, 1).mul(255).byte()
        image_tensor = image_tensor.permute(1, 2, 0).contiguous()
        data = bytes(image_tensor.view(-1).tolist())
        return Image.frombytes("RGB", (image_tensor.shape[1], image_tensor.shape[0]), data)

    if seed is not None:
        torch.manual_seed(seed)

    if mode == "single":
        single_output_dir = os.path.join(output_dir, "single")
        os.makedirs(single_output_dir, exist_ok=True)
        prompt = f"{condition} small size on a dark skin tone"
        image_tensor = pipe(prompt, num_inference_steps=50, guidance_scale=7.5, output_type="pt").images[0]
        image = tensor_to_pil(image_tensor)
        image.save(os.path.join(single_output_dir, f"{condition}.png"))
    elif mode == "dataset":
        if output_layout == "flat":
            dataset_output_dir = os.path.join(output_dir, output_subdir or condition)
        else:
            dataset_output_dir = os.path.join(output_dir, "dataset", output_subdir or condition)
        os.makedirs(dataset_output_dir, exist_ok=True)
        prompts = [
            f"{condition} small size on a dark skin tone",
            f"{condition} large size on a light skin tone",
            f"{condition} medium size on a medium skin tone"
        ]
        guidance_scales = [3.0, 5.0, 7.5, 10.0]
        num_inference_steps_list = [30, 50, 70, 100]

        for i in range(num_images):
            prompt = random.choice(prompts)
            guidance_scale = random.choice(guidance_scales)
            num_inference_steps = random.choice(num_inference_steps_list)
            image_tensor = pipe(
                prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                output_type="pt",
            ).images[0]
            image = tensor_to_pil(image_tensor)
            image.save(os.path.join(dataset_output_dir, f"{i+1}.png"))


def _parse_checkpoint_overrides(raw: Optional[str]) -> Dict[str, str]:
    if not raw:
        return {}
    overrides: Dict[str, str] = {}
    for item in raw.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(f"Invalid checkpoint override '{item}'. Use key=path format.")
        key, path = item.split("=", 1)
        overrides[key.strip()] = path.strip()
    return overrides


def _resolve_checkpoint_path(
    class_key: str,
    checkpoints_root: str,
    checkpoint_name: str,
    overrides: Dict[str, str],
) -> str:
    if class_key in overrides:
        return overrides[class_key]
    return os.path.join(checkpoints_root, class_key, checkpoint_name)


def generate_all_classes(
    classes: list,
    mode: str,
    output_dir: str,
    num_images_per_class: int,
    checkpoints_root: str,
    checkpoint_name: str,
    checkpoint_overrides: Dict[str, str],
    base_model_path: Optional[str] = None,
    seed: Optional[int] = None,
    output_layout: str = "nested",
):
    for idx, class_key in enumerate(classes):
        if class_key not in DEFAULT_CLASS_PROMPTS:
            raise ValueError(f"Unknown class '{class_key}'. Available: {sorted(DEFAULT_CLASS_PROMPTS)}")
        condition = DEFAULT_CLASS_PROMPTS[class_key]
        model_path = _resolve_checkpoint_path(class_key, checkpoints_root, checkpoint_name, checkpoint_overrides)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Checkpoint not found for {class_key}: {model_path}")
        class_seed = None if seed is None else seed + idx
        generate_images(
            condition=condition,
            mode=mode,
            output_dir=output_dir,
            num_images=num_images_per_class,
            model_path=model_path,
            base_model_path=base_model_path,
            output_subdir=class_key,
            seed=class_seed,
            output_layout=output_layout,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", type=str, default=None, help="Single class prompt. If omitted, generate all classes.")
    parser.add_argument("--mode", type=str, choices=["single", "dataset"], required=True)
    parser.add_argument("--output_dir", type=str, default="generated_images")
    parser.add_argument("--num_images", type=int, default=10, help="Images for single-class mode")
    parser.add_argument("--num_images_per_class", type=int, default=1500, help="Images per class for multi-class mode")
    parser.add_argument("--model_path", type=str, default="Lora/weights/checkpoint-15000", help="Path to LoRA model weights")
    parser.add_argument("--base_model_path", type=str, default=None, help="Path or repo id for the base Stable Diffusion model")
    parser.add_argument("--classes", type=str, default=",".join(DEFAULT_CLASS_ORDER), help="Comma-separated class keys")
    parser.add_argument("--checkpoints_root", type=str, default="lora_7classes", help="Root dir containing class checkpoints")
    parser.add_argument("--checkpoint_name", type=str, default="checkpoint-5000", help="Checkpoint dir name under each class")
    parser.add_argument(
        "--checkpoint_overrides",
        type=str,
        default=None,
        help="Override checkpoints per class: akiec=/path,mel=/path",
    )
    parser.add_argument("--seed", type=int, default=None, help="Base random seed")
    parser.add_argument(
        "--output_layout",
        choices=["nested", "flat"],
        default="nested",
        help="Output layout: nested (output_dir/dataset/class) or flat (output_dir/class)",
    )
    args = parser.parse_args()

    if args.condition:
        generate_images(
            args.condition,
            args.mode,
            args.output_dir,
            args.num_images,
            args.model_path,
            args.base_model_path,
            output_layout=args.output_layout,
        )
    else:
        class_list = [item.strip() for item in args.classes.split(",") if item.strip()]
        overrides = _parse_checkpoint_overrides(args.checkpoint_overrides)
        generate_all_classes(
            classes=class_list,
            mode=args.mode,
            output_dir=args.output_dir,
            num_images_per_class=args.num_images_per_class,
            checkpoints_root=args.checkpoints_root,
            checkpoint_name=args.checkpoint_name,
            checkpoint_overrides=overrides,
            base_model_path=args.base_model_path,
            seed=args.seed,
            output_layout=args.output_layout,
        )
