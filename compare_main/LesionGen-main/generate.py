import torch
import os
import random
import argparse
import sys
from pathlib import Path
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

def generate_images(
    condition,
    mode,
    output_dir,
    num_images,
    model_path="Lora/weights/checkpoint-15000",
    base_model_path=None,
    output_layout="nested",
):
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

    def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
        if image_tensor.ndim == 4:
            image_tensor = image_tensor[0]
        image_tensor = image_tensor.detach().cpu().clamp(0, 1).mul(255).byte()
        image_tensor = image_tensor.permute(1, 2, 0).contiguous()
        data = bytes(image_tensor.view(-1).tolist())
        return Image.frombytes("RGB", (image_tensor.shape[1], image_tensor.shape[0]), data)

    if mode == "single":
        single_output_dir = os.path.join(output_dir, "single")
        os.makedirs(single_output_dir, exist_ok=True)
        prompt = f"{condition} small size on a dark skin tone"
        image_tensor = pipe(prompt, num_inference_steps=50, guidance_scale=7.5, output_type="pt").images[0]
        image = tensor_to_pil(image_tensor)
        image.save(os.path.join(single_output_dir, f"{condition}.png"))
    elif mode == "dataset":
        if output_layout == "flat":
            dataset_output_dir = os.path.join(output_dir, condition)
        else:
            dataset_output_dir = os.path.join(output_dir, "dataset", condition)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["single", "dataset"], required=True)
    parser.add_argument("--output_dir", type=str, default="generated_images")
    parser.add_argument("--num_images", type=int, default=10)
    parser.add_argument("--model_path", type=str, default="Lora/weights/checkpoint-15000", help="Path to LoRA model weights")
    parser.add_argument("--base_model_path", type=str, default=None, help="Path or repo id for the base Stable Diffusion model")
    parser.add_argument(
        "--output_layout",
        choices=["nested", "flat"],
        default="nested",
        help="Output layout: nested (output_dir/dataset/condition) or flat (output_dir/condition)",
    )
    args = parser.parse_args()

    generate_images(
        args.condition,
        args.mode,
        args.output_dir,
        args.num_images,
        args.model_path,
        args.base_model_path,
        output_layout=args.output_layout,
    )
