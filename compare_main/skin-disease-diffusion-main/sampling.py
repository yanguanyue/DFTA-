import argparse
from pathlib import Path
import os
import torch
import open_clip
from torchvision import utils
from models.vis_token_extractor import VisTokenExtractor
from models.diffusion.diffusion_pipeline import DiffusionPipeline

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
default_hf_home = "/root/autodl-tmp/model/hf_home"
if not Path(default_hf_home).exists():
    default_hf_home = "/root/autodl-tmp/.hf"
os.environ.setdefault("HF_HOME", default_hf_home)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("WANDB_MODE", "offline")


def parse_args():
    parser = argparse.ArgumentParser(description="Sample images from diffusion model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Diffusion checkpoint path")
    parser.add_argument("--vae-checkpoint", type=str, default="", help="Override VAE checkpoint path")
    parser.add_argument("--output-root", type=str, default="/root/autodl-tmp/output/generate/skin-disease-diffusion")
    parser.add_argument("--num-samples-per-class", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--num-classes", type=int, default=7)
    parser.add_argument("--steps", type=int, default=750)
    parser.add_argument("--use-ddim", action="store_true", default=True)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="Resume by skipping existing files")
    parser.add_argument("--img-size", type=str, default="8,32,32", help="Latent image size as C,H,W")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp16", "fp32"], help="Sampling precision")
    return parser.parse_args()


def parse_img_size(raw: str):
    parts = [int(p) for p in raw.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError("--img-size must be C,H,W (e.g., 8,32,32)")
    return tuple(parts)


CLASS_ABBRS = ["akiec", "bcc", "bkl", "df", "nv", "mel", "vasc"]


def _get_existing_count(output_dir: Path) -> int:
    return len(list(output_dir.glob("*.png")))


def _index_to_alpha(index: int) -> str:
    if index < 0:
        raise ValueError("Index must be non-negative")
    letters = []
    index += 1
    while index > 0:
        index -= 1
        letters.append(chr(ord('a') + (index % 26)))
        index //= 26
    return "".join(reversed(letters))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_fp16 = (args.precision == "fp16") and device.type == "cuda"

    clip_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _, _ = open_clip.create_model_and_transforms(
        model_name='ViT-H-14',
        pretrained='laion2b_s32b_b79k',
        device=clip_device,
        jit=False,
        precision='fp32',
    )

    vis_backbone = model.visual
    vis_backbone.eval().requires_grad_(False)
    for p in vis_backbone.parameters():
        p.requires_grad = False

    vis_extractor = VisTokenExtractor(
        backbone=vis_backbone,
        layer_ids=[5, 11, 17, 23, 31],
        k=32,
        proj_dim=1024,
        device=clip_device,
    ).eval()

    latent_ckpt = args.vae_checkpoint.strip()
    if latent_ckpt:
        latent_path = Path(latent_ckpt)
        if not latent_path.exists():
            raise FileNotFoundError(f"VAE checkpoint not found: {latent_path}")
        pipeline = DiffusionPipeline.load_from_checkpoint(
            args.checkpoint,
            vis_extractor=vis_extractor,
            latent_embedder_checkpoint=str(latent_path),
            weights_only=False,
        )
    else:
        pipeline = DiffusionPipeline.load_from_checkpoint(
            args.checkpoint,
            vis_extractor=vis_extractor,
            weights_only=False,
        )
        pipeline.to(device)
        if use_fp16:
            pipeline = pipeline.half()

    img_size = parse_img_size(args.img_size)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    class_ids = list(range(args.num_classes))
    existing_counts = {}
    next_indices = {}
    for class_id in class_ids:
        class_name = CLASS_ABBRS[class_id] if class_id < len(CLASS_ABBRS) else f"class_{class_id}"
        output_dir = output_root / class_name
        output_dir.mkdir(parents=True, exist_ok=True)
        if args.resume:
            existing_counts[class_id] = _get_existing_count(output_dir)
            next_indices[class_id] = existing_counts[class_id]
        else:
            existing_counts[class_id] = 0
            next_indices[class_id] = 0

    remaining = {
        class_id: max(0, args.num_samples_per_class - existing_counts[class_id])
        for class_id in class_ids
    }

    print("Round-robin generation across classes...")
    print(f"Target per class: {args.num_samples_per_class}")

    while any(count > 0 for count in remaining.values()):
        for class_id in class_ids:
            remaining_count = remaining[class_id]
            if remaining_count <= 0:
                continue

            class_name = CLASS_ABBRS[class_id] if class_id < len(CLASS_ABBRS) else f"class_{class_id}"
            output_dir = output_root / class_name
            current_batch_size = min(args.batch_size, remaining_count)

            condition = torch.tensor([class_id] * current_batch_size, device=device)
            if use_fp16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    samples = pipeline.sample(
                        num_samples=current_batch_size,
                        img_size=img_size,
                        steps=args.steps,
                        use_ddim=args.use_ddim,
                        guidance_scale=args.guidance_scale,
                        condition=condition,
                    ).detach()
            else:
                samples = pipeline.sample(
                    num_samples=current_batch_size,
                    img_size=img_size,
                    steps=args.steps,
                    use_ddim=args.use_ddim,
                    guidance_scale=args.guidance_scale,
                    condition=condition,
                ).detach()

            start_idx = next_indices[class_id]
            for i, sample in enumerate(samples):
                img_idx = start_idx + i
                alpha_idx = _index_to_alpha(img_idx)
                img = (sample - sample.min()) / (sample.max() - sample.min())
                utils.save_image(img, output_dir / f'{class_name}_{alpha_idx}.png')

            next_indices[class_id] += current_batch_size
            remaining[class_id] -= current_batch_size
            generated_so_far = args.num_samples_per_class - remaining[class_id]
            print(
                f"Class {class_id}: {generated_so_far}/{args.num_samples_per_class} images generated "
                f"(remaining {remaining[class_id]})"
            )

    print("Completed! All classes meet target counts.")


if __name__ == "__main__":
    main()