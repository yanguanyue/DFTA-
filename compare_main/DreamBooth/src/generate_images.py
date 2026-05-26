import argparse
import math
import os
from time import perf_counter

import torch


parser = argparse.ArgumentParser(description="Input to generate images script")
parser.add_argument(
    "--pretrained_model_name_or_path",
    type=str,
    default=None,
    required=True,
    help="Path to pretrained model in local file, e.g., '/home/amytai/ura-2024-oustan/malignant-model'.",
)

parser.add_argument(
    "--prompt",
    type=str,
    default=None,
    required=False,
    help="Prompt to use in the image generation, e.g., 'melanoma'",
)

parser.add_argument(
    "--prompts",
    type=str,
    default=None,
    help=(
        "Comma-separated prompts for multi-class generation, e.g., "
        "'melanoma,nevus,basal cell carcinoma'"
    ),
)

parser.add_argument(
    "--prompt_map",
    type=str,
    default=None,
    help=(
        "Comma-separated class=prompt pairs, e.g., "
        "'akiec=actinic keratosis,bcc=basal cell carcinoma'"
    ),
)

parser.add_argument(
    "--num_images",
    type=int,
    default=100,
    help="Number of images to generate (per class if --prompts is used)",
)

parser.add_argument("--gpu_id", type=int, default=2, help="GPU device ID")

parser.add_argument(
    "--output_folder",
    type=str,
    default=None,
    required=True,
    help="Path to output folder, e.g., '/home/amytai/ura-2024-oustan/data/jpeg/generated'.",
)

parser.add_argument(
    "--class_subdir",
    type=str,
    default="",
    help="Optional subfolder under each class folder (e.g., 'images').",
)

parser.add_argument(
    "--dry_run",
    action="store_true",
    help="If set, only prints planned outputs without loading the model.",
)

args = parser.parse_args()

start_time = perf_counter()
print("Starting script")

device = "cuda:{}".format(args.gpu_id) if args.gpu_id >= 0 else "cpu"

prompt_items = []
if args.prompt_map:
    pairs = [p.strip() for p in args.prompt_map.split(",") if p.strip()]
    for pair in pairs:
        if "=" not in pair:
            raise ValueError("Invalid --prompt_map entry: '{}' (expected class=prompt)".format(pair))
        class_name, prompt = pair.split("=", 1)
        class_name = class_name.strip()
        prompt = prompt.strip()
        if not class_name or not prompt:
            raise ValueError("Invalid --prompt_map entry: '{}'".format(pair))
        prompt_items.append((class_name, prompt))
elif args.prompts:
    prompt_items = [(p.strip(), p.strip()) for p in args.prompts.split(",") if p.strip()]
elif args.prompt:
    prompt_items = [(args.prompt, args.prompt)]
else:
    raise ValueError("Either --prompt, --prompts, or --prompt_map must be provided.")

def build_output_dir(base_dir: str, class_name: str, subdir: str) -> str:
    if subdir:
        return os.path.join(base_dir, class_name, subdir)
    return os.path.join(base_dir, class_name)

if args.dry_run:
    for class_name, prompt in prompt_items:
        output_dir = build_output_dir(args.output_folder, class_name, args.class_subdir)
        print(
            f"[DRY RUN] Would generate {args.num_images} images for '{prompt}' "
            f"into {output_dir}"
        )
else:
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.float16,
        local_files_only=True,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)
    # pipe.safety_checker = lambda images, clip_input: (images, False) # Remove safety check

    for class_name, prompt in prompt_items:
        output_dir = build_output_dir(args.output_folder, class_name, args.class_subdir)
        os.makedirs(output_dir, exist_ok=True)

        for start_idx in range(0, args.num_images, 4):
            batch_size = min(4, args.num_images - start_idx)
            images = pipe(prompt, num_images_per_prompt=batch_size).images

            for idx, image in enumerate(images):
                image.save(os.path.join(output_dir, f"{class_name}-{(start_idx + idx)}.jpg"))
    
end_time = perf_counter()
elapsed_time = end_time-start_time

print(f'Total time: {elapsed_time}')
