"""Generate images per class using val masks for pairing.

This script does NOT condition the generator on the input masks. It pairs each
synthesized image with a mask sampled from the validation set, as requested.
"""

import argparse
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import PIL.Image
import torch

import dnnlib
import legacy


CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def _find_latest_snapshot(run_dir: Path) -> Path:
    snapshots = sorted(run_dir.glob("network-snapshot-*.pkl"))
    if not snapshots:
        raise FileNotFoundError(f"No snapshots found in {run_dir}")

    def _kimg(p: Path) -> int:
        match = re.search(r"network-snapshot-(\d+)\.pkl", p.name)
        return int(match.group(1)) if match else -1

    snapshots.sort(key=_kimg)
    return snapshots[-1]


def _find_latest_run(runs_root: Path, class_name: str) -> Path:
    class_root = runs_root / f"ham10000_{class_name}_mask_512"
    if not class_root.exists():
        raise FileNotFoundError(f"Run directory not found: {class_root}")
    run_dirs = [p for p in class_root.iterdir() if p.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No run subdirectories found in {class_root}")
    run_dirs.sort(key=lambda p: p.name)
    return run_dirs[-1]


def _load_masks(mask_dir: Path) -> List[Path]:
    masks = sorted(mask_dir.glob("*.png"))
    if not masks:
        raise FileNotFoundError(f"No masks found in {mask_dir}")
    return masks


def _generate_image(
    G,
    device: torch.device,
    rng: np.random.RandomState,
    truncation_psi: float,
) -> torch.Tensor:
    z = torch.from_numpy(rng.randn(1, G.z_dim)).to(device)
    label = torch.zeros([1, G.c_dim], device=device)

    if hasattr(G, "transfer") and G.transfer != "none":
        defect_z = torch.from_numpy(rng.randn(1, G.z_dim)).to(device)
        ws = G.mapping(z, None)
        defect_ws = G.defect_mapping(defect_z, label, truncation_psi=truncation_psi)
        if G.transfer in ["res_block", "res_block_match_dis", "res_block_uni_dis"]:
            img, _ = G.synthesis(
                ws,
                defect_ws,
                noise_mode="const",
                output_mask=True,
                fix_residual_to_zero=False,
            )
        else:
            img = G.synthesis(
                ws,
                defect_ws,
                noise_mode="const",
                fix_residual_to_zero=False,
            )
    else:
        img = G(z, label, truncation_psi=truncation_psi, noise_mode="const")

    return img


def _save_image(img: torch.Tensor, out_path: Path) -> None:
    img = ((img.permute(0, 2, 3, 1) + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
    PIL.Image.fromarray(img[0].cpu().numpy(), "RGB").save(out_path)


def _save_mask(mask_path: Path, out_path: Path, size: Tuple[int, int]) -> None:
    mask = PIL.Image.open(mask_path).convert("L").resize(size, PIL.Image.NEAREST)
    mask.save(out_path)


def generate_per_class(
    class_name: str,
    runs_root: Path,
    mask_root: Path,
    out_root: Path,
    num: int,
    truncation_psi: float,
    seed: int,
) -> Dict[str, str]:
    run_dir = _find_latest_run(runs_root, class_name)
    snapshot = _find_latest_snapshot(run_dir)
    mask_dir = mask_root / class_name
    masks = _load_masks(mask_dir)

    image_out = out_root / class_name / "image"
    mask_out = out_root / class_name / "mask"
    image_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    with dnnlib.util.open_url(str(snapshot)) as f:
        G = legacy.load_network_pkl(f)["G_ema"].to(device)  # type: ignore

    rng = np.random.RandomState(seed)
    img_size = (G.img_resolution, G.img_resolution)

    for idx in range(num):
        mask_path = masks[rng.randint(0, len(masks))]
        img = _generate_image(G, device, rng, truncation_psi)

        _save_image(img, image_out / f"{idx:05d}.png")
        _save_mask(mask_path, mask_out / f"{idx:05d}.png", img_size)

    return {
        "class": class_name,
        "run_dir": str(run_dir),
        "snapshot": str(snapshot),
        "mask_dir": str(mask_dir),
        "out_dir": str(out_root / class_name),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate images per class using val masks.")
    parser.add_argument("--runs-root", type=Path, default=Path("/root/autodl-tmp/DFMGAN-main/runs"))
    parser.add_argument("--mask-root", type=Path, default=Path("/root/autodl-tmp/data/HAM10000/input/val/HAM10000_seg_class"))
    parser.add_argument("--out-root", type=Path, default=Path("/root/autodl-tmp/DFMGAN-main/generated_val_masks"))
    parser.add_argument("--num", type=int, default=1500)
    parser.add_argument("--trunc", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary = []
    for class_name in CLASSES:
        info = generate_per_class(
            class_name=class_name,
            runs_root=args.runs_root,
            mask_root=args.mask_root,
            out_root=args.out_root,
            num=args.num,
            truncation_psi=args.trunc,
            seed=args.seed + CLASSES.index(class_name),
        )
        summary.append(info)

    print("Generation completed:")
    for item in summary:
        print(
            f"{item['class']}: snapshot={item['snapshot']} mask_dir={item['mask_dir']} out={item['out_dir']}"
        )


if __name__ == "__main__":
    main()
