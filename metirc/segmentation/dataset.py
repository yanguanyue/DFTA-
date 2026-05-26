import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _list_images(folder: Path) -> List[Path]:
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


@dataclass
class Sample:
    image_path: Path
    mask_path: Path
    class_name: str


class JointTransform:
    def __init__(self, image_size: int, is_train: bool = True):
        self.image_size = image_size
        self.is_train = is_train

    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.is_train:
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if random.random() < 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

        image = TF.resize(image, [self.image_size, self.image_size])
        mask = TF.resize(mask, [self.image_size, self.image_size], interpolation=TF.InterpolationMode.NEAREST)

        image_arr = np.array(image, dtype=np.float32, copy=True) / 255.0
        if image_arr.ndim == 2:
            image_arr = np.stack([image_arr] * 3, axis=-1)
        image = torch.tensor(image_arr, dtype=torch.float32).permute(2, 0, 1).contiguous()
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        mask_arr = np.array(mask, dtype=np.uint8, copy=True)
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[..., 0]
        mask = torch.tensor(mask_arr, dtype=torch.uint8)
        mask = (mask > 0).long()
        return image, mask


def _build_image_index(image_dir: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for img in _list_images(image_dir):
        index[img.stem] = img
    return index


def _resolve_pair(mask_path: Path, image_index: Dict[str, Path]) -> Optional[Path]:
    mask_stem = mask_path.stem
    if mask_stem.endswith("_segmentation"):
        mask_key = mask_stem.replace("_segmentation", "")
    else:
        mask_key = mask_stem
    return image_index.get(mask_key)


def _collect_samples(image_root: Path, mask_root: Path) -> List[Sample]:
    class_names = sorted([p.name for p in image_root.iterdir() if p.is_dir()])
    samples: List[Sample] = []
    for cls in class_names:
        img_dir = image_root / cls
        mask_dir = mask_root / cls
        if not mask_dir.exists():
            continue
        image_index = _build_image_index(img_dir)
        for mask_path in _list_images(mask_dir):
            image_path = _resolve_pair(mask_path, image_index)
            if image_path is None:
                continue
            samples.append(Sample(image_path=image_path, mask_path=mask_path, class_name=cls))
    return samples


class SegmentationDataset(Dataset):
    def __init__(
        self,
        image_root: Path,
        mask_root: Path,
        transform: Optional[Callable[[Image.Image, Image.Image], Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> None:
        self.image_root = image_root
        self.mask_root = mask_root
        self.transform = transform
        self.samples = _collect_samples(image_root, mask_root)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        image = Image.open(sample.image_path).convert("RGB")
        mask = Image.open(sample.mask_path).convert("L")
        if self.transform:
            image, mask = self.transform(image, mask)
        return {"image": image, "mask": mask}


def resolve_roots(dataset_root: Path, split: str, layout: str) -> Tuple[Path, Path]:
    if layout == "ham10000":
        image_root = dataset_root / split / "HAM10000_img_class"
        mask_root = dataset_root / split / "HAM10000_seg_class"
    elif layout == "mixed":
        image_root = dataset_root / split / "images"
        mask_root = dataset_root / split / "masks"
    else:
        raise ValueError(f"Unknown layout: {layout}")
    return image_root, mask_root
