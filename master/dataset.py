import cv2
import json
import random
import numpy as np
from pathlib import Path

from torch.utils.data import Dataset
from PIL import Image
import albumentations
import torch


class MyDataset(Dataset):
    def __init__(self, prompt_path: Path | None = None, drop_prompt_prob: float = 0.05):
        self.data = []
        self.drop_prompt_prob = drop_prompt_prob
        prompt_path = prompt_path or (Path(__file__).resolve().parent / 'data' / 'prompt.json')
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        with open(prompt_path, 'rt') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                source_path = self._resolve_path(prompt_path.parent, item.get('source', ''))
                target_path = self._resolve_path(prompt_path.parent, item.get('target', ''))
                item['source'] = str(source_path)
                item['target'] = str(target_path)
                self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        if random.random() < self.drop_prompt_prob:
            prompt = ""

        source = Image.open(source_filename).convert('L')
        source_array = np.array(source)
        threshold = 127
        binary_array = np.where(source_array > threshold, 255, 0).astype(np.uint8)
        binary_image = Image.fromarray(binary_array)
        source = binary_image.convert('RGB')

        target = Image.open(target_filename).convert('RGB')

        source = np.array(source).astype(np.uint8)
        target = np.array(target).astype(np.uint8)

        preprocess = self.transform()(image=target, mask=source)
        source, target = preprocess['mask'], preprocess['image']

        ############ Mask-Image Pair ############
        source = source.astype(np.float32) / 255.0
        target = target.astype(np.float32) / 127.5 - 1.0

        return dict(
            jpg=torch.tensor(target, dtype=torch.float32).contiguous(),
            txt=prompt,
            hint=torch.tensor(source, dtype=torch.float32).contiguous(),
        )

    def transform(self, size=512):
        transforms = albumentations.Compose(
                        [
                            albumentations.Resize(height=size, width=size)
                        ]
                    )
        return transforms

    @staticmethod
    def _resolve_path(root: Path, filename: str) -> Path:
        if not filename:
            return root
        path = Path(filename)
        if path.is_absolute():
            return path
        return (root / path).resolve()
