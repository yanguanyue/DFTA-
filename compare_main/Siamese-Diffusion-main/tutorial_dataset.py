import cv2
import json
import random
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import albumentations
import torch


class MyDataset(Dataset):
    def __init__(self, prompt_json: str | None = None, size: int = 512):
        self.data = []
        self.size = size
        root = self._resolve_prompt_json(prompt_json)
        with open(root, 'rt') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        p = random.random()
        if p > 0.95:
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

    def transform(self, size=None):
        if size is None:
            size = self.size
        transforms = albumentations.Compose(
                        [
                            albumentations.Resize(height=size, width=size)
                        ]
                    )
        return transforms

    @staticmethod
    def _resolve_prompt_json(prompt_json: str | None) -> str:
        if prompt_json:
            return prompt_json
        repo_root = Path(__file__).resolve().parent
        local_prompt = repo_root / 'data' / 'prompt.json'
        if local_prompt.exists():
            return str(local_prompt)
        fallback_prompt = Path('/root/autodl-tmp/main/Siamese-Diffusion-main/data/prompt.json')
        if fallback_prompt.exists():
            return str(fallback_prompt)
        raise FileNotFoundError('prompt.json not found; please pass --prompt-json')
