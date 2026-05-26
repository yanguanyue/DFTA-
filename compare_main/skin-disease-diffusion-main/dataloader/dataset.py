import torch.utils.data as data 
import torch 
from torch import nn
from pathlib import Path 
from torchvision import transforms as T
from torchvision.io import read_image
import pandas as pd 
import numpy as np
import albumentations as A
from PIL import Image

import random
from sklearn.model_selection import train_test_split


class SimpleDataset2D(data.Dataset):
    def __init__(
        self,
        path_root,
        item_pointers =[],
        crawler_ext = 'tif', # other options are ['jpg', 'jpeg', 'png', 'tiff'],
        transform = None,
        image_resize = None,
        augment_horizontal_flip = False,
        augment_vertical_flip = False, 
        image_crop = None,
    ):
        super().__init__()
        self.path_root = Path(path_root)
        self.crawler_ext = crawler_ext
        if len(item_pointers):
            self.item_pointers = item_pointers
        else:
            self.item_pointers = self.run_item_crawler(self.path_root, self.crawler_ext) 

        if transform is None: 
            self.transform = T.Compose([
                T.CenterCrop(image_crop) if image_crop is not None else nn.Identity(),  # 먼저 crop
                T.Resize(image_resize) if image_resize is not None else nn.Identity(),   # 그 다음 resize
                T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
                T.RandomVerticalFlip() if augment_vertical_flip else nn.Identity(),
                T.ConvertImageDtype(torch.float),
                T.Normalize(mean=0.5, std=0.5) # WARNING: mean and std are not the target values but rather the values to subtract and divide by: [0, 1] -> [0-0.5, 1-0.5]/0.5 -> [-1, 1]
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.item_pointers)

    def __getitem__(self, index):
        rel_path_item = self.item_pointers[index]
        path_item = self.path_root/rel_path_item
        # img = Image.open(path_item) 
        if isinstance(self.transform, A.Compose):
            # Albumentations (expects numpy)
            img = self.load_item(path_item)
            img_np = np.array(img)
            result = self.transform(image=img_np)
            img_tensor = result['image']
        else:
            # torchvision (use tensor pipeline)
            img_tensor = read_image(str(path_item))
            img_tensor = self.transform(img_tensor)


        return {'uid':rel_path_item.stem, 'source': img_tensor}
    
    def load_item(self, path_item):
        return Image.open(path_item).convert('RGB') 
        # return cv2.imread(str(path_item), cv2.IMREAD_UNCHANGED) # NOTE: Only CV2 supports 16bit RGB images 
    
    @classmethod
    def run_item_crawler(cls, path_root, extension, **kwargs):
        return [path.relative_to(path_root) for path in Path(path_root).rglob(f'*.{extension}')]

    def get_weights(self):
        """Return list of class-weights for WeightedSampling"""
        return None 
    
    
class HAM10000Dataset(SimpleDataset2D):
    """
    HAM10000 7-class Dataset.
    디렉터리 구조:
        ham10k/
            actinic_keratoses/*.jpg
            basal_cell_carcinoma/*.jpg
            ...
            vascular_lesion/*.jpg
    """

    CLASS2IDX = {
        # Full names
        "actinic_keratoses":    0,
        "basal_cell_carcinoma": 1,
        "benign_keratosis":     2,
        "dermatofibroma":       3,
        "melanocytic_nevi":     4,
        "melanoma":             5,
        "vascular_lesion":      6,
        # Common abbreviations (HAM10000 class folders)
        "akiec": 0,
        "bcc":   1,
        "bkl":   2,
        "df":    3,
        "nv":    4,
        "mel":   5,
        "vasc":  6,
    }

    def __init__(
        self,
        path_root: str,
        split: str = "train",        # "train" | "val" | "all"
        val_ratio: float = 0.1,
        random_seed: int = 42,
        **kwargs,                    # SimpleDataset2D 인자 그대로 전달
    ):
        # kwargs.setdefault("crawler_ext", "jpg")  # 확장자만 고정
        # super().__init__(path_root=path_root, **kwargs)
#####################################################
        # crawler_ext 제거하고 item_pointers를 직접 생성
        kwargs.pop("crawler_ext", None)

        # 먼저 빈 item_pointers로 부모 클래스 초기화
        super().__init__(path_root=path_root, item_pointers=[], **kwargs)
        
        # jpg와 png 파일 모두 찾기
        from pathlib import Path
        path = Path(path_root)
        item_pointers = []
        for ext in ['jpg', 'jpeg', 'png']:
            item_pointers.extend(path.rglob(f'*.{ext}'))
        
        # 상대 경로로 변환하고 정렬
        self.item_pointers = [f.relative_to(path) for f in sorted(item_pointers)]
##############################################         
        if split != "all":
            train_idx, val_idx = train_test_split(
                list(range(len(self.item_pointers))),
                test_size=val_ratio,
                random_state=random_seed,
                shuffle=True,
                stratify=[p.parent.name for p in self.item_pointers],
            )
            keep = train_idx if split == "train" else val_idx
            self.item_pointers = [self.item_pointers[i] for i in keep]

    def __getitem__(self, idx):
        rel_path = self.item_pointers[idx]
        path_img = Path(self.path_root) / rel_path
        # Transform 처리 - if문으로 분기
        if isinstance(self.transform, A.Compose):
            # Albumentations
            img = self.load_item(path_img)
            img_np = np.array(img)
            result = self.transform(image=img_np)
            img_tensor = result['image']
        else:
            # torchvision (use tensor pipeline)
            img_tensor = read_image(str(path_img))
            img_tensor = self.transform(img_tensor)

        return {
            "uid":   rel_path.stem,
            "source": img_tensor,           # SimpleDataset2D 기본 transform
            "target": self.CLASS2IDX[path_img.parent.name],
        }
        
    def get_weights(self):
        if not hasattr(self, "_weights"):
            counts = {c:0 for c in self.CLASS2IDX}
            for p in self.item_pointers:
                counts[p.parent.name] += 1
            freq = {k: v/sum(counts.values()) for k,v in counts.items()}
            self._weights = [1/freq[p.parent.name] for p in self.item_pointers]
        return self._weights

    