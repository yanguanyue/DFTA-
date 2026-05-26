import sys

sys.path.append("../")
from torch.utils.data import Dataset, DataLoader
import albumentations
from PIL import Image
import numpy as np
from omegaconf import OmegaConf
from pathlib import Path
import cv2
from natsort import natsorted

workspace = Path("~/.workspace").expanduser().as_posix()


class PolypBase(Dataset):
    def __init__(
        self,
        config=None,
        name=None,
        size=384,
        images_dir=None,
        masks_dir=None,
        image_exts=None,
        mask_suffix="",
        mask_ext=".png",
        max_samples=None,
        seed=23,
        shuffle=True,
        **kwargs,
    ):
        self.kwargs = kwargs
        self.config = config or OmegaConf.create()
        if not type(self.config) == dict:
            self.config = OmegaConf.to_container(self.config)

        self.name = name
        self.size = size
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_exts = image_exts or [".png"]
        self.mask_suffix = mask_suffix
        self.mask_ext = mask_ext
        self.max_samples = max_samples
        self.seed = seed
        self.shuffle = shuffle

        self.preprocessor = albumentations.Compose(
            [
                albumentations.LongestMaxSize(max_size=self.size),
                albumentations.PadIfNeeded(
                    min_height=self.size,
                    min_width=self.size,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    mask_value=0,
                ),
            ]
        )

        self._prepare()

    def _prepare(self):
        if self.images_dir is None or self.masks_dir is None:
            if self.name is None:
                raise ValueError("Either name or images_dir/masks_dir must be provided")
            self.root = Path(workspace).joinpath("datasets/diffusion_datasets", self.name).as_posix()
            self.images_dir = Path(self.root).joinpath("images").as_posix()
            self.masks_dir = Path(self.root).joinpath("masks").as_posix()
            dataset_label = self.name
        else:
            dataset_label = Path(self.images_dir).name

        print(f"Preparing dataset {dataset_label}")

        images_list = []
        for ext in self.image_exts:
            images_list.extend(Path(self.images_dir).rglob(f"*{ext}"))

        self.images_list_absolute = [file_path.as_posix() for file_path in images_list]
        self.images_list_absolute = natsorted(self.images_list_absolute)

        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(self.images_list_absolute)

        if self.max_samples is not None:
            self.images_list_absolute = self.images_list_absolute[: int(self.max_samples)]

        self.masks_list_absolute = [
            Path(self.masks_dir)
            .joinpath(f"{Path(p).stem}{self.mask_suffix}{self.mask_ext}")
            .as_posix()
            for p in self.images_list_absolute
        ]

    def __getitem__(self, i):
        data = {}
        image = Image.open(self.images_list_absolute[i]).convert("RGB")
        image = np.array(image).astype(np.uint8)

        mask = Image.open(self.masks_list_absolute[i]).convert("L")
        mask = np.array(mask).astype(np.uint8)

        _preprocessor = self.preprocessor(image=image, mask=mask)
        image, mask = _preprocessor["image"], _preprocessor["mask"]

        image = (image / 127.5 - 1.0).astype(np.float32)
        mask = mask / 255

        data["image"], data["segmentation"] = image, mask

        return data

    def __len__(self):
        return len(self.images_list_absolute)

    def getitem(self, i):
        return self.__getitem__(i)


class PolypClassSample(PolypBase):
    def __init__(
        self,
        images_root,
        masks_root,
        classes,
        total_samples=100,
        size=384,
        image_exts=None,
        mask_suffix="",
        mask_ext=".png",
        seed=23,
        **kwargs,
    ):
        kwargs.pop("max_samples", None)
        self.images_root = images_root
        self.masks_root = masks_root
        self.classes = classes
        self.total_samples = int(total_samples)
        self.seed = seed
        kwargs.pop("max_samples", None)
        super().__init__(
            size=size,
            images_dir=images_root,
            masks_dir=masks_root,
            image_exts=image_exts,
            mask_suffix=mask_suffix,
            mask_ext=mask_ext,
            max_samples=None,
            seed=seed,
            shuffle=False,
            **kwargs,
        )

    def _prepare(self):
        rng = np.random.default_rng(self.seed)
        images_list_absolute = []

        classes = list(self.classes)
        if len(classes) == 0:
            raise ValueError("classes must not be empty")

        base_count = max(1, self.total_samples // len(classes))
        remainder = max(0, self.total_samples - base_count * len(classes))

        for idx, cls_name in enumerate(classes):
            images_dir = Path(self.images_root).joinpath(cls_name)
            masks_dir = Path(self.masks_root).joinpath(cls_name)
            images_list = []
            for ext in self.image_exts:
                images_list.extend(images_dir.rglob(f"*{ext}"))
            images_list = [file_path.as_posix() for file_path in images_list]
            images_list = natsorted(images_list)
            rng.shuffle(images_list)

            take_count = base_count + (1 if idx < remainder else 0)
            images_list_absolute.extend(images_list[:take_count])

        rng.shuffle(images_list_absolute)
        self.images_list_absolute = images_list_absolute[: self.total_samples]

        self.masks_list_absolute = [
            Path(self.masks_root)
            .joinpath(Path(p).parent.name)
            .joinpath(f"{Path(p).stem}{self.mask_suffix}{self.mask_ext}")
            .as_posix()
            for p in self.images_list_absolute
        ]


if __name__ == "__main__":
    trainset = PolypBase(name="public_polyp_train", size=352, train=True, load_mask=True, load_hc=False)
    _data = trainset.getitem(0)

    data_loader = DataLoader(trainset, batch_size=1)

    for data in data_loader:
        image = data["image"]
        print(image.shape)  # B H W C
        print(data["image_name"])
        break
