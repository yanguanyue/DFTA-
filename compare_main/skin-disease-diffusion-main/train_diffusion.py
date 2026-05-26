from __future__ import annotations

import os
import argparse
from pathlib import Path
import shutil

# Hugging Face 镜像（必须在导入 open_clip / huggingface_hub 之前设置）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
default_hf_home = "/root/autodl-tmp/model/hf_home"
if not Path(default_hf_home).exists():
    default_hf_home = "/root/autodl-tmp/.hf"
os.environ.setdefault("HF_HOME", default_hf_home)
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_SILENT", "true")


def parse_args():
    parser = argparse.ArgumentParser(description="Train diffusion pipeline")
    parser.add_argument("--debug", action="store_true", help="Enable lightweight debug run")
    parser.add_argument("--fast-dev-run", action="store_true", help="Lightning fast_dev_run")
    parser.add_argument("--disable-wandb-images", action="store_true", help="Disable wandb image logging")
    parser.add_argument("--sample-every-n-steps", type=int, default=55, help="Sampling/logging interval")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--max-epochs", type=int, default=1001, help="Max training epochs")
    parser.add_argument("--min-epochs", type=int, default=100, help="Min training epochs")
    parser.add_argument("--max-steps", type=int, default=None, help="Max training steps")
    parser.add_argument("--limit-train-batches", type=float, default=1.0, help="Fraction of train batches")
    parser.add_argument("--limit-val-batches", type=float, default=1.0, help="Fraction of val batches")
    parser.add_argument("--log-every-n-steps", type=int, default=10, help="Logging frequency")
    parser.add_argument("--devices", type=str, default="0", help="GPU device ids, e.g. '0' or '0,1'")
    parser.add_argument("--data-path", type=str, default="/root/autodl-tmp/data/HAM10000/input/train/HAM10000_img_class")
    parser.add_argument("--output-dir", type=str, default=str(Path.cwd() / "runs"))
    parser.add_argument("--vae-checkpoint", type=str, required=True, help="Path to trained VAE checkpoint")
    return parser.parse_args()


args = parse_args()
vae_checkpoint_path = Path(args.vae_checkpoint)
if not vae_checkpoint_path.exists():
    raise FileNotFoundError(f"VAE checkpoint not found: {vae_checkpoint_path}")

from datetime import datetime
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from dataloader.datamodule import SimpleDataModule
from dataloader.dataset import HAM10000Dataset
from models.vae import VAE, DiagonalGaussianDistribution
from models.vis_token_extractor import VisTokenExtractor
from torch.utils.data import DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import BasicTransform
import cv2
import subprocess
import shutil
import wandb
import torch
import open_clip
import torch.nn as nn
import numpy as np
import numpy.core.multiarray as _np_multiarray

# 兼容：某些 numpy 构建缺少 multiarray.inexact，wandb 内部会访问该属性
if not hasattr(_np_multiarray, "inexact"):
    _np_multiarray.inexact = np.inexact
from torchvision import transforms as T

from models.diffusion.diffusion_pipeline import DiffusionPipeline
from models.diffusion.unet import UNet
from models.diffusion.gaussian_scheduler import GaussianNoiseScheduler
from models.diffusion.label_embedder import LabelEmbedder
from models.diffusion.time_embedder import TimeEmbedding


DATA_PATH = args.data_path


"""
OpenCLIP ViT-H/14
"""

clip_device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, _, preprocess = open_clip.create_model_and_transforms(
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
    
GLOBAL_VIS_BACKBONE = vis_backbone



vis_extractor = VisTokenExtractor(
    backbone=GLOBAL_VIS_BACKBONE,
    layer_ids=[5,11,17,23,31],
    k=32,
    proj_dim=1024,
    device=clip_device,
).eval()

"""
Data augmentation
"""
class FromPIL(BasicTransform):
    """Convert PIL Image to numpy array"""
    @property
    def targets(self):
        return {"image": self.apply}

    def apply(self, img, **params):
        return np.array(img)

    def get_transform_init_args_names(self):
        return []

strong_aug = T.Compose([
    T.RandomRotation(13),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.Resize((256, 256)),
    T.ConvertImageDtype(torch.float),
    T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])

val_aug = T.Compose([
    T.Resize((256, 256)),
    T.ConvertImageDtype(torch.float),
    T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])


train_batch_size = args.batch_size
num_workers = args.num_workers
limit_train_batches = args.limit_train_batches
limit_val_batches = args.limit_val_batches
max_epochs = args.max_epochs
min_epochs = args.min_epochs
log_every_n_steps = args.log_every_n_steps
num_sanity_val_steps = 2
sampler_num_samples = 7000

if args.debug:
    train_batch_size = min(train_batch_size, 4)
    num_workers = 0
    limit_train_batches = min(limit_train_batches, 0.05)
    limit_val_batches = min(limit_val_batches, 0.2)
    max_epochs = min(max_epochs, 1)
    min_epochs = 1
    log_every_n_steps = 1
    num_sanity_val_steps = 0
    sampler_num_samples = min(sampler_num_samples, 200)
    if args.max_steps is None:
        args.max_steps = 5

train_ds = HAM10000Dataset(
        path_root=DATA_PATH,
        split="train",
        val_ratio=0.1,
        random_seed=42,
        transform= strong_aug
    )


val_ds = HAM10000Dataset(
        path_root=DATA_PATH,
        split="val",
        image_crop=450,  
        image_resize=(256, 256),
        val_ratio=0.1,
        random_seed=42,
    transform=val_aug,
    )

dm = SimpleDataModule(
    ds_train=train_ds,
    ds_val=val_ds,
    batch_size=train_batch_size,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=False,
    weights=train_ds.get_weights(),
    balanced_epoch=True,
    sampler_num_samples=sampler_num_samples
)

current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
output_root = Path(args.output_dir)
output_root.mkdir(parents=True, exist_ok=True)
path_run_dir = output_root / str(current_time)
path_run_dir.mkdir(parents=True, exist_ok=True)
accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'

cond_embedder = LabelEmbedder
cond_embedder_kwargs = {
    'emb_dim': 1024,
    'num_classes': 7 # class number
}

time_embedder = TimeEmbedding
time_embedder_kwargs ={
    'emb_dim': 1024 # stable diffusion uses 4*model_channels (model_channels is about 256)
}

noise_scheduler = GaussianNoiseScheduler
noise_scheduler_kwargs = {
    'timesteps': 1000,
    'beta_start': 0.002, # 0.0001, 0.0015
    'beta_end': 0.02, # 0.01, 0.0195
    'schedule_strategy': 'scaled_linear'
}

noise_estimator = UNet
noise_estimator_kwargs = {
    'in_ch':8, #4ch
    'out_ch':8, #4ch
    'spatial_dims':2,
    'hid_chs':  [256, 256, 512, 1024],
    'kernel_sizes':[3, 3, 3, 3],
    'strides':     [1, 2, 2, 2],
    'time_embedder':time_embedder,
    'time_embedder_kwargs': time_embedder_kwargs,
    'cond_embedder':cond_embedder,
    'cond_embedder_kwargs': cond_embedder_kwargs,
    'deep_supervision': False,
    'use_res_block':True,
    'use_attention':'none',
    'use_vis_adapter': True # vis_adapter
    }

latent_embedder = VAE
latent_embedder_checkpoint = args.vae_checkpoint

pipeline = DiffusionPipeline(
    noise_estimator=noise_estimator,
    noise_estimator_kwargs=noise_estimator_kwargs,
    noise_scheduler=noise_scheduler,
    noise_scheduler_kwargs = noise_scheduler_kwargs,
    latent_embedder=latent_embedder,
    latent_embedder_checkpoint = latent_embedder_checkpoint,
    vis_extractor = vis_extractor,    # vis_extractor
    beta          = 0.02,
    estimator_objective='x_T',
    estimate_variance=False,
    use_self_conditioning=False,
    use_ema=False,
    classifier_free_guidance_dropout=0.1, 
    do_input_centering=False,
    clip_x0=False,
    sample_every_n_steps=args.sample_every_n_steps,
)
if args.disable_wandb_images:
    pipeline._disable_wandb_image_log = True

to_monitor = "train/loss" if args.debug else "val/loss"
min_max = "min"
save_and_sample_every = 100

logger = WandbLogger(
        project="skin-lesion-diffusion-vis",
        name=f"diffusion_{current_time}",
        log_model=False,
    )

early_stopping = EarlyStopping(
        monitor=to_monitor,
        min_delta=0.0, 
        patience=30, 
        mode='min'
    )
checkpointing = ModelCheckpoint(
    dirpath=str(path_run_dir), 
    monitor=to_monitor,
    # every_n_train_steps=20,
    save_last=True,
    save_top_k=2,
    mode=min_max,
)

devices = [int(d.strip()) for d in args.devices.split(",") if d.strip().isdigit()]
devices = devices if accelerator == 'gpu' else 1

trainer = Trainer(
        accelerator=accelerator,
    devices=devices,
        default_root_dir=str(path_run_dir),
        callbacks=[checkpointing, early_stopping],
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
    log_every_n_steps=log_every_n_steps,
    limit_val_batches=limit_val_batches,
    limit_train_batches=limit_train_batches,
        logger=logger,
        min_epochs=min_epochs,
    max_epochs=max_epochs,
    max_steps=args.max_steps,
    num_sanity_val_steps=num_sanity_val_steps,
    fast_dev_run=args.fast_dev_run,
    )

trainer.fit(pipeline, datamodule=dm)

last_ckpt = checkpointing.last_model_path
if last_ckpt and Path(last_ckpt).exists():
    shutil.copy2(last_ckpt, output_root / "diffusion_last.ckpt")
if checkpointing.best_model_path:
    best_path = Path(checkpointing.best_model_path)
    if best_path.exists():
        shutil.copy2(best_path, output_root / "diffusion_best.ckpt")