from __future__ import annotations
from datetime import datetime
from pathlib import Path
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import argparse
import torch
from dataloader.datamodule import SimpleDataModule
from dataloader.dataset import HAM10000Dataset
from models.vae import VAE, DiagonalGaussianDistribution
import shutil
import os

IS_COLAB = False
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
default_hf_home = "/root/autodl-tmp/model/hf_home"
if not Path(default_hf_home).exists():
    default_hf_home = "/root/autodl-tmp/.hf"
os.environ.setdefault("HF_HOME", default_hf_home)
torch.set_float32_matmul_precision('high' if IS_COLAB else 'medium')


def parse_args():
    parser = argparse.ArgumentParser(description="Train VAE for skin-disease diffusion")
    parser.add_argument("--data-path", type=str, default="/root/autodl-tmp/data/HAM10000/input/train/HAM10000_img_class")
    parser.add_argument("--output-dir", type=str, default=str(Path.cwd() / "runs"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=1001)
    parser.add_argument("--min-epochs", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--limit-train-batches", type=float, default=1.0)
    parser.add_argument("--limit-val-batches", type=float, default=1.0)
    parser.add_argument("--log-every-n-steps", type=int, default=20)
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


args = parse_args()
current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
output_root = Path(args.output_dir)
output_root.mkdir(parents=True, exist_ok=True)
path_run_dir = output_root / current_time
path_run_dir.mkdir(parents=True, exist_ok=True)

DATA_PATH = args.data_path

model = VAE(
        in_channels=3,
        out_channels=3,
        emb_channels=8,  #4ch or 8ch
        spatial_dims=2,
        hid_chs=[64, 128, 256, 512],
        kernel_sizes=[3, 3, 3, 3],
        strides=[1, 2, 2, 2],
        deep_supervision=1,
        use_attention="none",
        loss=torch.nn.MSELoss,          # reconstruction loss
        embedding_loss_weight=1e-6, # KL term weight
        sample_every_n_steps=500,

    )

use_wandb = os.environ.get("WANDB_MODE", "offline") != "disabled"
logger = (
        WandbLogger(
            project="skin-lesion-vae",
            name=f"vae_{current_time}",
            log_model=False,
        )
        if use_wandb
        else False
    )

to_monitor = "train/L1" if args.debug else "val/L1"
mode = "min"
save_and_sample_every = 100

early_stopping = EarlyStopping(
        monitor=to_monitor,
        min_delta=0.01,
        patience=30,
        mode=mode,
    )

checkpointing = ModelCheckpoint(
            dirpath=str(path_run_dir),
            monitor=to_monitor,
            # every_n_train_steps=280,
            save_last=True,
            save_top_k=2,
            mode=mode,
        )

train_ds = HAM10000Dataset(
        path_root=DATA_PATH,
        split="train",
        image_crop=450,  # 고정 크기로 center crop
        image_resize=(256, 256),  # 정사각형으로 리사이즈
        val_ratio=0.1,
        random_seed=42,
        augment_horizontal_flip=True,
        augment_vertical_flip=True,
    )

val_ds = HAM10000Dataset(
        path_root=DATA_PATH,
        split="val",
        image_crop=450,  # 고정 크기로 center crop
        image_resize=(256, 256),  # 정사각형으로 리사이즈
        val_ratio=0.1,
        random_seed=42,
    )

train_batch_size = args.batch_size
num_workers = args.num_workers
limit_train_batches = args.limit_train_batches
limit_val_batches = args.limit_val_batches
max_epochs = args.max_epochs
min_epochs = args.min_epochs
log_every_n_steps = args.log_every_n_steps
num_sanity_val_steps = 2

if args.debug:
    train_batch_size = min(train_batch_size, 4)
    num_workers = 0
    limit_train_batches = min(limit_train_batches, 0.05)
    limit_val_batches = min(limit_val_batches, 0.2)
    max_epochs = min(max_epochs, 1)
    min_epochs = 1
    log_every_n_steps = 1
    num_sanity_val_steps = 0
    if args.max_steps is None:
        args.max_steps = 5

dm = SimpleDataModule(
    ds_train=train_ds,
    ds_val=val_ds,
    batch_size=train_batch_size,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=False,  # DDP spawn 최적화
    weights=train_ds.get_weights(),
)

accelerator = "gpu" if torch.cuda.is_available() else "cpu"
devices = [0] if torch.cuda.is_available() else 1

trainer = Trainer(
    accelerator=accelerator,
    devices=devices,
    precision=32,
    default_root_dir=str(path_run_dir),
    callbacks=[checkpointing, early_stopping],
    enable_checkpointing=True,
    check_val_every_n_epoch=1,
    log_every_n_steps=log_every_n_steps,
    limit_val_batches=limit_val_batches,
    limit_train_batches=limit_train_batches,
    max_epochs=max_epochs,
    min_epochs=min_epochs,
    max_steps=args.max_steps,
    num_sanity_val_steps=num_sanity_val_steps,
    logger=logger,
    gradient_clip_val=1.0,
    accumulate_grad_batches=1,
    fast_dev_run=args.fast_dev_run,
)

trainer.fit(model, datamodule=dm)

last_ckpt = checkpointing.last_model_path
if last_ckpt and Path(last_ckpt).exists():
    shutil.copy2(last_ckpt, output_root / "vae_last.ckpt")
if checkpointing.best_model_path:
    best_path = Path(checkpointing.best_model_path)
    if best_path.exists():
        shutil.copy2(best_path, output_root / "vae_best.ckpt")
