import argparse
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DFTA (Dual-Flow Trajectory Alignment) with flow matching.")
    parser.add_argument("--config", type=str, default=str(Path(__file__).resolve().parent / "config" / "flow_matching.yaml"))
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--resume-ckpt", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--logger-freq", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--sd-locked", action="store_true", default=False)
    parser.add_argument("--only-mid-control", action="store_true", default=False)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--output-dir", type=str, default="/root/autodl-tmp/checkpoint/flow")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda-visible-devices", type=str, default=None)
    parser.add_argument("--disable-checkpoint", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    os.environ.setdefault("TMPDIR", "/root/autodl-tmp")

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    # Fix missing iJIT_NotifyEvent by preloading stub library if present.
    _ijit_stub = Path(__file__).resolve().parent.parent / "libijit_stub.so"
    if _ijit_stub.exists():
        os.environ.setdefault("LD_PRELOAD", str(_ijit_stub))

    import share  # noqa: F401

    import torch
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader
    from dataset import MyDataset
    from model.log import ImageLogger
    from pytorch_lightning.callbacks import ModelCheckpoint
    from cldm.model import create_model, load_state_dict

    pl.seed_everything(args.seed, workers=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    config_path = Path(args.config)
    model = create_model(str(config_path)).cpu()
    if args.resume:
        state_dict = load_state_dict(args.resume, location="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"Loaded checkpoint with {len(missing)} missing and {len(unexpected)} unexpected keys")
    else:
        print("Training from scratch (no --resume provided)")

    model.learning_rate = args.learning_rate
    model.sd_locked = args.sd_locked
    model.only_mid_control = args.only_mid_control

    dataset = MyDataset()
    dataloader = DataLoader(
        dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    callbacks = []
    if args.logger_freq and args.logger_freq > 0:
        callbacks.append(ImageLogger(batch_frequency=args.logger_freq, log_first_step=True))

    if not args.disable_checkpoint:
        checkpoint_cb = ModelCheckpoint(
            every_n_train_steps=500,
            save_top_k=1,
            save_last=True,
            monitor="train/loss",
            mode="min",
            filename="step-{step}",
        )
        callbacks.append(checkpoint_cb)

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=args.devices,
        callbacks=callbacks,
        enable_checkpointing=not args.disable_checkpoint,
        deterministic=True,
        max_steps=args.max_steps,
        default_root_dir=args.output_dir,
    )

    if args.resume_ckpt:
        trainer.fit(model, dataloader, ckpt_path=args.resume_ckpt)
    else:
        trainer.fit(model, dataloader)


if __name__ == "__main__":
    main()
