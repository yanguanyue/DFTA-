import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train Siamese-Diffusion with flow environment defaults.")
	parser.add_argument(
		"--pretrained-ckpt",
		type=str,
		default="/root/autodl-tmp/checkpoint/compare_models/Siamese/PRETRAINED/merged_pytorch_model.pth",
	)
	parser.add_argument(
		"--config",
		type=str,
		default=str(Path(__file__).resolve().parent / "models" / "cldm_v15.yaml"),
	)
	parser.add_argument("--prompt-json", type=str, default=None)
	parser.add_argument(
		"--output-dir",
		type=str,
		default="/root/autodl-tmp/checkpoint/compare_models/Siamese",
	)
	parser.add_argument("--batch-size", type=int, default=1)
	parser.add_argument("--num-workers", type=int, default=4)
	parser.add_argument("--logger-freq", type=int, default=200)
	parser.add_argument("--learning-rate", type=float, default=1e-5)
	parser.add_argument("--max-steps", type=int, default=3000)
	parser.add_argument("--devices", type=int, default=1)
	parser.add_argument("--accelerator", type=str, default="gpu")
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--gpu-ids", type=str, default="0")
	parser.add_argument("--resume-ckpt", type=str, default=None)
	parser.add_argument("--save-merged-path", type=str, default=None)
	parser.add_argument("--image-size", type=int, default=512)
	parser.add_argument("--sd-locked", action="store_true", default=False)
	parser.add_argument("--only-mid-control", action="store_true", default=False)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	if args.gpu_ids:
		os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

	_ijit_stub = Path(__file__).resolve().parent.parent / "libijit_stub.so"
	if _ijit_stub.exists():
		os.environ.setdefault("LD_PRELOAD", str(_ijit_stub))

	import share  # noqa: F401
	import torch
	import pytorch_lightning as pl
	from torch.utils.data import DataLoader
	from pytorch_lightning.callbacks import ModelCheckpoint
	from tutorial_dataset import MyDataset
	from cldm.logger import ImageLogger
	from cldm.model import create_model, load_state_dict

	pl.seed_everything(args.seed, workers=True)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

	pretrained_ckpt = Path(args.pretrained_ckpt)
	if not pretrained_ckpt.exists():
		raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_ckpt}")

	model = create_model(args.config).cpu()
	model.load_state_dict(load_state_dict(str(pretrained_ckpt), location="cpu"), strict=False)

	model.learning_rate = args.learning_rate
	model.sd_locked = args.sd_locked
	model.only_mid_control = args.only_mid_control

	dataset = MyDataset(prompt_json=args.prompt_json, size=args.image_size)
	dataloader = DataLoader(
		dataset,
		num_workers=args.num_workers,
		batch_size=args.batch_size,
		shuffle=True,
		drop_last=True,
	)
	logger = ImageLogger(batch_frequency=args.logger_freq)
	checkpoint_cb = ModelCheckpoint(
		every_n_train_steps=500,
		save_top_k=1,
		save_last=True,
		monitor="train/loss",
		mode="min",
		filename="step-{step}",
	)

	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	trainer = pl.Trainer(
		accelerator=args.accelerator,
		devices=args.devices,
		callbacks=[logger, checkpoint_cb],
		deterministic=True,
		max_steps=args.max_steps,
		default_root_dir=str(output_dir),
	)

	trainer.fit(model, dataloader, ckpt_path=args.resume_ckpt)

	merged_path = args.save_merged_path
	if not merged_path:
		merged_path = str(output_dir / "merged_pytorch_model.pth")
	merged_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
	torch.save(merged_state, merged_path)
	print(f"[√] Saved merged weights to {merged_path}")


if __name__ == "__main__":
	main()
