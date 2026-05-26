# SD1.5 LoRA + T2I-Adapter Training (HAM10000)

This script trains an **SD1.5 LoRA** while conditioning the UNet with a local **T2I-Adapter** using segmentation masks on the HAM10000 skin lesion dataset.

## Project Structure

```
Controlnet+T2i_adapter/
├── train_sd15_lora_controlnet_depth.py   # ControlNet training script
├── train_sd15_lora_t2i_adapter.py        # Main training script
├── generate_ham10000_lora_images.py      # Image generation script
└── README_lora_t2i_adapter.md            # This file
```

## Script

- `train_sd15_lora_t2i_adapter.py`

## Expected Inputs

- **Local SD1.5 diffusers model**: `/root/autodl-tmp/model/sd15-diffusers`
- **Local T2I-Adapter**: `/root/autodl-tmp/model/t2i-adapter`
- **CSVs** with `img_path`, `seg_path`, `llava_prompt` columns

## Data Format

The CSV files should contain the following columns:
- `img_path`: Path to input images
- `seg_path`: Path to segmentation masks
- `llava_prompt`: Text prompt for training

## Example Usage

```bash
/root/autodl-tmp/environment/skin/bin/python /root/autodl-tmp/compare_main/Controlnet+T2i_adapter/train_sd15_lora_t2i_adapter.py \
  --pretrained_model_name_or_path /root/autodl-tmp/model/sd15-diffusers \
  --adapter_path /root/autodl-tmp/model/t2i-adapter \
  --csv_paths /root/autodl-tmp/data/HAM10000/input/metadata_train_llava.csv,/root/autodl-tmp/data/HAM10000/input/metadata_val_llava.csv,/root/autodl-tmp/data/HAM10000/input/metadata_test_llava.csv \
  --output_dir /root/autodl-tmp/lora_t2i_output \
  --resolution 512 \
  --train_batch_size 1 \
  --max_train_steps 1000 \
  --mixed_precision fp16
```

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--pretrained_model_name_or_path` | Required | Path to SD1.5 diffusers model |
| `--adapter_path` | Required | Path to T2I-Adapter weights |
| `--csv_paths` | Required | Comma-separated CSV paths |
| `--image_root` | `/root/autodl-tmp` | Root path for resolving relative paths |
| `--output_dir` | `/root/autodl-tmp/checkpoint/compare_models/T2i_adapter` | Output directory |
| `--resolution` | 512 | Image resolution |
| `--train_batch_size` | 1 | Batch size |
| `--gradient_accumulation_steps` | 1 | Gradient accumulation steps |
| `--learning_rate` | 1e-4 | Learning rate |
| `--max_train_steps` | 1000 | Maximum training steps |
| `--mixed_precision` | fp16 | Mixed precision training |
| `--adapter_conditioning_scale` | 1.0 | T2I-Adapter conditioning scale |
| `--checkpointing_steps` | 1000 | Save checkpoint every N steps |
| `--seed` | 42 | Random seed |

## Output

- **LoRA weights**: `pytorch_lora_weights.safetensors`
- **Optional checkpoints**: `checkpoint-*` directories

## Notes

- Segmentation masks are used as the T2I-Adapter conditioning input.
- The script uses the `llava_prompt` column from CSVs as text conditioning.
- The T2I-Adapter provides additional conditioning to the UNet model during training.
