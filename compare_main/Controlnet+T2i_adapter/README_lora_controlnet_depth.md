# SD1.5 LoRA + ControlNet Depth Training (HAM10000)

This script trains a **LoRA** on SD1.5 while using **ControlNet Depth** conditioning (mask as control image) on the HAM10000 skin lesion dataset.

## Project Structure

```
Controlnet+T2i_adapter/
├── train_sd15_lora_controlnet_depth.py   # Main training script
├── train_sd15_lora_t2i_adapter.py        # T2I-Adapter training script
├── generate_ham10000_lora_images.py      # Image generation script
└── README_lora_controlnet_depth.md       # This file
```

## Script

- `train_sd15_lora_controlnet_depth.py`

## Inputs

- **SD1.5 diffusers model**: `/root/autodl-tmp/model/sd15-diffusers`
- **ControlNet depth model**: `/root/autodl-tmp/model/sd-controlnet-depth`
- **CSVs**: `metadata_train_llava.csv`, `metadata_val_llava.csv`, `metadata_test_llava.csv`
- The script uses the `llava_prompt` column for text prompts.

## Data Format

The CSV files should contain the following columns:
- `img_path`: Path to input images
- `seg_path`: Path to segmentation masks
- `llava_prompt`: Text prompt for training

## Example Run

```bash
/root/autodl-tmp/environment/skin/bin/python /root/autodl-tmp/compare_main/Controlnet+T2i_adapter/train_sd15_lora_controlnet_depth.py \
  --pretrained_model_name_or_path /root/autodl-tmp/model/sd15-diffusers \
  --controlnet_path /root/autodl-tmp/model/sd-controlnet-depth \
  --csv_paths /root/autodl-tmp/data/HAM10000/input/metadata_train_llava.csv,/root/autodl-tmp/data/HAM10000/input/metadata_val_llava.csv,/root/autodl-tmp/data/HAM10000/input/metadata_test_llava.csv \
  --output_dir /root/autodl-tmp/lora_controlnet_output \
  --resolution 512 \
  --train_batch_size 1 \
  --max_train_steps 1000 \
  --mixed_precision fp16
```

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--pretrained_model_name_or_path` | Required | Path to SD1.5 diffusers model |
| `--controlnet_path` | Required | Path to ControlNet depth model |
| `--csv_paths` | Required | Comma-separated CSV paths |
| `--image_root` | `/root/autodl-tmp` | Root path for resolving relative paths |
| `--output_dir` | `/root/autodl-tmp/checkpoint/compare_models/ControlNet` | Output directory |
| `--resolution` | 512 | Image resolution |
| `--train_batch_size` | 1 | Batch size |
| `--gradient_accumulation_steps` | 1 | Gradient accumulation steps |
| `--learning_rate` | 1e-4 | Learning rate |
| `--max_train_steps` | 1000 | Maximum training steps |
| `--mixed_precision` | fp16 | Mixed precision training |

## Notes

- Segmentation masks (`seg_path`) are used as the ControlNet conditioning image.
- Output LoRA weights are saved to `pytorch_lora_weights.safetensors` in the output directory.
- The script uses the `llava_prompt` column from CSVs as text conditioning.
