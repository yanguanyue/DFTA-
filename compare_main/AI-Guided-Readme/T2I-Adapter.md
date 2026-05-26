# T2I-Adapter LoRA

## Overview

T2I-Adapter LoRA is a lightweight adapter-based approach for skin lesion image generation. It combines T2I-Adapter with Stable Diffusion 1.5, using depth maps as conditioning signals and training only LoRA weights for efficient fine-tuning.

## Directory Structure

- `compare_main/Controlnet+T2i_adapter/`: Main code directory
  - `train_sd15_lora_t2i_adapter.py`: T2I-Adapter LoRA training script
  - `generate_ham10000_lora_images.py`: Image generation script
- `scripts/compare_T2i_adapter.sh`: One-click training + generation script
- `checkpoint/compare_models/T2i_adapter/`: LoRA checkpoint output
- `output/generate/T2i_Adapter/`: Generated image output
- `data/metadata_*_llava.csv`: Training/validation/test CSV files

## One-Click Script: compare_T2i_adapter.sh

### Script Location
```
scripts/compare_T2i_adapter.sh
```

### Features

The script provides a complete end-to-end workflow with the following capabilities:

1. **Automatic Model Download**
   - Checks if Stable Diffusion 1.5 exists locally
   - Checks if T2I-Adapter (ZoeDepth) exists locally
   - Automatically downloads from ModelScope if not found
   - Supports multiple repo ID fallbacks
   - Installs modelscope package if not available

2. **Training Pipeline**
   - Trains T2I-Adapter LoRA on SD1.5
   - Uses CSV files with image paths, segmentation masks, and prompts
   - Trains only LoRA weights (parameter-efficient)
   - Configurable training steps (default: 15000)
   - Supports checkpoint saving every 1000 steps

3. **Multi-CSV Support**
   - Reads from train, validation, and test CSV files
   - Supports relative path resolution
   - Auto-remaps `data/local/...` paths to `data/...`

4. **Generation Pipeline**
   - Uses validation set prompts and masks for generation
   - Generates specified images per class (default: 1500)
   - Supports class-first or image-first output layout
   - Outputs both images and masks

5. **Test Mode Support**
   - Set `TEST_MODE=1` for quick validation:
     - 2 training steps
     - 1 image per class
     - 2 inference steps

### Usage

```bash
# Full training + generation (15000 steps, 1500 images per class)
bash /root/autodl-tmp/scripts/compare_T2i_adapter.sh

# Test mode (2 steps, 1 image per class)
TEST_MODE=1 bash /root/autodl-tmp/scripts/compare_T2i_adapter.sh

# Training only (skip generation)
RUN_ENABLED=false bash /root/autodl-tmp/scripts/compare_T2i_adapter.sh

# Generation only (requires existing LoRA)
TRAIN_ENABLED=false bash /root/autodl-tmp/scripts/compare_T2i_adapter.sh

# Custom training and generation parameters
MAX_TRAIN_STEPS=10000 NUM_IMAGES_PER_CLASS=500 bash /root/autodl-tmp/scripts/compare_T2i_adapter.sh
```

### Key Parameters (Environment Variables)

| Variable | Description | Default |
|----------|-------------|---------|
| `TRAIN_ENABLED` | Enable training stage | `true` |
| `RUN_ENABLED` | Enable generation stage | `true` |
| `TEST_MODE` | Test mode (quick validation) | `0` |
| `MAX_TRAIN_STEPS` | Training steps | `15000` |
| `NUM_IMAGES_PER_CLASS` | Images per class | `1500` |
| `NUM_INFERENCE_STEPS` | Inference steps | `30` |
| `GUIDANCE_SCALE` | Classifier-free guidance | `7.5` |
| `ADAPTER_SCALE` | Adapter conditioning scale | `1.0` |

## Data Format

The script expects CSV files with the following columns:

| Column | Description |
|--------|-------------|
| `img_path` | Original image path (relative) |
| `seg_path` | Segmentation mask path (relative) |
| `llava_prompt` | LLaVA-generated text prompt |
| `prompt` | Fallback text prompt |

Path resolution:
- Relative paths are resolved against `IMAGE_ROOT`
- `data/local/...` paths auto-mapped to `data/...`

## Supported Classes

The script generates images for all 7 HAM10000 classes:
```
akiec, bcc, bkl, df, mel, nv, vasc
```

## Output Structure

Default `class_first` layout:
```
output/generate/T2i_Adapter/
├── akiec/
│   ├── image/
│   │   └── *.png
│   └── mask/
│       └── *.png
├── bcc/
│   └── ...
└── meta/
    └── metadata_t2i_adapter.jsonl
```

Alternative `image_first` layout:
```
output/generate/T2i_Adapter/
├── image/
│   ├── akiec/*.png
│   └── bcc/*.png
└── mask/
    ├── akiec/*.png
    └── bcc/*.png
```

## Model Paths

| Model | Default Path |
|-------|--------------|
| SD1.5 | `model/AI-ModelScope/stable-diffusion-v1-5` |
| T2I-Adapter | `model/AI-ModelScope/t2iadapter_zoedepth_sd15v1` |
| LoRA Output | `checkpoint/compare_models/T2i_adapter/` |

## Checkpoint Output

After training, the following files are created:
```
checkpoint/compare_models/T2i_adapter/
├── pytorch_lora_weights.safetensors  # LoRA weights
├── train_args.json                   # Training configuration
└── logs/                            # Training logs
```

## FAQ

### 1) Training produces OOM errors
- Reduce training batch size (currently fixed at 1)
- Ensure sufficient GPU memory
- Check if other processes are using GPU

### 2) Generated images are low quality
- Increase `NUM_INFERENCE_STEPS`
- Adjust `GUIDANCE_SCALE`
- Ensure training completed sufficient steps

### 3) Prompt truncation warnings
- LLaVA prompts exceeding 77 tokens are truncated by CLIP
- This is expected behavior and does not affect generation
- Prompts can be shortened in CSV files

### 4) Mask path resolution fails
- Check `IMAGE_ROOT` environment variable
- Ensure CSV paths are relative to project root
- Use absolute paths in CSV if needed

### 5) Output directory naming
- Note: Output directory is `T2i_Adapter` (not `T2I_Adapter`)
- Check correct case sensitivity in paths
