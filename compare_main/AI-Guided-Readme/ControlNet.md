# ControlNet Depth LoRA

## Overview

ControlNet Depth LoRA is a depth-map conditioned image generation approach for skin lesion synthesis. It uses ControlNet with depth maps as conditioning signals and trains only LoRA weights for efficient fine-tuning on Stable Diffusion 1.5.

## Directory Structure

- `compare_main/Controlnet+T2i_adapter/`: Main code directory
  - `train_sd15_lora_controlnet_depth.py`: ControlNet LoRA training script
  - `generate_ham10000_lora_images.py`: Image generation script
- `scripts/compare_ControlNet.sh`: One-click training + generation script
- `checkpoint/compare_models/Controlnet/`: LoRA checkpoint output
- `output/generate/ControlNet/`: Generated image output
- `data/metadata_*_llava.csv`: Training/validation/test CSV files

## One-Click Script: compare_ControlNet.sh

### Script Location
```
scripts/compare_ControlNet.sh
```

### Features

The script provides a complete end-to-end workflow with the following capabilities:

1. **Automatic Model Download**
   - Checks if Stable Diffusion 1.5 exists locally
   - Checks if ControlNet Depth model exists locally
   - Automatically downloads from ModelScope if not found
   - Supports multiple repo ID fallbacks:
     - `lllyasviel/sd-controlnet-depth`
     - `AI-ModelScope/sd-controlnet-depth`
     - `modelscope/sd-controlnet-depth`
   - Installs modelscope package if not available

2. **Training Pipeline**
   - Trains ControlNet LoRA on SD1.5
   - Uses depth maps as conditioning signals
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
bash /root/autodl-tmp/scripts/compare_ControlNet.sh

# Test mode (2 steps, 1 image per class)
TEST_MODE=1 bash /root/autodl-tmp/scripts/compare_ControlNet.sh

# Training only (skip generation)
RUN_ENABLED=false bash /root/autodl-tmp/scripts/compare_ControlNet.sh

# Generation only (requires existing LoRA)
TRAIN_ENABLED=false bash /root/autodl-tmp/scripts/compare_ControlNet.sh

# Custom training and generation parameters
MAX_TRAIN_STEPS=10000 NUM_IMAGES_PER_CLASS=500 bash /root/autodl-tmp/scripts/compare_ControlNet.sh
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
| `CONTROLNET_SCALE` | ControlNet conditioning scale | `1.0` |

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
output/generate/ControlNet/
├── akiec/
│   ├── image/
│   │   └── *.png
│   └── mask/
│       └── *.png
├── bcc/
│   └── ...
└── meta/
    └── metadata_controlnet_depth.jsonl
```

Alternative `image_first` layout:
```
output/generate/ControlNet/
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
| ControlNet Depth | `model/AI-ModelScope/sd-controlnet-depth` |
| LoRA Output | `checkpoint/compare_models/Controlnet/` |

## Checkpoint Output

After training, the following files are created:
```
checkpoint/compare_models/Controlnet/
├── pytorch_lora_weights.safetensors  # LoRA weights
├── train_args.json                   # Training configuration
└── logs/                            # Training logs
```

## FAQ

### 1) LoRA not loading correctly
- Check if `pytorch_lora_weights.safetensors` exists
- Ensure path matches `--lora_controlnet_dir`
- Verify file is not corrupted

### 2) Generated images don't match prompts
- Ensure `prompts_from_val_only` is enabled (uses val prompts)
- Check if CLIP truncates long prompts (>77 tokens)
- Verify mask paths are correct

### 3) Model loading failures
- Ensure SD1.5 and ControlNet directories exist
- Scripts auto-download but need network access
- Check ModelScope mirror availability

### 4) Mask path resolution issues
- CSV paths starting with `data/local/...` are auto-mapped
- Ensure `IMAGE_ROOT` points to project root
- Use absolute paths if needed

### 5) Output structure confusion
- Default is `class_first`: `<class>/image` and `<class>/mask`
- Use `--output_layout image_first` for alternative layout
- Check metadata JSONL for generation details

### 6) Difference from T2I-Adapter
- ControlNet uses dedicated ControlNet model
- T2I-Adapter uses lightweight adapter architecture
- Both use depth conditioning but different implementations
