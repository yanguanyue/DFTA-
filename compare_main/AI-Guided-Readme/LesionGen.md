# LesionGen

## Overview

LesionGen is a skin lesion generation framework using a two-stage LoRA training approach. It first trains a base LoRA on the entire dataset, then fine-tunes separate LoRAs for each of the 7 lesion classes.

## Directory Structure

- `compare_main/LesionGen-main/`: Main code directory
  - `train_lora.sh`: Base LoRA training script
  - `train_lora_7classes.sh`: Per-class LoRA training script
  - `generate_all.py`: Generation script
- `scripts/compare_LesionGen.sh`: One-click training + generation script
- `checkpoint/compare_models/LesionGen/`: Model checkpoint output
- `output/generate/LesionGen/`: Generated image output

## One-Click Script: compare_LesionGen.sh

### Script Location
```
scripts/compare_LesionGen.sh
```

### Features

The script provides a complete end-to-end workflow with the following capabilities:

1. **Automatic Model Download**
   - Checks for cached Stable Diffusion v1-4
   - Searches in both hub and model directories
   - Downloads from ModelScope if not cached
   - Supports multiple repo ID fallbacks

2. **Two-Stage Training Pipeline**
   - **Stage 1 - Base LoRA**: Trains base LoRA on full dataset
     - ~14k training steps
     - Creates foundation for class-specific training
     - Output: `ham10000_lora/`
   - **Stage 2 - Class LoRAs**: Fine-tunes 7 separate LoRAs
     - Resumes from base LoRA
     - Trains class-specific adapters
     - Output: `lora_7classes/`

3. **Flexible Generation**
   - Uses pre-trained class LoRAs
   - Supports different generation modes
   - Configurable output layout
   - Generates dataset-style outputs

4. **Environment Configuration**
   - Sets HuggingFace home directory
   - Configures local files only mode
   - Sets ModelScope cache directory
   - Creates required directories

### Usage

```bash
# Full training (base + 7 class LoRAs) + generation
bash /root/autodl-tmp/scripts/compare_LesionGen.sh

# Training only (skip generation)
TRAIN_ENABLED=true RUN_ENABLED=false bash /root/autodl-tmp/scripts/compare_LesionGen.sh

# Generation only (requires existing LoRAs)
TRAIN_ENABLED=false RUN_ENABLED=true NUM_IMAGES_PER_CLASS=1500 MODE=dataset OUTPUT_LAYOUT=flat bash /root/autodl-tmp/scripts/compare_LesionGen.sh
```

### Key Parameters (Environment Variables)

| Variable | Description | Default |
|----------|-------------|---------|
| `TRAIN_ENABLED` | Enable training stage | `true` |
| `RUN_ENABLED` | Enable generation stage | `true` |
| `NUM_IMAGES_PER_CLASS` | Images per class | `1500` |
| `MODE` | Generation mode | `dataset` |
| `OUTPUT_LAYOUT` | Output directory layout | `flat` |
| `MODELSCOPE_CACHE` | ModelScope cache path | `model/` |
| `CHECKPOINTS_ROOT` | LoRA checkpoint path | `checkpoint/compare_models/LesionGen/lora_7classes` |

## Output Layouts

### Flat Layout
```
output/generate/LesionGen/
├── akiec_*.png
├── bcc_*.png
└── ...
```

### Class-First Layout
```
output/generate/LesionGen/
├── akiec/
│   └── *.png
├── bcc/
│   └── *.png
└── ...
```

## Supported Classes

```
akiec, bcc, bkl, df, mel, nv, vasc
```

## Checkpoint Paths

After training, checkpoints are saved to:
```
checkpoint/compare_models/LesionGen/
├── ham10000_lora/           # Base LoRA
└── lora_7classes/          # Per-class LoRAs
    ├── akiec/
    ├── bcc/
    ├── bkl/
    ├── df/
    ├── mel/
    ├── nv/
    └── vasc/
```

## Model Path

| Model | Default Path |
|-------|--------------|
| Base SD1.4 | Auto-detected from cache or downloaded |

## FAQ

### 1) ModelScope cache not found
- Script automatically downloads if not cached
- Check network connectivity
- Verify write permissions to model directory

### 2) Generation produces wrong classes
- Ensure 7-class LoRAs are trained
- Check CHECKPOINTS_ROOT path
- Verify checkpoint-5000 exists

### 3) Need faster training for testing
- Modify training scripts to reduce steps
- Use smaller batch sizes
- Skip class LoRA training for quick test

### 4) Output layout confusion
- Use `OUTPUT_LAYOUT=flat` or `OUTPUT_LAYOUT=class_first`
- Check both layouts to find your images

### 5) Training fails partway through
- Check base LoRA trained successfully
- Verify disk space for checkpoints
- Ensure GPU memory sufficient

## Notes

- Two-stage training: base → class-specific
- Uses LoRA for parameter-efficient fine-tuning
- 7 classes require 7 separate class LoRAs
- Generation requires all class LoRAs present
