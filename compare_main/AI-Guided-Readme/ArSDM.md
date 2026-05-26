# ArSDM

## Overview

ArSDM (Architecture for Sepsis Detection Model) is an advanced skin lesion generation framework supporting two-stage training: base model training and class-specific refinement. This document covers the complete comparison workflow with detailed script functionality.

## Directory Structure

- `compare_main/ArSDM-main/`: Main code directory
  - `run_stage2_from_base.sh`: Two-stage training script
  - `generate_stage2_samples.py`: Image generation script
  - `configs/`: YAML configuration files for each class (akiec, bcc, bkl, df, mel, nv, vasc)
- `scripts/compare_ArSDM.sh`: One-click training + generation script
- `checkpoint/compare_models/ArSDM/`: Training checkpoint output directory
- `output/generate2/ArSDM/`: Generated image output directory

## One-Click Script: compare_ArSDM.sh

### Script Location
```
scripts/compare_ArSDM.sh
```

### Features

The script provides a comprehensive two-stage training and generation workflow:

1. **Automatic Model Preparation**
   - Detects and validates CUDA environment
   - Sets `OMP_NUM_THREADS=1` for optimal performance
   - Supports pre-trained base model loading (if available)

2. **Two-Stage Training Pipeline**
   - **Stage 1 - Base Model Training**: Trains foundation model on full dataset
     - Configurable base training steps (`BASE_MAX_STEPS`, default: 5000)
     - Supports resuming from existing base checkpoint
   - **Stage 2 - Class-Specific Fine-tuning**: Trains LoRA for each of 7 classes
     - Uses separate YAML configs for each class
     - Configurable stage2 training steps (`STAGE2_MAX_STEPS`, default: 3000)

3. **Multi-Class Generation**
   - Loads checkpoints for each class
   - Generates specified number of images per class
   - Supports DDIM sampling with configurable steps

4. **Test Mode Support**
   - Set `TEST_MODE=1` for quick validation:
     - 5 training steps for both base and stage2
     - 1 image per class
     - batch size = 1
     - DDIM steps = 5

### Supported Classes

The script automatically handles all 7 HAM10000 classes:
- akiec (Actinic keratosis)
- bcc (Basal cell carcinoma)
- bkl (Benign keratosis)
- df (Dermatofibroma)
- mel (Melanoma)
- nv (Melanocytic nevi)
- vasc (Vascular lesions)

### Usage

```bash
# Full training + generation (5000 base steps, 3000 stage2 steps, 1500 images per class)
bash /root/autodl-tmp/scripts/compare_ArSDM.sh

# Test mode (5 steps, 1 image per class)
TEST_MODE=1 bash /root/autodl-tmp/scripts/compare_ArSDM.sh

# Training only (skip generation)
RUN_ENABLED=false bash /root/autodl-tmp/scripts/compare_ArSDM.sh

# Generation only (skip training)
TRAIN_ENABLED=false bash /root/autodl-tmp/scripts/compare_ArSDM.sh

# Custom training steps
BASE_MAX_STEPS=3000 STAGE2_MAX_STEPS=2000 bash /root/autodl-tmp/scripts/compare_ArSDM.sh

# Custom generation quantity
NUM_IMAGES_PER_CLASS=500 bash /root/autodl-tmp/scripts/compare_ArSDM.sh
```

### Key Parameters (Environment Variables)

| Variable | Description | Default |
|----------|-------------|---------|
| `TRAIN_ENABLED` | Enable training stage | `true` |
| `RUN_ENABLED` | Enable generation stage | `true` |
| `TEST_MODE` | Test mode (quick validation) | `0` |
| `BASE_MAX_STEPS` | Base model training steps | `5000` |
| `STAGE2_MAX_STEPS` | Stage2 fine-tuning steps | `3000` |
| `NUM_IMAGES_PER_CLASS` | Images per class | `1500` |
| `BATCH_SIZE` | Generation batch size | `8` |
| `DDIM_STEPS` | DDIM sampling steps | `20` |
| `PRETRAINED_BASE_CKPT` | Pre-trained base checkpoint path | Auto-detected |

### YAML Configuration Files

The script uses class-specific YAML configs:
```
configs/HAM10000_akiec.yaml
configs/HAM10000_bcc.yaml
configs/HAM10000_bkl_lora_512.yaml
configs/HAM10000_df_lora_512.yaml
configs/HAM10000_mel_lora_512.yaml
configs/HAM10000_nv_lora_512.yaml
configs/HAM10000_vasc_lora_512.yaml
```

## Training Workflow

### Stage 1: Base Model Training
- Trains on full HAM10000 dataset without class labels
- Creates foundation model for subsequent class-specific fine-tuning
- Output: Base checkpoint saved to `checkpoint/compare_models/ArSDM/PRETRAINED_BASE_CKPT/`

### Stage 2: Class-Specific Fine-tuning
- Loads base model as starting point
- Fine-tunes separate LoRA for each of 7 classes
- Uses transfer learning with `res_block_match_dis` mode
- Output: Per-class checkpoints in `checkpoint/compare_models/ArSDM/`

## Inference Workflow

Entry: `compare_main/ArSDM-main/generate_stage2_samples.py`

Core behavior:
- Loads checkpoints for each class sequentially
- Generates specified number of images per class
- Supports DDIM sampling for high-quality output

Output structure:
```
output/generate2/ArSDM/
├── akiec/
├── bcc/
├── bkl/
├── df/
├── mel/
├── nv/
└── vasc/
```

## Output Paths

- Base checkpoint: `checkpoint/compare_models/ArSDM/PRETRAINED_BASE_CKPT/ArSDM_ada_refine.ckpt`
- Class checkpoints: `checkpoint/compare_models/ArSDM/ham10000_<class>_lora_512/...`
- Generated images: `output/generate2/ArSDM/<class>/...`

## FAQ

### 1) Training is too slow
- Reduce `BASE_MAX_STEPS` and `STAGE2_MAX_STEPS`
- Use `TEST_MODE=1` first to verify pipeline

### 2) Out of memory errors
- Reduce `BATCH_SIZE` (default: 8)
- Reduce `DDIM_STEPS` (default: 20)

### 3) Want to skip base training
Set `PRETRAINED_BASE_CKPT` to existing base checkpoint path to resume from previous training.

### 4) Need to train specific classes only
Modify the `STAGE2_CONFIGS` array in the script to include only desired class configs.
