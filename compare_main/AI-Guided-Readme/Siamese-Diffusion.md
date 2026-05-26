# Siamese-Diffusion

## Overview

Siamese-Diffusion is a skin lesion image synthesis framework using Siamese architecture for paired image generation. This document covers the complete comparison workflow including data sources, training, inference, model preparation, and output directories.

## Directory Structure

- `compare_main/Siamese-Diffusion-main/`: Main code directory
  - `tutorial_train.py`: Training entry point (adapted for flow environment)
  - `tutorial_inference.py`: Inference entry point (generates 7 classes)
  - `tutorial_dataset.py`: Training data loader (reads from `prompt.json`)
  - `models/cldm_v15.yaml`: Model configuration (points to local CLIP model)
- `scripts/compare_Siamese.sh`: One-click training + generation script
- `checkpoint/compare_models/Siamese/`: Training checkpoint output directory
- `output/generate2/Siamese/`: Generated image output directory
- `data/HAM10000/input/train/`: Training data source (images + masks)

## One-Click Script: compare_Siamese.sh

### Script Location
```
scripts/compare_Siamese.sh
```

### Features

The script provides a complete end-to-end workflow with the following capabilities:

1. **Automatic Environment Configuration**
   - Sets HuggingFace offline mode (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`)
   - Configures `HF_HOME` for model caching
   - Automatically detects Python interpreter (defaults to flow environment)

2. **Training Pipeline**
   - Loads pre-trained ControlNet weights from specified path
   - Reads training data from `prompt.json` (mixed training across all classes)
   - Supports configurable training steps, batch size, and image resolution
   - Saves merged model weights after training

3. **Inference Pipeline**
   - Generates images per class based on prompt.json entries
   - Configurable number of images per class (default: 1500)
   - Supports DDIM sampling with adjustable steps

4. **Test Mode Support**
   - Set `TEST_MODE=1` for quick validation (5 steps, 1 image per class)

### Usage

```bash
# Full training + generation (1500 images per class)
bash /root/autodl-tmp/scripts/compare_Siamese.sh

# Test mode (5 training steps, 1 image per class)
TEST_MODE=1 bash /root/autodl-tmp/scripts/compare_Siamese.sh

# Training only (skip generation)
RUN_ENABLED=false bash /root/autodl-tmp/scripts/compare_Siamese.sh

# Generation only (skip training)
TRAIN_ENABLED=false bash /root/autodl-tmp/scripts/compare_Siamese.sh
```

### Key Parameters (Environment Variables)

| Variable | Description | Default |
|----------|-------------|---------|
| `TRAIN_ENABLED` | Enable training stage | `true` |
| `RUN_ENABLED` | Enable generation stage | `true` |
| `TEST_MODE` | Test mode (quick validation) | `0` |
| `MAX_STEPS` | Training steps | `3000` |
| `NUM_IMAGES_PER_CLASS` | Images per class | `1500` |
| `DDIM_STEPS` | Sampling steps | `50` |
| `BATCH_SIZE` | Batch size | `1` |
| `IMAGE_SIZE` | Image resolution | `512` |
| `GPU_IDS` | GPU device ID | `0` |

## Data Source and Organization

Training uses samples from `prompt.json`, **not trained separately by class**. The `prompt.json` points to:

- `data/HAM10000/input/train/HAM10000_img_class/<class>/...`
- `data/HAM10000/input/train/HAM10000_seg_class/<class>/...`

Each sample contains:
- `source`: Mask image
- `target`: Original image
- `prompt`: Text prompt

If `compare_main/Siamese-Diffusion-main/data/prompt.json` does not exist, it automatically falls back to:
```
/root/autodl-tmp/main/Siamese-Diffusion-main/data/prompt.json
```

## Model Preparation

### 1) Pre-trained Weights (Required)
Before training, load from:
```
checkpoint/compare_models/Siamese/PRETRAINED/merged_pytorch_model.pth
```
The script initializes ControlNet weights from this file.

### 2) CLIP Text Model (Required)
The repository uses CLIP text encoder:
```
model/clip-vit-large-patch14
```
`models/cldm_v15.yaml` has been modified to point to local path to avoid network download.

### 3) Hugging Face Offline Mode (Enabled)
`compare_Siamese.sh` sets:
- `HF_HOME=/root/autodl-tmp/model/hf_home`
- `HF_HUB_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`

## Training Workflow

Entry: `compare_main/Siamese-Diffusion-main/tutorial_train.py`

Core behavior:
- Reads all samples from `prompt.json` (mixed training)
- Loads pre-trained weights
- Saves merged weights to:
  - `checkpoint/compare_models/Siamese/merged_pytorch_model.pth`

## Inference and Generation Workflow

Entry: `compare_main/Siamese-Diffusion-main/tutorial_inference.py`

Core behavior:
- Splits `prompt.json` by class
- Generates specified number of images per class (default 1500)
- Outputs to:
  - `output/generate2/Siamese/<class>/images/*.png`

Supported classes:
```
akiec, bcc, bkl, df, mel, nv, vasc
```

## Output Paths

- Training weights: `checkpoint/compare_models/Siamese/merged_pytorch_model.pth`
- Generated results: `output/generate2/Siamese/<class>/images/*.png`
- Training logs: `checkpoint/compare_models/Siamese/lightning_logs/...`

## FAQ

### 1) Training prompts unable to download CLIP
Already configured to point to local `model/clip-vit-large-patch14`, and script enables offline mode.

### 2) Prompt `prompt.json not found`
Ensure:
- `compare_main/Siamese-Diffusion-main/data/prompt.json` exists, or
- `/root/autodl-tmp/main/Siamese-Diffusion-main/data/prompt.json` exists.

### 3) Need to train separately by class?
Current training is **mixed training**. If you need to train by class separately, I can help automatically split `prompt.json` and generate 7 training configs and scripts.
