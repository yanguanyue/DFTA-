# LF-VAR Training and Generation Guide

## Overview

LF-VAR is a controllable deep learning framework for skin lesion image synthesis and generation. This project is based on the MICCAI 2025 paper, using VQVAE + VAR Transformer architecture for skin lesion image generation.

## Directory Structure

```
/root/autodl-tmp/
├── scripts/
│   └── compare_LF-VAR.sh          # Main training and generation script
├── compare_main/
│   └── LF-VAR-main/
│       ├── main/
│       │   ├── train.py           # Training script
│       │   ├── infer.py           # Inference script
│       │   ├── trainer.py         # Trainer
│       │   ├── models/            # Model definitions
│       │   └── utils/             # Utility functions
│       └── sh/
│           └── compare_MAIN.sh    # Original training script
├── checkpoint/
│   └── compare_models/
│       └── LF-VAR/               # Model checkpoint save path
├── output/
│   └── generate/
│       └── LF-VAR/               # Generated image output path
└── data/
    └── HAM10000/
        └── input/                # Dataset path
```

## One-Click Script: compare_LF-VAR.sh

### Script Location
```
scripts/compare_LF-VAR.sh
```

### Features

The script provides a complete end-to-end workflow with the following capabilities:

1. **Automatic Model Download**
   - Downloads pre-trained VAE checkpoint from HuggingFace mirror
   - Downloads pre-trained VAR checkpoint from HuggingFace mirror
   - Uses `hf-mirror.com` for faster downloads in China region
   - Checks if models exist before downloading to avoid re-downloading

2. **Two-Phase Execution**
   - **Phase 1 - Training**: Trains VAR model on HAM10000 dataset
     - Configurable epochs, batch size, and model depth
     - Uses PyTorch distributed training (`torch.distributed.run`)
     - Supports mixed precision training (fp16)
   - **Phase 2 - Generation**: Generates images for all 7 classes
     - Loads trained checkpoint
     - Generates specified number of images per class
     - Uses distributed inference

3. **Two Running Modes**
   - **Test mode** (`test`): Quick validation
     - 1 training epoch
     - Batch size = 2
     - 1 image generated per class
   - **Full mode** (`full`): Production training
     - 200 training epochs
     - Batch size = 35
     - 1500 images generated per class

4. **Distributed Training Support**
   - Uses `torch.distributed.run` for single-GPU training
   - Automatically generates random master port
   - Supports configurable GPU selection

5. **Environment Configuration**
   - Sets HuggingFace endpoint to mirror
   - Automatically creates required output directories

### Usage

```bash
# Enter project directory
cd /root/autodl-tmp

# Test mode (1 epoch + 1 image per class)
bash scripts/compare_LF-VAR.sh test

# Full mode (200 epochs + 1500 images per class)
bash scripts/compare_LF-VAR.sh full
```

### Key Parameters (Environment Variables)

| Parameter | Description | Test Mode | Full Mode |
|-----------|-------------|-----------|-----------|
| TRAIN_EPOCHS | Training epochs | 1 | 200 |
| TRAIN_BS | Batch size | 2 | 35 |
| GPU_SUM | GPU count | 1 | 1 |
| NUM_IMAGES | Images per class | 1 | 1500 |

### Training Parameters (Hardcoded)

| Parameter | Description | Value |
|-----------|-------------|-------|
| `--depth` | Model depth | 16 |
| `--fp16` | Mixed precision | 1 |
| `--alng` | AdaLN gamma init | 1e-3 |
| `--wpe` | Position encoding weight | 0.1 |

## Dataset

### Data Paths

```
/root/autodl-tmp/data/HAM10000/input/
├── train_val/
│   └── HAM10000_img_class/
│       ├── akiec/
│       ├── bcc/
│       ├── bkl/
│       ├── df/
│       ├── mel/
│       ├── nv/
│       └── vasc/
├── test/
│   └── HuggingFace/
└── metadata.csv
```

### Class Description

HAM10000 dataset contains 7 skin lesion classes:

| Class ID | Name | Description |
|----------|------|-------------|
| 0 | akiec | Actinic keratosis / Bowen's disease |
| 1 | bcc | Basal cell carcinoma |
| 2 | bkl | Benign keratosis-like lesions |
| 3 | df | Dermatofibroma |
| 4 | mel | Melanoma |
| 5 | nv | Melanocytic nevi |
| 6 | vasc | Vascular lesions |

## Model Architecture

### VQVAE (Vector Quantized Variational Autoencoder)

- **V**: 4096 (vocabulary size)
- **Cvae**: 32 (channel number)
- **ch**: 160 (base channel number)
- **share_quant_resi**: 4

### VAR Transformer

- **depth**: 16
- **patch_nums**: 1_2_3_4_5_6_8_10_13_16
- **patch_size**: 16

## Output Description

### Checkpoint Output

After training, checkpoints are saved to:
```
checkpoint/compare_models/LF-VAR/
├── ar-ckpt-best.pth   # Best checkpoint
├── ar-ckpt-last.pth   # Latest checkpoint
└── log.txt            # Training log
```

### Generated Image Output

Generated images are saved to:
```
output/generate/LF-VAR/
├── akiec/
├── bcc/
├── bkl/
├── df/
├── mel/
├── nv/
└── vasc/
```

Each class folder contains generated PNG images with filename format: `{class_name}_{index}.png`

## Pre-trained Models

The script automatically downloads the following pre-trained models:

1. **VAE**: `vae_ch160v4096z32.pth`
   - Source: HuggingFace FoundationVision/var
   - Path: `/root/autodl-tmp/compare_main/LF-VAR-main/main/`

2. **VAR**: `var_d16.pth`
   - Source: HuggingFace FoundationVision/var

## Notes

- Training uses distributed PyTorch (`torch.distributed.run`)
- Model automatically selects available GPU via `CUDA_VISIBLE_DEVICES`
- Mixed precision training (fp16) is enabled by default for memory efficiency
