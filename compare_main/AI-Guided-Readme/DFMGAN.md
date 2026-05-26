# DFMGAN (Defect-Aware Feature Manipulation GAN)

## Overview

DFMGAN (Few-Shot Defect Image Generation) is a GAN-based approach for skin lesion image synthesis. It uses a two-stage training approach: first training a base model on defect-free images, then applying defect-aware feature manipulation for each class.

## Directory Structure

- `compare_main/DFMGAN-main/`: Main code directory
  - `train.py`: Training script (StyleGAN3-based)
  - `generate.py`: Generation script
  - `dataset_tool.py`: Dataset preparation tool
- `scripts/compare_DFMGAN.sh`: One-click training + generation script
- `checkpoint/compare_models/DFMGAN/`: Model checkpoint output
- `output/generate/DFMGAN/`: Generated image output

## One-Click Script: compare_DFMGAN.sh

### Script Location
```
scripts/compare_DFMGAN.sh
```

### Features

The script provides a complete end-to-end workflow with the following capabilities:

1. **Two-Stage Training**
   - **Stage 1 - Base Model**: Trains StyleGAN3 on defect-free images
     - Configurable training kimg (default: 80)
     - Creates foundation generator for all classes
   - **Stage 2 - Class Transfer**: Fine-tunes for each class with masks
     - Uses defect-aware feature manipulation
     - Configurable class training kimg (default: 20)
     - Generates 7 separate class-specific models

2. **Automatic Dataset Preparation**
   - Builds mask dataset ZIP files for each class
   - Uses `dataset_tool.py` with source mask support
   - Resizes images to 512x512
   - Supports limiting max images

3. **Image Generation with Masks**
   - Generates both images and corresponding masks
   - Supports configurable number of images per class
   - Automatically separates masks to dedicated directory

4. **Test Mode Support**
   - Set `TEST_MODE=1` for quick validation:
     - 1 kimg for base training
     - 1 kimg per class
     - 1 image per class

5. **Checkpoint Management**
   - Auto-detects checkpoint directories
   - Supports legacy directory naming
   - Falls back to multiple directory patterns

### Usage

```bash
# Full training + generation (80 kimg base, 20 kimg per class, 1500 images per class)
bash /root/autodl-tmp/scripts/compare_DFMGAN.sh

# Test mode (1 kimg, 1 image per class)
TEST_MODE=1 bash /root/autodl-tmp/scripts/compare_DFMGAN.sh

# Training only (skip generation)
RUN_ENABLED=false bash /root/autodl-tmp/scripts/compare_DFMGAN.sh

# Generation only (requires existing checkpoints)
TRAIN_ENABLED=false bash /root/autodl-tmp/scripts/compare_DFMGAN.sh

# Custom training kimg
BASE_KIMG=40 CLASS_KIMG=10 bash /root/autodl-tmp/scripts/compare_DFMGAN.sh

# Custom generation quantity
NUM_IMAGES_PER_CLASS=500 bash /root/autodl-tmp/scripts/compare_DFMGAN.sh
```

### Key Parameters (Environment Variables)

| Variable | Description | Default |
|----------|-------------|---------|
| `TRAIN_ENABLED` | Enable training stage | `true` |
| `RUN_ENABLED` | Enable generation stage | `true` |
| `TEST_MODE` | Test mode (quick validation) | `0` |
| `BASE_KIMG` | Base model training kimg | `80` |
| `CLASS_KIMG` | Per-class training kimg | `20` |
| `SNAP` | Snapshot interval | `10` |
| `NUM_IMAGES_PER_CLASS` | Images per class | `1500` |
| `MAX_IMAGES` | Max images in dataset (optional) | (unlimited) |

## Data Requirements

The script expects the following directory structure:

```
data/HAM10000/input/train/
├── HAM10000_img_class/
│   ├── akiec/
│   ├── bcc/
│   ├── bkl/
│   ├── df/
│   ├── mel/
│   ├── nv/
│   └── vasc/
└── HAM10000_seg_class/
    ├── akiec/
    ├── bcc/
    ├── bkl/
    ├── df/
    ├── mel/
    ├── nv/
    └── vasc/
```

## Supported Classes

```
akiec, bcc, bkl, df, mel, nv, vasc
```

## Output Structure

```
output/generate/DFMGAN/
├── akiec/
│   ├── image/
│   │   └── *.png
│   └── mask/
│       └── *.png
├── bcc/
│   └── ...
└── vasc/
    └── ...
```

## Checkpoint Paths

After training, checkpoints are saved to:
```
checkpoint/compare_models/DFMGAN/
├── ham10000_base/           # Base model
│   └── */network-snapshot-*.pkl
├── ham10000_akiec_mask_512/  # Class-specific models
├── ham10000_bcc_mask_512/
├── ham10000_bkl_mask_512/
├── ham10000_df_mask_512/
├── ham10000_mel_mask_512/
├── ham10000_nv_mask_512/
└── ham10000_vasc_mask_512/
```

## Mask ZIP Files

During training, mask datasets are created:
```
data/HAM10000/input/train/dfmgan_mask_zips/
├── ham10000_akiec_mask_512.zip
├── ham10000_bcc_mask_512.zip
└── ...
```

## FAQ

### 1) No masks in generated output
- Ensure `--gen-mask` flag is enabled (default)
- Check if training used mask ZIP files
- Verify mask paths in generation

### 2) Checkpoint not found errors
- Script looks for `ham10000_<class>_mask_512` first
- Falls back to legacy `ham10000_<class>` naming
- Check if training completed for all classes

### 3) Generated masks are black
- Ensure source mask exists in dataset_tool processing
- Verify `--source-mask` flag was used
- Check if mask ZIP contains valid masks

### 4) Training takes too long
- Reduce `BASE_KIMG` and `CLASS_KIMG`
- Use `TEST_MODE=1` for quick validation

### 5) Out of memory during generation
- Reduce `NUM_IMAGES_PER_CLASS`
- Process fewer classes at a time

## Notes

- Uses StyleGAN3 architecture
- Transfer learning with `res_block_match_dis` mode
- Supports incremental training from base
- All 7 classes require separate training runs
