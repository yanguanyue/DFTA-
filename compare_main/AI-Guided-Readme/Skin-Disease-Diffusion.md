# Skin-Disease Diffusion

## Overview

Skin-Disease Diffusion is a skin lesion image synthesis framework based on VAE + Diffusion architecture. It uses a two-stage training approach: first training a VAE for image compression, then training a diffusion model for high-quality image generation.

## Directory Structure

- `compare_main/skin-disease-diffusion-main/`: Main code directory
  - `train_vae.py`: VAE training script
  - `train_diffusion.py`: Diffusion model training script
  - `sampling.py`: Image generation/sampling script
- `scripts/compare_skin-disease-diffusion.sh`: One-click training + generation script
- `checkpoint/compare_models/skin-disease-diffusionr/`: Training checkpoint output
  - `vae/vae_last.ckpt`: VAE model checkpoint
  - `diffusion/diffusion_last.ckpt`: Diffusion model checkpoint
- `output/generate/skin-disease-diffusion/`: Generated image output directory
- `data/HAM10000/input/train/HAM10000_img_class/`: Training data source

## One-Click Script: compare_skin-disease-diffusion.sh

### Script Location
```
scripts/compare_skin-disease-diffusion.sh
```

### Features

The script provides a complete end-to-end workflow with the following capabilities:

1. **Automatic Model Download**
   - Checks if Stable Diffusion v1.5 exists locally
   - Automatically downloads from ModelScope if not found
   - Supports multiple repo ID fallbacks (AI-ModelScope, modelscope, official)
   - Installs modelscope package if not available

2. **Two-Stage Training Pipeline**
   - **Stage 1 - VAE Training**: Trains VAE for image compression
     - Converts images to latent representations
     - Configurable training steps (default: 15000)
     - Saves checkpoint to `vae/vae_last.ckpt`
   - **Stage 2 - Diffusion Training**: Trains diffusion model
     - Uses trained VAE for latent space operations
     - Configurable training steps (default: 15000)
     - Saves checkpoint to `diffusion/diffusion_last.ckpt`

3. **Image Sampling/Generation**
   - Uses DDIM sampling for high-quality generation
   - Generates specified number of images per class (default: 1500)
   - Configurable guidance scale and sampling steps
   - Supports fp16 precision for memory efficiency

4. **Test Mode Support**
   - Set `TEST_MODE=1` for quick validation:
     - 5 training steps for both VAE and diffusion
     - 1 image per class
     - 10 sampling steps
     - batch size = 1

5. **Flexible Configuration**
   - Training-only or generation-only modes
   - All parameters configurable via environment variables
   - Automatic directory creation

### Usage

```bash
# Full training + generation (15000 VAE steps, 15000 diffusion steps, 1500 images per class)
bash /root/autodl-tmp/scripts/compare_skin-disease-diffusion.sh

# Test mode (5 steps, 1 image per class)
TEST_MODE=1 bash /root/autodl-tmp/scripts/compare_skin-disease-diffusion.sh

# Training only (skip generation)
RUN_ENABLED=false bash /root/autodl-tmp/scripts/compare_skin-disease-diffusion.sh

# Generation only (requires existing checkpoints)
TRAIN_ENABLED=false bash /root/autodl-tmp/scripts/compare_skin-disease-diffusion.sh

# Custom training steps
MAX_TRAIN_STEPS=5000 bash /root/autodl-tmp/scripts/compare_skin-disease-diffusion.sh

# Custom generation parameters
NUM_IMAGES_PER_CLASS=500 SAMPLE_STEPS=100 BATCH_SIZE=4 bash /root/autodl-tmp/scripts/compare_skin-disease-diffusion.sh
```

### Key Parameters (Environment Variables)

| Variable | Description | Default |
|----------|-------------|---------|
| `TRAIN_ENABLED` | Enable training stage | `true` |
| `RUN_ENABLED` | Enable generation stage | `true` |
| `TEST_MODE` | Test mode (quick validation) | `0` |
| `MAX_TRAIN_STEPS` | Training steps (per stage) | `15000` |
| `NUM_IMAGES_PER_CLASS` | Images per class | `1500` |
| `SAMPLE_STEPS` | Sampling/DDIM steps | `750` |
| `BATCH_SIZE` | Generation batch size | `8` |
| `GUIDANCE_SCALE` | Classifier-free guidance | `3.0` |
| `SAMPLE_PRECISION` | Sampling precision | `fp16` |

## Supported Classes

The script generates images for all 7 HAM10000 classes:
```
akiec, bcc, bkl, df, mel, nv, vasc
```

## Output Structure

```
output/generate/skin-disease-diffusion/
├── akiec/
│   ├── akiec_a.png
│   ├── akiec_b.png
│   └── ...
├── bcc/
├── bkl/
├── df/
├── mel/
├── nv/
└── vasc/
```

Files are named with pattern: `{class_name}_{letter_index}.png`

## Checkpoint Paths

After training, checkpoints are saved to:
```
checkpoint/compare_models/skin-disease-diffusionr/
├── vae/
│   └── vae_last.ckpt
└── diffusion/
    └── diffusion_last.ckpt
```

## FAQ

### 1) Sampling is very slow
- Sampling with default 750 steps is intentionally slow for quality
- Reduce `SAMPLE_STEPS` for faster generation
- Increase `BATCH_SIZE` if GPU memory allows

### 2) Out of memory (OOM) errors
- Reduce `BATCH_SIZE` (try 1)
- Use `fp16` precision (already default)
- Reduce `SAMPLE_STEPS`

### 3) Generation produces black/blank images
- Ensure both VAE and diffusion checkpoints exist
- Check if training completed successfully
- Try regenerating with different random seed

### 4) Model download fails
- Script uses ModelScope mirror automatically
- Check network connectivity
- Ensure modelscope package is installed

### 5) Need to skip training and only generate
Set `TRAIN_ENABLED=false` but ensure checkpoints exist:
- `checkpoint/compare_models/skin-disease-diffusionr/vae/vae_last.ckpt`
- `checkpoint/compare_models/skin-disease-diffusionr/diffusion/diffusion_last.ckpt`
