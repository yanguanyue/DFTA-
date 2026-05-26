# Derm-T2IM (Dermatology Text-to-Image Model)

## Overview

Derm-T2IM is a specialized text-to-image generation model for dermatology images. It is fine-tuned from Stable Diffusion v1-5 specifically for generating skin lesion images from text descriptions.

## Directory Structure

- `compare_main/Derm-T2IM/`: Main code directory
  - `inferance.py`: Inference script
- `scripts/compare_Derm_T2IM.sh`: One-click generation script
- `checkpoint/compare_models/Derm-T2IM/`: Model checkpoint
  - `Derm-T2IM.safetensors`: Model weights
- `output/generate/Derm-T2IM/`: Generated image output

## One-Click Script: compare_Derm_T2IM.sh

### Script Location
```
scripts/compare_Derm_T2IM.sh
```

### Features

The script provides a complete generation workflow with the following capabilities:

1. **Automatic Model Validation**
   - Checks if Derm-T2IM weights exist
   - Verifies model file integrity
   - Exits with error if model not found

2. **Diffusers Configuration Download**
   - Automatically downloads SD1.5 diffusers config
   - Caches to HuggingFace home directory
   - Uses mirror endpoint for faster downloads

3. **Class-Specific Generation**
   - Generates images for all 7 HAM10000 classes
   - Uses predefined prompts for each disease
   - Generates 1 image per class (default)

4. **Environment Configuration**
   - Sets HuggingFace home directory
   - Configures mirror endpoint (default: hf-mirror.com)
   - Supports custom mirror via environment variable

5. **Clean Output**
   - Removes old output before generation
   - Creates fresh output directory per run
   - Clear logging of generation progress

### Usage

```bash
# Generate images (1 per class)
bash /root/autodl-tmp/scripts/compare_Derm_T2IM.sh

# Use custom mirror
HF_ENDPOINT_URL="https://custom-mirror.com" bash /root/autodl-tmp/scripts/compare_Derm_T2IM.sh
```

### Key Parameters (Environment Variables)

| Variable | Description | Default |
|----------|-------------|---------|
| `HF_ENDPOINT_URL` | HuggingFace mirror URL | `https://hf-mirror.com` |
| `HF_HOME_PATH` | HuggingFace cache directory | `model/hf_home` |

## Generation Prompts

The script uses predefined prompts for each class:

| Class | Prompt |
|-------|--------|
| akiec | An image of a skin area with actinic keratoses or intraepithelial carcinoma. |
| bcc | An image of a skin area with basal cell carcinoma. |
| bkl | An image of a skin area with benign keratosis-like lesions. |
| df | An image of a skin area with dermatofibroma. |
| mel | An image of a skin area with melanoma. |
| nv | An image of a skin area with melanocytic nevi. |
| vasc | An image of a skin area with a vascular lesion. |

## Supported Classes

```
akiec, bcc, bkl, df, mel, nv, vasc
```

## Output Structure

```
output/generate/Derm-T2IM/
└── inference/
    ├── akiec/
    │   └── 00000.png
    ├── bcc/
    │   └── 00000.png
    ├── bkl/
    │   └── 00000.png
    ├── df/
    │   └── 00000.png
    ├── mel/
    │   └── 00000.png
    ├── nv/
    │   └── 00000.png
    └── vasc/
        └── 00000.png
```

## Model Path

| Model | Path |
|-------|------|
| Derm-T2IM weights | `checkpoint/compare_models/Derm-T2IM/Derm-T2IM.safetensors` |

## FAQ

### 1) Local config not found error
- Ensure `model/hf_home` has cached diffusers config
- Check mirror URL is accessible
- For fully offline, pre-cache on machine with network

### 2) CUDA/Out of Memory errors
- Reduce batch_size in generation script
- Current defaults are conservative (batch_size=1, n=1)

### 3) Model file not found
- Verify `Derm-T2IM.safetensors` exists in checkpoint directory
- Download from HuggingFace if missing:
  https://huggingface.co/MAli-Farooq/Derm-T2IM

### 4) Generation is slow
- Current config uses single image generation
- Adjust `n` and `batch_size` parameters in script

## Notes

- This is a pre-trained model (no training in this workflow)
- Uses diffusers library for loading
- Single file weights (.safetensors)
- Optimized for dermatology domain
