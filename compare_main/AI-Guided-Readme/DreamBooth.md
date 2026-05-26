# DreamBooth

## Overview

DreamBooth is a deep learning framework for subject-driven image generation. This implementation uses a mixed training approach on HAM10000 skin lesion dataset, training a single model on all classes and using prompts to distinguish between classes during generation.

## Directory Structure

- `compare_main/DreamBooth/`: Main code directory
  - `train_all_dreambooth.sh`: Training script
  - `src/generate_images.py`: Generation script
- `scripts/compare_DreamBooth.sh`: One-click training + generation script
- `checkpoint/compare_models/DreamBooth/`: Model checkpoint output
- `output/generate/DreamBooth/`: Generated image output
- `data/HAM10000/input/train/HAM10000_img_class/`: Training data source

## One-Click Script: compare_DreamBooth.sh

### Script Location
```
scripts/compare_DreamBooth.sh
```

### Features

The script provides a complete end-to-end workflow with the following capabilities:

1. **Automatic Model Download**
   - Checks if Stable Diffusion 1.5 exists locally
   - Automatically downloads from ModelScope if not found
   - Supports multiple repo ID fallbacks
   - Installs modelscope package if not available

2. **Automatic Mixed Dataset Preparation**
   - Automatically builds mixed training directory
   - Creates symbolic links from all class directories
   - Avoids duplicating images
   - Uses unified prompt ("skin lesion") for all training

3. **Training Pipeline**
   - Trains a single DreamBooth model on mixed dataset
   - Uses subject-driven generation approach
   - Configurable training steps (default: 15000)
   - Model learns "skin lesion" concept as a whole

4. **Prompt-Based Generation**
   - During generation, uses different prompts for each class
   - Maps each class to specific disease description
   - Generates images per class based on class-specific prompts

5. **Test Mode Support**
   - Set `TEST_MODE=1` for quick validation:
     - 2 training steps
     - 1 image per class

### Usage

```bash
# Full training + generation (15000 steps, 1500 images per class)
bash /root/autodl-tmp/scripts/compare_DreamBooth.sh

# Test mode (2 steps, 1 image per class)
TEST_MODE=1 bash /root/autodl-tmp/scripts/compare_DreamBooth.sh

# Training only (skip generation)
RUN_ENABLED=false bash /root/autodl-tmp/scripts/compare_DreamBooth.sh

# Generation only (requires existing model)
TRAIN_ENABLED=false bash /root/autodl-tmp/scripts/compare_DreamBooth.sh

# Custom training steps
MAX_TRAIN_STEPS=10000 bash /root/autodl-tmp/scripts/compare_DreamBooth.sh
```

### Key Parameters (Environment Variables)

| Variable | Description | Default |
|----------|-------------|---------|
| `TRAIN_ENABLED` | Enable training stage | `true` |
| `RUN_ENABLED` | Enable generation stage | `true` |
| `TEST_MODE` | Test mode (quick validation) | `0` |
| `MAX_TRAIN_STEPS` | Training steps | `15000` |
| `NUM_IMAGES_PER_CLASS` | Images per class | `1500` |
| `OUTPUT_NAME` | Model output directory name | `ham10000-mix-model` |
| `SINGLE_PROMPT` | Unified training prompt | `skin lesion` |
| `GPU_ID` | GPU device ID | `0` |

## Class Prompt Mapping

During generation, each class is mapped to a specific disease description:

| Class | Generation Prompt |
|-------|-------------------|
| akiec | actinic keratosis and intraepithelial carcinoma |
| bcc | basal cell carcinoma |
| bkl | benign keratosis-like lesion |
| df | dermatofibroma |
| mel | melanoma |
| nv | melanocytic nevus |
| vasc | vascular lesion |

## Supported Classes

```
akiec, bcc, bkl, df, mel, nv, vasc
```

## Output Structure

```
output/generate/DreamBooth/
├── akiec/
│   └── *.png
├── bcc/
│   └── *.png
├── bkl/
│   └── *.df`
├── mel/
│   └── *.png
├── nv/
│   └── *.png
└── vasc/
    └── *.png
```

## Model Path

| Model | Default Path |
|-------|--------------|
| SD1.5 | `model/AI-ModelScope/stable-diffusion-v1-5` |
| Checkpoint | `checkpoint/compare_models/DreamBooth/ham10000-mix-model/` |

## Checkpoint Output

After training, the model is saved to:
```
checkpoint/compare_models/DreamBooth/ham10000-mix-model/
```

## FAQ

### 1) Model doesn't distinguish classes well
- This is expected behavior for mixed training
- Class distinction relies on prompts during generation
- Ensure prompts are descriptive enough

### 2) Generated images don't match expected class
- Check prompt mapping in generation script
- Ensure prompts are specific to each disease
- Try adjusting guidance scale

### 3) Training too slow
- Use `TEST_MODE=1` first to verify pipeline
- Reduce `MAX_TRAIN_STEPS` if needed

### 4) Need class-specific models instead of mixed
- Current implementation uses mixed training
- For class-specific models, would need separate training runs
- Can modify script to train 7 separate models

### 5) Memory issues during training
- Reduce training batch size in inner script
- Ensure no other GPU processes running

## Notes

- Mixed training learns "overall distribution" of skin lesions
- Class differentiation depends on generation-time prompts
- This approach is efficient but may have lower class-specific quality
