# DreamBooth Training

This directory contains scripts for training Stable Diffusion 1.5 (SD1.5) using DreamBooth on the HAM10000 skin lesion dataset.

## Project Structure

```
DreamBooth/
├── train_all_dreambooth.sh    # Main training script
├── README.md                   # This file
├── requirements.txt            # Python dependencies
└── .gitignore
```

## Training Script

The main script `train_all_dreambooth.sh` trains DreamBooth LoRA models on HAM10000 dataset.

### Usage

```bash
cd /root/autodl-tmp/compare_main/DreamBooth
bash train_all_dreambooth.sh
```

### Configuration

You can customize the training by setting environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `ROOT_PATH` | Project root | Root path for data and models |
| `PYTHON` | `/root/miniconda3/envs/flow/bin/python` | Python interpreter |
| `DATA_ROOT` | `$ROOT_PATH/data/HAM10000/input/train/HAM10000_img_class` | Training data root |
| `MODEL_CACHE_DIR` | `$ROOT_PATH/model/AI-ModelScope/stable-diffusion-v1-5` | SD1.5 model path |
| `OUTPUT_ROOT` | `$ROOT_PATH/checkpoint/compare_models/DreamBooth` | Output directory |
| `MAX_TRAIN_STEPS` | `1500` | Maximum training steps |
| `SINGLE_MODEL` | `0` | Train single mixed model instead of per-class |
| `SINGLE_PROMPT` | `"skin lesion"` | Prompt for single model training |

### Train All 7 Classes

By default, the script trains separate models for each of the 7 HAM10000 classes:

- `akiec` - Actinic keratosis and intraepithelial carcinoma
- `bcc` - Basal cell carcinoma
- `bkl` - Benign keratosis-like lesion
- `df` - Dermatofibroma
- `mel` - Melanoma
- `nv` - Melanocytic nevus
- `vasc` - Vascular lesion

Output models are saved to `$OUTPUT_ROOT/<class>-model/`

### Train Single Mixed Model

To train a single model on all classes combined:

```bash
SINGLE_MODEL=1 OUTPUT_NAME="ham10000-mix-model" bash train_all_dreambooth.sh
```

### Training Parameters

The script uses the following default training parameters:

- Resolution: 512
- Train batch size: 1
- Gradient accumulation steps: 1
- Learning rate: 5e-6
- LR scheduler: constant
- LR warmup steps: 0
- Max train steps: 15000 (configurable via `MAX_TRAIN_STEPS`)

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Key dependencies:
- `diffusers>=0.27.2`
- `accelerate>=0.29.3`
- `peft`
- `torch>=2.3.0`
- `transformers`

## Cache Configuration

The script automatically configures Hugging Face cache directories:

- `HF_HOME`: `$ROOT_PATH/cache/huggingface`
- `TRANSFORMERS_CACHE`: `$ROOT_PATH/cache/transformers`
- `DIFFUSERS_CACHE`: `$ROOT_PATH/cache/diffusers`
- `TORCH_HOME`: `$ROOT_PATH/cache/torch`

You can override these by setting the corresponding environment variables before running the script.
