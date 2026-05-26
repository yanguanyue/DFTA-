# Classifier Training

This directory contains classification model training code for the HAM10000 skin lesion dataset. It supports training various deep learning architectures on both real and synthetic (mixed) datasets.

## Project Structure

```
metirc/classifier/
├── train_mixed_models.sh       # Main training script for mixed datasets
├── mix_synthetic.py             # Mixed dataset construction script
├── summarize_improvements.py   # Results summarization
├── README.md                    # This file
```

## Models

This directory supports training the following model architectures:
- **ResNet18** - Lightweight residual network
- **ResNet50** - Deeper residual network with better capacity
- **EfficientNet-B0** - Efficient convolutional neural network


## Dataset Preparation

Organize your dataset in the following structure:

```
dataset/
├── train/
│   ├── class1/
│   ├── class2/
│   └── ...
└── val/
    ├── class1/
    ├── class2/
    └── ...
```

## Mixed Dataset Construction

To create a mixed dataset combining real and synthetic images:

```bash
python metirc/classifier/mix_synthetic.py \
  --real-train /root/autodl-tmp/data/HAM10000/input/train/HAM10000_img_class \
  --real-val /root/autodl-tmp/data/HAM10000/input/val/HAM10000_img_class \
  --synthetic-root /root/autodl-tmp/output/generate/<model_name> \
  --output-root /root/autodl-tmp/output/mixed_datasets/classifier/<model_name> \
  --majority-cap 500
```

## Training Commands

### Using the Training Script

The main script `train_mixed_models.sh` trains all model architectures on both baseline and mixed datasets:

```bash
cd /root/autodl-tmp/metirc/classifier
bash train_mixed_models.sh
```

This script supports the following synthetic models:
- Controlnet
- T2I-Adapter
- Derm-T2IM
- DreamBooth
- LF-VAR
- LesionGen
- Siamese
- DFMGAN
- flow
- skin-disease-diffusion
- ArSDM

### Using PyTorch Classification Framework

For individual model training, you can use the [pytorch-classification-extended](https://github.com/adarsh-crafts/pytorch-classification-extended) framework:

#### Prerequisites

```bash
git clone https://github.com/adarsh-crafts/pytorch-classification-extended.git
cd pytorch-classification-extended
pip install -r requirements.txt
```

#### Example: ResNet18
```bash
python customdata.py \
    -a resnet18 \
    -d "path_to_dataset_folder" \
    --pretrained \
    --epochs 30 \
    --schedule 15 25 \
    --gamma 0.1 \
    --lr 0.001 \
    -c "checkpoints/my_model/resnet18"
```
Example: EfficientNet-B0
```python
python customdata.py `
    -a efficientnet_b0 `
    -d "path_to_dataset_folder" `
    --pretrained `
    --epochs 30 `
    --schedule 15 25 `
    --gamma 0.1 `
    --lr 0.001 `
    --train-batch 64 `
    --test-batch 64 `
    -c "checkpoints/my_model/efficientnet_b0"
```

## Hyperparameters

| Parameter | ResNet18 | ResNet50 | ViT-B/16 | EfficientNet-B0 | VGG16 |
|-----------|----------|----------|----------|-----------------|-------|
| Epochs | `30` | `30` | `30` | `30` | `30` |
| schedule | `15 25` | `15 25` | `15 25` | `15 25` | `15 25` |
| gamma | `0.1` | `0.1` | `0.1` | `0.1` | `0.1` |
| lr | `0.001` | `0.001` | `0.001` | `0.001` | `0.001` |
| train-batch | [default] | `64` | `64` | `64` | `64` |
| test-batch | [default] | `64` | `64` | `64` | `64` |

## Hardware

- **GPU**: NVIDIA RTX 4060



## Inference

To use the trained models for inference:

```python
import torch
from torchvision import transforms
from PIL import Image

# Load model
model = torch.load('checkpoints/resnet18/best_model.pth')
model.eval()

# Prepare image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

image = Image.open('path/to/image.jpg')
input_tensor = transform(image).unsqueeze(0)

# Inference
with torch.no_grad():
    output = model(input_tensor)
    prediction = output.argmax(dim=1)
```

## Output Directories

- Checkpoints: `/root/autodl-tmp/output/classifier/checkpoints/`
- Logs: `/root/autodl-tmp/output/classifier/logs/`
- Mixed datasets: `/root/autodl-tmp/output/mixed_datasets/classifier/`
