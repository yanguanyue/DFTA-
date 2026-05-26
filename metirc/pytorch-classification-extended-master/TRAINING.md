# Training Recipes

This document contains detailed training recipes for reproducing the results reported in the main README.

## Table of Contents
- [Custom-data](#Custom-data)
- [CIFAR-10](#cifar-10)
- [CIFAR-100](#cifar-100)
- [ImageNet](#imagenet)
- [General Training Tips](#general-training-tips)

## Custom-data

#### VGG16
```bash
python customdata.py -a vgg16 --dataset dataset --checkpoint checkpoints/dataset/vgg16 --epochs 164 --schedule 81 122 --gamma 0.1 
```

### ResNet-50
```bash
python customdata.py -a resnet50 --data dataset --epochs 90 --schedule 31 61 --gamma 0.1 -c checkpoints/dataset/resnet50
```

## CIFAR-10

#### AlexNet
```bash
python cifar.py -a alexnet --epochs 164 --schedule 81 122 --gamma 0.1 --checkpoint checkpoints/cifar10/alexnet 
```

#### VGG19 (BN)
```bash
python cifar.py -a vgg19_bn --epochs 164 --schedule 81 122 --gamma 0.1 --checkpoint checkpoints/cifar10/vgg19_bn 
```

#### ResNet-110
```bash
python cifar.py -a resnet --depth 110 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar10/resnet-110 
```

#### ResNet-1202
```bash
python cifar.py -a resnet --depth 1202 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar10/resnet-1202 
```

#### PreResNet-110
```bash
python cifar.py -a preresnet --depth 110 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar10/preresnet-110 
```

#### ResNeXt-29, 8x64d
```bash
python cifar.py -a resnext --depth 29 --cardinality 8 --widen-factor 4 --schedule 150 225 --wd 5e-4 --gamma 0.1 --checkpoint checkpoints/cifar10/resnext-8x64d 
```

#### ResNeXt-29, 16x64d
```bash
python cifar.py -a resnext --depth 29 --cardinality 16 --widen-factor 4 --schedule 150 225 --wd 5e-4 --gamma 0.1 --checkpoint checkpoints/cifar10/resnext-16x64d 
```

#### WRN-28-10-drop
```bash
python cifar.py -a wrn --depth 28 --widen-factor 10 --drop 0.3 --epochs 200 --schedule 60 120 160 --wd 5e-4 --gamma 0.2 --checkpoint checkpoints/cifar10/WRN-28-10-drop
```

#### DenseNet-BC (L=100, k=12)
**Note**: 
* DenseNet uses weight decay value `1e-4`. Larger weight decay (`5e-4`) is harmful for accuracy (95.46% vs. 94.05%) 
* Official batch size is 64. There is no significant difference using batch size 64 or 128 (95.46% vs 95.11%)
```bash
python cifar.py -a densenet --depth 100 --growthRate 12 --train-batch 64 --epochs 300 --schedule 150 225 --wd 1e-4 --gamma 0.1 --checkpoint checkpoints/cifar10/densenet-bc-100-12
```

#### DenseNet-BC (L=190, k=40) 
```bash
python cifar.py -a densenet --depth 190 --growthRate 40 --train-batch 64 --epochs 300 --schedule 150 225 --wd 1e-4 --gamma 0.1 --checkpoint checkpoints/cifar10/densenet-bc-L190-k40
```

---

## CIFAR-100

#### AlexNet
```bash
python cifar.py -a alexnet --dataset cifar100 --checkpoint checkpoints/cifar100/alexnet --epochs 164 --schedule 81 122 --gamma 0.1 
```

#### VGG19 (BN)
```bash
python cifar.py -a vgg19_bn --dataset cifar100 --checkpoint checkpoints/cifar100/vgg19_bn --epochs 164 --schedule 81 122 --gamma 0.1 
```

#### ResNet-110
```bash
python cifar.py -a resnet --dataset cifar100 --depth 110 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar100/resnet-110 
```

#### ResNet-1202
```bash
python cifar.py -a resnet --dataset cifar100 --depth 1202 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar100/resnet-1202 
```

#### PreResNet-110
```bash
python cifar.py -a preresnet --dataset cifar100 --depth 110 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar100/preresnet-110 
```

#### ResNeXt-29, 8x64d
```bash
python cifar.py -a resnext --dataset cifar100 --depth 29 --cardinality 8 --widen-factor 4 --checkpoint checkpoints/cifar100/resnext-8x64d --schedule 150 225 --wd 5e-4 --gamma 0.1
```

#### ResNeXt-29, 16x64d
```bash
python cifar.py -a resnext --dataset cifar100 --depth 29 --cardinality 16 --widen-factor 4 --checkpoint checkpoints/cifar100/resnext-16x64d --schedule 150 225 --wd 5e-4 --gamma 0.1
```

#### WRN-28-10-drop
```bash
python cifar.py -a wrn --dataset cifar100 --depth 28 --widen-factor 10 --drop 0.3 --epochs 200 --schedule 60 120 160 --wd 5e-4 --gamma 0.2 --checkpoint checkpoints/cifar100/WRN-28-10-drop
```

#### DenseNet-BC (L=100, k=12)
```bash
python cifar.py -a densenet --dataset cifar100 --depth 100 --growthRate 12 --train-batch 64 --epochs 300 --schedule 150 225 --wd 1e-4 --gamma 0.1 --checkpoint checkpoints/cifar100/densenet-bc-100-12
```

#### DenseNet-BC (L=190, k=40) 
```bash
python cifar.py -a densenet --dataset cifar100 --depth 190 --growthRate 40 --train-batch 64 --epochs 300 --schedule 150 225 --wd 1e-4 --gamma 0.1 --checkpoint checkpoints/cifar100/densenet-bc-L190-k40
```

---

## ImageNet

### ResNet-18
```bash
python imagenet.py -a resnet18 --data ~/dataset/ILSVRC2012/ --epochs 90 --schedule 31 61 --gamma 0.1 -c checkpoints/imagenet/resnet18
```

### ResNet-50
```bash
python imagenet.py -a resnet50 --data ~/dataset/ILSVRC2012/ --epochs 90 --schedule 31 61 --gamma 0.1 -c checkpoints/imagenet/resnet50
```

### ResNeXt-50 (32x4d)
*Originally trained on 8 GPUs*
```bash
python imagenet.py -a resnext50 --base-width 4 --cardinality 32 --data ~/dataset/ILSVRC2012/ --epochs 90 --schedule 31 61 --gamma 0.1 -c checkpoints/imagenet/resnext50-32x4d
```

### EfficientNet-B0
```bash
python imagenet.py -a efficientnet_b0 --data ~/dataset/ILSVRC2012/ --epochs 350 --schedule 175 262 --gamma 0.1 --lr 0.256 --wd 1e-5 -c checkpoints/imagenet/efficientnet_b0
```

### Vision Transformer (ViT-B/16)
```bash
python imagenet.py -a vit_b_16 --data ~/dataset/ILSVRC2012/ --epochs 300 --schedule 150 225 --gamma 0.1 --lr 3e-3 --opt adamw --wd 0.3 -c checkpoints/imagenet/vit_b_16
```

---

## General Training Tips

### Command Line Arguments

**Common Arguments:**
- `-a, --arch`: Model architecture (e.g., `resnet`, `vgg19_bn`, `densenet`)
- `--depth`: Model depth (for depth-configurable architectures)
- `--epochs`: Number of training epochs
- `--schedule`: Learning rate decay milestones (epochs)
- `--gamma`: Learning rate decay factor
- `--lr`: Initial learning rate (default: 0.1)
- `--wd`: Weight decay (default: 1e-4)
- `--train-batch`: Training batch size (default: 128)
- `--test-batch`: Testing batch size (default: 100)
- `-c, --checkpoint`: Path to save checkpoints
- `--resume`: Path to checkpoint to resume training
- `--gpu-id`: GPU device ID(s) to use

**Architecture-Specific:**
- `--cardinality`: Number of groups for ResNeXt
- `--widen-factor`: Width multiplier for WRN and ResNeXt
- `--growthRate`: Growth rate for DenseNet
- `--drop`: Dropout rate for WRN

### Multi-GPU Training

**DataParallel (Single Node):**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python cifar.py -a resnet --depth 110 --gpu-id 0,1,2,3
```

**DistributedDataParallel (Recommended):**
```bash
python -m torch.distributed.launch --nproc_per_node=4 cifar.py -a resnet --depth 110
```

### Resuming Training

To resume from a checkpoint:
```bash
python cifar.py -a resnet --depth 110 --resume checkpoints/cifar10/resnet-110/checkpoint.pth.tar
```

### Monitoring Training

Training logs are automatically saved to the checkpoint directory. To visualize training curves:
```python
from utils.logger import Logger
logger = Logger('checkpoints/cifar10/resnet-110')
logger.plot()
```

### Memory Optimization

If you encounter out-of-memory errors:
1. Reduce batch size: `--train-batch 64`
2. Use gradient accumulation (modify training script)
3. Enable mixed precision training (requires apex or native PyTorch AMP)
4. Use smaller model variants

### Best Practices

1. **Learning Rate**: Start with 0.1 for SGD, 3e-4 for Adam/AdamW
2. **Weight Decay**: Use 1e-4 for most models, 5e-4 for ResNeXt/WRN
3. **Batch Size**: 128 for CIFAR, 256 for ImageNet (adjust based on GPU memory)
4. **Data Augmentation**: Random crop, horizontal flip for CIFAR; standard ImageNet augmentation
5. **Warmup**: Consider LR warmup for large batch training or Vision Transformers

### Reproducing Results

For exact reproduction:
1. Use the same PyTorch and CUDA versions
2. Set random seeds: `--manualSeed 42`
3. Use the provided hyperparameters exactly
4. Ensure proper data preprocessing

### Troubleshooting

**Poor Convergence:**
- Check learning rate (try reducing by 10x)
- Verify data normalization
- Check for gradient explosion (reduce LR or add gradient clipping)

**Overfitting:**
- Increase weight decay
- Add dropout (for supported architectures)
- Use data augmentation
- Consider early stopping

**Underfitting:**
- Increase model capacity (depth/width)
- Train for more epochs
- Reduce regularization (weight decay, dropout)

---