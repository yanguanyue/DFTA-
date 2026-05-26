"""
Universal Grad-CAM script

Author: Adarsh Pritam
Date: 2025-11-11
"""
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import torchvision.models as models
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse

def get_target_layer(model, arch):
    """
    Automatically determine the target layer for GradCAM based on architecture.
    
    Args:
        model: PyTorch model
        arch: Architecture name (string)
    
    Returns:
        List containing the target layer
    """
    # ResNet family (resnet18, resnet34, resnet50, resnet101, resnet152, resnext50, resnext101)
    if 'resnet' in arch or 'resnext' in arch:
        return [model.layer4[-1]]
    
    # VGG family (vgg11, vgg13, vgg16, vgg19)
    elif 'vgg' in arch:
        return [model.features[-1]]
    
    # DenseNet family (densenet121, densenet161, densenet169, densenet201)
    elif 'densenet' in arch:
        return [model.features[-1]]
    
    # MobileNet family (mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large)
    elif 'mobilenet' in arch:
        return [model.features[-1]]
    
    # EfficientNet family (efficientnet_b0 through efficientnet_b7)
    elif 'efficientnet' in arch:
        return [model.features[-1]]
    
    # Inception family (inception_v3)
    elif 'inception' in arch:
        return [model.Mixed_7c]
    
    # SqueezeNet family (squeezenet1_0, squeezenet1_1)
    elif 'squeezenet' in arch:
        return [model.features[-1]]
    
    # Wide ResNet
    elif 'wide_resnet' in arch:
        return [model.layer4[-1]]
    
    # ShuffleNet family (shufflenet_v2_x0_5, shufflenet_v2_x1_0)
    elif 'shufflenet' in arch:
        return [model.conv5]
    
    # Vision Transformer (vit_b_16, vit_b_32, vit_l_16, vit_l_32)
    elif 'vit' in arch:
        # ViT requires different handling - use the last encoder block
        return [model.encoder.layers[-1].ln_1]
    
    # AlexNet
    elif 'alexnet' in arch:
        return [model.features[-1]]
    
    # GoogLeNet
    elif 'googlenet' in arch:
        return [model.inception5b]
    
    # RegNet family (regnet_y_400mf, regnet_x_400mf, etc.)
    elif 'regnet' in arch:
        return [model.trunk_output[-1]]
    
    # ConvNeXt family (convnext_tiny, convnext_small, etc.)
    elif 'convnext' in arch:
        return [model.features[-1][-1]]
    
    # Default fallback - try to find the last convolutional layer
    else:
        print(f"Warning: Architecture '{arch}' not explicitly supported. Attempting to find last conv layer...")
        # Try to find features module
        if hasattr(model, 'features'):
            return [model.features[-1]]
        # Try to find last layer4 (common in ResNet-like architectures)
        elif hasattr(model, 'layer4'):
            return [model.layer4[-1]]
        else:
            raise ValueError(f"Could not automatically determine target layer for architecture: {arch}")

def load_model(checkpoint_path, arch, num_classes, device='cpu'):
    """
    Load model from checkpoint with proper architecture and class adaptation.
    
    Args:
        checkpoint_path: Path to .pth.tar checkpoint
        arch: Model architecture name
        num_classes: Number of output classes
        device: Device to load model on
    
    Returns:
        Loaded model
    """
    # Create base model
    print(f"=> Creating model '{arch}'")
    model = models.__dict__[arch](weights=None)
    
    # Adapt final layer for num_classes (same logic as training script)
    print(f"=> Modifying final layer for {num_classes} classes")
    try:
        # ResNet, ResNeXt, Wide ResNet, RegNet
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, num_classes)
    except AttributeError:
        try:
            # VGG, AlexNet
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = torch.nn.Linear(num_ftrs, num_classes)
        except (AttributeError, IndexError):
            try:
                # EfficientNet, MobileNetV3
                num_ftrs = model.classifier[1].in_features
                model.classifier[1] = torch.nn.Linear(num_ftrs, num_classes)
            except (AttributeError, IndexError):
                try:
                    # Vision Transformer (ViT)
                    num_ftrs = model.heads.head.in_features
                    model.heads.head = torch.nn.Linear(num_ftrs, num_classes)
                except (AttributeError, IndexError):
                    try:
                        # DenseNet
                        num_ftrs = model.classifier.in_features
                        model.classifier = torch.nn.Linear(num_ftrs, num_classes)
                    except (AttributeError, IndexError):
                        try:
                            # SqueezeNet
                            model.classifier[1] = torch.nn.Conv2d(512, num_classes, kernel_size=1)
                            model.num_classes = num_classes
                        except (AttributeError, IndexError):
                            try:
                                # GoogLeNet, Inception
                                num_ftrs = model.fc.in_features
                                model.fc = torch.nn.Linear(num_ftrs, num_classes)
                            except AttributeError:
                                raise ValueError(f"Could not adapt final layer for architecture: {arch}")
    
    # Load checkpoint
    print(f"=> Loading checkpoint from '{checkpoint_path}'")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Remove 'module.' prefix if present (from DataParallel)
    state_dict = checkpoint["state_dict"]
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.eval()
    model = model.to(device)
    
    print(f"=> Loaded checkpoint (epoch {checkpoint.get('epoch', 'unknown')}, "
          f"acc: {checkpoint.get('acc', 'unknown'):.2f}%)")
    
    return model

def reshape_transform_vit(tensor, height=14, width=14):
    """
    Reshapes the output of a ViT layer to a 2D spatial format.
    
    Args:
        tensor: The output tensor from the ViT layer 
                (shape: [B, N, D] e.g., [1, 197, 768])
        height: The height of the patch grid (e.g., 14 for 224x224 / 16x16)
        width: The width of the patch grid (e.g., 14 for 224x224 / 16x16)
    
    Returns:
        Reshaped tensor (shape: [B, D, H, W] e.g., [1, 768, 14, 14])
    """
    # Input tensor shape: [B, N, D]
    # N = num_patches (H*W) + 1 (CLS token)
    
    # Remove the CLS token
    # [B, N, D] -> [B, N-1, D] (e.g., [1, 196, 768])
    result = tensor[:, 1:, :]
    
    # Permute to put channels (D) first
    # [B, N-1, D] -> [B, D, N-1] (e.g., [1, 768, 196])
    result = result.transpose(1, 2)
    
    batch_size, embedding_dim, num_patches = result.shape
    
    # Reshape to 2D spatial format
    # [B, D, N-1] -> [B, D, H, W] (e.g., [1, 768, 14, 14])
    result = result.reshape(batch_size, embedding_dim, height, width)
    
    return result

def generate_gradcam(image_path, model, arch, target_class=None, device='cpu'):
    """
    Generate Grad-CAM visualization for an image.
    
    Args:
        image_path: Path to input image
        model: Loaded PyTorch model
        arch: Model architecture name
        target_class: Target class for CAM (None = use prediction)
        device: Device to run inference on
    
    Returns:
        rgb_img, visualization, predicted_class
    """
    # Get target layer based on architecture
    target_layers = get_target_layer(model, arch)
    print(f"=> Using target layer: {target_layers}")
    
    # Load and preprocess image
    rgb_img = cv2.imread(image_path)
    if rgb_img is None:
        raise ValueError(f"Could not load image from: {image_path}")
    
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img = np.float32(rgb_img) / 255.0
    
    # Preprocess for model
    input_tensor = preprocess_image(
        rgb_img,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ).to(device)
    
    # Get model prediction
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = output.argmax(dim=1).item()
        confidence = torch.softmax(output, dim=1)[0, predicted_class].item()
    
    print(f"=> Predicted class: {predicted_class} (confidence: {confidence:.2%})")
    
    # Use target class or prediction
    target = target_class if target_class is not None else predicted_class
    
    # Create GradCAM object
    # Create GradCAM object
    transform = None
    if 'vit' in arch:
        # 224x224 image and 16x16 patch size = 14x14 grid
        image_size = 224 
        patch_size = 16 
        num_patches_side = image_size // patch_size
        
        # Use a lambda to pass the H/W arguments to the transform
        transform = lambda x: reshape_transform_vit(x, height=num_patches_side, width=num_patches_side)
        print(f"=> Applying ViT reshape_transform (target grid: {num_patches_side}x{num_patches_side})")

    cam = GradCAM(model=model, 
                target_layers=target_layers, 
                reshape_transform=transform) # Pass the transform here
    
    # Generate Grad-CAM for target class
    targets = [ClassifierOutputTarget(target)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    # Overlay CAM on image
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    return rgb_img, visualization, predicted_class, confidence

def main():
    parser = argparse.ArgumentParser(description='Universal Grad-CAM for PyTorch models')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth.tar)')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--arch', type=str, default='resnet50',
                        help='Model architecture (default: resnet50)')
    parser.add_argument('--num-classes', type=int, default=5,
                        help='Number of classes (default: 5)')
    parser.add_argument('--target-class', type=int, default=None,
                        help='Target class for CAM (default: use prediction)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save visualization (default: display only)')
    parser.add_argument('--gpu-id', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    
    args = parser.parse_args()
    
    # Set device
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=> Using device: {device}")
    
    # Load model
    model = load_model(args.checkpoint, args.arch, args.num_classes, device)
    
    # Generate Grad-CAM
    rgb_img, visualization, predicted_class, confidence = generate_gradcam(
        args.image, model, args.arch, args.target_class, device
    )
    
    # Visualize
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(rgb_img)
    # plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(visualization)
    target = args.target_class if args.target_class is not None else predicted_class
    plt.title(f'Grad-CAM - Class {target}\n(Predicted: {predicted_class}, Conf: {confidence:.1%})')
    plt.axis('off')
    
    plt.tight_layout()
    
    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f"=> Saved visualization to: {args.output}")
    else:
        plt.show()

if __name__ == '__main__':
    main()