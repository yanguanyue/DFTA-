'''
Model Evaluation Script for ImageNet Classification
Calculates comprehensive metrics for research paper
'''
from __future__ import print_function

import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import models.imagenet as customized_models

from sklearn.metrics import (confusion_matrix, classification_report, 
                            accuracy_score, precision_recall_fscore_support,
                            roc_auc_score, average_precision_score, roc_curve,
                            precision_recall_curve)
from utils import Bar, AverageMeter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Models
default_model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

customized_models_names = sorted(name for name in customized_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(customized_models.__dict__[name]))

for name in customized_models.__dict__:
    if name.islower() and not name.startswith("__") and callable(customized_models.__dict__[name]):
        models.__dict__[name] = customized_models.__dict__[name]

model_names = default_model_names + customized_models_names

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch ImageNet Model Evaluation')
parser.add_argument('-d', '--data', default='path to dataset', type=str,
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--test-batch', default=200, type=int, metavar='N',
                    help='test batchsize (default: 200)')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names))
parser.add_argument('--checkpoint', required=True, type=str, metavar='PATH',
                    help='path to checkpoint folder (e.g., checkpoints/A1_classification/efficientnet_b0)')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--model-file', default='model_best.pth.tar', type=str,
                    help='checkpoint filename (default: model_best.pth.tar)')

args = parser.parse_args()

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

def main():
    # Data loading
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    num_classes = len(val_dataset.classes)
    class_names = val_dataset.classes
    print(f'==> Found {num_classes} classes: {class_names}')

    # Load checkpoint
    checkpoint_path = os.path.join(args.checkpoint, args.model_file)
    print(f'==> Loading checkpoint from {checkpoint_path}')
    
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
    
    checkpoint = torch.load(checkpoint_path)
    
    # Create model
    print(f"=> Creating model '{args.arch}'")
    model = models.__dict__[args.arch]()
    
    # Adapt final layer to num_classes if needed
    try:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    except AttributeError:
        try:
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs, num_classes)
        except (AttributeError, IndexError):
            try:
                num_ftrs = model.classifier[1].in_features
                model.classifier[1] = nn.Linear(num_ftrs, num_classes)
            except (AttributeError, IndexError):
                try:
                    num_ftrs = model.heads.head.in_features
                    model.heads.head = nn.Linear(num_ftrs, num_classes)
                except (AttributeError, IndexError):
                    num_ftrs = model.classifier.in_features
                    model.classifier = nn.Linear(num_ftrs, num_classes)
    
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(checkpoint['state_dict'])
    
    print(f'==> Loaded checkpoint (epoch {checkpoint["epoch"]}, best_acc: {checkpoint["best_acc"]:.2f}%)')
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    cudnn.benchmark = True

    # Evaluate
    print('\n==> Starting evaluation...')
    all_labels, all_predictions, all_probabilities = evaluate(val_loader, model, use_cuda, num_classes)

    # Calculate and save metrics
    print('\n==> Calculating metrics...')
    save_metrics(all_labels, all_predictions, all_probabilities, 
                 class_names, args.checkpoint, num_classes)

    print(f'\n==> Evaluation complete! Results saved in {args.checkpoint}')

def evaluate(val_loader, model, use_cuda, num_classes):
    """Evaluate model and collect predictions"""
    model.eval()
    
    all_labels = []
    all_predictions = []
    all_probabilities = []
    
    batch_time = AverageMeter()
    end = time.time()
    
    bar = Bar('Evaluating', max=len(val_loader))
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            
            # Get predictions
            outputs = model(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)
            
            # Store results
            all_labels.extend(targets.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            
            # Measure time
            batch_time.update(time.time() - end)
            end = time.time()
            
            # Progress
            bar.suffix = '({batch}/{size}) Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:}'.format(
                batch=batch_idx + 1,
                size=len(val_loader),
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
            )
            bar.next()
    
    bar.finish()
    
    return np.array(all_labels), np.array(all_predictions), np.array(all_probabilities)

def save_metrics(labels, predictions, probabilities, class_names, save_dir, num_classes):
    """Calculate and save all metrics"""
    
    # Create output file
    output_file = os.path.join(save_dir, 'evaluation_metrics.txt')
    
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MODEL EVALUATION METRICS\n")
        f.write("="*80 + "\n\n")
        
        # Overall Accuracy
        accuracy = accuracy_score(labels, predictions)
        f.write(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
        
        # Per-class metrics
        f.write("="*80 + "\n")
        f.write("PER-CLASS METRICS\n")
        f.write("="*80 + "\n\n")
        
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average=None, labels=range(num_classes), zero_division=0
        )
        
        f.write(f"{'Class':<30} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n")
        f.write("-"*80 + "\n")
        
        for i, class_name in enumerate(class_names):
            f.write(f"{class_name:<30} {precision[i]:<12.4f} {recall[i]:<12.4f} "
                   f"{f1[i]:<12.4f} {support[i]:<10}\n")
        
        # Macro and Weighted averages
        f.write("\n" + "="*80 + "\n")
        f.write("AVERAGED METRICS\n")
        f.write("="*80 + "\n\n")
        
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            labels, predictions, average='macro', zero_division=0
        )
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted', zero_division=0
        )
        
        f.write("Macro Average:\n")
        f.write(f"  Precision: {macro_precision:.4f}\n")
        f.write(f"  Recall:    {macro_recall:.4f}\n")
        f.write(f"  F1-Score:  {macro_f1:.4f}\n\n")
        
        f.write("Weighted Average:\n")
        f.write(f"  Precision: {weighted_precision:.4f}\n")
        f.write(f"  Recall:    {weighted_recall:.4f}\n")
        f.write(f"  F1-Score:  {weighted_f1:.4f}\n\n")
        
        # ROC-AUC (One-vs-Rest for multiclass)
        f.write("="*80 + "\n")
        f.write("ROC-AUC SCORES (One-vs-Rest)\n")
        f.write("="*80 + "\n\n")
        
        try:
            # Binarize labels for ROC-AUC
            from sklearn.preprocessing import label_binarize
            labels_bin = label_binarize(labels, classes=range(num_classes))
            
            # Calculate ROC-AUC for each class
            roc_auc_per_class = []
            f.write(f"{'Class':<30} {'ROC-AUC':<12}\n")
            f.write("-"*80 + "\n")
            
            for i, class_name in enumerate(class_names):
                if len(np.unique(labels_bin[:, i])) > 1:  # Check if class exists in test set
                    roc_auc = roc_auc_score(labels_bin[:, i], probabilities[:, i])
                    roc_auc_per_class.append(roc_auc)
                    f.write(f"{class_name:<30} {roc_auc:<12.4f}\n")
                else:
                    f.write(f"{class_name:<30} {'N/A (no samples)':<12}\n")
            
            # Macro and Weighted ROC-AUC
            if len(roc_auc_per_class) > 0:
                macro_roc_auc = roc_auc_score(labels_bin, probabilities, average='macro', multi_class='ovr')
                weighted_roc_auc = roc_auc_score(labels_bin, probabilities, average='weighted', multi_class='ovr')
                
                f.write("\n")
                f.write(f"Macro ROC-AUC:    {macro_roc_auc:.4f}\n")
                f.write(f"Weighted ROC-AUC: {weighted_roc_auc:.4f}\n\n")
        except Exception as e:
            f.write(f"Error calculating ROC-AUC: {str(e)}\n\n")
        
        # PR-AUC (Precision-Recall AUC)
        f.write("="*80 + "\n")
        f.write("PR-AUC SCORES (Precision-Recall AUC)\n")
        f.write("="*80 + "\n\n")
        
        try:
            pr_auc_per_class = []
            f.write(f"{'Class':<30} {'PR-AUC':<12}\n")
            f.write("-"*80 + "\n")
            
            for i, class_name in enumerate(class_names):
                if len(np.unique(labels_bin[:, i])) > 1:
                    pr_auc = average_precision_score(labels_bin[:, i], probabilities[:, i])
                    pr_auc_per_class.append(pr_auc)
                    f.write(f"{class_name:<30} {pr_auc:<12.4f}\n")
                else:
                    f.write(f"{class_name:<30} {'N/A (no samples)':<12}\n")
            
            # Macro and Weighted PR-AUC
            if len(pr_auc_per_class) > 0:
                macro_pr_auc = average_precision_score(labels_bin, probabilities, average='macro')
                weighted_pr_auc = average_precision_score(labels_bin, probabilities, average='weighted')
                
                f.write("\n")
                f.write(f"Macro PR-AUC:    {macro_pr_auc:.4f}\n")
                f.write(f"Weighted PR-AUC: {weighted_pr_auc:.4f}\n\n")
        except Exception as e:
            f.write(f"Error calculating PR-AUC: {str(e)}\n\n")
        
        # Confusion Matrix
        f.write("="*80 + "\n")
        f.write("CONFUSION MATRIX\n")
        f.write("="*80 + "\n\n")
        
        cm = confusion_matrix(labels, predictions)
        
        # Save confusion matrix as text
        f.write("Confusion Matrix (rows=true labels, cols=predictions):\n\n")
        f.write(f"{'':>20}")
        for class_name in class_names:
            f.write(f"{class_name[:15]:>17}")
        f.write("\n")
        
        for i, class_name in enumerate(class_names):
            f.write(f"{class_name[:18]:>20}")
            for j in range(num_classes):
                f.write(f"{cm[i, j]:>17}")
            f.write("\n")
        
        # Classification Report
        f.write("\n" + "="*80 + "\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("="*80 + "\n\n")
        
        report = classification_report(labels, predictions, target_names=class_names, 
                                       digits=4, zero_division=0)
        f.write(report)
        
        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print(f'==> Metrics saved to {output_file}')
    
    # Save confusion matrix as CSV for easy plotting
    cm_file = os.path.join(save_dir, 'confusion_matrix.csv')
    np.savetxt(cm_file, cm, delimiter=',', fmt='%d')
    print(f'==> Confusion matrix saved to {cm_file}')
    
    # Save class names
    classes_file = os.path.join(save_dir, 'class_names.txt')
    with open(classes_file, 'w') as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")
    print(f'==> Class names saved to {classes_file}')
    
    # Generate and save confusion matrix plot
    try:
        plt.figure(figsize=(max(10, num_classes), max(8, num_classes*0.8)))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count'})
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        cm_plot_file = os.path.join(save_dir, 'confusion_matrix.png')
        plt.savefig(cm_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'==> Confusion matrix plot saved to {cm_plot_file}')
    except Exception as e:
        print(f'Warning: Could not save confusion matrix plot: {str(e)}')

if __name__ == '__main__':
    main()