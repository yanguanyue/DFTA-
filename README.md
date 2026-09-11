# Skin Lesion Image Generation and Evaluation - DFTA 🏥

This repository presents **DFTA (Dual-Flow Trajectory Alignment)**, a skin lesion image generation method based on Flow Matching. It provides a comprehensive framework for skin lesion image generation, evaluation, and downstream task validation using HAM10000, ISIC2017, and Dermofit. The framework includes training, generation-quality assessment, downstream classification and segmentation, cross-dataset evaluation, and a complete A--I/DFTA ablation study. Training and generation scripts are implemented for ArSDM, ControlNet, T2I-Adapter, DreamBooth, DFMGAN, LesionGen, Derm-T2IM, LF-VAR, Siamese Diffusion, and Skin-Disease-Diffusion. The current paper evaluation pipeline, however, includes eight of these baselines (all except LF-VAR and Skin-Disease-Diffusion) together with DFTA.

## Publication

**Accepted and in production** in the *International Journal of Imaging Systems and Technology*.

- **Title:** DFTA: Dual-Flow Trajectory Alignment for Medical Pathology Image Synthesis
- **Journal:** *International Journal of Imaging Systems and Technology*
- **DOI:** [10.1002/ima.70447](https://doi.org/10.1002/ima.70447)

> **Note:** The article is currently in production. The DOI link may not resolve until the publisher completes DOI activation and online publication.

---

## 🏗️ Architecture Overview

<p align="center">
<img src="image/DFTA Architecture Overview.png" alt="DFTA Architecture Overview" width="900"/>
</p>
<p align="center"><em>Figure: Dual-Flow Trajectory Alignment (DFTA) architecture — parallel Image-Flow and Mask-Flow paths with trajectory consistency guidance and Online Straight Euler Augmentation.</em></p>

---

## Directory Conventions (Default Paths) 📁

| Purpose | Path |
|---------|------|
| Generated Images | `/root/autodl-tmp/output/generate/<MODEL>/<CLASS>/images` |
| Main Evaluation Output | `/root/autodl-tmp/output/generate_evaluation` |
| Supplementary Evaluation Output | `/root/autodl-tmp/output/generate_evaluation/{2017,dermofit}` |
| Complete Ablation Output | `/root/autodl-tmp/output/ablation_complete/evaluation` |
| Model Cache | `/root/autodl-tmp/model` |
| Real Data | `/root/autodl-tmp/data/HAM10000/input` |

---

## Core Academic and Visual Content 📊

### 🔑 Key Achievements

The values below are read from `output/generate_evaluation`. Generation metrics are reported on the full generated sets; downstream values are means over five seeds.

| Task | Metric | DFTA (Ours) | Strongest non-DFTA result | Difference |
|------|--------|-------------|---------------------------|------------|
| **Generation Quality** | KID ↓ | **0.0408** | 0.0925 (Derm-T2IM) | **-0.0517** |
| | Density ↑ | **0.5859** | 0.5738 (LesionGen) | **+0.0121** |
| | Coverage ↑ | **0.7442** | 0.5576 (ArSDM) | **+0.1866** |
| **Classification** (ResNet18) | BAcc / Macro-F1 ↑ | **75.50 / 76.90** | 74.70 / **76.97** (Siamese-Diffusion) | **+0.80 / -0.07 pp** |
| **Classification** (ResNet50) | BAcc / Macro-F1 ↑ | 78.23 / 79.89 | **78.59** (ArSDM) / **80.16** (DFMGAN) | **-0.36 / -0.27 pp** |
| **Classification** (EfficientNet-B0) | BAcc / Macro-F1 ↑ | **72.31 / 74.19** | 70.39 (LesionGen) / 72.30 (Derm-T2IM) | **+1.92 / +1.89 pp** |
| **Segmentation** (SegFormer) | mDice / mIoU ↑ | **94.27 / 89.67** | 94.14 / 89.50 (T2I-Adapter) | **+0.13 / +0.17 pp** |
| **Segmentation** (UNet) | mDice / mIoU ↑ | **89.32 / 82.28** | 87.67 / 79.96 (ArSDM) | **+1.65 / +2.32 pp** |

### Supplementary Dataset Results

DFTA results on the two supplementary datasets are summarized below. Downstream values are five-seed means.

| Dataset | KID ↓ | Density ↑ | Coverage ↑ |
|---------|-------|-----------|------------|
| ISIC2017 | 0.0775 | 0.6960 | 0.9493 |
| Dermofit | 0.0782 | 0.7905 | 0.8750 |

| Dataset | Backbone | BAcc / Macro-F1 ↑ |
|---------|----------|-------------------|
| ISIC2017 | ResNet18 | 70.83 / 72.84 |
| ISIC2017 | ResNet50 | 76.80 / 77.67 |
| ISIC2017 | EfficientNet-B0 | 69.12 / 70.62 |
| Dermofit | ResNet18 | 82.94 / 83.66 |
| Dermofit | ResNet50 | 86.94 / 86.79 |
| Dermofit | EfficientNet-B0 | 77.77 / 79.62 |

| Dataset | Backbone | mDice / mIoU ↑ |
|---------|----------|----------------|
| ISIC2017 | SegFormer | 87.26 / 79.64 |
| ISIC2017 | UNet | 79.73 / 70.39 |
| Dermofit | SegFormer | 91.85 / 85.69 |
| Dermofit | UNet | 85.83 / 76.41 |

### 📈 Quantitative Analysis: Fidelity vs. Diversity Trade-off

The figure below illustrates the performance comparison across all evaluated models on HAM10000 dataset. DFTA achieves the **optimal balance** between structural fidelity (Density) and sample diversity (Coverage), with the **lowest KID score** (darkest color), indicating minimal distribution discrepancy from real data.

<p align="center">
<img src="image/Quantitative Performance Comparison.png" alt="Quantitative Performance Comparison" width="650"/>
</p>
<p align="center"><em>Figure: Performance scatter plot on HAM10000 — DFTA occupies the top-right corner (highest fidelity & diversity) with the lowest KID.</em></p>

### 🖼️ Qualitative Analysis: Visual Comparison Across Methods

Qualitative results demonstrate DFTA's superior capability in modeling complex dermatopathological structures while preserving fine-grained local texture details. Compared to baseline methods, DFTA maintains better morphological consistency, generates more coherent pathological textures, and achieves a superior balance between local details and global anatomy.

<p align="center">
<img src="image/Qualitative Comparison of Generated Skin Lesion Images.png" alt="Qualitative Comparison of Generated Skin Lesion Images" width="900"/>
</p>
<p align="center"><em>Figure: Qualitative comparison across methods and lesion categories. Methods marked with (M) use lesion masks as input. DFTA produces the most realistic and structurally consistent results.</em></p>


---

## Pretrained Models 🤖

We provide pretrained weights and a fully trained model (10000 steps) for one-click download:

🔗 **Model Hub**: [https://huggingface.co/yanguanyue/DFTA-10000-steps](https://huggingface.co/yanguanyue/DFTA-10000-steps)

The model includes:
- **Pretrained weights**: `PRETRAINED/merged_pytorch_model.pth` (from Siamese polyp model)
- **Trained checkpoint**: `lightning_logs/version_4/checkpoints/last.ckpt` (10000 steps)

---

## Quick Start ⚡

```bash
# Environment
conda env create -f main/environment.yaml
conda activate flow

# 1. Download data
bash scripts/dataset_downloader.sh

# 2. Train Flow model
bash scripts/compare_Main.sh

# 3. Normalize all generated inputs, then run the complete HAM10000 evaluation
python scripts/generate_evaluation/prepare_inputs.py
bash scripts/generate_evaluation/run_all.sh

# 4. Run the supplementary ISIC2017 and Dermofit evaluations
bash scripts/generate_evaluation/2017/run_all.sh
bash scripts/generate_evaluation/dermofit/run_all.sh

# 5. Evaluate the complete A--I ablation sequence
bash scripts/ablation/complete/run_all_ai.sh all
```

## 1. Environment Setup 🛠️

### 1.1 Create Conda Environment

Use the provided environment file at `main/environment.yaml`:

```bash
# Install dependencies from environment file
conda env create -f main/environment.yaml

# Activate the environment (the yaml declares `name: flow`)
conda activate flow
```

### 1.2 Directory Structure

```
/root/autodl-tmp/
├── main/                          # Flow-matching code and configuration
│   ├── train.py                   # Training entry (PyTorch Lightning)
│   ├── inference.py               # Inference entry
│   ├── generate.py                # Batch generation for HAM10000
│   ├── dataset.py                 # Data loading logic
│   ├── config/flow_matching.yaml  # Model configuration
│   ├── modules/                   # Model modules (cldm/, ldm/)
│   ├── share.py                   # Common settings
│   └── data/prompt.json           # Training/prompt list
├── scripts/                        # Shell scripts for all tasks
├── checkpoint/flow/                # Training checkpoints (Lightning)
├── output/generate/flow/          # Generated images (by class)
├── output/generate_evaluation/    # Main evaluation results
├── output/crossdataset/           # Downloaded ISIC2017 and Dermofit result snapshots
├── output/ablation_complete/      # Complete A--I ablation results
├── compare_main/                  # Comparison models
└── metirc/                        # Classification, segmentation, and metric utilities
```

### 1.3 Key Scripts

| Script | Purpose |
|--------|---------|
| `scripts/compare_Main.sh` | One-click Flow training + generation |
| `scripts/dataset_downloader.sh` | Download HAM10000 dataset |
| `scripts/generate_evaluation/run_all.sh` | Run the complete main evaluation |
| `scripts/generate_evaluation/run_metric.sh` | Run generation quality metrics |
| `scripts/generate_evaluation/run_classifier.sh` | Train repeated downstream classifiers |
| `scripts/generate_evaluation/run_segmentation.sh` | Train repeated downstream segmenters |

---

## 2. Data Download 📥

Download and prepare the HAM10000 dataset using the provided script.

### Script
```bash
bash scripts/dataset_downloader.sh
```

This script will:
- Download the original HAM10000 data (with mirror support)
- Split into train/val sets
- Resize images to 512×512
- Organize by class into `HAM10000_img_class` / `HAM10000_seg_class`

### Expected Output
After running, the following directories should exist:
- `data/HAM10000/input/train/HAM10000_img_class`
- `data/HAM10000/input/val/HAM10000_img_class`

### HAM10000 Classes
The dataset contains 7 skin lesion classes:
| Code | Full Name |
|------|-----------|
| akiec | Actinic Keratoses |
| bcc | Basal Cell Carcinoma |
| bkl | Benign Keratosis |
| df | Dermatofibroma |
| mel | Melanoma |
| nv | Melanocytic Nevi |
| vasc | Vascular Lesions |

---

## 3. Data Preprocessing 🔧

### 3.1 Hair Removal (DullRazor)

Optional preprocessing to remove hair artifacts from dermoscopy images.

```bash
bash scripts/run_dullrazor_ham10000.sh
```

This processes images in `data/HAM10000/input` for cleaner generation inputs.

---

## 4. Flow Model Training 🚀

Flow-Matching (ControlLDM) training and generation pipeline using `scripts/compare_Main.sh`.

### 4.1 One-Click Training + Generation

Run the complete pipeline with one command:

```bash
bash scripts/compare_Main.sh
```

This script handles:
- Training the Flow-Matching model
- Auto-finding the latest checkpoint
- Generating images for all 7 HAM10000 classes

### 4.2 Control Generation Only

```bash
# Skip training, only run generation
TRAIN_ENABLED=false bash scripts/compare_Main.sh
```

### 4.3 Quick Test Mode

```bash
# Test mode: 5 training steps, 1 image per class
TEST_MODE=1 bash scripts/compare_Main.sh
```

### 4.4 Using Pre-trained Weights

```bash
# Continue training with pre-trained weights
USE_PRETRAINED=1 bash scripts/compare_Main.sh

# Use pre-trained for generation only (skip training)
TRAIN_ENABLED=false USE_PRETRAINED=1 bash scripts/compare_Main.sh
```

### 4.5 Key Parameters (Environment Variables)

| Variable | Description | Default |
|----------|-------------|---------|
| `TRAIN_ENABLED` | Enable/disable training stage | `true` |
| `RUN_ENABLED` | Enable/disable generation stage | `true` |
| `TEST_MODE` | Quick test mode (5 steps, 1 img/class) | `false` |
| `USE_PRETRAINED` | Load pre-trained weights for initialization | `true` |
| `NUM_IMAGES_PER_CLASS` | Number of images to generate per class | 1500 |
| `MAX_STEPS` | Maximum training steps | 10000 |
| `NUM_WORKERS` | Data loading workers | 4 |
| `RESUME_CKPT` | Specify checkpoint path to resume from | - |

### 4.6 Input Data Format

`main/data/prompt.json` - One JSON per line:
```json
{"source": "/path/to/mask.png", "target": "/path/to/image.jpg", "prompt": "A photo of skin lesion"}
```

### 4.7 Output Locations

| Output | Path |
|--------|------|
| Training Checkpoint | `checkpoint/flow/lightning_logs/...` |
| Merged Model | `checkpoint/flow/PRETRAINED/merged_pytorch_model.pth` |
| Generated Images | `output/generate/flow/<class>/images/` |

---

## 5. Comparison Model Training 📈

Ten comparison models have training and generation scripts. The current paper/main evaluation configured by `scripts/generate_evaluation/prepare_inputs.py` and `scripts/generate_evaluation/models.sh` uses ArSDM, ControlNet, DFMGAN, Derm-T2IM, DreamBooth, LesionGen, Siamese-Diffusion, and T2I-Adapter, plus DFTA (`flow`). LF-VAR and Skin-Disease-Diffusion are implemented in the repository but are not included in that current evaluation pipeline.

### 5.1 Model Comparison Table

| # | Model | Script | Description |
|---|-------|--------|-------------|
| 1 | **DreamBooth** | `scripts/compare_DreamBooth.sh` | Subject-driven generation using mixed training across all classes |
| 2 | **ControlNet Depth** | `scripts/compare_ControlNet.sh` | Depth-map conditioned generation with LoRA fine-tuning |
| 3 | **T2I-Adapter** | `scripts/compare_T2i_adapter.sh` | Lightweight adapter using depth maps as conditioning |
| 4 | **ArSDM** | `scripts/compare_ArSDM.sh` | Two-stage training: base model + class-specific LoRA |
| 5 | **Siamese-Diffusion** | `scripts/compare_Siamese.sh` | Siamese architecture for paired image generation |
| 6 | **LF-VAR** | `scripts/compare_LF-VAR.sh` | Latent Variable AutoRegressive model for synthesis |
| 7 | **Skin-Disease-Diffusion** | `scripts/compare_skin-disease-diffusion.sh` | Specialized diffusion for skin disease images |
| 8 | **LesionGen** | `scripts/compare_LesionGen.sh` | Lesion-specific generation framework |
| 9 | **DFMGAN** | `scripts/compare_DFMGAN.sh` | Deep Feature Matching GAN approach |
| 10 | **Derm-T2IM** | `scripts/compare_Derm_T2IM.sh` | Text-to-image model for dermoscopy |

### 5.2 Common Usage Pattern

```bash
# Full training + generation (method-specific training defaults)
bash scripts/compare_<Model>.sh

# Test mode (quick validation)
TEST_MODE=1 bash scripts/compare_<Model>.sh

# Training only (skip generation)
RUN_ENABLED=false bash scripts/compare_<Model>.sh

# Generation only (requires existing model)
TRAIN_ENABLED=false bash scripts/compare_<Model>.sh

# Custom parameters
MAX_TRAIN_STEPS=10000 NUM_IMAGES_PER_CLASS=500 bash scripts/compare_<Model>.sh
```

### 5.3 Key Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TRAIN_ENABLED` | Enable training stage | `true` |
| `RUN_ENABLED` | Enable generation stage | `true` |
| `TEST_MODE` | Quick test mode | `false` |
| `MAX_TRAIN_STEPS` | Training steps | Method-specific; inspect the selected script |
| `NUM_IMAGES_PER_CLASS` | Images per class | 1500 |

The comparison scripts do not all use the same optimization-step budget. The released main evaluation compares their final generated outputs under the same input normalization, generated-image quota, real reference split, and metric implementation; it should not be described as an equal-training-step comparison. DFTA's current `scripts/compare_Main.sh` default and the reported checkpoint use 10,000 training steps.

---

## 6. Medical Text Prompts 💬

Some comparison models (e.g., ControlNet, T2I-Adapter) require text prompts for generation. This project provides pre-generated prompts for all HAM10000 classes.

### Pre-provided Prompts

The project includes ready-to-use prompts in CSV format:
- `data/metadata_train_llava.csv`
- `data/metadata_val_llava.csv`
- `data/metadata_test_llava.csv`

These contain image paths, segmentation masks, and disease descriptions for each class.

### Custom Prompt Generation (Optional)

If you want to generate custom medical prompts using LLaVA, run:

```bash
bash scripts/run_generate_llava_med.sh
```

This will:
1. Clone LLaVA repository if not present
2. Generate medical prompts from HAM10000 metadata
3. Save to `/root/autodl-tmp/model` and `/root/autodl-tmp/data`

#### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_ID` | LLaVA model identifier | - |
| `MODEL_DIR` | Local model directory | - |
| `LIMIT` | Maximum prompts to generate | - |

---

## 7. Latest Main Evaluation Pipeline 📊

The current main experiment is implemented under `scripts/generate_evaluation/` and writes results to `output/generate_evaluation/`. It supersedes the older standalone `scripts/run_metric.sh`, `scripts/run_classifier.sh`, and `scripts/run_segmentation.sh` workflow.

Before evaluation, normalize the generated outputs from all configured methods:

```bash
python scripts/generate_evaluation/prepare_inputs.py
```

This creates a consistent `<MODEL>/<CLASS>/images` and optional `masks` layout under `output/generate_evaluation/normalized/`, and records input counts in `input_report.csv` and `input_report.json`. The configured methods are ArSDM, Controlnet, DFMGAN, Derm-T2IM, DreamBooth, LesionGen, Siamese, T2I-Adapter, and `flow` (DFTA).

Run every main-evaluation stage with:

```bash
bash scripts/generate_evaluation/run_all.sh
```

The command runs generation metrics, five-seed classification, and five-seed segmentation in sequence.

### Metrics Included

| Metric | Description |
|--------|-------------|
| KID | Kernel Inception Distance |
| Density & Coverage | Distribution coverage metrics |

### 7.1 Generation Quality

```bash
bash scripts/generate_evaluation/run_metric.sh
```

KID, Density, and Coverage are computed for every method against the HAM10000 validation split. The paper uses the aggregate `summary` object from each per-method JSON file.

### 7.2 Generation Evaluation Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `NORMALIZED_ROOT` | Normalized generated images | `/root/autodl-tmp/output/generate_evaluation/normalized` |
| Real image root | Evaluation reference images | `/root/autodl-tmp/data/HAM10000/input` |
| Real split | Evaluation split | `val` |
| `EVAL_ROOT` | Evaluation output root | `/root/autodl-tmp/output/generate_evaluation` |
| `BATCH_SIZE` | Batch size | 16 |
| Model list | Defined in `scripts/generate_evaluation/models.sh` | All configured models |

### 7.3 Generation Output

- Per-method JSON results: `output/generate_evaluation/generation_metrics/<MODEL>/metrics_<MODEL>.json`
- Each JSON file contains per-class values and the aggregate `summary` used in the paper.

### 7.4 Pre-downloaded Weights

To avoid repeated downloads, ensure Inception weights exist at:
```
/root/autodl-tmp/model/inception_v3_google-0cc3c7bd.pth
```

Set `TORCH_HOME=/root/autodl-tmp/model` to cache models.

---

## 8. Latest Main Downstream Classification 🔬

Classification mixes real training images with synthetic samples using the same class-aware protocol for every generation method. Minority classes (`akiec`, `bcc`, `df`, and `vasc`) use up to 1,500 synthetic images per class, while majority classes (`bkl`, `mel`, and `nv`) use up to 500. A real-only baseline is evaluated through the same training code.

### 8.1 Run Classifier

```bash
bash scripts/generate_evaluation/run_classifier.sh
```

### 8.2 Current Protocol

| Variable | Description | Default |
|----------|-------------|---------|
| Real training data | Original training split | `data/HAM10000/input/train/HAM10000_img_class` |
| Real validation data | Original validation split | `data/HAM10000/input/val/HAM10000_img_class` |
| Synthetic data | Normalized generated images | `output/generate_evaluation/normalized` |
| Mixed datasets | Real and synthetic training sets | `output/generate_evaluation/classifier/mixed` |
| Output root | Per-seed classifier results | `output/generate_evaluation/classifier` |
| `EPOCHS` | Training epochs | 30 |
| `TRAIN_BATCH` | Training batch size | 64 |
| `SEEDS` | Repeated-run seeds | `42 123 2024 3407 7777` |

Each of ResNet18, ResNet50, and EfficientNet-B0 is retrained independently for every augmentation method and seed. The reported paper metrics are BAcc and Macro-F1; Accuracy, Macro-AUC, Macro-Precision, and Macro-Recall remain available in the result files but are not used in the current paper tables.

### 8.3 Supported Models

- ResNet18
- ResNet50
- EfficientNet-B0

### 8.4 Output

- Per-seed results: `output/generate_evaluation/classifier/runs/`
- Aggregate means and standard deviations: `output/generate_evaluation/classifier/statistics/summary.csv`

---

## 9. Latest Main Downstream Segmentation 🩺

Segmentation uses only methods that provide paired synthetic images and masks: ArSDM, Controlnet, DFMGAN, Siamese, T2I-Adapter, and DFTA (`flow`). The mixer uses the same 1,500/500 minority/majority caps as the main classification protocol. UNet and SegFormer are trained independently over the five fixed seeds, and results are reported as mDice and mIoU.

### 9.1 Run Segmentation

```bash
bash scripts/generate_evaluation/run_segmentation.sh
```

### 9.2 Current Protocol

| Variable | Description | Default |
|----------|-------------|---------|
| Real data root | Original image--mask pairs | `data/HAM10000/input` |
| Synthetic data | Normalized generated image--mask pairs | `output/generate_evaluation/normalized` |
| Mixed datasets | Real and synthetic segmentation sets | `output/generate_evaluation/segmentation/mixed` |
| Output root | Per-seed segmentation results | `output/generate_evaluation/segmentation` |
| `MAX_STEPS` | Training max steps | 15000 |
| `VAL_INTERVAL` | Validation interval | 3000 |
| `BATCH_SIZE` | Batch size | 4 |
| `IMAGE_SIZE` | Image resolution | 512 |
| `SEEDS` | Repeated-run seeds | `42 123 2024 3407 7777` |

The current script also defines a real-only baseline using the original HAM10000 image--mask pairs and the same optimization settings as the augmented runs.

### 9.3 Output

- Per-seed results: `output/generate_evaluation/segmentation/runs/`
- Aggregate means and standard deviations: `output/generate_evaluation/segmentation/statistics/summary.csv`


---

## 10. Supplementary Dataset Evaluation 🌍

The current paper evaluates generation quality and within-dataset downstream augmentation on ISIC2017 and Dermofit. These are not the older zero-shot ISIC2017/PH2 scripts under `scripts/crossdataset/`. Each latest `run_all.sh` pipeline prepares the dataset and normalized synthetic inputs, computes generation metrics, constructs a real-only baseline, retrains every downstream architecture over five seeds, and aggregates the results.

### 10.1 Run ISIC2017

```bash
bash scripts/generate_evaluation/2017/run_all.sh
```

Results are written to:

- Runtime path configured by the script: `output/generate_evaluation/2017/`
- Generation metrics: `generation_metrics/<MODEL>/metrics_<MODEL>.json`
- Classification summary: `classifier/statistics/summary.csv`
- Segmentation summary: `segmentation/statistics/summary.csv`

ISIC2017 uses three HAM10000-compatible labels. The manifest builder maps the official ground-truth columns as follows: `melanoma=1` to `mel`, `seborrheic_keratosis=1` to `bkl`, and samples negative for both columns to `nv`. The configured synthetic-image caps are `bkl=1500`, `mel=1500`, and `nv=500`. Classification uses ResNet18, ResNet50, and EfficientNet-B0; segmentation uses UNet and SegFormer.

### 10.2 Run Dermofit

```bash
bash scripts/generate_evaluation/dermofit/run_all.sh
```

Results are written to:

- Runtime path configured by the script: `output/generate_evaluation/dermofit/`
- Generation metrics: `generation_metrics/<MODEL>/metrics_<MODEL>.json`
- Classification summary: `classifier/statistics/summary.csv`
- Segmentation summary: `segmentation/statistics/summary.csv`

Dermofit uses manifests already normalized to the following seven HAM10000-compatible labels:

| Normalized label | Clinical category used by the experiment |
|------------------|------------------------------------------|
| `akiec` | Actinic keratosis or intraepithelial carcinoma |
| `bcc` | Basal cell carcinoma |
| `bkl` | Benign keratosis-like lesion |
| `df` | Dermatofibroma |
| `mel` | Melanoma |
| `nv` | Melanocytic nevus |
| `vasc` | Vascular lesion |

The configured synthetic-image caps are `akiec=150`, `bcc=50`, `bkl=50`, `df=150`, `mel=150`, `nv=50`, and `vasc=150`. It uses the same three classifiers, two segmenters, five seeds, 30 classification epochs, and 15,000 segmentation steps as ISIC2017. The released experiment directory contains these normalized manifests and validates all seven labels, but it does not contain the earlier raw-Dermofit-to-manifest conversion script. Consequently, any raw source labels that were merged or excluded cannot be reconstructed from this repository alone and should not be inferred from the normalized class names.

In this Windows workspace, the downloaded result snapshots used by the paper are stored under `output/crossdataset/2017/` and `output/crossdataset/dermofit/`. The Linux evaluation scripts themselves retain the runtime roots shown above.

---

## 11. Ablation Study 🧪

The latest ablation study evaluates groups A--I and DFTA under a common input-validation and downstream protocol. The DFTA candidate pool comes from the main `flow` experiment, while `group_paths.sh` enumerates the ablation variants A--I only. Although the main DFTA generation directory contains 10,500 images, every ablation component selects the same 7,500-image class quota used for A--I.

### 11.0 Complete Ablation Configurations

The current evaluation retains the ten-row sequence used in the paper:

| Group | Configuration |
|-------|---------------|
| A | Diffusion baseline with a correct mask |
| B | Flow Matching with a zero mask |
| C | Flow Matching with a class-wise mismatched mask |
| D | Single-branch Flow Matching with the correct mask |
| E | D with Conditional Stochastic Flow Sampling (CSFS) |
| F | D plus the image-flow branch |
| G | F plus trajectory alignment |
| H | G plus Online Straight Euler Augmentation (OSEA) |
| I | G plus CSFS |
| DFTA | G plus OSEA and CSFS |

All rows use the same data split, training budget, effective generated-image quota, and downstream protocol. Generation quality, classification, and segmentation are reported separately. For DFTA, the additional images in the 10,500-image source pool are not used in the quota-controlled ablation comparison.

### 11.1 Prepare the A--I Inputs

The complete evaluation expects generated inputs under `output/ablation_complete/A` through `output/ablation_complete/I`, as defined by `scripts/ablation/complete/group_paths.sh`. The older `scripts/ablation/run_ablation_experiments.sh` runner implements a separate two-model/three-mode experiment and does not generate the paper's A--I sequence; it should therefore not be used to reproduce the current ablation table.

The expected generated counts are 1,500 for `akiec`, `bcc`, `df`, and `vasc`, and 500 for `bkl`, `mel`, and `nv`. The complete evaluation scripts validate these inputs before computing metrics; segmentation additionally requires masks.

### 11.2 Run the Complete Evaluation

```bash
bash scripts/ablation/complete/run_all_ai.sh all
```

The wrapper runs generation metrics, classification, and segmentation for A--I. Individual stages can also be selected with `metric`, `classifier`, or `segmentation`.

Generation evaluation computes KID, Density, and Coverage against the HAM10000 validation split. Every ablation row, including DFTA, effectively uses 7,500 generated images: 1,500 each for `akiec`, `bcc`, `df`, and `vasc`, and 500 each for `bkl`, `mel`, and `nv`. DFTA is generated as a larger 10,500-image candidate pool (1,500 per class), but only the quota-matched 7,500-image subset is used in each ablation component. Thus, the A--I/DFTA comparison is controlled for the number of evaluated synthetic images and their class composition; the source-directory size must not be confused with the effective evaluation sample count.

### 11.3 Ablation Downstream Tasks

```bash
bash scripts/ablation/complete/run_classifier_ai.sh
bash scripts/ablation/complete/run_segmentation_ai.sh
```

Classification evaluates ResNet18, ResNet50, and EfficientNet-B0 with seeds 42, 123, 2024, 3407, and 7777. Segmentation evaluates UNet and SegFormer with the same seeds. The paper reports BAcc/Macro-F1 for classification and mDice/mIoU for segmentation.

Outputs are saved under:
- `output/ablation_complete/` (generated images and configurations)
- `output/ablation_complete/evaluation/generation_metrics/`
- `output/ablation_complete/evaluation/classifier/statistics/summary.csv`
- `output/ablation_complete/evaluation/segmentation/statistics/summary.csv`

---

## 12. Open Source and Academic Standards 🌐












---
