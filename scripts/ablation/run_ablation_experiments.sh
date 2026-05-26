#!/usr/bin/env bash
set -euo pipefail

##############################
# DFTA Ablation Study (Optimized: 2 Trained Models + 3 Generation Modes)
#
# Train only 2 ablation models:
#   Model A: Single-Flow (No Image-Flow, No OSEA)
#   Model B: Dual-Flow with Trajectory Alignment (No OSEA)
#
# Mode 3 uses the existing full DFTA model from checkpoint/flow (already trained)
#
# Generation Modes (renumbered 1/2/3):
#   Mode 1:  Single-Flow + CSFS Stochastic Sampling (original mode 2)
#   Mode 2:  Dual-Flow (No Aug) + CSFS Stochastic Sampling (original mode 4)
#   Mode 3:  Dual-Flow (Full/Pretrained from checkpoint/flow) + Deterministic Sampling (original mode 5)
#
# Usage:
#   ./run_ablation_experiments.sh [train|generate|all] [options]
#
# Examples:
#   ./run_ablation_experiments.sh train          # Train 2 ablation models only
#   ./run_ablation_experiments.sh generate       # Generate modes 1/2/3
#   ./run_ablation_experiments.sh all            # Train + Generate
#   TRAIN_ENABLED=false ./run_ablation_experiments.sh all  # Skip training, just generate
#
# Reference format: scripts/compare_Main.sh
##############################

ABLATION_ENABLED=${ABLATION_ENABLED:-true}
TRAIN_ENABLED=${TRAIN_ENABLED:-true}
GENERATE_ENABLED=${GENERATE_ENABLED:-true}
TEST_MODE=${TEST_MODE:-0}
USE_PRETRAINED=${USE_PRETRAINED:-1}

ROOT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=${PYTHON:-/root/miniconda3/envs/flow/bin/python}
ABLATION_CKPT_ROOT="$ROOT_PATH/checkpoint/ablation"
FLOW_CKPT_ROOT="$ROOT_PATH/checkpoint/flow"
OUTPUT_ROOT="$ROOT_PATH/output/ablation"
MAIN_DIR="$ROOT_PATH/main"
PROMPT_JSON="$ROOT_PATH/main/data/prompt.json"

MAX_STEPS=${MAX_STEPS:-10000}
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_WORKERS=${NUM_WORKERS:-17}
DEVICES=${DEVICES:-1}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

NUM_IMAGES_PER_CLASS=${NUM_IMAGES_PER_CLASS:-1500}
DDIM_STEPS=${DDIM_STEPS:-50}
SEED=${SEED:-42}

PRETRAINED_CKPT=${PRETRAINED_CKPT:-"$FLOW_CKPT_ROOT/PRETRAINED/merged_pytorch_model.pth"}
RESUME_CKPT=${RESUME_CKPT:-""}

CLASS_LIST="akiec,bcc,bkl,df,mel,nv,vasc"

export HF_HOME="$ROOT_PATH/model/hf_home"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1

if [ ! -x "$PYTHON" ]; then
  PYTHON=python
fi

if [ "$TEST_MODE" = "1" ]; then
  MAX_STEPS=5
  NUM_IMAGES_PER_CLASS=1
  DDIM_STEPS=5
  BATCH_SIZE=1
fi

mkdir -p "$ABLATION_CKPT_ROOT" "$OUTPUT_ROOT"

show_help() {
  cat << 'EOF'
DFTA Ablation Study Runner (2 Models + 3 Generation Modes)

Usage: ./run_ablation_experiments.sh <command> [options]

Commands:
  train           Train 2 ablation models (Model A & B)
  generate        Run inference for modes 1/2/3
  generate-single Run inference for a specific mode (1/2/3)
  all             Train and then generate (default if no command given)
  list            List available models and generation modes
  help            Show this help message

Options (can also be set as environment variables):
  TRAIN_ENABLED=true|false      Enable/disable training (default: true)
  GENERATE_ENABLED=true|false   Enable/disable generation (default: true)
  TEST_MODE=0|1                 Test mode with reduced steps/samples
  USE_PRETRAINED=0|1            Load pretrained model before training (default: 1)
  MAX_STEPS=N                   Training steps per model (default: 10000)
  BATCH_SIZE=N                  Batch size (default: 1)
  DEVICES=N                     Number of GPUs (default: 1)
  CUDA_VISIBLE_DEVICES=X        GPU device IDs (default: 0)
  NUM_IMAGES_PER_CLASS=N        Images per class for generation (default: 1500)
  DDIM_STEPS=N                  Inference steps (default: 50)
  PRETRAINED_CKPT=path          Path to pretrained checkpoint
  RESUME_CKPT=path              Path to resume from specific checkpoint

Options for 'generate-single' command:
  MODE=<1|2|3>                  Generation mode (required, or use positional arg)
  CLASS_LIST=classes            Comma-separated class list (e.g., "mel,nv")
                                Default: all classes (akiec,bcc,bkl,df,mel,nv,vasc)

Examples:
  # Train 2 ablation models (with pretrained model loading by default)
  ./run_ablation_experiments.sh train

  # Generate images for modes 1/2/3 (including Mode 3 using existing model)
  ./run_ablation_experiments.sh generate

  # Full pipeline: train 2 models + generate modes 1/2/3
  ./run_ablation_experiments.sh all

  # Quick test run
  TEST_MODE=1 ./run_ablation_experiments.sh all

  # Only generate (skip training, assuming checkpoints exist)
  TRAIN_ENABLED=false ./run_ablation_experiments.sh generate

  # Use specific GPU and more samples
  CUDA_VISIBLE_DEVICES=1 NUM_IMAGES_PER_CLASS=500 ./run_ablation_experiments.sh generate

  # ===== Single Mode Generation Examples =====
  
  # Generate Mode 1 only (all classes)
  ./run_ablation_experiments.sh generate-single 1
  
  # Generate Mode 3 for melanoma class only
  MODE=3 CLASS_LIST=mel ./run_ablation_experiments.sh generate-single
  
  # Generate Mode 2 for specific classes
  MODE=2 CLASS_LIST="mel,nv,bcc" ./run_ablation_experiments.sh generate-single
  
  # Quick test single mode with 1 image per class
  MODE=1 TEST_MODE=1 ./run_ablation_experiments.sh generate-single
  
  # Generate multiple classes in Mode 2
  MODE=2 CLASS_LIST="akiec,mel" NUM_IMAGES_PER_CLASS=100 ./run_ablation_experiments.sh generate-single

  # Train from scratch without pretrained model
  USE_PRETRAINED=0 ./run_ablation_experiments.sh train

  # Use custom pretrained checkpoint
  PRETRAINED_CKPT=/path/to/custom_model.pth ./run_ablation_experiments.sh train

Models to Train (2):
  Model A (01_single_flow):     Single-Flow, No Image-Flow, No OSEA
  Model B (02_dual_flow_no_aug): Dual-Flow + Trajectory Alignment, No OSEA

Generation Modes (3):
  Mode 1:  Model A + CSFS Stochastic Sampling (noise_scale=0.1)
  Mode 2:  Model B + CSFS Stochastic Sampling (noise_scale=0.1)
  Mode 3:  Pretrained Full Model (from checkpoint/flow) + Deterministic

Note: Mode 3 does NOT require training - it uses your existing full DFTA model!
EOF
}

list_modes() {
  cat << 'EOF'
Available Ablation Configurations
================================

TRAINING MODELS (2 to train):
-----------------------------
Model A: 01_single_flow
  - Architecture: Single Mask-Flow branch only
  - Components:   No Image-Flow fusion weight (α_img), No OSEA augmentation
  - Output:       checkpoint/ablation/01_single_flow/
  - Supports:     CSFS stochastic sampling in inference (Mode 1)

Model B: 02_dual_flow_no_aug
  - Architecture: Complete dual-flow (Mask-Flow + Image-Flow) with fusion weights
  - Components:   With trajectory alignment loss, No OSEA augmentation
  - Output:       checkpoint/ablation/02_dual_flow_no_aug/
  - Supports:     CSFS stochastic sampling in inference (Mode 2)

PRETRAINED MODEL (existing, no training needed):
-----------------------------------------------
Full DFTA Model: checkpoint/flow/
  - Architecture: Complete dual-flow (Mask-Flow + Image-Flow)
  - Components:   With trajectory alignment loss, With OSEA augmentation
  - Source:       Your previously trained model (already exists!)
  - Used by:      Mode 3 only

GENERATION MODES (3 total):
---------------------------
Mode 1: single_flow_csfs
  Model:    A (Single-Flow - newly trained)
  Method:   CSFS Stochastic Sampling (noise_scale=0.1)
  Tests:    Impact of stochastic sampling on single-stream model

Mode 2: dual_flow_no_aug_csfs
  Model:    B (Dual-Flow No Aug - newly trained)
  Method:   CSFS Stochastic Sampling (noise_scale=0.1)
  Tests:    Combined effect of dual-flow + stochastic sampling (without OSEA)

Mode 3: dual_flow_full_det
  Model:    Full DFTA (from checkpoint/flow - already trained!)
  Method:   Deterministic Euler Integration
  Tests:    Complete DFTA model performance (dual-flow + OSEA + deterministic)

Output Directory Structure:
  output/ablation/
  ├── single_flow_csfs/          (Mode 1 - Model A)
  ├── dual_flow_no_aug_csfs/     (Mode 2 - Model B)
  └── dual_flow_full_det/        (Mode 3 - Existing Full Model)

Training Efficiency:
  - Original approach: 5 models × 10000 steps = 50000 steps
  - This approach:      2 models × 10000 steps = 20000 steps (60% reduction!)
  - Mode 3 reuses your existing trained model
  - All models load from pretrained checkpoint by default (USE_PRETRAINED=1)

Pretrained Model Loading:
  - Default: checkpoint/flow/PRETRAINED/merged_pytorch_model.pth
  - Both Model A and Model B will start training from this pretrained model
  - Set USE_PRETRAINED=0 to train completely from scratch
  - Use PRETRAINED_CKPT=/path/to/model.pth to specify custom path
EOF
}

train_models() {
  echo "=========================================="
  echo "Stage: Training 2 Ablation Models"
  echo "=========================================="

  if [ "$TRAIN_ENABLED" != true ]; then
    echo "[Skip] Training disabled (TRAIN_ENABLED=false)"
    return 0
  fi

  local configs=(
    "01_single_flow:Ablation/config_single_flow.yaml"
    "02_dual_flow_no_aug:Ablation/config_dual_flow_no_aug.yaml"
  )

  for config_entry in "${configs[@]}"; do
    IFS=':' read -r model_name config_file <<< "$config_entry"
    local output_dir="$ABLATION_CKPT_ROOT/$model_name"

    echo ""
    echo "------------------------------------------"
    echo "Training: $model_name"
    echo "Config: $config_file"
    echo "Output: $output_dir"
    echo "Steps: $MAX_STEPS"

    if [ "$USE_PRETRAINED" = "1" ] && [ -f "$PRETRAINED_CKPT" ]; then
      echo "Pretrained: $PRETRAINED_CKPT (will be loaded)"
    else
      echo "Pretrained: Not found or disabled - training from scratch"
    fi

    if [ -n "$RESUME_CKPT" ] && [ -f "$RESUME_CKPT" ]; then
      echo "Resume from: $RESUME_CKPT"
    fi
    echo "------------------------------------------"

    PYTHONPATH="${MAIN_DIR}:${PYTHONPATH:-}" TRAIN_ARGS=(
      "$PYTHON" "$MAIN_DIR/train.py" \
        --config "$MAIN_DIR/$config_file" \
        --output-dir "$output_dir" \
        --batch-size "$BATCH_SIZE" \
        --num-workers "$NUM_WORKERS" \
        --max-steps "$MAX_STEPS" \
        --devices "$DEVICES" \
        --seed "$SEED" \
        --cuda-visible-devices "$CUDA_VISIBLE_DEVICES"
    )

    if [ "$USE_PRETRAINED" = "1" ] && [ -f "$PRETRAINED_CKPT" ]; then
      TRAIN_ARGS+=(--resume "$PRETRAINED_CKPT")
    fi

    if [ -n "$RESUME_CKPT" ] && [ -f "$RESUME_CKPT" ]; then
      TRAIN_ARGS+=(--resume-ckpt "$RESUME_CKPT")
    fi

    "${TRAIN_ARGS[@]}"

    echo "✓ Training completed: $model_name"
  done

  echo ""
  echo "✓ All ablation models trained successfully!"
  echo ""
  echo "Note: Mode 3 will use the existing full DFTA model from:"
  echo "  $FLOW_CKPT_ROOT"
}

generate_images() {
  echo ""
  echo "=========================================="
  echo "Stage: Generating Images (Modes 1/2/3)"
  echo "=========================================="

  if [ "$GENERATE_ENABLED" != true ]; then
    echo "[Skip] Generation disabled (GENERATE_ENABLED=false)"
    return 0
  fi

  echo ""
  echo "Generation Configuration:"
  echo "  - Mode 1/2: Using ablation checkpoints from $ABLATION_CKPT_ROOT"
  echo "  - Mode 3:   Using pretrained model from $FLOW_CKPT_ROOT"
  echo "  - Classes:  $CLASS_LIST"
  echo "  - Samples/class: $NUM_IMAGES_PER_CLASS"
  echo "  - DDIM steps: $DDIM_STEPS"
  echo ""

  local modes=(1 2 3)
  for mode in "${modes[@]}"; do
    MODE="$mode" generate_single_mode
  done

  echo ""
  echo "✓ All generation modes completed!"
  echo "Results saved to: $OUTPUT_ROOT"
}

generate_single_mode() {
  local target_mode="${MODE:-${1:-}}"
  local target_class_list="${CLASS_LIST:-akiec,bcc,bkl,df,mel,nv,vasc}"

  if [ -z "$target_mode" ]; then
    echo "Error: Mode is required for 'generate-single' command"
    echo ""
    echo "Usage:"
    echo "  ./run_ablation_experiments.sh generate-single <mode_id>"
  echo "  MODE=<1|2|3> ./run_ablation_experiments.sh generate-single"
    echo ""
    echo "Available modes:"
  echo "  1 - single_flow_csfs     (Single-Flow + CSFS Stochastic)"
  echo "  2 - dual_flow_no_aug_csfs(Dual-Flow No Aug + CSFS Stochastic)"
  echo "  3 - dual_flow_full_det   (Full DFTA Model + Deterministic)"
    exit 1
  fi

  if ! [[ "$target_mode" =~ ^(1|2|3)$ ]]; then
    echo "Error: Invalid mode '$target_mode'. Must be 1, 2, or 3"
    exit 1
  fi

  local actual_mode=""
  case "$target_mode" in
    1) actual_mode=1;;
    2) actual_mode=2;;
    3) actual_mode=3;;
  esac

  local mode_names=(
    [1]="single_flow_csfs"
    [2]="dual_flow_no_aug_csfs"
    [3]="dual_flow_full_det"
  )

  local mode_descriptions=(
    [1]="Single-Flow + CSFS Stochastic Sampling (noise_scale=0.1)"
    [2]="Dual-Flow (No Aug) + CSFS Stochastic Sampling (noise_scale=0.1)"
    [3]="Dual-Flow (Full/Pretrained) + Deterministic Sampling"
  )

  echo ""
  echo "=========================================="
  echo "Single Mode Generation"
  echo "=========================================="
  echo "Mode ${target_mode}: ${mode_descriptions[$target_mode]}"
  echo "Classes: ${target_class_list}"
  echo "Images/class: ${NUM_IMAGES_PER_CLASS}"
  echo "DDIM steps: ${DDIM_STEPS}"

  if [ "$target_mode" = "3" ]; then
    echo "Model Source: checkpoint/flow (Pretrained Full DFTA) ⭐"
  else
    local model_name=""
    case "$target_mode" in
      1) model_name="01_single_flow";;
      2) model_name="02_dual_flow_no_aug";;
    esac
    echo "Model Source: checkpoint/ablation/${model_name}"
  fi

  echo "Output: ${OUTPUT_ROOT}/${mode_names[$target_mode]}/"
  echo "=========================================="

  PYTHONPATH="${MAIN_DIR}:${PYTHONPATH:-}" "$PYTHON" "$MAIN_DIR/Ablation/inference_ablation.py" \
    --mode "$actual_mode" \
    --ablation-ckpt-root "$ABLATION_CKPT_ROOT" \
    --flow-ckpt-root "$FLOW_CKPT_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --prompt-json "$PROMPT_JSON" \
    --device "cuda" \
    --batch-size "$BATCH_SIZE" \
    --num-workers 0 \
    --max-samples "$((NUM_IMAGES_PER_CLASS * 7))" \
    --ddim-steps "$DDIM_STEPS" \
    --seed "$SEED" \
    --image-size 512 \
    --class-list "$target_class_list" \
    --num-per-class "$NUM_IMAGES_PER_CLASS"

  echo ""
  echo "✓ Mode ${target_mode} (${mode_names[$target_mode]}) completed!"
  echo "Results saved to: ${OUTPUT_ROOT}/${mode_names[$target_mode]}/"
  echo "Generated classes: ${target_class_list}"
}

main() {
  local command="${1:-all}"
  shift || true

  case "$command" in
    train)
      train_models
      ;;
    generate)
      generate_images
      ;;
    generate-single|gen-single|gen)
      generate_single_mode "$@"
      ;;
    all)
      if [ "$ABLATION_ENABLED" = true ]; then
        train_models
        generate_images
      else
        echo "[Skip] Ablation study disabled (ABLATION_ENABLED=false)"
      fi
      ;;
    list)
      list_modes
      ;;
    help|--help|-h)
      show_help
      ;;
    *)
      echo "Error: Unknown command '$command'"
      echo "Use './run_ablation_experiments.sh help' for usage information"
      exit 1
      ;;
  esac
}

main "$@"