#!/usr/bin/env bash
set -euo pipefail

##############################
# Siamese-Diffusion
COMPARE_SIAMESE=true
TRAIN_ENABLED=${TRAIN_ENABLED:-true}
RUN_ENABLED=${RUN_ENABLED:-true}
TEST_MODE=${TEST_MODE:-0}
##############################

ROOT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RUN_ROOT="$ROOT_PATH/compare_main/Siamese-Diffusion-main"
PYTHON=${PYTHON:-/root/miniconda3/envs/flow/bin/python}
CHECKPOINT_ROOT="$ROOT_PATH/checkpoint/compare_models/Siamese"
OUTPUT_DIR="$ROOT_PATH/output/generate/Siamese"
PRETRAINED_CKPT="$CHECKPOINT_ROOT/PRETRAINED/merged_pytorch_model.pth"
DEFAULT_PROMPT_JSON="$RUN_ROOT/data/prompt.json"
FALLBACK_PROMPT_JSON="$ROOT_PATH/main/Siamese-Diffusion-main/data/prompt.json"
PROMPT_JSON=${PROMPT_JSON:-$DEFAULT_PROMPT_JSON}

MAX_STEPS=${MAX_STEPS:-3000}
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_IMAGES_PER_CLASS=${NUM_IMAGES_PER_CLASS:-1500}
DDIM_STEPS=${DDIM_STEPS:-50}
GPU_IDS=${GPU_IDS:-0}
DEVICES=${DEVICES:-1}
NUM_WORKERS=${NUM_WORKERS:-4}
IMAGE_SIZE=${IMAGE_SIZE:-512}

export HF_HOME="$ROOT_PATH/model/hf_home"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [ ! -x "$PYTHON" ]; then
  PYTHON=python
fi

if [ "$TEST_MODE" = "1" ]; then
  MAX_STEPS=5
  NUM_IMAGES_PER_CLASS=1
  DDIM_STEPS=5
  BATCH_SIZE=1
fi

if [ ! -f "$PROMPT_JSON" ]; then
  if [ -f "$FALLBACK_PROMPT_JSON" ]; then
    PROMPT_JSON="$FALLBACK_PROMPT_JSON"
  fi
fi

mkdir -p "$CHECKPOINT_ROOT" "$OUTPUT_DIR"

if [ "$COMPARE_SIAMESE" = true ]; then
  echo "Comparing Siamese-Diffusion..."

  if [ "$TRAIN_ENABLED" = true ]; then
    echo "[Stage] Training Siamese-Diffusion (steps=${MAX_STEPS})..."
    if [ ! -f "$PRETRAINED_CKPT" ]; then
      echo "[x] Pretrained checkpoint not found: $PRETRAINED_CKPT"
      exit 1
    fi

    "$PYTHON" "$RUN_ROOT/tutorial_train.py" \
      --pretrained-ckpt "$PRETRAINED_CKPT" \
      --prompt-json "$PROMPT_JSON" \
      --output-dir "$CHECKPOINT_ROOT" \
      --batch-size "$BATCH_SIZE" \
      --num-workers "$NUM_WORKERS" \
      --max-steps "$MAX_STEPS" \
      --devices "$DEVICES" \
      --accelerator gpu \
      --gpu-ids "$GPU_IDS" \
      --image-size "$IMAGE_SIZE" \
      --save-merged-path "$CHECKPOINT_ROOT/merged_pytorch_model.pth"
  fi

  if [ "$RUN_ENABLED" = true ]; then
    echo "[Stage] Generating images (per class=${NUM_IMAGES_PER_CLASS})..."
    "$PYTHON" "$RUN_ROOT/tutorial_inference.py" \
      --ckpt-path "$CHECKPOINT_ROOT/merged_pytorch_model.pth" \
      --prompt-json "$PROMPT_JSON" \
      --output-dir "$OUTPUT_DIR" \
      --batch-size "$BATCH_SIZE" \
      --num-workers "$NUM_WORKERS" \
      --num-per-class "$NUM_IMAGES_PER_CLASS" \
      --ddim-steps "$DDIM_STEPS" \
      --gpu-ids "$GPU_IDS" \
      --image-size "$IMAGE_SIZE"
  fi
fi
