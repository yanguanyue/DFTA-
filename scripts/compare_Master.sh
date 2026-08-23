#!/usr/bin/env bash
set -euo pipefail

##############################
# Flow-Matching (main)
COMPARE_MAIN=true
TRAIN_ENABLED=${TRAIN_ENABLED:-true}
RUN_ENABLED=${RUN_ENABLED:-true}
TEST_MODE=${TEST_MODE:-0}
USE_PRETRAINED=${USE_PRETRAINED:-1}
##############################

ROOT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MASTER_PATH="$ROOT_PATH/master"
PYTHON=${PYTHON:-/root/miniconda3/envs/flow/bin/python}
CHECKPOINT_ROOT="$ROOT_PATH/checkpoint/flow"
OUTPUT_ROOT="$ROOT_PATH/output/generate/flow"
PROMPT_JSON="$MASTER_PATH/data/prompt.json"

MAX_STEPS=${MAX_STEPS:-10000}
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_IMAGES_PER_CLASS=${NUM_IMAGES_PER_CLASS:-1500}
DDIM_STEPS=${DDIM_STEPS:-50}
DEVICES=${DEVICES:-1}
NUM_WORKERS=${NUM_WORKERS:-4}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

PRETRAINED_CKPT=${PRETRAINED_CKPT:-"$CHECKPOINT_ROOT/PRETRAINED/merged_pytorch_model.pth"}
RESUME_CKPT=${RESUME_CKPT:-""}

CLASS_LIST="akiec,bcc,bkl,df,mel,nv,vasc"

export HF_HOME="$ROOT_PATH/model/hf_home"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-0}
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-0}
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

mkdir -p "$CHECKPOINT_ROOT" "$OUTPUT_ROOT"

if [ "$COMPARE_MAIN" = true ]; then
  echo "Comparing flow-matching (main)..."

  if [ "$TRAIN_ENABLED" = true ]; then
    echo "[Stage] Training flow-matching (steps=${MAX_STEPS})..."
    TRAIN_ARGS=(
      "$MASTER_PATH/train.py"
      --output-dir "$CHECKPOINT_ROOT"
      --batch-size "$BATCH_SIZE"
      --num-workers "$NUM_WORKERS"
      --max-steps "$MAX_STEPS"
      --devices "$DEVICES"
      --cuda-visible-devices "$CUDA_VISIBLE_DEVICES"
    )

    if [ "$USE_PRETRAINED" = "1" ]; then
      if [ ! -f "$PRETRAINED_CKPT" ]; then
        echo "[x] Pretrained checkpoint not found: $PRETRAINED_CKPT"
        exit 1
      fi
      TRAIN_ARGS+=(--resume "$PRETRAINED_CKPT")
    fi

    if [ -n "$RESUME_CKPT" ] && [ -f "$RESUME_CKPT" ]; then
      TRAIN_ARGS+=(--resume-ckpt "$RESUME_CKPT")
    fi

    "$PYTHON" "${TRAIN_ARGS[@]}"
  fi

  if [ "$RUN_ENABLED" = true ]; then
    echo "[Stage] Resolving latest checkpoint..."
    LATEST_CKPT=$(ls -t "$CHECKPOINT_ROOT"/lightning_logs/version_*/checkpoints/last.ckpt 2>/dev/null | head -n 1 || true)
    if [ -z "$LATEST_CKPT" ]; then
      echo "[x] No checkpoint found in $CHECKPOINT_ROOT"
      exit 1
    fi

  echo "[Stage] Generating images with inference.py (per class=${NUM_IMAGES_PER_CLASS})..."
  "$PYTHON" "$MASTER_PATH/inference.py" \
    --ckpt "$LATEST_CKPT" \
    --prompt-json "$PROMPT_JSON" \
    --output-root "$OUTPUT_ROOT" \
    --class-list "$CLASS_LIST" \
    --num-per-class "$NUM_IMAGES_PER_CLASS" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --ddim-steps "$DDIM_STEPS" \
    --device "cuda" \
    --image-size 512 \
    --stochastic \
    --noise-scale 0.1
  fi
fi
