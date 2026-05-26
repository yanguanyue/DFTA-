#!/usr/bin/env bash
set -euo pipefail

##############################
# ArSDM
COMPARE_ARSDM=true
TRAIN_ENABLED=${TRAIN_ENABLED:-true}
RUN_ENABLED=${RUN_ENABLED:-true}
TEST_MODE=${TEST_MODE:-0}
##############################

ROOT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RUN_ROOT="$ROOT_PATH/compare_main/ArSDM-main"
PYTHON=${PYTHON:-/root/miniconda3/envs/flow/bin/python}
CHECKPOINT_ROOT="$ROOT_PATH/checkpoint/compare_models/ArSDM"
OUTPUT_DIR="$ROOT_PATH/output/generate2/ArSDM"
export OMP_NUM_THREADS=1

BASE_MAX_STEPS=${BASE_MAX_STEPS:-5000}
STAGE2_MAX_STEPS=${STAGE2_MAX_STEPS:-3000}
NUM_IMAGES_PER_CLASS=${NUM_IMAGES_PER_CLASS:-1500}
BATCH_SIZE=${BATCH_SIZE:-8}
DDIM_STEPS=${DDIM_STEPS:-20}
PRETRAINED_BASE_CKPT=${PRETRAINED_BASE_CKPT:-$CHECKPOINT_ROOT/PRETRAINED_BASE_CKPT/ArSDM_ada_refine.ckpt}

STAGE2_CONFIGS=(
  "$RUN_ROOT/configs/HAM10000_akiec.yaml"
  "$RUN_ROOT/configs/HAM10000_bcc.yaml"
  "$RUN_ROOT/configs/HAM10000_bkl_lora_512.yaml"
  "$RUN_ROOT/configs/HAM10000_df_lora_512.yaml"
  "$RUN_ROOT/configs/HAM10000_mel_lora_512.yaml"
  "$RUN_ROOT/configs/HAM10000_nv_lora_512.yaml"
  "$RUN_ROOT/configs/HAM10000_vasc_lora_512.yaml"
)

if [ ! -x "$PYTHON" ]; then
  PYTHON=python
fi

if [ "$TEST_MODE" = "1" ]; then
  BASE_MAX_STEPS=5
  STAGE2_MAX_STEPS=5
  NUM_IMAGES_PER_CLASS=1
  BATCH_SIZE=1
  DDIM_STEPS=5
fi

mkdir -p "$CHECKPOINT_ROOT" "$OUTPUT_DIR"

if [ "$COMPARE_ARSDM" = true ]; then
  echo "Comparing ArSDM..."

  if [ "$TRAIN_ENABLED" = true ]; then
    echo "[Stage] Training base + stage2 (base_steps=${BASE_MAX_STEPS}, stage2_steps=${STAGE2_MAX_STEPS})..."
    ROOT_DIR="$RUN_ROOT" \
    PYTHON="$PYTHON" \
    LOG_ROOT="$CHECKPOINT_ROOT" \
    BASE_MAX_STEPS="$BASE_MAX_STEPS" \
    STAGE2_MAX_STEPS="$STAGE2_MAX_STEPS" \
    PRETRAINED_BASE_CKPT="$PRETRAINED_BASE_CKPT" \
    "$RUN_ROOT/run_stage2_from_base.sh"
  fi

  if [ "$RUN_ENABLED" = true ]; then
    echo "[Stage] Generating images (per class=${NUM_IMAGES_PER_CLASS})..."
    "$PYTHON" "$RUN_ROOT/generate_stage2_samples.py" \
      --configs "${STAGE2_CONFIGS[@]}" \
      --output_root "$OUTPUT_DIR" \
      --log_root "$CHECKPOINT_ROOT" \
      --num_images "$NUM_IMAGES_PER_CLASS" \
      --batch_size "$BATCH_SIZE" \
      --ddim_steps "$DDIM_STEPS"
  fi
fi
