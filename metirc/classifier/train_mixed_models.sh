#!/usr/bin/env bash
set -euo pipefail

PYTHON=/root/autodl-tmp/environment/skin/bin/python
REPO=/root/autodl-tmp/pytorch-classification-extended-master
REAL_TRAIN=/root/autodl-tmp/data/HAM10000/input/train/HAM10000_img_class
REAL_VAL=/root/autodl-tmp/data/HAM10000/input/val/HAM10000_img_class
OUTPUT_ROOT=/root/autodl-tmp/output/mixed_datasets/classifier
CHECKPOINT_ROOT=/root/autodl-tmp/output/classifier/checkpoints/ham10000_mix
LOG_ROOT=/root/autodl-tmp/output/classifier/logs

MODELS=(baseline Controlnet T2I-Adapter Derm-T2IM DreamBooth LF-VAR LesionGen Siamese DFMGAN flow skin-disease-diffusion ArSDM)
ARCHS=(resnet18 resnet50 efficientnet_b0 vgg16 vit_b_16)

mkdir -p "$OUTPUT_ROOT" "$CHECKPOINT_ROOT" "$LOG_ROOT"

# Baseline (real only): use a symlinked dataset structure
if [ ! -d "$OUTPUT_ROOT/baseline" ]; then
  mkdir -p "$OUTPUT_ROOT/baseline"
  ln -sfn "$REAL_TRAIN" "$OUTPUT_ROOT/baseline/train"
  ln -sfn "$REAL_VAL" "$OUTPUT_ROOT/baseline/val"
fi

for MODEL in "${MODELS[@]}"; do
  DATASET_DIR="$OUTPUT_ROOT/$MODEL"

  if [ "$MODEL" != "baseline" ]; then
  "$PYTHON" /root/autodl-tmp/metirc/classifier/mix_synthetic.py \
      --real-train "$REAL_TRAIN" \
      --real-val "$REAL_VAL" \
  --synthetic-root "/root/autodl-tmp/output/generate/$MODEL" \
      --output-root "$DATASET_DIR" \
      --majority-cap 500
  fi

  for ARCH in "${ARCHS[@]}"; do
    CKPT_DIR="$CHECKPOINT_ROOT/$MODEL/$ARCH"
    LOG_FILE="$LOG_ROOT/${MODEL}_${ARCH}.log"
    mkdir -p "$CKPT_DIR"

    if [ -f "$CKPT_DIR/model_best.pth.tar" ]; then
      echo "[SKIP] $MODEL / $ARCH already has model_best.pth.tar"
      continue
    fi

    EXTRA_OPTS=()
    if [ "$ARCH" = "vit_b_16" ]; then
      EXTRA_OPTS+=(--train-batch 16 --test-batch 16)
    else
      EXTRA_OPTS+=(--train-batch 64 --test-batch 64)
    fi

    PRETRAINED=(--pretrained)

    cd "$REPO"
    nohup "$PYTHON" customdata.py -a "$ARCH" -d "$DATASET_DIR" \
      "${PRETRAINED[@]}" \
      --epochs 30 --schedule 15 25 --gamma 0.1 --lr 0.001 --gpu-id 0 -c "$CKPT_DIR" \
      "${EXTRA_OPTS[@]}" \
      > "$LOG_FILE" 2>&1
  done

done
