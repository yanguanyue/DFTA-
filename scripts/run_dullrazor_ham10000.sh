#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/root/autodl-tmp"
INPUT_DIR="${ROOT_DIR}/data/HAM10000/input"

python "${ROOT_DIR}/data/util/batch_dullrazor.py" \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${INPUT_DIR}" \
  --skip-keywords "seg,mask,segmentation,huggingface,train_val"
