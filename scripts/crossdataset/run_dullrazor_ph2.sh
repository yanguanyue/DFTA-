#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
INPUT_ROOT="${ROOT_DIR}/data/PH2/input"
IMG_CLASS_DIR="${INPUT_ROOT}/img_class"
IMG_DIR="${INPUT_ROOT}/PH2_img"

python "${ROOT_DIR}/data/util/batch_dullrazor.py" \
  --input-dir "${IMG_CLASS_DIR}" \
  --output-dir "${IMG_CLASS_DIR}" \
  --skip-keywords "seg,mask,segmentation,seg_class"

python "${ROOT_DIR}/data/util/batch_dullrazor.py" \
  --input-dir "${IMG_DIR}" \
  --output-dir "${IMG_DIR}" \
  --skip-keywords "seg,mask,segmentation,seg_class"
