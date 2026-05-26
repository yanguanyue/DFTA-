#!/usr/bin/env bash
set -euo pipefail

##############################
# DFMGAN
# https://github.com/Ldhlwh/DFMGAN
COMPARE_DFMGAN=true
TRAIN_ENABLED=${TRAIN_ENABLED:-true}
RUN_ENABLED=${RUN_ENABLED:-true}
TEST_MODE=${TEST_MODE:-0}
##############################

PYTHON=${PYTHON:-/root/miniconda3/envs/flow/bin/python}
ROOT=/root/autodl-tmp
RUN_ROOT=$ROOT/compare_main/DFMGAN-main
DATA_ROOT=$ROOT/data/HAM10000/input
TRAIN_DIR=$DATA_ROOT/train/HAM10000_img_class
MASK_DIR=$DATA_ROOT/train/HAM10000_seg_class
MASK_ZIP_ROOT=$DATA_ROOT/train/dfmgan_mask_zips
CHECKPOINT_ROOT=$ROOT/checkpoint/compare_models/DFMGAN
BASE_OUT=$CHECKPOINT_ROOT/ham10000_base
OUTPUT_DIR=$ROOT/output/generate/DFMGAN

CLASSES=(akiec bcc bkl df mel nv vasc)

BASE_KIMG=${BASE_KIMG:-80}
CLASS_KIMG=${CLASS_KIMG:-20}
SNAP=${SNAP:-10}
NUM_IMAGES_PER_CLASS=${NUM_IMAGES_PER_CLASS:-1500}
MAX_IMAGES=${MAX_IMAGES:-}

if [ ! -x "$PYTHON" ]; then
  PYTHON=python
fi

if [ "$TEST_MODE" = "1" ]; then
  BASE_KIMG=1
  CLASS_KIMG=1
  SNAP=1
  NUM_IMAGES_PER_CLASS=1
  MAX_IMAGES=20
fi

latest_pkl() {
  "$PYTHON" - "$1" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
if not root.exists():
    print("")
    sys.exit(0)

run_dirs = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
for run_dir in run_dirs:
    pkls = sorted(run_dir.glob("network-snapshot-*.pkl"))
    if pkls:
        print(pkls[-1])
        sys.exit(0)
print("")
PY
}

resolve_base_dir() {
  if [ -d "$CHECKPOINT_ROOT/ham10000_base_512" ]; then
    echo "$CHECKPOINT_ROOT/ham10000_base_512"
    return
  fi
  echo "$BASE_OUT"
}

resolve_class_dir() {
  local cls=$1
  local preferred="$CHECKPOINT_ROOT/ham10000_${cls}_mask_512"
  if [ -d "$preferred" ]; then
    echo "$preferred"
    return
  fi
  echo "$CHECKPOINT_ROOT/ham10000_${cls}"
}

mkdir -p "$CHECKPOINT_ROOT" "$OUTPUT_DIR"
mkdir -p "$MASK_ZIP_ROOT"

prepare_mask_zip() {
  local cls=$1
  local img_dir="$TRAIN_DIR/$cls"
  local mask_dir="$MASK_DIR/$cls"
  local zip_path="$MASK_ZIP_ROOT/ham10000_${cls}_mask_512.zip"

  if [ ! -d "$img_dir" ]; then
    echo "[x] Image dir not found: $img_dir"
    exit 1
  fi
  if [ ! -d "$mask_dir" ]; then
    echo "[x] Mask dir not found: $mask_dir"
    exit 1
  fi

  if [ ! -f "$zip_path" ]; then
    echo "[Stage] Building mask dataset for $cls..." >&2
    DATASET_ARGS=(--source "$img_dir" --source-mask "$mask_dir" --dest "$zip_path" --width 512 --height 512 --resize-filter lanczos)
    if [ -n "$MAX_IMAGES" ]; then
      DATASET_ARGS+=(--max-images "$MAX_IMAGES")
    fi
    "$PYTHON" "$RUN_ROOT/dataset_tool.py" "${DATASET_ARGS[@]}" 1>&2
  fi

  echo "$zip_path"
}

if [ "$COMPARE_DFMGAN" = true ]; then
  echo "Comparing DFMGAN..."

  if [ "$TRAIN_ENABLED" = true ]; then
    if [ ! -d "$TRAIN_DIR" ]; then
      echo "[x] Training data not found at $TRAIN_DIR"
      exit 1
    fi

    echo "[Stage] Training base model (${BASE_KIMG} kimg)..."
    "$PYTHON" "$RUN_ROOT/train.py" \
      --outdir "$BASE_OUT" \
      --data "$TRAIN_DIR" \
      --gpus 1 \
      --kimg "$BASE_KIMG" \
      --snap "$SNAP" \
      --metrics none

    BASE_DIR=$(resolve_base_dir)
    BASE_PKL=$(latest_pkl "$BASE_DIR")
    if [ -z "$BASE_PKL" ]; then
      echo "[x] Base checkpoint not found under $BASE_DIR"
      exit 1
    fi

    for cls in "${CLASSES[@]}"; do
      CLASS_DATA=$(prepare_mask_zip "$cls")
  CLASS_OUT=$(resolve_class_dir "$cls")

      echo "[Stage] Training class ${cls} (${CLASS_KIMG} kimg)..."
      "$PYTHON" "$RUN_ROOT/train.py" \
        --outdir "$CLASS_OUT" \
        --data "$CLASS_DATA" \
        --gpus 1 \
        --kimg "$CLASS_KIMG" \
        --snap "$SNAP" \
        --metrics none \
        --resume "$BASE_PKL" \
        --transfer res_block_match_dis
    done
  fi

  if [ "$RUN_ENABLED" = true ]; then
    echo "[Stage] Generating images..."
    for cls in "${CLASSES[@]}"; do
      CLASS_OUT=$(resolve_class_dir "$cls")
      CLASS_PKL=$(latest_pkl "$CLASS_OUT")
      if [ -z "$CLASS_PKL" ]; then
        echo "[x] No checkpoint found for class $cls under $CLASS_OUT"
        exit 1
      fi

      OUT_IMAGE="$OUTPUT_DIR/$cls/image"
      OUT_MASK="$OUTPUT_DIR/$cls/mask"
      rm -rf "$OUT_IMAGE" "$OUT_MASK"
      mkdir -p "$OUT_IMAGE" "$OUT_MASK"
      "$PYTHON" "$RUN_ROOT/generate.py" \
        --network "$CLASS_PKL" \
        --output "$OUT_IMAGE" \
        --gen-mask \
        --num "$NUM_IMAGES_PER_CLASS"
      if ls "$OUT_IMAGE"/*_mask.png >/dev/null 2>&1; then
        mv "$OUT_IMAGE"/*_mask.png "$OUT_MASK"/
      fi
    done
  fi
fi
