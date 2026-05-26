#!/usr/bin/env bash
set -euo pipefail

##############################
# DreamBooth
# https://github.com/huggingface/diffusers
COMPARE_DREAMBOOTH=true
TRAIN_ENABLED=${TRAIN_ENABLED:-true}
RUN_ENABLED=${RUN_ENABLED:-true}
TEST_MODE=${TEST_MODE:-0}
##############################

ROOT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RUN_ROOT="$ROOT_PATH/compare_main/DreamBooth"
PYTHON=${PYTHON:-/root/miniconda3/envs/flow/bin/python}
MODEL_DIR="$ROOT_PATH/model/AI-ModelScope/stable-diffusion-v1-5"
CHECKPOINT_ROOT="$ROOT_PATH/checkpoint/compare_models/DreamBooth"
OUTPUT_DIR="$ROOT_PATH/output/generate/DreamBooth"
DATA_ROOT="$ROOT_PATH/data/HAM10000/input/train/HAM10000_img_class"
MIXED_DIR="$ROOT_PATH/data/HAM10000/input/train/ham10000_img_mix"
OUTPUT_NAME=${OUTPUT_NAME:-"ham10000-mix-model"}
SINGLE_PROMPT=${SINGLE_PROMPT:-"skin lesion"}
GPU_ID=${GPU_ID:-0}

MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-15000}
NUM_IMAGES_PER_CLASS=${NUM_IMAGES_PER_CLASS:-1500}

if [ ! -x "$PYTHON" ]; then
  PYTHON=python
fi

if [ "$TEST_MODE" = "1" ]; then
  MAX_TRAIN_STEPS=2
  NUM_IMAGES_PER_CLASS=1
fi

mkdir -p "$CHECKPOINT_ROOT" "$OUTPUT_DIR"

build_mixed_dir() {
  if [ -d "$MIXED_DIR" ] && [ "$(ls -A "$MIXED_DIR" 2>/dev/null)" ]; then
    echo "$MIXED_DIR"
    return
  fi
  echo "[Stage] Building mixed training directory at $MIXED_DIR..."
  mkdir -p "$MIXED_DIR"
  find "$DATA_ROOT" -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) -print0 | \
    while IFS= read -r -d '' img; do
      base=$(basename "$img")
      ln -sf "$img" "$MIXED_DIR/$base"
    done
  echo "$MIXED_DIR"
}

ensure_modelscope() {
  "$PYTHON" - <<'PY'
try:
    import modelscope  # noqa: F401
    print("[√] modelscope available")
except Exception:
    raise SystemExit(1)
PY
}

install_modelscope() {
  echo "[Stage] Installing modelscope into current Python environment..."
  "$PYTHON" -m pip install --quiet modelscope
}

resolve_model_dir() {
  if [ -d "$MODEL_DIR" ]; then
    echo "$MODEL_DIR"
    return
  fi
  echo ""
}

BASE_MODEL_DIR=$(resolve_model_dir)
if [ -z "$BASE_MODEL_DIR" ]; then
  echo "[ModelScope] stable-diffusion-v1-5 not found at $MODEL_DIR, downloading..."
  if ! ensure_modelscope; then
    install_modelscope
  fi
  export MODEL_DIR
  "$PYTHON" - <<'PY'
from modelscope import snapshot_download
import os

model_dir = os.environ.get("MODEL_DIR")
repo_candidates = [
    "AI-ModelScope/stable-diffusion-v1-5",
    "modelscope/stable-diffusion-v1-5",
    "stable-diffusion-v1-5",
]

last_error = None
for repo_id in repo_candidates:
  try:
    snapshot_download(repo_id, local_dir=model_dir)
    print(f"✅ Base model cached via ModelScope: {repo_id}")
    last_error = None
    break
  except Exception as exc:
    last_error = exc

if last_error is not None:
    raise last_error
PY
  BASE_MODEL_DIR=$(resolve_model_dir)
fi

if [ -z "$BASE_MODEL_DIR" ]; then
  echo "[x] Base model not found at $MODEL_DIR"
  exit 1
fi

echo "[√] Base model exists: $BASE_MODEL_DIR"

declare -A PROMPTS
PROMPTS[akiec]="actinic keratosis and intraepithelial carcinoma"
PROMPTS[bcc]="basal cell carcinoma"
PROMPTS[bkl]="benign keratosis-like lesion"
PROMPTS[df]="dermatofibroma"
PROMPTS[mel]="melanoma"
PROMPTS[nv]="melanocytic nevus"
PROMPTS[vasc]="vascular lesion"

CLASSES=(akiec bcc bkl df mel nv vasc)

if [ "$COMPARE_DREAMBOOTH" = true ]; then
  echo "Comparing DreamBooth..."

  if [ "$TRAIN_ENABLED" = true ]; then
    echo "[Stage] Training DreamBooth (steps=${MAX_TRAIN_STEPS})..."
    MIXED_INSTANCE_DIR=$(build_mixed_dir)
    MODEL_DIR="$BASE_MODEL_DIR" \
    PYTHON="$PYTHON" \
    ROOT_PATH="$ROOT_PATH" \
    REPO_ROOT="$RUN_ROOT" \
    DATA_ROOT="$DATA_ROOT" \
    OUTPUT_ROOT="$CHECKPOINT_ROOT" \
    MAX_TRAIN_STEPS="$MAX_TRAIN_STEPS" \
    SINGLE_MODEL=1 \
    MIXED_INSTANCE_DIR="$MIXED_INSTANCE_DIR" \
    OUTPUT_NAME="$OUTPUT_NAME" \
    SINGLE_PROMPT="$SINGLE_PROMPT" \
      bash "$RUN_ROOT/train_all_dreambooth.sh"
  fi

  if [ "$RUN_ENABLED" = true ]; then
    echo "[Stage] Generating images..."
    MODEL_PATH="$CHECKPOINT_ROOT/$OUTPUT_NAME"
    if [ ! -d "$MODEL_PATH" ]; then
      echo "[x] Model not found at $MODEL_PATH"
      exit 1
    fi

    PROMPT_MAP="akiec=${PROMPTS[akiec]},bcc=${PROMPTS[bcc]},bkl=${PROMPTS[bkl]},df=${PROMPTS[df]},mel=${PROMPTS[mel]},nv=${PROMPTS[nv]},vasc=${PROMPTS[vasc]}"

    "$PYTHON" "$RUN_ROOT/src/generate_images.py" \
      --pretrained_model_name_or_path "$MODEL_PATH" \
      --prompt_map "$PROMPT_MAP" \
      --num_images "$NUM_IMAGES_PER_CLASS" \
      --gpu_id "$GPU_ID" \
      --output_folder "$OUTPUT_DIR"
  fi
fi
