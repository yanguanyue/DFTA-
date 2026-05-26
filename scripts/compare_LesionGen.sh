#!/usr/bin/env bash
set -euo pipefail

##############################
# LesionGen
# https://github.com/jfayyad/LesionGen
COMPARE_LesionGen=true
TRAIN_ENABLED=${TRAIN_ENABLED:-true}
RUN_ENABLED=${RUN_ENABLED:-true}
##############################
root_path=$(pwd)
OUTPUT_DIR=$root_path/output/generate/LesionGen
RUN_ROOT_PATH=$root_path/compare_main/LesionGen-main
HF_HOME_PATH=${HF_HOME_PATH:-"$root_path/model/hf_home"}
BASE_MODEL_REPO="CompVis/stable-diffusion-v1-4"
MODELSCOPE_CACHE=${MODELSCOPE_CACHE:-"$root_path/model"}
MODELSCOPE_REPO_ID=${MODELSCOPE_REPO_ID:-"modelscope/stable-diffusion-v1-4"}
CHECKPOINTS_ROOT=${CHECKPOINTS_ROOT:-"$root_path/checkpoint/compare_models/LesionGen/lora_7classes"}
CHECKPOINT_NAME=${CHECKPOINT_NAME:-"checkpoint-5000"}
NUM_IMAGES_PER_CLASS=${NUM_IMAGES_PER_CLASS:-1500}
MODE=${MODE:-dataset}
OUTPUT_LAYOUT=${OUTPUT_LAYOUT:-flat}

mkdir -p "$OUTPUT_DIR"
mkdir -p "$HF_HOME_PATH"

export HF_HOME="$HF_HOME_PATH"
export DIFFUSERS_LOCAL_FILES_ONLY=${DIFFUSERS_LOCAL_FILES_ONLY:-1}
export MODELSCOPE_CACHE

resolve_model_dir() {
    local hub_dir
    hub_dir=$(find "$MODELSCOPE_CACHE/hub" -maxdepth 2 -type d -name "models--*stable-diffusion-v1-4*" 2>/dev/null | head -n 1)
    if [ -n "$hub_dir" ]; then
        echo "$hub_dir"
        return
    fi
    find "$MODELSCOPE_CACHE" -maxdepth 3 -type d -path "*/stable-diffusion-v1-4" 2>/dev/null | head -n 1
}

echo "Checking base model cache..."
BASE_MODEL_DIR=$(resolve_model_dir || true)
if [ -z "$BASE_MODEL_DIR" ]; then
    echo "ModelScope cache not found in $MODELSCOPE_CACHE. Downloading stable-diffusion-v1-4..."
    python - <<'PY'
from modelscope import snapshot_download
import os

preferred = os.getenv("MODELSCOPE_REPO_ID")
fallbacks = [preferred, "AI-ModelScope/stable-diffusion-v1-4", "modelscope/stable-diffusion-v1-4", "stable-diffusion-v1-4"]
cache_dir = os.getenv("MODELSCOPE_CACHE")

last_error = None
for repo_id in [item for item in fallbacks if item]:
    try:
        snapshot_download(repo_id, cache_dir=cache_dir)
        print(f"✅ Base model cached via ModelScope: {repo_id}")
        last_error = None
        break
    except Exception as exc:
        last_error = exc

if last_error is not None:
    raise last_error
PY
    BASE_MODEL_DIR=$(resolve_model_dir || true)
fi

if [ -z "$BASE_MODEL_DIR" ]; then
    echo "[x] ModelScope cache directory not found under $MODELSCOPE_CACHE"
    exit 1
fi

echo "[√] Base model exists: $BASE_MODEL_DIR"

if [ "$COMPARE_LesionGen" = true ]; then
    echo "Comparing LesionGen..."

    if [ "$TRAIN_ENABLED" = true ]; then
        echo "[Stage] Training base LoRA (14k steps)..."
        bash "$RUN_ROOT_PATH/train_lora.sh"

        echo "[Stage] Training 7-class LoRA (resume from base)..."
        bash "$RUN_ROOT_PATH/train_lora_7classes.sh"
    fi

    if [ "$RUN_ENABLED" = true ]; then
        echo "[Stage] Generating images with 7-class LoRA..."
        if [ ! -d "$CHECKPOINTS_ROOT" ]; then
            echo "[x] LoRA checkpoints not found at $CHECKPOINTS_ROOT"
            exit 1
        fi

        python "$RUN_ROOT_PATH/generate_all.py" \
            --mode "$MODE" \
            --output_dir "$OUTPUT_DIR" \
            --num_images_per_class "$NUM_IMAGES_PER_CLASS" \
            --checkpoints_root "$CHECKPOINTS_ROOT" \
            --checkpoint_name "$CHECKPOINT_NAME" \
            --base_model_path "$BASE_MODEL_DIR" \
            --output_layout "$OUTPUT_LAYOUT"
    fi
fi
