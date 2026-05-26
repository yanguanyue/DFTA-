#!/usr/bin/env bash
set -euo pipefail

##############################
# Skin-Disease-Diffusion
COMPARE_SKIN_DIFFUSION=true
TRAIN_ENABLED=${TRAIN_ENABLED:-true}
RUN_ENABLED=${RUN_ENABLED:-true}
TEST_MODE=${TEST_MODE:-0}
##############################

ROOT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RUN_ROOT="$ROOT_PATH/compare_main/skin-disease-diffusion-main"
PYTHON=${PYTHON:-/root/miniconda3/envs/flow/bin/python}
MODEL_DIR="$ROOT_PATH/model/AI-ModelScope"
SD_MODEL_DIR="$MODEL_DIR/stable-diffusion-v1-5"
CHECKPOINT_ROOT="$ROOT_PATH/checkpoint/compare_models/skin-disease-diffusionr"
VAE_DIR="$CHECKPOINT_ROOT/vae"
DIFFUSION_DIR="$CHECKPOINT_ROOT/diffusion"
OUTPUT_DIR="$ROOT_PATH/output/generate/skin-disease-diffusion"
DATA_ROOT="$ROOT_PATH/data/HAM10000/input/train/HAM10000_img_class"
HF_HOME_DIR="$ROOT_PATH/model/hf_home"

MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-15000}
NUM_IMAGES_PER_CLASS=${NUM_IMAGES_PER_CLASS:-1500}
SAMPLE_STEPS=${SAMPLE_STEPS:-750}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-3.0}
BATCH_SIZE=${BATCH_SIZE:-8}
SAMPLE_PRECISION=${SAMPLE_PRECISION:-fp16}
VAE_CKPT="$VAE_DIR/vae_last.ckpt"

if [ ! -x "$PYTHON" ]; then
  PYTHON=python
fi

if [ "$TEST_MODE" = "1" ]; then
  MAX_TRAIN_STEPS=5
  NUM_IMAGES_PER_CLASS=1
  SAMPLE_STEPS=10
  BATCH_SIZE=1
fi

mkdir -p "$CHECKPOINT_ROOT" "$VAE_DIR" "$DIFFUSION_DIR" "$OUTPUT_DIR" "$MODEL_DIR" "$HF_HOME_DIR"

export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="$HF_HOME_DIR"

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

ensure_model() {
  local model_path="$1"
  local repo_candidates="$2"
  if [ -d "$model_path" ]; then
    echo "[√] Model exists: $model_path"
    return
  fi

  echo "[ModelScope] $model_path not found, downloading..."
  if ! ensure_modelscope; then
    install_modelscope
  fi

  MODEL_PATH="$model_path" REPO_CANDIDATES="$repo_candidates" "$PYTHON" - <<'PY'
import os
from modelscope import snapshot_download

model_path = os.environ.get("MODEL_PATH")
repo_candidates = [c for c in os.environ.get("REPO_CANDIDATES", "").split("|") if c]

last_error = None
for repo_id in repo_candidates:
    try:
        snapshot_download(repo_id, local_dir=model_path)
        print(f"✅ Model cached via ModelScope: {repo_id}")
        last_error = None
        break
    except Exception as exc:
        last_error = exc

if last_error is not None:
    raise last_error
PY

  if [ ! -d "$model_path" ]; then
    echo "[x] Model not found at $model_path"
    exit 1
  fi
}

if [ "$COMPARE_SKIN_DIFFUSION" = true ]; then
  echo "Comparing skin-disease diffusion..."

  ensure_model "$SD_MODEL_DIR" "AI-ModelScope/stable-diffusion-v1-5|modelscope/stable-diffusion-v1-5|stable-diffusion-v1-5"

  if [ "$TRAIN_ENABLED" = true ]; then
    echo "[Stage] Training VAE (steps=${MAX_TRAIN_STEPS})..."
    VAE_DEBUG_FLAG=""
    if [ "$TEST_MODE" = "1" ]; then
      VAE_DEBUG_FLAG="--debug"
    fi

    "$PYTHON" "$RUN_ROOT/train_vae.py" \
      --data-path "$DATA_ROOT" \
      --output-dir "$VAE_DIR" \
      --max-steps "$MAX_TRAIN_STEPS" \
      $VAE_DEBUG_FLAG

    if [ ! -f "$VAE_CKPT" ]; then
      echo "[x] VAE checkpoint not found at $VAE_CKPT"
      exit 1
    fi

    echo "[Stage] Training diffusion (steps=${MAX_TRAIN_STEPS})..."
    DIFFUSION_DEBUG_FLAG=""
    if [ "$TEST_MODE" = "1" ]; then
      DIFFUSION_DEBUG_FLAG="--debug"
    fi

    "$PYTHON" "$RUN_ROOT/train_diffusion.py" \
      --data-path "$DATA_ROOT" \
      --output-dir "$DIFFUSION_DIR" \
      --vae-checkpoint "$VAE_CKPT" \
      --max-steps "$MAX_TRAIN_STEPS" \
      $DIFFUSION_DEBUG_FLAG
  fi

  if [ "$RUN_ENABLED" = true ]; then
    echo "[Stage] Sampling images (per class=${NUM_IMAGES_PER_CLASS})..."
    DIFFUSION_CKPT="$DIFFUSION_DIR/diffusion_last.ckpt"
    if [ ! -f "$DIFFUSION_CKPT" ]; then
      echo "[x] Diffusion checkpoint not found at $DIFFUSION_CKPT"
      exit 1
    fi
    if [ ! -f "$VAE_CKPT" ]; then
      echo "[x] VAE checkpoint not found at $VAE_CKPT"
      exit 1
    fi

    SAMPLE_RESUME_FLAG=""
    if [ "$TEST_MODE" = "1" ]; then
      SAMPLE_RESUME_FLAG="--resume"
    fi

    "$PYTHON" "$RUN_ROOT/sampling.py" \
      --checkpoint "$DIFFUSION_CKPT" \
      --vae-checkpoint "$VAE_CKPT" \
      --output-root "$OUTPUT_DIR" \
      --num-samples-per-class "$NUM_IMAGES_PER_CLASS" \
      --batch-size "$BATCH_SIZE" \
      --steps "$SAMPLE_STEPS" \
      --guidance-scale "$GUIDANCE_SCALE" \
      --precision "$SAMPLE_PRECISION" \
      --use-ddim \
      $SAMPLE_RESUME_FLAG
  fi
fi
