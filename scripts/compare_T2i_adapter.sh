#!/usr/bin/env bash
set -euo pipefail

##############################
# T2I-Adapter LoRA (HAM10000)
COMPARE_T2I_ADAPTER=true
TRAIN_ENABLED=${TRAIN_ENABLED:-true}
RUN_ENABLED=${RUN_ENABLED:-true}
TEST_MODE=${TEST_MODE:-0}
##############################

ROOT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RUN_ROOT="$ROOT_PATH/compare_main/Controlnet+T2i_adapter"
PYTHON=${PYTHON:-/root/miniconda3/envs/flow/bin/python}
MODEL_DIR="$ROOT_PATH/model/AI-ModelScope"
SD_MODEL_DIR="$MODEL_DIR/stable-diffusion-v1-5"
ADAPTER_DIR="$MODEL_DIR/t2iadapter_zoedepth_sd15v1"
CHECKPOINT_ROOT="$ROOT_PATH/checkpoint/compare_models/T2i_adapter"
OUTPUT_DIR="$ROOT_PATH/output/generate/T2i_Adapter"
CSV_TRAIN="$ROOT_PATH/data/metadata_train_llava.csv"
CSV_VAL="$ROOT_PATH/data/metadata_val_llava.csv"
CSV_TEST="$ROOT_PATH/data/metadata_test_llava.csv"
IMAGE_ROOT="$ROOT_PATH"

MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-15000}
NUM_IMAGES_PER_CLASS=${NUM_IMAGES_PER_CLASS:-1500}
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-30}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-7.5}
ADAPTER_SCALE=${ADAPTER_SCALE:-1.0}

if [ ! -x "$PYTHON" ]; then
  PYTHON=python
fi

if [ "$TEST_MODE" = "1" ]; then
  MAX_TRAIN_STEPS=2
  NUM_IMAGES_PER_CLASS=1
  NUM_INFERENCE_STEPS=2
fi

mkdir -p "$CHECKPOINT_ROOT" "$OUTPUT_DIR" "$MODEL_DIR"

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

if [ "$COMPARE_T2I_ADAPTER" = true ]; then
  echo "Comparing T2I-Adapter..."

  ensure_model "$SD_MODEL_DIR" "AI-ModelScope/stable-diffusion-v1-5|modelscope/stable-diffusion-v1-5|stable-diffusion-v1-5"
  ensure_model "$ADAPTER_DIR" "AI-ModelScope/t2iadapter_zoedepth_sd15v1|modelscope/t2iadapter_zoedepth_sd15v1|t2iadapter_zoedepth_sd15v1"

  if [ "$TRAIN_ENABLED" = true ]; then
    echo "[Stage] Training T2I-Adapter LoRA (steps=${MAX_TRAIN_STEPS})..."
    "$PYTHON" "$RUN_ROOT/train_sd15_lora_t2i_adapter.py" \
      --pretrained_model_name_or_path "$SD_MODEL_DIR" \
      --adapter_path "$ADAPTER_DIR" \
      --csv_paths "$CSV_TRAIN,$CSV_VAL,$CSV_TEST" \
      --output_dir "$CHECKPOINT_ROOT" \
      --resolution 512 \
      --train_batch_size 1 \
      --max_train_steps "$MAX_TRAIN_STEPS" \
      --checkpointing_steps 1000 \
      --mixed_precision fp16
  fi

  if [ "$RUN_ENABLED" = true ]; then
    echo "[Stage] Generating images (per class=${NUM_IMAGES_PER_CLASS})..."
    if [ ! -f "$CHECKPOINT_ROOT/pytorch_lora_weights.safetensors" ]; then
      echo "[x] LoRA weights not found at $CHECKPOINT_ROOT/pytorch_lora_weights.safetensors"
      exit 1
    fi

    "$PYTHON" "$RUN_ROOT/generate_ham10000_lora_images.py" \
      --sd_model "$SD_MODEL_DIR" \
      --adapter_path "$ADAPTER_DIR" \
      --lora_t2i_dir "$CHECKPOINT_ROOT" \
      --csv_train "$CSV_TRAIN" \
      --csv_val "$CSV_VAL" \
      --csv_test "$CSV_TEST" \
      --image_root "$IMAGE_ROOT" \
      --output_t2i "$OUTPUT_DIR" \
      --num_per_class "$NUM_IMAGES_PER_CLASS" \
      --num_inference_steps "$NUM_INFERENCE_STEPS" \
      --guidance_scale "$GUIDANCE_SCALE" \
      --adapter_conditioning_scale "$ADAPTER_SCALE" \
      --output_layout class_first \
      --prompts_from_val_only \
      --run_t2i
  fi
fi
