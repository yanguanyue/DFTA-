#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT_PATH=$(cd "$SCRIPT_DIR/../.." && pwd)

export PYTHONPATH="$SCRIPT_DIR/external/diffusers/src:${PYTHONPATH:-}"

export TRAIN_DIR="$ROOT_PATH/data/HAM10000/input/train/HAM10000_img_class"
export OUTPUT_DIR="$ROOT_PATH/checkpoint/compare_models/LesionGen/ham10000_lora"
export HF_HOME="$ROOT_PATH/model/hf_home"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"

export MODELSCOPE_CACHE="$ROOT_PATH/model"
export MODELSCOPE_REPO_ID="${MODELSCOPE_REPO_ID:-modelscope/stable-diffusion-v1-4}"

resolve_model_dir() {
	local hub_dir
	hub_dir=$(find "$MODELSCOPE_CACHE/hub" -maxdepth 2 -type d -name "models--*stable-diffusion-v1-4*" 2>/dev/null | head -n 1)
	if [ -n "$hub_dir" ]; then
		echo "$hub_dir"
		return
	fi
	find "$MODELSCOPE_CACHE" -maxdepth 3 -type d -path "*/stable-diffusion-v1-4" 2>/dev/null | head -n 1
}

BASE_MODEL_DIR=$(resolve_model_dir || true)
if [ -z "$BASE_MODEL_DIR" ]; then
	echo "[ModelScope] Downloading stable-diffusion-v1-4 to $MODELSCOPE_CACHE"
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
export MODEL_NAME="$BASE_MODEL_DIR"

mkdir -p "$OUTPUT_DIR"

BASE_CAPTION_TEXT=${BASE_CAPTION_TEXT:-"a high quality image of a skin lesion"}
python "$SCRIPT_DIR/tools/create_metadata_jsonl.py" \
	--image_dir "$TRAIN_DIR" \
	--caption "$BASE_CAPTION_TEXT" \
	--overwrite

ACCELERATE_CMD=${ACCELERATE_CMD:-"python -m accelerate.commands.launch"}
BASE_MAX_TRAIN_STEPS=${BASE_MAX_TRAIN_STEPS:-14000}
BASE_CHECKPOINTING_STEPS=${BASE_CHECKPOINTING_STEPS:-2000}
BASE_VALIDATION_EPOCHS=${BASE_VALIDATION_EPOCHS:-5}

$ACCELERATE_CMD "$SCRIPT_DIR/external/diffusers/examples/text_to_image/train_text_to_image_lora.py" \
--pretrained_model_name_or_path=$MODEL_NAME \
--train_data_dir=$TRAIN_DIR \
--image_column=image \
--caption_column=text \
--resolution=256 --center_crop --random_flip \
--validation_prompt="generate a high quality image of melanoma skin lesion" \
--num_validation_images=1 \
--validation_epochs=$BASE_VALIDATION_EPOCHS \
--train_batch_size=1 \
--max_train_steps=$BASE_MAX_TRAIN_STEPS \
--learning_rate=5e-06 \
--lr_scheduler="constant" --lr_warmup_steps=0 \
--gradient_accumulation_steps=4 \
--gradient_checkpointing \
--mixed_precision="fp16" \
--checkpointing_steps=$BASE_CHECKPOINTING_STEPS \
--rank=64 \
--prediction_type="epsilon" \
--output_dir=$OUTPUT_DIR
