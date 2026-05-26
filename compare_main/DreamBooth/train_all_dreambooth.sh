#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_PATH=${ROOT_PATH:-"$(cd "$SCRIPT_DIR/../.." && pwd)"}
REPO_ROOT=${REPO_ROOT:-"$SCRIPT_DIR"}
PYTHON=${PYTHON:-/root/miniconda3/envs/flow/bin/python}

# Hugging Face mirror (可按需修改)
export HF_ENDPOINT=${HF_ENDPOINT:-"https://hf-mirror.com"}

# redirect caches to the data partition so large downloads don't fill the system disk
export HF_HOME=${HF_HOME:-"$ROOT_PATH/cache/huggingface"}
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-"$ROOT_PATH/cache/transformers"}
export DIFFUSERS_CACHE=${DIFFUSERS_CACHE:-"$ROOT_PATH/cache/diffusers"}
export TORCH_HOME=${TORCH_HOME:-"$ROOT_PATH/cache/torch"}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-"$ROOT_PATH/cache"}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-"$ROOT_PATH/cache/pip"}

if [ ! -x "$PYTHON" ]; then
  PYTHON=python
fi

ACCELERATE_LAUNCH=("$PYTHON" -m accelerate.commands.launch)

DATA_ROOT=${DATA_ROOT:-"$ROOT_PATH/data/HAM10000/input/train/HAM10000_img_class"}
MODEL_CACHE_DIR=${MODEL_CACHE_DIR:-"$ROOT_PATH/model/AI-ModelScope/stable-diffusion-v1-5"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$ROOT_PATH/checkpoint/compare_models/DreamBooth"}
SINGLE_MODEL=${SINGLE_MODEL:-0}
MIXED_INSTANCE_DIR=${MIXED_INSTANCE_DIR:-""}
SINGLE_PROMPT=${SINGLE_PROMPT:-"skin lesion"}
OUTPUT_NAME=${OUTPUT_NAME:-"ham10000-mix-model"}

# 2) 训练参数
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-15000}

COMMON_ARGS=(
  --pretrained_model_name_or_path "${MODEL_CACHE_DIR}"
  --resolution 512
  --train_batch_size 1
  --gradient_accumulation_steps 1
  --learning_rate 5e-6
  --lr_scheduler "constant"
  --lr_warmup_steps 0
  --max_train_steps "${MAX_TRAIN_STEPS}"
)

# 3) 七类训练（目录名 -> prompt -> 输出目录名）
declare -A PROMPTS
PROMPTS[akiec]="actinic keratosis and intraepithelial carcinoma"
PROMPTS[bcc]="basal cell carcinoma"
PROMPTS[bkl]="benign keratosis-like lesion"
PROMPTS[df]="dermatofibroma"
PROMPTS[mel]="melanoma"
PROMPTS[nv]="melanocytic nevus"
PROMPTS[vasc]="vascular lesion"

if [ "$SINGLE_MODEL" = "1" ]; then
  INSTANCE_DIR=${MIXED_INSTANCE_DIR:-"$DATA_ROOT"}
  OUTPUT_DIR="${OUTPUT_ROOT}/${OUTPUT_NAME}"
  mkdir -p "${OUTPUT_DIR}"

  echo "\n===== Training ${OUTPUT_NAME} ====="
  "${ACCELERATE_LAUNCH[@]}" "${REPO_ROOT}/src/train_dreambooth.py" \
    --instance_data_dir "${INSTANCE_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --instance_prompt "${SINGLE_PROMPT}" \
    "${COMMON_ARGS[@]}"
else
  for cls in akiec bcc bkl df mel nv vasc; do
    INSTANCE_DIR="${DATA_ROOT}/${cls}"
    OUTPUT_DIR="${OUTPUT_ROOT}/${cls}-model"
    mkdir -p "${OUTPUT_DIR}"

    echo "\n===== Training ${cls} ====="
    "${ACCELERATE_LAUNCH[@]}" "${REPO_ROOT}/src/train_dreambooth.py" \
      --instance_data_dir "${INSTANCE_DIR}" \
      --output_dir "${OUTPUT_DIR}" \
      --instance_prompt "${PROMPTS[$cls]}" \
      "${COMMON_ARGS[@]}"
  done
fi

echo "\nAll trainings finished. Models saved in: ${OUTPUT_ROOT}"