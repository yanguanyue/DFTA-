#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/root/autodl-tmp"
LLAVA_DIR="${ROOT_DIR}/data/util/LLaVA-main"
MODEL_CACHE="${ROOT_DIR}/model"
OUTPUT_DIR="${ROOT_DIR}/data"

if [ ! -d "${LLAVA_DIR}" ]; then
  mkdir -p "${ROOT_DIR}/compare_main"
  git clone https://github.com/haotian-liu/LLaVA.git "${LLAVA_DIR}"
fi

export HF_HOME="${MODEL_CACHE}"
export TRANSFORMERS_CACHE="${MODEL_CACHE}"

python "${ROOT_DIR}/data/util/generate_llava_med_prompts.py" \
  --data-root "${ROOT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --model-cache "${MODEL_CACHE}" \
  ${MODEL_ID:+--model-id "${MODEL_ID}"} \
  ${MODEL_DIR:+--model-dir "${MODEL_DIR}"} \
  ${LIMIT:+--limit "${LIMIT}"}
