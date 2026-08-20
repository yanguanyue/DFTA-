#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/models.sh"
PYTHON=${PYTHON:-/root/miniconda3/envs/flow/bin/python}
OUT=$EVAL_ROOT/generation_metrics
mkdir -p "$OUT"
for model in "${MODELS[@]}"; do
  test -f "$OUT/$model/metrics_${model}.json" && continue
  "$PYTHON" /root/autodl-tmp/metirc/metirc.py \
    --gen_root "$NORMALIZED_ROOT/$model" \
    --real_root /root/autodl-tmp/data/HAM10000/input \
    --real_split val --output_dir "$OUT/$model" --batch_size "${BATCH_SIZE:-16}"
done
