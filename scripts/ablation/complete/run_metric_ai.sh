#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/group_paths.sh"
PYTHON=${PYTHON:-/root/miniconda3/envs/flow/bin/python}
OUT=/root/autodl-tmp/output/ablation_complete/evaluation/generation_metrics
REPORT=$OUT/input_report.csv
mapping=(); for g in "${ABLATION_GROUPS[@]}"; do mapping+=("$g=${GROUP_ROOT[$g]}"); done
"$PYTHON" /root/autodl-tmp/metirc/statistics/check_ai_inputs.py --mapping "${mapping[@]}" --output "$REPORT"
mkdir -p "$OUT"
for g in "${ABLATION_GROUPS[@]}"; do
  "$PYTHON" /root/autodl-tmp/metirc/metirc.py \
    --gen_root "${GROUP_ROOT[$g]}" --real_root /root/autodl-tmp/data/HAM10000/input \
    --real_split val --output_dir "$OUT/$g" --batch_size "${BATCH_SIZE:-16}"
done
