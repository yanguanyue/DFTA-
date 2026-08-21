#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/group_paths.sh"
PYTHON=${PYTHON:-/root/miniconda3/envs/flow/bin/python}
ROOT=/root/autodl-tmp/output/ablation_complete/evaluation/segmentation
MIX=$ROOT/mixed
RUNS=$ROOT/runs
SEEDS=(${SEEDS:-42 123 2024 3407 7777})
MODELS=(${MODELS:-unet segformer})
mapping=(); for g in "${ABLATION_GROUPS[@]}"; do mapping+=("$g=${GROUP_ROOT[$g]}"); done
"$PYTHON" /root/autodl-tmp/metirc/statistics/check_ai_inputs.py --mapping "${mapping[@]}" --output "$ROOT/input_report.csv" --require-masks
mkdir -p "$MIX" "$RUNS"
for g in "${ABLATION_GROUPS[@]}"; do
  "$PYTHON" /root/autodl-tmp/metirc/segmentation/mix_synthetic_seg.py \
    --real-root /root/autodl-tmp/data/HAM10000/input --synthetic-root "${GROUP_ROOT[$g]}" \
    --output-root "$MIX/$g" --minority-cap 1500 --majority-cap 500 --seed 42
done
for g in "${ABLATION_GROUPS[@]}"; do
  for model in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      out="$RUNS/$g/$model/seed_$seed"; mkdir -p "$out"
      if test -f "$out/${model}_metrics.json"; then
        rm -f "$out/${model}_best.pt"
        continue
      fi
      "$PYTHON" /root/autodl-tmp/metirc/segmentation/train_segmentation.py \
        --dataset-root "$MIX/$g" --data-layout mixed --model "$model" \
        --batch-size "${BATCH_SIZE:-4}" --image-size "${IMAGE_SIZE:-512}" \
        --max-steps "${MAX_STEPS:-15000}" --val-interval "${VAL_INTERVAL:-3000}" \
        --num-workers "${WORKERS:-2}" --amp \
        --seed "$seed" --save-dir "$out" > "$out/stdout.log" 2>&1
      rm -f "$out/${model}_best.pt"
    done
  done
done
"$PYTHON" /root/autodl-tmp/metirc/statistics/analyze_repeated_runs.py \
  --task segmentation --input-root "$RUNS" --output-dir "$ROOT/statistics"
