#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/models.sh"
PYTHON=${PYTHON:-/root/miniconda3/envs/flow/bin/python}
ROOT=$EVAL_ROOT/segmentation
MIX=$ROOT/mixed
RUNS=$ROOT/runs
SEEDS=(${SEEDS:-42 123 2024 3407 7777})
SEG_MODELS=(${SEG_MODELS:-unet segformer})
mkdir -p "$MIX" "$RUNS"
for model in "${MASK_MODELS[@]}"; do
  "$PYTHON" /root/autodl-tmp/metirc/segmentation/mix_synthetic_seg.py \
    --real-root /root/autodl-tmp/data/HAM10000/input \
    --synthetic-root "$NORMALIZED_ROOT/$model" --output-root "$MIX/$model" \
    --minority-cap 1500 --majority-cap 500 --seed 42
done
# The baseline uses only the original HAM10000 train/validation pairs.
# All optimization settings and random seeds are identical to the augmented runs.
for model in baseline "${MASK_MODELS[@]}"; do
  if test "$model" = baseline; then
    dataset_root=/root/autodl-tmp/data/HAM10000/input
    data_layout=ham10000
  else
    dataset_root="$MIX/$model"
    data_layout=mixed
  fi
  for seg_model in "${SEG_MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      out="$RUNS/$model/$seg_model/seed_$seed"; mkdir -p "$out"
      if test -f "$out/${seg_model}_metrics.json"; then
        rm -f "$out/${seg_model}_best.pt"
        continue
      fi
      "$PYTHON" /root/autodl-tmp/metirc/segmentation/train_segmentation.py \
        --dataset-root "$dataset_root" --data-layout "$data_layout" --model "$seg_model" \
        --batch-size "${BATCH_SIZE:-4}" --image-size "${IMAGE_SIZE:-512}" \
        --max-steps "${MAX_STEPS:-15000}" --val-interval "${VAL_INTERVAL:-3000}" \
        --num-workers "${WORKERS:-2}" --amp \
        --seed "$seed" --save-dir "$out" > "$out/stdout.log" 2>&1
      rm -f "$out/${seg_model}_best.pt"
    done
  done
done
"$PYTHON" /root/autodl-tmp/metirc/statistics/analyze_repeated_runs.py \
  --task segmentation --input-root "$RUNS" --output-dir "$ROOT/statistics"
