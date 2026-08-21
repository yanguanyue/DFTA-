#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/group_paths.sh"
PYTHON=${PYTHON:-/root/miniconda3/envs/flow/bin/python}
REPO=/root/autodl-tmp/metirc/pytorch-classification-extended-master
REAL_TRAIN=/root/autodl-tmp/data/HAM10000/input/train/HAM10000_img_class
REAL_VAL=/root/autodl-tmp/data/HAM10000/input/val/HAM10000_img_class
ROOT=/root/autodl-tmp/output/ablation_complete/evaluation/classifier
MIX=$ROOT/mixed
RUNS=$ROOT/runs
SEEDS=(${SEEDS:-42 123 2024 3407 7777})
ARCHS=(${ARCHS:-resnet18 resnet50 efficientnet_b0})
WORKERS=${WORKERS:-8}
TRAIN_BATCH=${TRAIN_BATCH:-64}
TEST_BATCH=${TEST_BATCH:-64}
mapping=(); for g in "${ABLATION_GROUPS[@]}"; do mapping+=("$g=${GROUP_ROOT[$g]}"); done
"$PYTHON" /root/autodl-tmp/metirc/statistics/check_ai_inputs.py --mapping "${mapping[@]}" --output "$ROOT/input_report.csv"
mkdir -p "$MIX/baseline" "$RUNS"
ln -sfn "$REAL_TRAIN" "$MIX/baseline/train"
ln -sfn "$REAL_VAL" "$MIX/baseline/val"
for g in "${ABLATION_GROUPS[@]}"; do
  "$PYTHON" /root/autodl-tmp/metirc/classifier/mix_synthetic.py \
    --real-train "$REAL_TRAIN" --real-val "$REAL_VAL" --synthetic-root "${GROUP_ROOT[$g]}" \
    --output-root "$MIX/$g" --majority-cap 500 --seed 42
done
for g in baseline "${ABLATION_GROUPS[@]}"; do
  for arch in "${ARCHS[@]}"; do
    case "$arch" in
      resnet18) arch_workers="${RESNET18_WORKERS:-12}" ;;
      resnet50) arch_workers="${RESNET50_WORKERS:-16}" ;;
      efficientnet_b0) arch_workers="${EFFICIENTNET_WORKERS:-16}" ;;
      *) arch_workers="$WORKERS" ;;
    esac
    for seed in "${SEEDS[@]}"; do
      out="$RUNS/$g/$arch/seed_$seed"; mkdir -p "$out"
      if test -f "$out/evaluation_metrics.json"; then
        rm -f "$out/model_best.pth.tar" "$out/checkpoint.pth.tar"
        continue
      fi
      if ! test -f "$out/model_best.pth.tar"; then
      (cd "$REPO"; "$PYTHON" customdata.py -a "$arch" -d "$MIX/$g" --pretrained \
        --epochs "${EPOCHS:-30}" --schedule 15 25 --gamma 0.1 --lr 0.001 --gpu-id 0 \
        --manualSeed "$seed" -c "$out" -j "$arch_workers" --amp \
        --train-batch "$TRAIN_BATCH" --test-batch "$TEST_BATCH") \
        > "$out/stdout.log" 2>&1
      fi
      "$PYTHON" /root/autodl-tmp/metirc/classifier/evaluate_complete.py         --data "$MIX/$g" --arch "$arch" --checkpoint "$out/model_best.pth.tar"         --output "$out/evaluation_metrics.json" --batch-size "$TEST_BATCH" --workers "$arch_workers"
      rm -f "$out/model_best.pth.tar" "$out/checkpoint.pth.tar"
    done
  done
done
"$PYTHON" /root/autodl-tmp/metirc/statistics/analyze_repeated_runs.py \
  --task classifier --input-root "$RUNS" --output-dir "$ROOT/statistics"
