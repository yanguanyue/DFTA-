#!/usr/bin/env bash
set -euo pipefail
DIR=$(cd "$(dirname "$0")" && pwd)
source "$DIR/config.sh"
STATUS=$EVAL_ROOT/status.tsv; mkdir -p "$EVAL_ROOT/logs"; touch "$STATUS"
run_stage(){ local n=$1; shift; if grep -q "^$n[[:space:]]\+OK" "$STATUS"; then return; fi; printf '%s\tSTART\t%s\n' "$n" "$(date -Is)" >> "$STATUS"; "$@"; printf '%s\tOK\t%s\n' "$n" "$(date -Is)" >> "$STATUS"; }
run_stage prepare "$PYTHON" "$DIR/prepare_inputs.py" "$DIR/config.json"

mkdir -p "$EVAL_ROOT/generation_metrics"
for model in "${MODELS[@]}"; do
  out="$EVAL_ROOT/generation_metrics/$model"
  test -f "$out/metrics_${model}.json" && continue
  run_stage "metric_$model" "$PYTHON" /root/autodl-tmp/metirc/metirc.py --gen_root "$NORMALIZED_ROOT/$model" --real_root "$REAL_ROOT" --real_split val --output_dir "$out" --batch_size "${METRIC_BATCH:-16}" --class_list "$CLASS_CSV"
done

CR=$EVAL_ROOT/classifier; mkdir -p "$CR/mixed/baseline" "$CR/runs"
ln -sfn "$REAL_ROOT/train/HAM10000_img_class" "$CR/mixed/baseline/train"
ln -sfn "$REAL_ROOT/val/HAM10000_img_class" "$CR/mixed/baseline/val"
for model in "${MODELS[@]}"; do run_stage "mix_cls_$model" "$PYTHON" "$DIR/mix_classifier_images.py" --real-train "$REAL_ROOT/train/HAM10000_img_class" --real-val "$REAL_ROOT/val/HAM10000_img_class" --synthetic-root "$NORMALIZED_ROOT/$model" --output-root "$CR/mixed/$model" --majority-cap "$MAJORITY_CAP" --seed 42; done
for model in baseline "${MODELS[@]}"; do for arch in "${ARCHS[@]}"; do for seed in "${SEEDS[@]}"; do
  out="$CR/runs/$model/$arch/seed_$seed"; mkdir -p "$out"; test -f "$out/evaluation_metrics.json" && continue
  case "$arch" in resnet18) workers=12;; *) workers=16;; esac
  if ! test -f "$out/model_best.pth.tar"; then
    (cd /root/autodl-tmp/metirc/pytorch-classification-extended-master; "$PYTHON" customdata.py -a "$arch" -d "$CR/mixed/$model" --pretrained --epochs "$EPOCHS" --schedule 15 25 --gamma 0.1 --lr 0.001 --gpu-id 0 --manualSeed "$seed" -c "$out" -j "$workers" --train-batch "$TRAIN_BATCH" --test-batch "$TEST_BATCH") > "$out/stdout.log" 2>&1
  fi
  "$PYTHON" /root/autodl-tmp/metirc/classifier/evaluate_complete.py --data "$CR/mixed/$model" --arch "$arch" --checkpoint "$out/model_best.pth.tar" --output "$out/evaluation_metrics.json" --batch-size "$TEST_BATCH" --workers "$workers"
  rm -f "$out/model_best.pth.tar" "$out/checkpoint.pth.tar"; printf 'classifier_%s_%s_%s\tOK\t%s\n' "$model" "$arch" "$seed" "$(date -Is)" >> "$STATUS"
done; done; done
"$PYTHON" /root/autodl-tmp/metirc/statistics/analyze_repeated_runs.py --task classifier --input-root "$CR/runs" --output-dir "$CR/statistics"

SR=$EVAL_ROOT/segmentation; mkdir -p "$SR/mixed" "$SR/runs" "$SR/empty_synthetic"
# Real-only segmentation baseline. The same mixer and mixed layout are used,
# but both synthetic caps are zero, so train/val contain only real pairs.
run_stage "mix_seg_baseline" "$PYTHON" /root/autodl-tmp/metirc/segmentation/mix_synthetic_seg.py --real-root "$REAL_ROOT" --synthetic-root "$SR/empty_synthetic" --output-root "$SR/mixed/baseline" --minority-cap 0 --majority-cap 0 --seed 42
for model in "${MASK_MODELS[@]}"; do run_stage "mix_seg_$model" "$PYTHON" /root/autodl-tmp/metirc/segmentation/mix_synthetic_seg.py --real-root "$REAL_ROOT" --synthetic-root "$NORMALIZED_ROOT/$model" --output-root "$SR/mixed/$model" --minority-cap "$MINORITY_CAP" --majority-cap "$MAJORITY_CAP" --seed 42; done
for model in baseline "${MASK_MODELS[@]}"; do for seg in "${SEG_MODELS[@]}"; do for seed in "${SEEDS[@]}"; do
  out="$SR/runs/$model/$seg/seed_$seed"; mkdir -p "$out"; test -f "$out/${seg}_metrics.json" && continue
  "$PYTHON" /root/autodl-tmp/metirc/segmentation/train_segmentation.py --dataset-root "$SR/mixed/$model" --data-layout mixed --model "$seg" --batch-size "$SEG_BATCH" --image-size 512 --max-steps "$SEG_STEPS" --val-interval "$SEG_VAL_INTERVAL" --num-workers 2 --amp --seed "$seed" --save-dir "$out" > "$out/stdout.log" 2>&1
  rm -f "$out/${seg}_best.pt"; printf 'segmentation_%s_%s_%s\tOK\t%s\n' "$model" "$seg" "$seed" "$(date -Is)" >> "$STATUS"
done; done; done
"$PYTHON" /root/autodl-tmp/metirc/statistics/analyze_repeated_runs.py --task segmentation --input-root "$SR/runs" --output-dir "$SR/statistics"
printf 'ALL_OK\t%s\n' "$(date -Is)" >> "$STATUS"
