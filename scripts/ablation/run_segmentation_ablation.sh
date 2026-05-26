#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-/root/miniconda3/envs/flow/bin/python}
REAL_ROOT=/root/autodl-tmp/data/HAM10000/input
SYN_ROOT=/root/autodl-tmp/output/ablation
FLOW_ROOT=/root/autodl-tmp/output/generate/flow
MIX_ROOT=/root/autodl-tmp/output/ablation/metric/mixed_datasets/segmentation
OUT_ROOT=/root/autodl-tmp/output/ablation/metric/segmentation
OUTPUT_DIR=$OUT_ROOT/outputs
OUTPUT_JSON=$OUT_ROOT/summary/metrics_summary.json
OUTPUT_CSV=$OUT_ROOT/summary/metrics_summary.csv
OUTPUT_XLSX=$OUT_ROOT/summary/metrics_summary.xlsx
SAMPLE=${SAMPLE:-0}
DRY_RUN=${DRY_RUN:-0}
MAX_STEPS=${MAX_STEPS:-15000}
VAL_INTERVAL=${VAL_INTERVAL:-3000}
BATCH_SIZE=${BATCH_SIZE:-4}
IMAGE_SIZE=${IMAGE_SIZE:-512}

if [ ! -x "$PYTHON" ]; then
  PYTHON=python
fi

mkdir -p "$MIX_ROOT" "$OUT_ROOT/summary"

MODEL_LIST=$(
  "$PYTHON" - <<'PY'
from pathlib import Path
import sys

ablation_root = Path("/root/autodl-tmp/output/ablation")
flow_root = Path("/root/autodl-tmp/output/generate/flow")

def has_masks(model_dir: Path) -> bool:
  for cls_dir in model_dir.iterdir():
    if not cls_dir.is_dir():
      continue
    for images_dir, masks_dir in [
      (cls_dir / "images", cls_dir / "masks"),
      (cls_dir / "image", cls_dir / "mask"),
    ]:
      if images_dir.exists() and masks_dir.exists():
        if any(masks_dir.glob("*.png")) or any(masks_dir.glob("*.jpg")):
          return True
  return False

valid = []
if ablation_root.exists():
  for model_dir in sorted(ablation_root.iterdir()):
    if model_dir.is_dir() and model_dir.name != "metric" and has_masks(model_dir):
      valid.append(model_dir.name)

if flow_root.exists() and has_masks(flow_root):
  valid.append("flow")

print(" ".join(valid))
PY
)

if [ -z "$MODEL_LIST" ]; then
  echo "No models with masks found under $SYN_ROOT"
  exit 0
fi

MODELS=($MODEL_LIST)

if [ "$SAMPLE" = "1" ]; then
  MAX_STEPS=1
  VAL_INTERVAL=1
  BATCH_SIZE=1
  IMAGE_SIZE=64
  if [ ${#MODELS[@]} -gt 0 ]; then
    MODELS=(${MODELS[0]})
  fi
fi

for MODEL in "${MODELS[@]}"; do
  SYN_DATA_ROOT="$SYN_ROOT/$MODEL"
  if [ "$MODEL" = "flow" ]; then
    SYN_DATA_ROOT="$FLOW_ROOT"
  fi

  MIX_ARGS=(
    --real-root "$REAL_ROOT"
    --synthetic-root "$SYN_DATA_ROOT"
    --output-root "$MIX_ROOT/$MODEL"
    --minority-cap 1500
    --majority-cap 500
  )
  if [ "$DRY_RUN" = "1" ]; then
    MIX_ARGS+=(--dry-run)
  fi
  "$PYTHON" /root/autodl-tmp/metirc/segmentation/mix_synthetic_seg.py "${MIX_ARGS[@]}"

done

MODELS_CSV=$(IFS=, ; echo "${MODELS[*]}")

if [ "$DRY_RUN" = "1" ]; then
  echo "[DRY_RUN] Skip segmentation training and summary"
  exit 0
fi

"$PYTHON" /root/autodl-tmp/metirc/segmentation/run_generated_models.py \
  --mixed-root "$MIX_ROOT" \
  --models "$MODELS_CSV" \
  --output-json "$OUTPUT_JSON" \
  --output-dir "$OUTPUT_DIR" \
  --batch-size "$BATCH_SIZE" \
  --image-size "$IMAGE_SIZE" \
  --max-steps "$MAX_STEPS" \
  --val-interval "$VAL_INTERVAL"

"$PYTHON" /root/autodl-tmp/metirc/segmentation/summarize_segmentation.py \
  --input-json "$OUTPUT_JSON" \
  --output-csv "$OUTPUT_CSV" \
  --output-xlsx "$OUTPUT_XLSX"
