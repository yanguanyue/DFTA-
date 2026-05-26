#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-/root/miniconda3/envs/flow/bin/python}
REPO=/root/autodl-tmp/metirc/pytorch-classification-extended-master
REAL_TRAIN=/root/autodl-tmp/data/HAM10000/input/train/HAM10000_img_class
REAL_VAL=/root/autodl-tmp/data/HAM10000/input/val/HAM10000_img_class
SYN_ROOT=/root/autodl-tmp/output/ablation
FLOW_ROOT=/root/autodl-tmp/output/generate/flow
MIX_ROOT=/root/autodl-tmp/output/ablation/metric/mixed_datasets/classifier
OUT_ROOT=/root/autodl-tmp/output/ablation/metric/classifier
CHECKPOINT_ROOT=$OUT_ROOT/checkpoints/ham10000_mix
LOG_ROOT=$OUT_ROOT/logs
METRIC_DIR=$OUT_ROOT/metrics
SAMPLE=${SAMPLE:-0}
DRY_RUN=${DRY_RUN:-0}
EPOCHS=${EPOCHS:-30}
SCHEDULE=${SCHEDULE:-"15 25"}
TRAIN_BATCH=${TRAIN_BATCH:-64}
TEST_BATCH=${TEST_BATCH:-64}
WORKERS=${WORKERS:-0}

if [ ! -x "$PYTHON" ]; then
  PYTHON=python
fi

if ! "$PYTHON" - <<'PY'
import importlib
def needs_numpy_downgrade() -> bool:
  try:
    import numpy
  except Exception:
    return False
  major = int(numpy.__version__.split(".")[0])
  return major >= 2

missing = []
for name in ["progress", "openpyxl"]:
    try:
        importlib.import_module(name)
    except Exception:
        missing.append(name)
if missing or needs_numpy_downgrade():
    raise SystemExit(1)
PY
then
  if [ -f "$REPO/requirements.txt" ]; then
    "$PYTHON" -m pip install -r "$REPO/requirements.txt"
  else
    "$PYTHON" -m pip install progress
  fi
  "$PYTHON" -m pip install openpyxl
  "$PYTHON" -m pip install "numpy<2.0"
fi

ARCHS=(resnet18 resnet50 efficientnet_b0)

mkdir -p "$MIX_ROOT" "$CHECKPOINT_ROOT" "$LOG_ROOT" "$METRIC_DIR"

# Discover available synthetic models (ablation + flow)
MODEL_LIST=$(
  "$PYTHON" - <<'PY'
from pathlib import Path
import sys

ablation_root = Path("/root/autodl-tmp/output/ablation")
flow_root = Path("/root/autodl-tmp/output/generate/flow")

def has_images(model_dir: Path) -> bool:
  for cls_dir in model_dir.iterdir():
    if not cls_dir.is_dir():
      continue
    candidates = [cls_dir / "images", cls_dir / "image", cls_dir]
    for cand in candidates:
      if cand.exists() and any(cand.glob("*.png")):
        return True
  return False

valid = []
if ablation_root.exists():
  for model_dir in sorted(ablation_root.iterdir()):
    if not model_dir.is_dir() or model_dir.name == "metric":
      continue
    if has_images(model_dir):
      valid.append(model_dir.name)

if flow_root.exists() and has_images(flow_root):
  valid.append("flow")

print(" ".join(valid))
PY
)

MODELS=(baseline ${MODEL_LIST})

if [ "$SAMPLE" = "1" ]; then
  ARCHS=(resnet18)
  EPOCHS=1
  SCHEDULE="1 1"
  TRAIN_BATCH=2
  TEST_BATCH=2
  if [ ${#MODELS[@]} -gt 1 ]; then
    MODELS=(${MODELS[1]})
  fi
fi

# Baseline (real only)
if [ ! -d "$MIX_ROOT/baseline" ]; then
  mkdir -p "$MIX_ROOT/baseline"
  ln -sfn "$REAL_TRAIN" "$MIX_ROOT/baseline/train"
  ln -sfn "$REAL_VAL" "$MIX_ROOT/baseline/val"
fi

for MODEL in "${MODELS[@]}"; do
  DATASET_DIR="$MIX_ROOT/$MODEL"

  if [ "$MODEL" != "baseline" ]; then
    SYN_DATA_ROOT="$SYN_ROOT/$MODEL"
    if [ "$MODEL" = "flow" ]; then
      SYN_DATA_ROOT="$FLOW_ROOT"
    fi

    MIX_ARGS=(
      --real-train "$REAL_TRAIN"
      --real-val "$REAL_VAL"
      --synthetic-root "$SYN_DATA_ROOT"
      --output-root "$DATASET_DIR"
      --majority-cap 500
    )
    if [ "$DRY_RUN" = "1" ]; then
      MIX_ARGS+=(--dry-run)
    fi
    "$PYTHON" /root/autodl-tmp/metirc/classifier/mix_synthetic.py "${MIX_ARGS[@]}"
  fi

  if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY_RUN] Skip classifier training for $MODEL"
    continue
  fi

  for ARCH in "${ARCHS[@]}"; do
    CKPT_DIR="$CHECKPOINT_ROOT/$MODEL/$ARCH"
    LOG_FILE="$LOG_ROOT/${MODEL}_${ARCH}.log"
    mkdir -p "$CKPT_DIR"

    if [ -f "$CKPT_DIR/model_best.pth.tar" ]; then
      echo "[SKIP] $MODEL / $ARCH already has model_best.pth.tar"
      continue
    fi

    EXTRA_OPTS=()
    if [ "$ARCH" = "vit_b_16" ]; then
      EXTRA_OPTS+=(--train-batch 16 --test-batch 16)
    else
      EXTRA_OPTS+=(--train-batch "$TRAIN_BATCH" --test-batch "$TEST_BATCH")
    fi

    PRETRAINED=(--pretrained)

    cd "$REPO"
    nohup "$PYTHON" customdata.py -a "$ARCH" -d "$DATASET_DIR" \
      "${PRETRAINED[@]}" \
      --epochs "$EPOCHS" --schedule $SCHEDULE --gamma 0.1 --lr 0.001 --gpu-id 0 -c "$CKPT_DIR" \
      -j "$WORKERS" \
      "${EXTRA_OPTS[@]}" \
      > "$LOG_FILE" 2>&1
  done

done

if [ "$DRY_RUN" != "1" ]; then
  "$PYTHON" /root/autodl-tmp/metirc/classifier/summarize_improvements.py \
    --checkpoint-root "$CHECKPOINT_ROOT" \
    --output-dir "$METRIC_DIR"
fi
