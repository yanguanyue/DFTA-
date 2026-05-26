#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1
export TORCH_HOME=${TORCH_HOME:-/root/autodl-tmp/model}
PREFETCH_WEIGHTS=${PREFETCH_WEIGHTS:-1}
INCEPTION_WEIGHTS=${INCEPTION_WEIGHTS:-$TORCH_HOME/inception_v3_google-0cc3c7bd.pth}

GEN_ROOT=${GEN_ROOT:-/root/autodl-tmp/output/ablation}
FLOW_ROOT=${FLOW_ROOT:-/root/autodl-tmp/output/generate/flow}
REAL_ROOT=${REAL_ROOT:-/root/autodl-tmp/data/HAM10000/input}
REAL_SPLIT=${REAL_SPLIT:-val}
OUT_DIR=${OUT_DIR:-/root/autodl-tmp/output/ablation/metric/metrics}
BATCH_SIZE=${BATCH_SIZE:-16}
CLASS_LIST=${CLASS_LIST:-}
LIMIT=${LIMIT:-}
DC_BATCH_SIZE=${DC_BATCH_SIZE:-}
DIVERSITY_PAIRS=${DIVERSITY_PAIRS:-}
MODEL_LIST=${MODEL_LIST:-}

if [ ! -d "${REAL_ROOT}/${REAL_SPLIT}/HAM10000_img_class" ]; then
  echo "[Error] Missing real data: ${REAL_ROOT}/${REAL_SPLIT}/HAM10000_img_class"
  exit 1
fi

mkdir -p "$OUT_DIR"

if [ "$PREFETCH_WEIGHTS" = "1" ] && [ ! -f "$INCEPTION_WEIGHTS" ]; then
  echo "Prefetching Inception/LPIPS weights into $TORCH_HOME ..."
  python - <<'PY'
import torchvision
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

_ = torchvision.models.inception_v3(weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1)
_ = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True)
print("Prefetch done")
PY
elif [ "$PREFETCH_WEIGHTS" = "1" ]; then
  echo "Inception weights already cached: $INCEPTION_WEIGHTS"
fi

GEN_DIRS=("$GEN_ROOT"/*/)
if [ -d "$FLOW_ROOT" ]; then
  GEN_DIRS+=("$FLOW_ROOT")
fi

for gen_dir in "${GEN_DIRS[@]}"; do
  model=$(basename "$gen_dir")
  if [ -n "$MODEL_LIST" ]; then
    case ",${MODEL_LIST}," in
      *",${model},"*) ;;
      *)
        continue
        ;;
    esac
  fi

  case "$model" in
    metrics_cmmd_clip|metric|.ipynb_checkpoints)
      continue
      ;;
  esac

  echo "Computing metrics for: $model"

  cmd=(python /root/autodl-tmp/metirc/metirc.py \
    --gen_root "$gen_dir" \
    --real_root "$REAL_ROOT" \
    --real_split "$REAL_SPLIT" \
    --output_dir "$OUT_DIR" \
    --batch_size "$BATCH_SIZE")

  if [ -n "$CLASS_LIST" ]; then
    cmd+=(--class_list "$CLASS_LIST")
  fi
  if [ -n "$LIMIT" ]; then
    cmd+=(--limit "$LIMIT")
  fi
  if [ -n "$DC_BATCH_SIZE" ]; then
    cmd+=(--dc_batch_size "$DC_BATCH_SIZE")
  fi


  "${cmd[@]}"

done

echo "Outputs saved to: $OUT_DIR"
