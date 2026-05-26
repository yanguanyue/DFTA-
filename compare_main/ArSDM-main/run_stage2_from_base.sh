#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/autodl-tmp/compare_main/ArSDM-main}"
PYTHON="${PYTHON:-/root/miniconda3/envs/flow/bin/python}"
export PYTHONPATH="$ROOT_DIR"
export OMP_NUM_THREADS=1

BASE_CFG="$ROOT_DIR/configs/HAM10000_base_512.yaml"
BASE_EXP_NAME="HAM10000_base_512"
LOG_ROOT="${LOG_ROOT:-/root/autodl-tmp/checkpoint/compare_models/ArSDM}"
BASE_LOG_ROOT="$LOG_ROOT/${BASE_EXP_NAME}"
TMP_DIR="$ROOT_DIR/configs/stage2_tmp"

BASE_MAX_STEPS="${BASE_MAX_STEPS:-5000}"
STAGE2_MAX_STEPS="${STAGE2_MAX_STEPS:-3000}"
PRETRAINED_BASE_CKPT="${PRETRAINED_BASE_CKPT:-}"

# Controls
# FORCE_BASE_TRAIN=1  -> always (re)train base config
# RESUME_BASE=1       -> resume base run if checkpoint exists
# RESUME_STAGE2=1     -> resume stage2 runs if checkpoint exists (default: 1)
FORCE_BASE_TRAIN="${FORCE_BASE_TRAIN:-0}"
RESUME_BASE="${RESUME_BASE:-0}"
RESUME_STAGE2="${RESUME_STAGE2:-1}"
SKIP_STAGE2_IF_EXISTS="${SKIP_STAGE2_IF_EXISTS:-1}"

STAGE2_CONFIGS=(
  "$ROOT_DIR/configs/HAM10000_akiec.yaml"
  "$ROOT_DIR/configs/HAM10000_bcc.yaml"
  "$ROOT_DIR/configs/HAM10000_bkl_lora_512.yaml"
  "$ROOT_DIR/configs/HAM10000_df_lora_512.yaml"
  "$ROOT_DIR/configs/HAM10000_mel_lora_512.yaml"
  "$ROOT_DIR/configs/HAM10000_nv_lora_512.yaml"
  "$ROOT_DIR/configs/HAM10000_vasc_lora_512.yaml"
)

mkdir -p "$TMP_DIR"

echo "==============================="
echo "Step 1/2: Check base training"
echo "Config: $BASE_CFG"
echo "==============================="

LATEST_RUN=""
BASE_CKPT=""
# If a pretrained checkpoint is provided, use it as init (but still run base training)
PRETRAINED_INIT=""
if [ -n "$PRETRAINED_BASE_CKPT" ] && [ -f "$PRETRAINED_BASE_CKPT" ]; then
  echo "Found pretrained base checkpoint to initialize from: $PRETRAINED_BASE_CKPT"
  PRETRAINED_INIT="$PRETRAINED_BASE_CKPT"
fi

# Discover latest existing base run (used only if no pretrained init provided and to resume if requested)
if [ -d "$BASE_LOG_ROOT" ]; then
  LATEST_RUN=$(ls -td "$BASE_LOG_ROOT"/* 2>/dev/null | head -n 1 || true)
fi
if [ -n "$LATEST_RUN" ]; then
  BASE_CKPT="$LATEST_RUN/checkpoints/last.ckpt"
  if [ ! -f "$BASE_CKPT" ]; then
    BASE_CKPT=$(ls -t "$LATEST_RUN"/checkpoints/*.ckpt 2>/dev/null | head -n 1 || true)
  fi
fi

if [ "$FORCE_BASE_TRAIN" = "1" ] || [ -z "$BASE_CKPT" ]; then
  echo "No usable base checkpoint found (or FORCE_BASE_TRAIN=1). Training base..."
  BASE_TMP_CFG="$TMP_DIR/$(basename "$BASE_CFG")"
  "$PYTHON" - <<PY
import yaml
from pathlib import Path
cfg_path = Path("$BASE_CFG")
content = yaml.safe_load(cfg_path.read_text())
content.setdefault("lightning", {}).setdefault("trainer", {})["max_steps"] = int("$BASE_MAX_STEPS")
content.setdefault("exp", {})["logdir"] = "$LOG_ROOT"
ckpt_path = content.get("model", {}).get("params", {}).get("ckpt_path")
if ckpt_path and not Path(ckpt_path).exists():
    content.setdefault("model", {}).setdefault("params", {})["ckpt_path"] = None
Path("$BASE_TMP_CFG").write_text(yaml.safe_dump(content, sort_keys=False, allow_unicode=True))
PY
  "$PYTHON" "$ROOT_DIR/main.py" --config_file "$BASE_TMP_CFG"

  # Re-locate latest base checkpoint after training
  LATEST_RUN=$(ls -td "$BASE_LOG_ROOT"/* 2>/dev/null | head -n 1 || true)
  if [ -z "$LATEST_RUN" ]; then
    echo "ERROR: No base run found in $BASE_LOG_ROOT" >&2
    exit 1
  fi

  BASE_CKPT="$LATEST_RUN/checkpoints/last.ckpt"
  if [ ! -f "$BASE_CKPT" ]; then
    BASE_CKPT=$(ls -t "$LATEST_RUN"/checkpoints/*.ckpt 2>/dev/null | head -n 1 || true)
  fi
fi

if [ -z "$BASE_CKPT" ] || [ ! -f "$BASE_CKPT" ]; then
  echo "ERROR: No checkpoint found for base run in $BASE_LOG_ROOT" >&2
  exit 1
fi

if [ "$RESUME_BASE" = "1" ] && [ -n "$LATEST_RUN" ] && [ -f "$BASE_CKPT" ]; then
  echo "Resuming base training from: $BASE_CKPT"
  BASE_TMP_CFG="$TMP_DIR/$(basename "$BASE_CFG")"
  "$PYTHON" - <<PY
import yaml
from pathlib import Path
cfg_path = Path("$BASE_CFG")
ckpt_path = Path("$BASE_CKPT")
content = yaml.safe_load(cfg_path.read_text())
content.setdefault("exp", {})["resume"] = str(ckpt_path)
content.setdefault("exp", {})["logdir"] = "$LOG_ROOT"
content.setdefault("lightning", {}).setdefault("trainer", {})["max_steps"] = int("$BASE_MAX_STEPS")
base_init = content.get("model", {}).get("params", {}).get("ckpt_path")
if base_init and not Path(base_init).exists():
    content.setdefault("model", {}).setdefault("params", {})["ckpt_path"] = None
Path("$BASE_TMP_CFG").write_text(yaml.safe_dump(content, sort_keys=False, allow_unicode=True))
PY
  "$PYTHON" "$ROOT_DIR/main.py" --config_file "$BASE_TMP_CFG"
fi

echo "Using base checkpoint: $BASE_CKPT"

echo "==============================="
echo "Step 2/2: Run stage2 configs"
echo "==============================="

for cfg in "${STAGE2_CONFIGS[@]}"; do
  base_name=$(basename "$cfg")
  tmp_cfg="$TMP_DIR/$base_name"
  resume_flag="$TMP_DIR/${base_name%.yaml}_resume.txt"
  rm -f "$resume_flag"

  "$PYTHON" - <<PY
import glob
import os
import yaml
from pathlib import Path

cfg_path = Path("$cfg")
ckpt_path = Path("$BASE_CKPT")
resume_stage2 = "${RESUME_STAGE2}" == "1"

content = yaml.safe_load(cfg_path.read_text())
content.setdefault("model", {}).setdefault("params", {})["ckpt_path"] = str(ckpt_path)
content.setdefault("exp", {})["logdir"] = "$LOG_ROOT"
content.setdefault("lightning", {}).setdefault("trainer", {})["max_steps"] = int("$STAGE2_MAX_STEPS")

if resume_stage2:
  exp = content.setdefault("exp", {})
  exp_name = exp.get("exp_name")
  log_root = exp.get("logdir", "$LOG_ROOT")
  resume_ckpt = None
  if exp_name:
    exp_dir = os.path.join(log_root, exp_name)
    if os.path.isdir(exp_dir):
      runs = sorted(glob.glob(os.path.join(exp_dir, "*")), key=os.path.getmtime, reverse=True)
      if runs:
        last_ckpt = os.path.join(runs[0], "checkpoints", "last.ckpt")
        if os.path.isfile(last_ckpt):
          resume_ckpt = last_ckpt
        else:
          cand = sorted(glob.glob(os.path.join(runs[0], "checkpoints", "*.ckpt")), key=os.path.getmtime, reverse=True)
          if cand:
            resume_ckpt = cand[0]
  if resume_ckpt:
    exp["resume"] = resume_ckpt
    Path("$resume_flag").write_text(resume_ckpt)

Path("$tmp_cfg").write_text(yaml.safe_dump(content, sort_keys=False, allow_unicode=True))
PY

  echo "==============================="
  echo "Running stage2: $tmp_cfg"
  echo "==============================="
  if [ "$SKIP_STAGE2_IF_EXISTS" = "1" ] && [ -s "$resume_flag" ]; then
    echo "Found existing checkpoint, skip stage2: $(cat "$resume_flag")"
  else
    "$PYTHON" "$ROOT_DIR/main.py" --config_file "$tmp_cfg"
  fi
  echo "Done: $tmp_cfg"
  echo
  sleep 2

done
