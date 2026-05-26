#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/autodl-tmp/compare_main/ArSDM-main}"
PYTHON="${PYTHON:-/root/miniconda3/envs/flow/bin/python}"
SCRIPT="$ROOT_DIR/generate_stage2_samples.py"
LOG_DIR="${LOG_DIR:-/root/autodl-tmp/checkpoint/compare_models/ArSDM/generated_logs}"
LOG_FILE="$LOG_DIR/log.txt"
PID_FILE="$ROOT_DIR/generate_stage2.pid"

mkdir -p "${LOG_DIR}"

# Pass through any extra args to the script
ARGS="$@"

# Run in background with nohup, record pid
nohup "$PYTHON" "$SCRIPT" $ARGS > "$LOG_FILE" 2>&1 &
PID=$!

echo $PID > "$PID_FILE"
echo "Started generation (pid=$PID). Logs -> $LOG_FILE"
