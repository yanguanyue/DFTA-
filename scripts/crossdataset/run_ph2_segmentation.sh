#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-/root/miniconda3/envs/flow/bin/python}

if [ ! -x "$PYTHON" ]; then
  PYTHON=python
fi

"$PYTHON" /root/autodl-tmp/scripts/crossdataset/eval_ph2_segmentation.py "$@"
