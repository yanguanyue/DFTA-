#!/usr/bin/env bash
set -euo pipefail
DIR=/root/autodl-tmp/scripts/ablation/complete
"$DIR/run_classifier_ai.sh"
"$DIR/run_segmentation_ai.sh"
