#!/usr/bin/env bash
set -euo pipefail
DIR=$(cd "$(dirname "$0")" && pwd)
"$DIR/run_metric.sh"
"$DIR/run_classifier.sh"
"$DIR/run_segmentation.sh"
