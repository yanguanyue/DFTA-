#!/usr/bin/env bash
set -euo pipefail
DIR=$(cd "$(dirname "$0")" && pwd)
case "${1:-all}" in
  metric) "$DIR/run_metric_ai.sh" ;;
  classifier) "$DIR/run_classifier_ai.sh" ;;
  segmentation) "$DIR/run_segmentation_ai.sh" ;;
  all) "$DIR/run_metric_ai.sh"; "$DIR/run_classifier_ai.sh"; "$DIR/run_segmentation_ai.sh" ;;
  *) echo "Usage: $0 [metric|classifier|segmentation|all]"; exit 2 ;;
esac
