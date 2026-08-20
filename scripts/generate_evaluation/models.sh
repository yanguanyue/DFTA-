#!/usr/bin/env bash
MODELS=(
  ArSDM Controlnet DFMGAN Derm-T2IM DreamBooth LesionGen
  Siamese T2I-Adapter flow
)
MASK_MODELS=(ArSDM Controlnet DFMGAN Siamese T2I-Adapter flow)
CLASSES=(akiec bcc bkl df mel nv vasc)
SOURCE_ROOT=/root/autodl-tmp/output/generate
EVAL_ROOT=/root/autodl-tmp/output/generate_evaluation
NORMALIZED_ROOT=$EVAL_ROOT/normalized
