PYTHON=/root/autodl-tmp/experiments/dermofit_compare_20260805/venv_infer_cu124/bin/python
EVAL_ROOT=/root/autodl-tmp/output/generate_evaluation/2017
NORMALIZED_ROOT=$EVAL_ROOT/normalized; REAL_ROOT=$EVAL_ROOT/real_data
MODELS=(ArSDM Controlnet DFMGAN Derm-T2IM DreamBooth LesionGen Siamese T2I-Adapter flow)
MASK_MODELS=(ArSDM Controlnet DFMGAN Siamese T2I-Adapter flow)
CLASSES=(bkl mel nv); CLASS_CSV=bkl,mel,nv
MAJORITY_CAP=500; MINORITY_CAP=1500
SEEDS=(${SEEDS:-42 123 2024 3407 7777}); ARCHS=(${ARCHS:-resnet18 resnet50 efficientnet_b0}); SEG_MODELS=(${SEG_MODELS:-unet segformer})
EPOCHS=${EPOCHS:-30}; TRAIN_BATCH=${TRAIN_BATCH:-64}; TEST_BATCH=${TEST_BATCH:-64}; SEG_BATCH=${SEG_BATCH:-4}; SEG_STEPS=${SEG_STEPS:-15000}; SEG_VAL_INTERVAL=${SEG_VAL_INTERVAL:-3000}
