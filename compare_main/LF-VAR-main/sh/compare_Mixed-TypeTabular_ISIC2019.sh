#!/bin/bash
##############################
# MixedType Tabular Data Synthesis (ISIC2019)
INIT_MixedTypeTabular=false
Training_MixedTypeTabular=false
Synthesis_MixedTypeTabular=true

home_dir="$HOME"
root_path=$(pwd)
OUTPUT_DIR=$root_path/data/compare_results/MixedTypeTabular
RUN_ROOT_PATH=$root_path/compare_models/run/MixedTypeTabular
DATA_REP_ROOT=$root_path/compare_models/reps/MixedTypeTabular

mkdir -p $OUTPUT_DIR

conda activate tabsyn

if [ "$INIT_MixedTypeTabular" = true ]; then
    mkdir -p $root_path/compare_models/reps/
    cd $root_path/compare_models/reps/

    if [ -d "MixedTypeTabular" ]; then
        echo "[√] MixedTypeTabular repo already Downloaded."
    else
        echo "Downloading MixedTypeTabular repo..."
        git clone https://github.com/amazon-science/tabsyn.git MixedTypeTabular
    fi

    cd $RUN_ROOT_PATH
    python convert_category_to_number.py --path $DATA_REP_ROOT/data/ISIC2019/radiomics_final_test.csv
    python convert_category_to_number.py --path $DATA_REP_ROOT/data/ISIC2019/radiomics_final_train.csv
    python convert_category_to_number.py --path $DATA_REP_ROOT/data/ISIC2019/radiomics_final_trainval.csv
    python convert_category_to_number.py --path $DATA_REP_ROOT/data/ISIC2019/radiomics_final_val.csv

    cd $DATA_REP_ROOT
    python process_dataset.py --dataname ISIC2019
fi

if [ "$Training_MixedTypeTabular" = true ]; then
    cd $DATA_REP_ROOT
    echo "Training MixedTypeTabular [vae]..."
    python main.py --dataname ISIC2019 --method vae --mode train
    echo "Training MixedTypeTabular [tabsyn]..."
    python main.py --dataname ISIC2019 --method tabsyn --mode train
fi

if [ "$Synthesis_MixedTypeTabular" = true ]; then
    cd $DATA_REP_ROOT
    python main.py --dataname ISIC2019 --method tabsyn --mode sample --save_path $OUTPUT_DIR"/ISIC2019_tabsyn.csv"
fi
