#!/bin/bash
set -e
##############################
# MAIN (ISIC2019 + TabSyn)
COMPARE_MAIN=true
inference_MAIN=true
cross_inference_MAIN_tabsyn=true
##############################
root_path=$(pwd)
OUTPUT_DIR=$root_path/data/compare_results/main_isic2019_tabsyn
DATASET_DIR=$root_path/data/local/ISIC2019/input
RUN_ROOT_PATH=$root_path/main

mkdir -p $OUTPUT_DIR

if [ "$COMPARE_MAIN" = true ]; then
    echo "Comparing MAIN (ISIC2019 + TabSyn)..."

    if [ "$inference_MAIN" = true ]; then
      cd $OUTPUT_DIR/
      rm -rf akiec bcc bkl df mel nv vasc

      cd $RUN_ROOT_PATH
      MASTER_PORT=$((RANDOM % (65535 - 1024 + 1) + 1024))
      CUDA_VISIBLE_DEVICES=$gpu_id python infer.py \
      --depth=16 --bs=2 --ep=200 --fp16=1 --alng=1e-3 --wpe=0.1 \
      --local_out_dir_path $OUTPUT_DIR --data_path $DATASET_DIR
    fi

    if [ "$cross_inference_MAIN_tabsyn" = true ]; then
      tabsyn_csv=$root_path/data/compare_results/MixedTypeTabular/ISIC2019_tabsyn.csv
      cd $RUN_ROOT_PATH
      MASTER_PORT=$((RANDOM % (65535 - 1024 + 1) + 1024))
      CUDA_VISIBLE_DEVICES=$gpu_id torchrun --nproc_per_node=$gpu_sum --nnodes=1 --node_rank=0 \
      --master_addr="127.0.0.1" --master_port=$MASTER_PORT infer.py \
      --depth=16 --bs=35 --ep=200 --fp16=1 --alng=1e-3 --wpe=0.1 \
      --local_out_dir_path $OUTPUT_DIR --data_path $DATASET_DIR \
      --tabsyn_csv_path $tabsyn_csv --cross_infer true
    fi
fi
