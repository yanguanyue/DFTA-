#!/bin/bash
set -e

##############################
# LF-VAR Compare Script
##############################

# Mode: test (1 image per category) or full (1500 images per category)
MODE=${1:-"test"}

home_dir="$HOME"
root_path=$(pwd)
RUN_ROOT_PATH=$root_path/compare_main/LF-VAR-main/main
checkpoint_path=$root_path/checkpoint/compare_models/LF-VAR
output_path=$root_path/output/generate/LF-VAR
dataset_dir=$root_path/data/HAM10000/input

mkdir -p $checkpoint_path
mkdir -p $output_path

echo "========================================="
echo "LF-VAR Training and Generation Pipeline"
echo "Mode: $MODE"
echo "Checkpoint dir: $checkpoint_path"
echo "Output dir: $output_path"
echo "========================================="

# Download pretrained VAE checkpoint if not exists
cd $RUN_ROOT_PATH
export HF_ENDPOINT="https://hf-mirror.com"

if [ ! -f "vae_ch160v4096z32.pth" ]; then
    echo "Downloading VAE checkpoint from mirror..."
    wget -c -O "vae_ch160v4096z32.pth" "https://hf-mirror.com/FoundationVision/var/resolve/main/vae_ch160v4096z32.pth"
    echo "VAE checkpoint download complete."
else
    echo "VAE checkpoint exists."
fi

# Download VAR pretrained checkpoint if not exists
if [ ! -f "var_d16.pth" ]; then
    echo "Downloading VAR checkpoint from mirror..."
    wget -c -O "var_d16.pth" "https://hf-mirror.com/FoundationVision/var/resolve/main/var_d16.pth"
    echo "VAR checkpoint download complete."
else
    echo "VAR checkpoint exists."
fi

##############################
# Phase 1: Training
##############################
echo ""
echo "========================================="
echo "Phase 1: Training LF-VAR Model"
echo "========================================="

cd $RUN_ROOT_PATH

if [ "$MODE" = "test" ]; then
    echo "Running in TEST mode - 5 epochs for quick validation"
    TRAIN_EPOCHS=1
    TRAIN_BS=2
    GPU_SUM=1
    GPU_ID=0
else
    echo "Running in FULL mode - 200 epochs"
    TRAIN_EPOCHS=200
    TRAIN_BS=35
    GPU_SUM=1
    GPU_ID=0
fi

echo "Training with epochs=$TRAIN_EPOCHS, batch_size=$TRAIN_BS"
echo "Checkpoints will be saved to: $checkpoint_path"

PYTHON_PATH=$(which python)
MASTER_PORT=$((RANDOM % (65535 - 1024 + 1) + 1024))

$PYTHON_PATH -m torch.distributed.run --nproc_per_node=$GPU_SUM --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=$MASTER_PORT train.py \
    --depth=16 \
    --bs=$TRAIN_BS \
    --ep=$TRAIN_EPOCHS \
    --fp16=1 \
    --alng=1e-3 \
    --wpe=0.1 \
    --local_out_dir_path=$checkpoint_path \
    --data_path=$dataset_dir

echo "Training phase completed!"
echo "Checkpoints saved to: $checkpoint_path"

##############################
# Phase 2: Image Generation
##############################
echo ""
echo "========================================="
echo "Phase 2: Generating Images"
echo "========================================="

cd $RUN_ROOT_PATH

MASTER_PORT=$((RANDOM % (65535 - 1024 + 1) + 1024))

if [ "$MODE" = "test" ]; then
    echo "Running in TEST mode - generating 1 image per category"
    CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON_PATH -m torch.distributed.run --nproc_per_node=$GPU_SUM --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=$MASTER_PORT infer.py \
        --depth=16 \
        --bs=1 \
        --ep=1 \
        --fp16=1 \
        --alng=1e-3 \
        --wpe=0.1 \
        --local_out_dir_path=$output_path \
        --data_path=$dataset_dir
else
    echo "Running in FULL mode - generating 1500 images per category"
    CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON_PATH -m torch.distributed.run --nproc_per_node=$GPU_SUM --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=$MASTER_PORT infer.py \
        --depth=16 \
        --bs=2 \
        --ep=200 \
        --fp16=1 \
        --alng=1e-3 \
        --wpe=0.1 \
        --local_out_dir_path=$output_path \
        --data_path=$dataset_dir
fi

echo ""
echo "========================================="
echo "Pipeline completed successfully!"
echo "Generated images saved to: $output_path"
echo "========================================="
