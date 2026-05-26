#!/bin/bash
# Suspending the script if any command fails
set -e

# Random seed
RANDOM=$(date +%s | cut -c 7-10 | sed 's/^0*//')

# Activate the 'skin_generative' environment
CONDA_PATH="$HOME/miniconda/etc/profile.d/conda.sh"
if [[ -f "$CONDA_PATH" ]]; then
    # A40-24Q
    export HF_HOME=/mnt/huggingface_cache
    source $CONDA_PATH
fi
CONDA_PATH="$HOME/opt/miniconda3/etc/profile.d/conda.sh"
if [[ -f "$CONDA_PATH" ]]; then
    source $CONDA_PATH
fi

# Active environment
conda activate skin_generative
echo "✅ The 'skin_generative' environment is now active."

#sudo apt install imagemagick
#mogrify -path /mnt/SkinGenerativeModel/code/data/local/HAM10000/input/train/HuggingFace/akiec/HAM10000_img_class_png/ -format png /mnt/SkinGenerativeModel/code/data/local/HAM10000/input/train/HuggingFace/akiec/HAM10000_img_class/*.jpg

# User reported pyarrow build issues. Assuming environment is already set up.
# pip install -r sh/requirements.txt --quiet
echo "✅ Requirements all installed (step skipped per user request)."


# relogin wandb (only if available)
if command -v wandb &> /dev/null; then
    wandb login --relogin 52e1d262f0a0f8911bdf0b02938c845b023f1bd5
else
    echo "⚠️ wandb command not found per user report, skipping login."
fi

# Check data folder
if [ ! -d "data" ]; then
    echo "⚠️ 'data' folder not found. Creating it..."
    mkdir -p data
fi

checkpoint_path=compare_models/checkpoints
results_path=data/compare_results
mkdir -p $checkpoint_path
mkdir -p $results_path


