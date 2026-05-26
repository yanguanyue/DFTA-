#!/bin/bash
####################################################
# Download Datasets (ISIC2017)
split_and_resize=true
img_resize=512
prepare_isic2017_class=true
prepare_ph2_class=true
check_dataset=true
####################################################
# Override this to use a faster mirror, e.g.
# export ISIC2017_BASE_URL="https://isic-challenge-data.s3.us-east-1.amazonaws.com/2017"
isic2017_base_url=${ISIC2017_BASE_URL:-"https://isic-challenge-data.s3.us-east-1.amazonaws.com/2017"}
# Compute paths relative to the script location so the script
# works when invoked from anywhere.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_path="$(cd "$script_dir/../.." && pwd)"
current_user=$(whoami)

# Check data folder
if [ ! -d "$root_path/data" ]; then
    echo "[!] 'data' folder not found, creating it now..."
    mkdir -p "$root_path/data"
fi

dataset_download_dir=$root_path/data/cloud/download
mkdir -p "$dataset_download_dir"

dataset_dir=$root_path/data
mkdir -p "$dataset_dir"

############################
# Helpers                  #
############################
download_and_verify() {
    local file=$1            # Filename
    local expected_hash=$2   # Expected hash value (optional)
    local url=$3             # Download URL

    if [ ! -f "$file" ]; then
        wget -O "$file" "$url"
    fi

    if [ -n "$expected_hash" ]; then
        current_hash=$(md5sum "$file" | awk '{ print $1 }')
        echo "MD5:"$current_hash

        if [ "$current_hash" == "$expected_hash" ]; then
            echo $file" download finished and valied."
        else
            echo $file" hash not mathch with "$expected_hash
            rm -f "$file"
            exit 1
        fi
    else
        echo "[i] Skip MD5 check for $file"
    fi
}

download_git_repo_archive() {
    local repo_url=$1
    local archive_path=$2

    if [ -f "$archive_path" ]; then
        echo "[√] Archive already exists: $archive_path"
        return
    fi

    if ! command -v git >/dev/null 2>&1; then
        echo "[!] git not found. Please install git to download $repo_url"
        exit 1
    fi

    local tmp_dir
    tmp_dir=$(mktemp -d)

    echo "Downloading $repo_url ..."
    git clone --depth 1 "$repo_url" "$tmp_dir/repo"
    tar -czf "$archive_path" -C "$tmp_dir/repo" .
    rm -rf "$tmp_dir"
    echo "[√] Saved archive to $archive_path"
}

dir_has_files() {
    local dir=$1
    if [ -d "$dir" ] && [ "$(ls -A "$dir" 2>/dev/null)" ]; then
        return 0
    fi
    return 1
}

count_files() {
    local dir=$1
    if [ ! -d "$dir" ]; then
        echo 0
        return
    fi
    find "$dir" -type f \( -name "*.jpg" -o -name "*_segmentation.png" \) | wc -l
}

isic2017_processed() {
    local all_img="$dataset_dir/ISIC2017/input/ISIC2017_img"
    local all_seg="$dataset_dir/ISIC2017/input/ISIC2017_seg"

    [ "$(count_files "$all_img")" -eq 2750 ] || return 1
    [ "$(count_files "$all_seg")" -eq 2750 ] || return 1
    return 0
}

ph2_processed() {
    local img_dir="$dataset_dir/PH2/input/PH2_img"
    local seg_dir="$dataset_dir/PH2/input/PH2_seg"

    if dir_has_files "$img_dir" && dir_has_files "$seg_dir"; then
        return 0
    fi
    return 1
}

############################
# Extra datasets (gitcode) #
############################
download_git_repo_archive \
    "https://gitcode.com/open-source-toolkit/3f18e.git" \
    "$dataset_download_dir/PH2.tar.gz"

############################
# ISIC2017                 #
############################
if isic2017_processed; then
    echo "[√] ISIC2017 already processed, skipping all steps."
else
    mkdir -p "$dataset_dir/ISIC2017/original" "$dataset_dir/ISIC2017/input"

    if dir_has_files "$dataset_dir/ISIC2017/original"; then
        echo "[√] ISIC2017 original data exists, skipping download."
    else
        echo "Downloading ISIC2017 datasets..."
        cd "$dataset_download_dir"

    download_and_verify "ISIC-2017_Training_Data.zip" "a14a7e622c67a358797ae59abb8a0b0c" "$isic2017_base_url/ISIC-2017_Training_Data.zip"
    download_and_verify "ISIC-2017_Training_Part1_GroundTruth.zip" "77fdbeb6fbec4139937224416b250f4c" "$isic2017_base_url/ISIC-2017_Training_Part1_GroundTruth.zip"
    download_and_verify "ISIC-2017_Validation_Data.zip" "8d6419d942112f709894c0d82f6c9038" "$isic2017_base_url/ISIC-2017_Validation_Data.zip"
    download_and_verify "ISIC-2017_Validation_Part1_GroundTruth.zip" "64d3e68fa2deeb8a5e89aa8dec2efd44" "$isic2017_base_url/ISIC-2017_Validation_Part1_GroundTruth.zip"
    download_and_verify "ISIC-2017_Test_v2_Data.zip" "5f6a0b5e1f2972bd1f5ea02680489f09" "$isic2017_base_url/ISIC-2017_Test_v2_Data.zip"
    download_and_verify "ISIC-2017_Test_v2_Part1_GroundTruth.zip" "b1742de6bd257faca3b2b21a4aa3b781" "$isic2017_base_url/ISIC-2017_Test_v2_Part1_GroundTruth.zip"

        cp "$dataset_download_dir"/ISIC-2017_Training_Data.zip "$dataset_dir/ISIC2017/original/ISIC-2017_Training_Data.zip"
        cp "$dataset_download_dir"/ISIC-2017_Training_Part1_GroundTruth.zip "$dataset_dir/ISIC2017/original/ISIC-2017_Training_Part1_GroundTruth.zip"
        cp "$dataset_download_dir"/ISIC-2017_Validation_Data.zip "$dataset_dir/ISIC2017/original/ISIC-2017_Validation_Data.zip"
        cp "$dataset_download_dir"/ISIC-2017_Validation_Part1_GroundTruth.zip "$dataset_dir/ISIC2017/original/ISIC-2017_Validation_Part1_GroundTruth.zip"
        cp "$dataset_download_dir"/ISIC-2017_Test_v2_Data.zip "$dataset_dir/ISIC2017/original/ISIC-2017_Test_v2_Data.zip"
        cp "$dataset_download_dir"/ISIC-2017_Test_v2_Part1_GroundTruth.zip "$dataset_dir/ISIC2017/original/ISIC-2017_Test_v2_Part1_GroundTruth.zip"

        cd "$dataset_dir/ISIC2017/original"
        unzip -d ISIC_2017_Training ISIC-2017_Training_Data.zip
        unzip -d ISIC_2017_Training ISIC-2017_Training_Part1_GroundTruth.zip
    wget -O ISIC-2017_Training_Part3_GroundTruth.csv "$isic2017_base_url/ISIC-2017_Training_Part3_GroundTruth.csv"
        unzip -d ISIC_2017_Validation ISIC-2017_Validation_Data.zip
        unzip -d ISIC_2017_Validation ISIC-2017_Validation_Part1_GroundTruth.zip
        unzip -d ISIC_2017_Test ISIC-2017_Test_v2_Data.zip
        unzip -d ISIC_2017_Test ISIC-2017_Test_v2_Part1_GroundTruth.zip

    wget -O ISIC-2017_Validation_Part3_GroundTruth.csv "$isic2017_base_url/ISIC-2017_Validation_Part3_GroundTruth.csv"
    wget -O ISIC-2017_Test_v2_Part3_GroundTruth.csv "$isic2017_base_url/ISIC-2017_Test_v2_Part3_GroundTruth.csv"

        rm ISIC-2017_Training_Data.zip
        rm ISIC-2017_Training_Part1_GroundTruth.zip
        rm ISIC-2017_Validation_Data.zip
        rm ISIC-2017_Validation_Part1_GroundTruth.zip
        rm ISIC-2017_Test_v2_Data.zip
        rm ISIC-2017_Test_v2_Part1_GroundTruth.zip
    fi

    if [ "$split_and_resize" = true ]; then
        echo "Resizing ISIC2017..."
        rm -rf "$dataset_dir/ISIC2017/input/ISIC2017_img"
        rm -rf "$dataset_dir/ISIC2017/input/ISIC2017_seg"

        python "$root_path/data/util/resize_images.py" --input-path "$dataset_dir/ISIC2017/original/ISIC_2017_Training/ISIC-2017_Training_Data" --output-path "$dataset_dir/ISIC2017/input/ISIC2017_img" --image-resize $img_resize $img_resize
        python "$root_path/data/util/resize_images.py" --input-path "$dataset_dir/ISIC2017/original/ISIC_2017_Training/ISIC-2017_Training_Part1_GroundTruth" --output-path "$dataset_dir/ISIC2017/input/ISIC2017_seg" --image-resize $img_resize $img_resize --is-seg True
        echo "[√] ISIC2017 training resize done."

        python "$root_path/data/util/resize_images.py" --input-path "$dataset_dir/ISIC2017/original/ISIC_2017_Validation/ISIC-2017_Validation_Data" --output-path "$dataset_dir/ISIC2017/input/ISIC2017_img" --image-resize $img_resize $img_resize
        python "$root_path/data/util/resize_images.py" --input-path "$dataset_dir/ISIC2017/original/ISIC_2017_Validation/ISIC-2017_Validation_Part1_GroundTruth" --output-path "$dataset_dir/ISIC2017/input/ISIC2017_seg" --image-resize $img_resize $img_resize --is-seg True
        echo "[√] ISIC2017 val resize done."

        python "$root_path/data/util/resize_images.py" --input-path "$dataset_dir/ISIC2017/original/ISIC_2017_Test/ISIC-2017_Test_v2_Data" --output-path "$dataset_dir/ISIC2017/input/ISIC2017_img" --image-resize $img_resize $img_resize
        python "$root_path/data/util/resize_images.py" --input-path "$dataset_dir/ISIC2017/original/ISIC_2017_Test/ISIC-2017_Test_v2_Part1_GroundTruth" --output-path "$dataset_dir/ISIC2017/input/ISIC2017_seg" --image-resize $img_resize $img_resize --is-seg True
        echo "[√] ISIC2017 test resize done."
    fi

    if [ "$prepare_isic2017_class" = true ]; then
        if [ -f "$dataset_dir/ISIC2017/original/ISIC-2017_Training_Part3_GroundTruth.csv" ]; then
            echo "Preparing ISIC2017 class folders from ground-truth CSV (train)..."
            python "$root_path/data/util/isic2017_prepare_from_csv.py" \
                --image_dir "$dataset_dir/ISIC2017/input/ISIC2017_img" \
                --seg_dir "$dataset_dir/ISIC2017/input/ISIC2017_seg" \
                --groundtruth_csv "$dataset_dir/ISIC2017/original/ISIC-2017_Training_Part3_GroundTruth.csv" \
                --output "$dataset_dir/ISIC2017/input" \
                --split all \
                --append_metadata
        else
            echo "[!] ISIC2017 class preparation skipped: training ground-truth CSV not found."
        fi

        if [ -f "$dataset_dir/ISIC2017/original/ISIC-2017_Validation_Part3_GroundTruth.csv" ]; then
            echo "Preparing ISIC2017 class folders from ground-truth CSV (val)..."
            python "$root_path/data/util/isic2017_prepare_from_csv.py" \
                --image_dir "$dataset_dir/ISIC2017/input/ISIC2017_img" \
                --seg_dir "$dataset_dir/ISIC2017/input/ISIC2017_seg" \
                --groundtruth_csv "$dataset_dir/ISIC2017/original/ISIC-2017_Validation_Part3_GroundTruth.csv" \
                --output "$dataset_dir/ISIC2017/input" \
                --split all \
                --append_metadata
        else
            echo "[!] ISIC2017 val class preparation skipped: validation ground-truth CSV not found."
        fi

        if [ -f "$dataset_dir/ISIC2017/original/ISIC-2017_Test_v2_Part3_GroundTruth.csv" ]; then
            echo "Preparing ISIC2017 class folders from ground-truth CSV (test)..."
            python "$root_path/data/util/isic2017_prepare_from_csv.py" \
                --image_dir "$dataset_dir/ISIC2017/input/ISIC2017_img" \
                --seg_dir "$dataset_dir/ISIC2017/input/ISIC2017_seg" \
                --groundtruth_csv "$dataset_dir/ISIC2017/original/ISIC-2017_Test_v2_Part3_GroundTruth.csv" \
                --output "$dataset_dir/ISIC2017/input" \
                --split all \
                --append_metadata
        else
            echo "[!] ISIC2017 test class preparation skipped: test ground-truth CSV not found."
        fi
    fi
fi

############################
# PH2                      #
############################
if ph2_processed; then
    echo "[√] PH2 already processed, skipping all steps."
else
    ph2_original_dir="$dataset_dir/PH2/original"
    ph2_input_dir="$dataset_dir/PH2/input"
    mkdir -p "$ph2_original_dir" "$ph2_input_dir"

    if [ ! -f "$dataset_download_dir/PH2.tar.gz" ]; then
        echo "[!] PH2 archive not found: $dataset_download_dir/PH2.tar.gz"
        exit 1
    fi

    if [ ! -d "$ph2_original_dir/PH2Dataset" ] || [ ! -f "$ph2_original_dir/PH2_dataset.txt" ]; then
        echo "Extracting PH2 dataset..."
        tmp_dir=$(mktemp -d)
        tar -xzf "$dataset_download_dir/PH2.tar.gz" -C "$tmp_dir"

        rar_path=$(find "$tmp_dir" -maxdepth 2 -name "PH2 Dataset.rar" | head -n 1)
        if [ -z "$rar_path" ]; then
            echo "[!] PH2 Dataset.rar not found inside PH2.tar.gz"
            rm -rf "$tmp_dir"
            exit 1
        fi

        if command -v unrar >/dev/null 2>&1; then
            unrar x -o+ "$rar_path" "$ph2_original_dir"
        elif command -v 7z >/dev/null 2>&1; then
            7z x -y "$rar_path" -o"$ph2_original_dir"
        else
            echo "[!] unrar/7z not found. Please install one of them to extract PH2 Dataset.rar"
            rm -rf "$tmp_dir"
            exit 1
        fi

        dataset_txt=$(find "$ph2_original_dir" -maxdepth 3 -name "PH2_dataset.txt" | head -n 1)
        if [ -n "$dataset_txt" ] && [ "$dataset_txt" != "$ph2_original_dir/PH2_dataset.txt" ]; then
            cp "$dataset_txt" "$ph2_original_dir/PH2_dataset.txt"
        fi

        rm -rf "$tmp_dir"
    fi

    if [ ! -d "$ph2_original_dir/PH2_Data" ] || [ ! -d "$ph2_original_dir/PH2_GroundTruth" ]; then
        if [ -f "$root_path/data/util/PH2_prepare.py" ]; then
            echo "Preparing PH2 images and masks..."
            python "$root_path/data/util/PH2_prepare.py" \
                --input-dir "$ph2_original_dir" \
                --output-dir "$ph2_original_dir"
        else
            echo "[!] PH2_prepare.py not found. Expected at data/util/PH2_prepare.py"
            exit 1
        fi
    fi

    if [ "$split_and_resize" = true ]; then
        echo "Resizing PH2..."
        rm -rf "$ph2_input_dir/PH2_img"
        rm -rf "$ph2_input_dir/PH2_seg"

        python "$root_path/data/util/resize_images.py" --input-path "$ph2_original_dir/PH2_Data" --output-path "$ph2_input_dir/PH2_img" --image-resize $img_resize $img_resize
        python "$root_path/data/util/resize_images.py" --input-path "$ph2_original_dir/PH2_GroundTruth" --output-path "$ph2_input_dir/PH2_seg" --image-resize $img_resize $img_resize --is-seg True
        echo "[√] PH2 resize done."
    fi

    if [ "$prepare_ph2_class" = true ]; then
        if [ -f "$ph2_original_dir/PH2_dataset.txt" ]; then
            echo "Preparing PH2 class folders from PH2_dataset.txt (merge Common/Atypical Nevus -> nv)..."
            rm -rf "$ph2_input_dir/img_class" "$ph2_input_dir/seg_class" "$ph2_input_dir/metadata.csv"
            python "$root_path/data/util/ph2_prepare_from_txt.py" \
                --image_dir "$ph2_input_dir/PH2_img" \
                --seg_dir "$ph2_input_dir/PH2_seg" \
                --groundtruth_txt "$ph2_original_dir/PH2_dataset.txt" \
                --output "$ph2_input_dir" \
                --split all
        else
            echo "[!] PH2 class preparation skipped: PH2_dataset.txt not found."
        fi
    fi
fi

############################
# Check dataset size       #
############################
if [ "$check_dataset" = true ]; then
    echo "Checking dataset size..."
    if [ -d "$dataset_dir/ISIC2017/input/ISIC2017_img" ]; then
        python "$root_path/data/util/check_image_size.py" --data-name ISIC2017_all_img --data-path "$dataset_dir/ISIC2017/input/ISIC2017_img/" --image-size $img_resize $img_resize --image-num 2750
        python "$root_path/data/util/check_image_size.py" --data-name ISIC2017_all_seg --data-path "$dataset_dir/ISIC2017/input/ISIC2017_seg/" --image-size $img_resize $img_resize --image-num 2750
    else
        echo "[!] ISIC2017 input not found, skipping size check."
    fi

    echo "✅ Dataset check finished."
fi
