#!/bin/bash
####################################################
# Download Datasets
split_and_resize=true
ham_train_rate=0.8
ham_val_rate=0.1
img_resize=512
train_rate=0.9

move_class=true
prepare_VAR=true
check_dataset=true
####################################################
root_path=$(pwd)
# Suspending the script if any command fails
current_user=$(whoami)

# Check data folder
if [ ! -d "$root_path/data" ]; then
    echo "[!] 'data' folder not found, creating it now..."
    mkdir -p "$root_path/data"
fi
dataset_download_dir=$root_path/data/cloud/download

if [ ! -d "$dataset_download_dir"  ]; then
  mkdir -p $dataset_download_dir
fi

dataset_dir=$root_path/data
mkdir -p dataset_dir

############################
# Download datasets        ##
############################
echo "Downloing original datasets..."
# Create folder for each dataset
mkdir -p $dataset_dir/HAM10000/original

# Define function to download and verify files
download_and_verify() {
    local file=$1            # Filename
    local expected_hash=$2   # Expected hash value
    local url=$3             # Download URL

    # Check if file exists
    if [ ! -f "$file" ]; then
        wget -O "$file" "$url"
    fi

    # Calculate current file's md5 hash value
    current_hash=$(md5sum "$file" | awk '{ print $1 }')
    echo "MD5:"$current_hash

    # Compare current calculated hash value with expected hash value
    if [ "$expected_hash" == "SKIP" ]; then
        echo $file" download finished (MD5 verification skipped)."
    elif [ "$current_hash" == "$expected_hash" ]; then
        echo $file" download finished and valied."
    else
        echo $file" hash not mathch with "$expected_hash
        rm -f "$file"  # Delete incomplete file
        exit 1
    fi
}

# HAM10000
cd $dataset_dir/HAM10000/original
## HAM10000 images
file_count=-1
if [ -d "HAM10000_images" ]; then
    file_count=$(ls -1 HAM10000_images | wc -l)
fi
if [ $file_count -ne 10015 ]; then
    cd $dataset_dir/HAM10000/original
    rm -rf HAM10000_images
    cd $dataset_download_dir

    download_and_verify "HAM10000_images_part_1.zip" "4639bfa73ab251610530a97c898e6e46" "https://dataverse.harvard.edu/api/access/datafile/3172585"
    download_and_verify "HAM10000_images_part_2.zip" "da43d6cc50f6613013be07e8986b384b" "https://dataverse.harvard.edu/api/access/datafile/3172584"

    cp $dataset_download_dir/HAM10000_images_part_1.zip $dataset_dir/HAM10000/original/HAM10000_images_part_1.zip
    cp $dataset_download_dir/HAM10000_images_part_2.zip $dataset_dir/HAM10000/original/HAM10000_images_part_2.zip
    cd $dataset_dir/HAM10000/original
    unzip -d HAM10000_images HAM10000_images_part_1.zip
    unzip -d HAM10000_images HAM10000_images_part_2.zip
    rm HAM10000_images_part_1.zip
    rm HAM10000_images_part_2.zip
else
    echo "[√] HAM10000 Images check pass, skipping download."
fi

mkdir -p $dataset_dir/HAM10000/input
cd $dataset_dir/HAM10000/input
if [ ! -f "./HAM10000_metadata.csv" ]; then
    cd $dataset_download_dir
    download_and_verify "HAM10000_metadata.csv" "SKIP" "https://dataverse.harvard.edu/api/access/datafile/4338392?format=original"
    cp $dataset_download_dir/HAM10000_metadata.csv $dataset_dir/HAM10000/input/HAM10000_metadata.csv
else
    echo "[√] HAM10000 meta check pass, skipping download."
fi

## HAM10000 GroundTruth
cd $dataset_dir/HAM10000/original
file_count=-1
if [ -d "HAM10000_GroundTruth" ]; then
    file_count=$(ls -1 HAM10000_GroundTruth | wc -l)
fi
if [ $file_count -ne 10015 ]; then
    rm -rf HAM10000_GroundTruth
    cd $dataset_download_dir
    download_and_verify "HAM10000_segmentations_lesion_tschandl.zip" "6e8d252e09cfdb0189199f15985a5b84" "https://dataverse.harvard.edu/api/access/datafile/3838943"

    cp $dataset_download_dir/HAM10000_segmentations_lesion_tschandl.zip $dataset_dir/HAM10000/original/HAM10000_segmentations_lesion_tschandl.zip
    cd $dataset_dir/HAM10000/original
    unzip -j -d HAM10000_GroundTruth HAM10000_segmentations_lesion_tschandl.zip
    rm HAM10000_segmentations_lesion_tschandl.zip
else
    echo "[√] HAM10000 GroundTruth check pass, skipping download."
fi

echo "✅ HAM10000 dataset is already download."

if [ ! -f "$root_path/data/util/HAM_split_and_resize.py" ]; then
    echo "Error: preprocessing scripts not found under $root_path/data/util"
    exit 1
fi

############################
# Split and resize         #
############################
if [ "$split_and_resize" = true ]; then
    echo "Splitting and resizing HAM10000..."
    echo "* trianning rate: $ham_train_rate"
    echo "* validation rate: $ham_val_rate"
    echo "* resizing image to $img_resize x $img_resize"
    cd $dataset_dir

    rm -rf $dataset_dir/HAM10000/input/test
    rm -rf $dataset_dir/HAM10000/input/train
    rm -rf $dataset_dir/HAM10000/input/val
    python $root_path/data/util/HAM_split_and_resize.py --input-path $dataset_dir/HAM10000/original/  --output-path $dataset_dir/HAM10000/input/ --image-resize $img_resize $img_resize --train-rate $ham_train_rate --val-rate $ham_val_rate
    echo "[√] HAM10000 split and resize done."
    echo "✅ Datasets split and resize finish."
fi
############################
# Move to class folder    ##
############################
if [ "$move_class" = true ]; then
    echo "Start to move images to the class folder..."

    train_val_img_dir="$dataset_dir/HAM10000/input/train_val/HAM10000_img_class/"
    mkdir -p "$train_val_img_dir"
    train_val_seg_dir="$dataset_dir/HAM10000/input/train_val/HAM10000_seg_class/"
    mkdir -p "$train_val_seg_dir"

    # HAM10000 train img
    output_dir="$dataset_dir/HAM10000/input/train/HAM10000_img_class/"
    python $root_path/data/util/class_copy.py --data-name HAM10000_train_img --csv-file $dataset_dir/HAM10000/input/HAM10000_metadata.csv --data-path $dataset_dir/HAM10000/input/train/HAM10000_img/ --out-path $output_dir
    file_count=$(find "$output_dir" -type f | wc -l)
    if [ "$file_count" -ne 8012 ]; then
      echo "Error：Folder: $output_dir files num: $file_count, != 8012"
      exit 1
    fi
    rsync -a "$output_dir" "$train_val_img_dir"

    # HAM10000 train seg
    output_dir="$dataset_dir/HAM10000/input/train/HAM10000_seg_class/"
    python $root_path/data/util/class_copy.py --data-name HAM10000_train_seg --csv-file $dataset_dir/HAM10000/input/HAM10000_metadata.csv --data-path $dataset_dir/HAM10000/input/train/HAM10000_seg/ --out-path $output_dir
    file_count=$(find "$output_dir" -type f | wc -l)
    if [ "$file_count" -ne 8012 ]; then
      echo "Error：Folder: $output_dir files num: $file_count, != 8012"
      exit 1
    fi
    rsync -a "$output_dir" "$train_val_seg_dir"

    # HAM10000 val img
    output_dir="$dataset_dir/HAM10000/input/val/HAM10000_img_class/"
    python $root_path/data/util/class_copy.py --data-name HAM10000_val_img --csv-file $dataset_dir/HAM10000/input/HAM10000_metadata.csv --data-path $dataset_dir/HAM10000/input/val/HAM10000_img/ --out-path $output_dir
    file_count=$(find "$output_dir" -type f | wc -l)
    if [ "$file_count" -ne 1001 ]; then
      echo "Error：Folder: $output_dir files num: $file_count, != 1001"
      exit 1
    fi
    rsync -a "$output_dir" "$train_val_img_dir"
    # Check $train_val_img_dir
    file_count=$(find "$train_val_img_dir" -type f | wc -l)
    if [ "$file_count" -ne 9013 ]; then
      echo "Error：Folder: $train_val_img_dir files num: $file_count, != 9013"
      exit 1
    fi


    # HAM10000 val seg
    output_dir="$dataset_dir/HAM10000/input/val/HAM10000_seg_class/"
    python $root_path/data/util/class_copy.py --data-name HAM10000_val_seg --csv-file $dataset_dir/HAM10000/input/HAM10000_metadata.csv --data-path $dataset_dir/HAM10000/input/val/HAM10000_seg/ --out-path $output_dir
    file_count=$(find "$output_dir" -type f | wc -l)
    if [ "$file_count" -ne 1001 ]; then
      echo "Error：Folder: $output_dir files num: $file_count, != 1001"
      exit 1
    fi
    rsync -a "$output_dir" "$train_val_seg_dir"
    # Check $train_val_seg_dir
    file_count=$(find "$train_val_seg_dir" -type f | wc -l)
    if [ "$file_count" -ne 9013 ]; then
      echo "Error：Folder: $train_val_seg_dir files num: $file_count, != 9013"
      exit 1
    fi

    # HAM10000 test img
    output_dir="$dataset_dir/HAM10000/input/test/HAM10000_img_class/"
    python $root_path/data/util/class_copy.py --data-name HAM10000_test_img --csv-file $dataset_dir/HAM10000/input/HAM10000_metadata.csv --data-path $dataset_dir/HAM10000/input/test/HAM10000_img/ --out-path $output_dir
    file_count=$(find "$output_dir" -type f | wc -l)
    if [ "$file_count" -ne 1002 ]; then
      echo "Error：Folder: $output_dir files num: $file_count, != 1002"
      exit 1
    fi

    # HAM10000 test seg
    output_dir="$dataset_dir/HAM10000/input/test/HAM10000_seg_class/"
    python $root_path/data/util/class_copy.py --data-name HAM10000_test_seg --csv-file $dataset_dir/HAM10000/input/HAM10000_metadata.csv --data-path $dataset_dir/HAM10000/input/test/HAM10000_seg/ --out-path $output_dir
    file_count=$(find "$output_dir" -type f | wc -l)
    if [ "$file_count" -ne 1002 ]; then
      echo "Error：Folder: $output_dir files num: $file_count, != 1002"
      exit 1
    fi

    # Modify directory hierarchy structure
    folders=("akiec" "bcc" "bkl" "df" "mel" "nv" "vasc")
    target_dirs=("train_val" "train" "test" "val")
    for target_dir in "${target_dirs[@]}"; do
        folder_temp=$dataset_dir/HAM10000/input/$target_dir/
        rm -r $folder_temp/HuggingFace 2>/dev/null
        for folder in "${folders[@]}"; do
            # Create new hierarchy structure
            mkdir -p $folder_temp/HuggingFace/$folder/HAM10000_img_class
            mkdir -p $folder_temp/HuggingFace/$folder/HAM10000_seg_class
            # Move corresponding content to new structure
            cp $folder_temp/HAM10000_img_class/$folder/* $folder_temp/HuggingFace/$folder/HAM10000_img_class/ 2>/dev/null
            cp $folder_temp/HAM10000_seg_class/$folder/* $folder_temp/HuggingFace/$folder/HAM10000_seg_class/ 2>/dev/null
        done
    done

    python $root_path/data/util/HAM_meta_generate.py --root-path $dataset_dir/HAM10000/input

    echo "✅ Dataset class move finish."

fi

############################
# Copy files for VAR    ##
############################

# Define function for file copying
copy_files_for_VAR() {
    local source_base=$1
    local target_base=$2

    # Iterate through subdirectories under HuggingFace directory
    for dir in "$source_base"/*/HAM10000_img_class; do
        # Get corresponding subdirectory name
        local sub_dir=$(basename "$(dirname "$dir")")
        # Build target path
        local target_dir="$target_base/$sub_dir"
        # Create target directory if it doesn't exist
        if [ ! -d "$target_dir" ]; then
            mkdir -p "$target_dir"
        fi
        # Copy files from source directory to target directory
        cp -r "$dir/"* "$target_dir/"
        echo "Copied files from $dir to $target_dir"
    done
}

if [ "$prepare_VAR" = true ]; then
  # Define source and target directories
  source_base=$dataset_dir/HAM10000/input/test/"HuggingFace"
  target_base=$dataset_dir/HAM10000/input/test/"VAR"
  copy_files_for_VAR "$source_base" "$target_base"

  source_base=$dataset_dir/HAM10000/input/val/"HuggingFace"
  target_base=$dataset_dir/HAM10000/input/val/"VAR"
  copy_files_for_VAR "$source_base" "$target_base"

  source_base=$dataset_dir/HAM10000/input/train/"HuggingFace"
  target_base=$dataset_dir/HAM10000/input/train/"VAR"
  copy_files_for_VAR "$source_base" "$target_base"

fi
############################
# Check resize dataset    ##
############################
if [ "$check_dataset" = true ]; then
    echo "Checking image size..."
    cd $root_path
    python $root_path/data/util/check_image_size.py --data-name HAM10000_train_img --data-path $dataset_dir/HAM10000/input/train/HAM10000_img/ --image-size $img_resize $img_resize --image-num 8012
    python $root_path/data/util/check_image_size.py --data-name HAM10000_train_seg --data-path $dataset_dir/HAM10000/input/train/HAM10000_seg/ --image-size $img_resize $img_resize --image-num 8012

    python $root_path/data/util/check_image_size.py --data-name HAM10000_val_img --data-path $dataset_dir/HAM10000/input/val/HAM10000_img/ --image-size $img_resize $img_resize --image-num 1001
    python $root_path/data/util/check_image_size.py --data-name HAM10000_val_seg --data-path $dataset_dir/HAM10000/input/val/HAM10000_seg/ --image-size $img_resize $img_resize --image-num 1001

    python $root_path/data/util/check_image_size.py --data-name HAM10000_test_img --data-path $dataset_dir/HAM10000/input/test/HAM10000_img/ --image-size $img_resize $img_resize --image-num 1002
    python $root_path/data/util/check_image_size.py --data-name HAM10000_test_seg --data-path $dataset_dir/HAM10000/input/test/HAM10000_seg/ --image-size $img_resize $img_resize --image-num 1002

    echo "✅ Dataset check pass."
fi
