import urllib.request
import os
import subprocess
import sys
import zipfile

base = "/root/autodl-tmp/data/ISIC2017/original"
input_base = "/root/autodl-tmp/data/ISIC2017/input"
download_dir = "/root/autodl-tmp/data/cloud/download"
util_dir = "/root/autodl-tmp/data/util"

os.makedirs(base, exist_ok=True)

csv_urls = {
    "ISIC-2017_Validation_Part3_GroundTruth.csv": "https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Validation_Part3_GroundTruth.csv",
    "ISIC-2017_Test_v2_Part3_GroundTruth.csv": "https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Test_v2_Part3_GroundTruth.csv",
}

print("=== Step 1: Download missing Part3 GroundTruth CSVs ===")
for fname, url in csv_urls.items():
    dest = os.path.join(base, fname)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"[√] {fname} already exists ({os.path.getsize(dest)} bytes), skipping.")
        continue
    print(f"Downloading {fname} ...")
    urllib.request.urlretrieve(url, dest)
    size = os.path.getsize(dest)
    print(f"  -> Saved {fname} ({size} bytes)")

print("\n=== Step 2: Ensure ISIC_2017_Validation is unzipped ===")
val_dir = os.path.join(base, "ISIC_2017_Validation")
if os.path.isdir(val_dir) and os.listdir(val_dir):
    print(f"[√] {val_dir} already exists, skipping unzip.")
else:
    print(f"ISIC_2017_Validation not found or empty, unzipping from cloud/download/ ...")
    val_data_zip = os.path.join(download_dir, "ISIC-2017_Validation_Data.zip")
    val_gt_zip = os.path.join(download_dir, "ISIC-2017_Validation_Part1_GroundTruth.zip")
    for zp, target_subdir in [(val_data_zip, "ISIC-2017_Validation_Data"), (val_gt_zip, "ISIC-2017_Validation_Part1_GroundTruth")]:
        if not os.path.isfile(zp):
            print(f"  [!] Zip not found: {zp}, skipping.")
            continue
        extract_to = os.path.join(val_dir, target_subdir)
        os.makedirs(extract_to, exist_ok=True)
        with zipfile.ZipFile(zp, 'r') as zf:
            zf.extractall(extract_to)
        print(f"  [√] Unzipped {os.path.basename(zp)} -> {extract_to}")

print("\n=== Step 3: Prepare val class folders ===")
val_csv = os.path.join(base, "ISIC-2017_Validation_Part3_GroundTruth.csv")
if os.path.exists(val_csv) and os.path.getsize(val_csv) > 0:
    val_img = os.path.join(input_base, "val/ISIC2017_img")
    val_seg = os.path.join(input_base, "val/ISIC2017_seg")
    if not os.path.isdir(val_img):
        print(f"  [!] Val image dir not found: {val_img}")
        print(f"  Skipping val class preparation (need resize first).")
    else:
        ret = subprocess.run([
            sys.executable, os.path.join(util_dir, "isic2017_prepare_from_csv.py"),
            "--image_dir", val_img,
            "--seg_dir", val_seg,
            "--groundtruth_csv", val_csv,
            "--output", input_base,
            "--split", "val",
        ])
        print(f"  Val preparation exit code: {ret.returncode}")
else:
    print(f"  [!] Val CSV not found: {val_csv}, skipping.")

print("\n=== Step 4: Prepare test class folders ===")
test_csv = os.path.join(base, "ISIC-2017_Test_v2_Part3_GroundTruth.csv")
if os.path.exists(test_csv) and os.path.getsize(test_csv) > 0:
    test_img = os.path.join(input_base, "test/ISIC2017_img")
    test_seg = os.path.join(input_base, "test/ISIC2017_seg")
    if not os.path.isdir(test_img):
        print(f"  [!] Test image dir not found: {test_img}")
        print(f"  Skipping test class preparation (need resize first).")
    else:
        ret = subprocess.run([
            sys.executable, os.path.join(util_dir, "isic2017_prepare_from_csv.py"),
            "--image_dir", test_img,
            "--seg_dir", test_seg,
            "--groundtruth_csv", test_csv,
            "--output", input_base,
            "--split", "test",
        ])
        print(f"  Test preparation exit code: {ret.returncode}")
else:
    print(f"  [!] Test CSV not found: {test_csv}, skipping.")

print("\n✅ All done! Check results:")
for split in ["val", "test"]:
    hf_dir = os.path.join(input_base, split, "HuggingFace")
    if os.path.isdir(hf_dir):
        classes = os.listdir(hf_dir)
        print(f"  {split}/HuggingFace/: {classes}")
    else:
        print(f"  {split}/HuggingFace/: NOT FOUND")
