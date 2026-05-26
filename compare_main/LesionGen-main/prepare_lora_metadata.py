import csv
import json
import os

SOURCE_CSV = "/root/autodl-tmp/data/HAM10000/input/metadata_train.csv"
TRAIN_DIR = "/root/autodl-tmp/data/HAM10000/input/train/HAM10000_img_class"
OUTPUT_JSONL = os.path.join(TRAIN_DIR, "metadata.jsonl")

if not os.path.exists(SOURCE_CSV):
    raise FileNotFoundError(f"Missing metadata file: {SOURCE_CSV}")
if not os.path.isdir(TRAIN_DIR):
    raise FileNotFoundError(f"Missing train directory: {TRAIN_DIR}")

prefix = "data/local/HAM10000/input/train/HAM10000_img_class/"

row_count = 0
with open(SOURCE_CSV, "r", newline="") as input_file, open(OUTPUT_JSONL, "w", newline="") as output_file:
    reader = csv.DictReader(input_file)
    if "img_path" not in reader.fieldnames or "prompt" not in reader.fieldnames:
        raise ValueError(f"Unexpected columns: {reader.fieldnames}")

    for row in reader:
        img_path = row["img_path"].replace(prefix, "")
        record = {
            "file_name": img_path,
            "text": row["prompt"],
        }
        output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        row_count += 1

print(f"Wrote {OUTPUT_JSONL} with {row_count} entries")
