import csv
import os

BASE_DIR = "/root/autodl-tmp/data/HAM10000/input"
SPLITS_DIR = os.path.join(BASE_DIR, "splits")

FILES = {
    "train": os.path.join(BASE_DIR, "metadata_train.csv"),
    "val": os.path.join(BASE_DIR, "metadata_val.csv"),
    "test": os.path.join(BASE_DIR, "metadata_test.csv"),
}

REPLACE_PREFIX = "data/local/HAM10000/input"

os.makedirs(SPLITS_DIR, exist_ok=True)

for split, csv_path in FILES.items():
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing {split} metadata: {csv_path}")

    output_path = os.path.join(SPLITS_DIR, f"{split}.csv")
    row_count = 0

    with open(csv_path, "r", newline="") as input_file, open(output_path, "w", newline="") as output_file:
        reader = csv.DictReader(input_file)
        if "img_path" not in reader.fieldnames or "class" not in reader.fieldnames:
            raise ValueError(f"Unexpected columns in {csv_path}: {reader.fieldnames}")

        writer = csv.DictWriter(output_file, fieldnames=["img_path", "dx"])
        writer.writeheader()

        for row in reader:
            img_path = row["img_path"].replace(REPLACE_PREFIX, BASE_DIR)
            writer.writerow({"img_path": img_path, "dx": row["class"]})
            row_count += 1

    print(f"Wrote {output_path} ({row_count} rows)")
