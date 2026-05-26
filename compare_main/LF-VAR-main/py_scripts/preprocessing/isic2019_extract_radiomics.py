import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Extract ISIC2019 radiomics features")
    parser.add_argument("--metadata", default="data/local/ISIC2019/input/metadata.csv")
    parser.add_argument("--settings", default="main/radiomics/feature_extract_setting.yaml")
    parser.add_argument("--output", default="data/local/ISIC2019/input/radiomics_final.csv")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for debugging")
    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    df = pd.read_csv(metadata_path)
    if not {"img_path", "seg_path", "class"}.issubset(df.columns):
        raise ValueError("metadata.csv must contain img_path, seg_path, class")

    if args.limit:
        df = df.head(args.limit)

    feature_rows = []
    feature_names = [
        "original_firstorder_Mean",
        "original_firstorder_Std",
        "original_firstorder_Minimum",
        "original_firstorder_Maximum",
        "original_shape2D_PixelSurface",
        "original_shape2D_Sphericity",
        "original_shape2D_Elongation",
        "original_shape2D_Perimeter",
    ]

    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = Path(row["img_path"])
        seg_path = Path(row["seg_path"])
        if not img_path.exists() or not seg_path.exists():
            continue

        with Image.open(img_path) as img:
            img = img.convert("RGB")
            img_array = np.asarray(img, dtype=np.float32)
        with Image.open(seg_path) as msk:
            msk = msk.convert("L")
            mask_array = np.asarray(msk, dtype=np.float32) > 0

        if mask_array.any():
            masked_pixels = img_array[mask_array]
            mean_val = float(masked_pixels.mean())
            std_val = float(masked_pixels.std())
            min_val = float(masked_pixels.min())
            max_val = float(masked_pixels.max())
            area = float(mask_array.sum())
        else:
            mean_val = std_val = min_val = max_val = 0.0
            area = 0.0

        perimeter = float(np.sum(np.abs(np.diff(mask_array.astype(np.int32), axis=0)))) + float(
            np.sum(np.abs(np.diff(mask_array.astype(np.int32), axis=1)))
        )
        sphericity = float(4 * np.pi * area / (perimeter ** 2 + 1e-6))
        elongation = float((mask_array.shape[1] + 1e-6) / (mask_array.shape[0] + 1e-6))

        row_values = [img_path.name]
        row_values.extend([
            mean_val,
            std_val,
            min_val,
            max_val,
            area,
            sphericity,
            elongation,
            perimeter,
        ])
        row_values.append(row["class"])
        feature_rows.append(row_values)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = ["file_name"] + feature_names + ["category"]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id"] + header)
        for idx, values in enumerate(feature_rows):
            writer.writerow([idx] + values)

    print(f"Saved radiomics CSV to {output_path}")


if __name__ == "__main__":
    main()
