import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def stratified_split(df, train_ratio, val_ratio, seed):
    rng = np.random.default_rng(seed)
    train_parts = []
    val_parts = []
    test_parts = []

    for category, group in df.groupby("category"):
        indices = group.index.to_numpy().copy()
        rng.shuffle(indices)
        total = len(indices)
        val_count = int(total * val_ratio)
        train_count = int(total * train_ratio)
        train_idx = indices[:train_count]
        val_idx = indices[train_count : train_count + val_count]
        test_idx = indices[train_count + val_count :]

        train_parts.append(df.loc[train_idx])
        val_parts.append(df.loc[val_idx])
        test_parts.append(df.loc[test_idx])

    return (
        pd.concat(train_parts, ignore_index=True),
        pd.concat(val_parts, ignore_index=True),
        pd.concat(test_parts, ignore_index=True),
    )


def main():
    parser = argparse.ArgumentParser(description="Prepare ISIC2019 CSV splits for TabSyn")
    parser.add_argument("--radiomics_csv", default="data/local/ISIC2019/input/radiomics_final.csv")
    parser.add_argument("--output_dir", default="compare_models/reps/MixedTypeTabular/data/ISIC2019")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    csv_path = Path(args.radiomics_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Radiomics CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "category" not in df.columns:
        raise ValueError("Radiomics CSV must contain a 'category' column")

    train_df, val_df, test_df = stratified_split(df, args.train_ratio, args.val_ratio, args.seed)
    trainval_df = pd.concat([train_df, val_df], ignore_index=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(output_dir / "radiomics_final_train.csv", index=False)
    val_df.to_csv(output_dir / "radiomics_final_val.csv", index=False)
    test_df.to_csv(output_dir / "radiomics_final_test.csv", index=False)
    trainval_df.to_csv(output_dir / "radiomics_final_trainval.csv", index=False)

    print(f"Saved TabSyn splits to {output_dir}")


if __name__ == "__main__":
    main()
