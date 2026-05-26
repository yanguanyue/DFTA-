import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Bootstrap TabSyn-like CSV from ISIC2019 radiomics")
    parser.add_argument("--radiomics_csv", default="data/local/ISIC2019/input/radiomics_final.csv")
    parser.add_argument("--output", default="data/compare_results/MixedTypeTabular/ISIC2019_tabsyn.csv")
    parser.add_argument("--samples_per_class", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    df = pd.read_csv(args.radiomics_csv)
    if "category" not in df.columns:
        raise ValueError("radiomics_csv must include 'category'")

    synthetic_rows = []
    for category, group in df.groupby("category"):
        if group.empty:
            continue
        indices = rng.choice(group.index.to_numpy(), size=args.samples_per_class, replace=True)
        synthetic_rows.append(df.loc[indices])

    out_df = pd.concat(synthetic_rows, ignore_index=True)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"Saved bootstrap TabSyn CSV to {output_path}")


if __name__ == "__main__":
    main()
