#!/usr/bin/env python3
import csv
import argparse
from pathlib import Path


def build_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    checkpoint_root = Path(args.checkpoint_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_root, output_dir


def read_best_valid_acc(log_path: Path):
    best = None
    if not log_path.exists():
        return None
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Learning Rate"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                valid_acc = float(parts[4])
            except ValueError:
                continue
            if best is None or valid_acc > best:
                best = valid_acc
    return best


def write_excel(csv_path: Path, xlsx_path: Path) -> None:
    try:
        from openpyxl import Workbook
    except Exception as exc:
        raise RuntimeError("openpyxl is required to write Excel files") from exc

    wb = Workbook()
    ws = wb.active

    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    for row in rows:
        ws.append(row)

    wb.save(xlsx_path)


def main():
    parser = argparse.ArgumentParser(description="Summarize classifier improvements")
    parser.add_argument(
        "--checkpoint-root",
        default="/root/autodl-tmp/output/classifier/checkpoints/ham10000_mix",
    )
    parser.add_argument(
        "--output-dir",
        default="/root/autodl-tmp/output/classifier/metrics",
    )
    args = parser.parse_args()

    checkpoint_root, output_dir = build_paths(args)
    records = []
    models = sorted([p.name for p in checkpoint_root.iterdir() if p.is_dir()])

    baseline_scores = {}
    if "baseline" in models:
        for arch_dir in (checkpoint_root / "baseline").iterdir():
            if not arch_dir.is_dir():
                continue
            best = read_best_valid_acc(arch_dir / "log.txt")
            if best is not None:
                baseline_scores[arch_dir.name] = best

    for model in models:
        for arch_dir in (checkpoint_root / model).iterdir():
            if not arch_dir.is_dir():
                continue
            arch = arch_dir.name
            best = read_best_valid_acc(arch_dir / "log.txt")
            if best is None:
                continue
            baseline = baseline_scores.get(arch)
            improvement = None
            if baseline is not None:
                improvement = best - baseline
            records.append({
                "model": model,
                "arch": arch,
                "best_valid_acc": best,
                "baseline_valid_acc": baseline,
                "improvement": improvement,
            })

    # Write CSV
    csv_path = output_dir / "improvements.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "arch", "best_valid_acc", "baseline_valid_acc", "improvement"],
        )
        writer.writeheader()
        for row in records:
            writer.writerow(row)

    # Write Markdown table
    md_path = output_dir / "improvements.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("| model | arch | best_valid_acc | baseline_valid_acc | improvement |\n")
        f.write("| --- | --- | --- | --- | --- |\n")
        for row in records:
            f.write(
                f"| {row['model']} | {row['arch']} | {row['best_valid_acc']:.4f} "
                f"| {row['baseline_valid_acc']:.4f} "
                f"| {row['improvement']:.4f} |\n"
                if row["baseline_valid_acc"] is not None
                else f"| {row['model']} | {row['arch']} | {row['best_valid_acc']:.4f} |  |  |\n"
            )

    xlsx_path = output_dir / "improvements.xlsx"
    write_excel(csv_path, xlsx_path)

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {xlsx_path}")


if __name__ == "__main__":
    main()
