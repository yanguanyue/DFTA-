#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize segmentation results")
    parser.add_argument("--input-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-xlsx", required=True, type=Path)
    args = parser.parse_args()

    with args.input_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "model", "status", "best_mDice", "best_mIoU", "step_run", "step_stop"],
        )
        writer.writeheader()
        for row in data:
            writer.writerow(row)

    args.output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    write_excel(args.output_csv, args.output_xlsx)


if __name__ == "__main__":
    main()
