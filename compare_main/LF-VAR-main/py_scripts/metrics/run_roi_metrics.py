import argparse
import json
from pathlib import Path
import subprocess

DEFAULT_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def run_single(command):
    print(" ".join(command))
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run ROI KID/IS/LPIPS for Derm-T2IM and main results")
    parser.add_argument("--root", default=".", help="Repo root path")
    parser.add_argument("--classes", nargs="*", default=DEFAULT_CLASSES)
    parser.add_argument("--output_dir", default="roi_metrics_results")
    parser.add_argument("--split", default="val", help="Dataset split to use (train/val/test/train_val)")
    parser.add_argument("--dataset_root", default="data/local/HAM10000/input", help="Dataset root path")
    parser.add_argument("--img_dir_name", default="HAM10000_img_class", help="Image class folder name")
    parser.add_argument("--seg_dir_name", default="HAM10000_seg_class", help="Segmentation class folder name")
    parser.add_argument("--gpu", action="store_true", default=True)

    args = parser.parse_args()
    root = Path(args.root)
    output_root = root / args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)

    script = root / "py_scripts" / "metrics" / "cal_metrics_roi_direct.py"

    # Derm-T2IM (generated names are numeric, use index mapping)
    derm_gen_root = root / "data" / "compare_results" / "Derm-T2IM" / "inference"
    for cls in args.classes:
        gen_dir = derm_gen_root / cls
        if not gen_dir.exists():
            continue
        real_img_dir = root / args.dataset_root / args.split / args.img_dir_name / cls
        real_mask_dir = root / args.dataset_root / args.split / args.seg_dir_name / cls
        out_file = output_root / f"Derm-T2IM_{cls}.json"
        run_single([
            "python",
            str(script),
            "--gen_dir",
            str(gen_dir),
            "--real_img_dir",
            str(real_img_dir),
            "--output",
            str(out_file),
            "--mode",
            "index",
            "--no_mask",
        ] + (["--gpu"] if args.gpu else []))

    # main (generated names include ISIC id, use name mapping)
    main_gen_root = root / "data" / "compare_results" / "main"
    for cls in args.classes:
        gen_dir = main_gen_root / cls
        if not gen_dir.exists():
            continue
        real_img_dir = root / args.dataset_root / args.split / args.img_dir_name / cls
        real_mask_dir = root / args.dataset_root / args.split / args.seg_dir_name / cls
        out_file = output_root / f"main_{cls}.json"
        run_single([
            "python",
            str(script),
            "--gen_dir",
            str(gen_dir),
            "--real_img_dir",
            str(real_img_dir),
            "--real_mask_dir",
            str(real_mask_dir),
            "--output",
            str(out_file),
            "--mode",
            "name",
        ] + (["--gpu"] if args.gpu else []))

    summary = {
        "output_dir": str(output_root),
        "classes": args.classes,
        "Derm-T2IM": {},
        "main": {},
    }

    for cls in args.classes:
        derm_file = output_root / f"Derm-T2IM_{cls}.json"
        if derm_file.exists():
            with derm_file.open("r", encoding="utf-8") as f:
                summary["Derm-T2IM"][cls] = json.load(f)["metrics"]
        main_file = output_root / f"main_{cls}.json"
        if main_file.exists():
            with main_file.open("r", encoding="utf-8") as f:
                summary["main"][cls] = json.load(f)["metrics"]

    summary_path = root / "roi_metrics_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    summary["summary_path"] = str(summary_path)
    summary["done"] = True
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
