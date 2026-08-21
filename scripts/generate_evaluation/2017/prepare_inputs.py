#!/usr/bin/env python3
import json, os, shutil, sys
from pathlib import Path

cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
out = Path(cfg["eval_root"])
normalized = out / "normalized"
real_root = out / "real_data"
exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

def files(folder):
    if not folder.is_dir(): return []
    return sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts), key=lambda p:p.name)

def choose(root, names, direct=False):
    for name in names:
        p=root/name
        if files(p): return p
    return root if direct and files(root) else None

def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink(): dst.unlink()
    os.symlink(src, dst)

for target in (normalized, real_root):
    if target.exists(): shutil.rmtree(target)
    target.mkdir(parents=True)

compat=Path(cfg["compat_root"])
for split in ("train","val","test"):
    for kind, alias in (("img_class","HAM10000_img_class"),("seg_class","HAM10000_seg_class")):
        src=compat/split/kind; dst=real_root/split/alias
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(src,dst)

rows=[]
expected=cfg["expected_counts"]
for model, source_text in cfg["models"].items():
    source=Path(source_text)
    if not source.is_dir(): raise FileNotFoundError(source)
    for cls in cfg["classes"]:
        root=source/cls
        image_dir=choose(root,["images","image"],True)
        mask_dir=choose(root,["masks","mask"])
        images=files(image_dir) if image_dir else []
        masks=files(mask_dir) if mask_dir else []
        paired=min(len(images),len(masks))
        if len(images) < int(expected[cls]):
            raise RuntimeError(f"{model}/{cls}: images={len(images)} expected>={expected[cls]}")
        for i,src in enumerate(images): link(src,normalized/model/cls/"images"/f"sample_{i:06d}.png")
        for i,src in enumerate(masks[:paired]): link(src,normalized/model/cls/"masks"/f"sample_{i:06d}.png")
        rows.append(dict(model=model,cls=cls,images=len(images),masks=len(masks),paired=paired))
(out/"input_report.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
print(json.dumps(rows,indent=2))
