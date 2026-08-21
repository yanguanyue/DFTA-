import os
import shutil
import time
import zipfile
from pathlib import Path

BASE = Path("/root/autodl-tmp/output/ablation_complete")
ARCHIVE = BASE / "E_H_I.zip"
STAGE = BASE / ".ehi_extract_new"
MAPPING = {
    "single_flow_csfs": "E",
    "dual_flow_full_det": "H",
    "dual_flow_no_aug_csfs": "I",
}
CLASSES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
MAJORITY = {"bkl", "mel", "nv"}


def log(message):
    print(time.strftime("%F %T"), message, flush=True)


def files_by_stem(folder):
    result = {}
    for path in folder.iterdir():
        if path.is_file():
            if path.stem in result:
                raise RuntimeError(f"duplicate stem: {folder}/{path.stem}")
            result[path.stem] = path
    return result


if not ARCHIVE.is_file():
    raise FileNotFoundError(ARCHIVE)
log(f"archive bytes={ARCHIVE.stat().st_size}")

for pattern in (".E_H_I.zip.part*", ".E_H_I_w6_part*", ".E_H_I_w12_part*"):
    for path in BASE.glob(pattern):
        if path.is_file():
            log(f"remove abandoned upload fragment {path.name}")
            path.unlink()

with zipfile.ZipFile(ARCHIVE) as zf:
    bad = zf.testzip()
    if bad:
        raise RuntimeError(f"corrupt member: {bad}")
    unpacked = sum(item.file_size for item in zf.infolist())
    free = shutil.disk_usage(BASE).free
    log(f"zip verified; unpacked={unpacked}, free={free}")
    if free < unpacked + 2 * 1024**3:
        raise RuntimeError("insufficient disk space for safe extraction")
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir()
    log("extracting")
    zf.extractall(STAGE)

sources = {}
for source_name in MAPPING:
    matches = [p for p in STAGE.rglob(source_name) if p.is_dir()]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {source_name}, found {matches}")
    sources[source_name] = matches[0]

for source_name, group in MAPPING.items():
    source = sources[source_name]
    log(f"validating and pruning {source_name} -> {group}")
    for class_name in CLASSES:
        images_dir = source / class_name / "images"
        masks_dir = source / class_name / "masks"
        if not images_dir.is_dir() or not masks_dir.is_dir():
            raise RuntimeError(f"missing folders: {source_name}/{class_name}")
        images = files_by_stem(images_dir)
        masks = files_by_stem(masks_dir)
        if set(images) != set(masks):
            raise RuntimeError(
                f"unpaired data: {source_name}/{class_name} "
                f"images={len(images)} masks={len(masks)}"
            )
        target = 500 if class_name in MAJORITY else 1500
        if len(images) < target:
            raise RuntimeError(
                f"too few pairs: {source_name}/{class_name}={len(images)} target={target}"
            )
        keep = set(sorted(images)[:target])
        for stem in set(images) - keep:
            images[stem].unlink()
            masks[stem].unlink()

for source_name, group in MAPPING.items():
    target = BASE / group
    if target.exists():
        if target.resolve().parent != BASE.resolve() or target.name not in {"E", "H", "I"}:
            raise RuntimeError(f"unsafe replacement target: {target}")
        shutil.rmtree(target)
    shutil.move(str(sources[source_name]), str(target))
    log(f"installed {group}")

if STAGE.exists():
    shutil.rmtree(STAGE)

for group in ("E", "H", "I"):
    for class_name in CLASSES:
        target = 500 if class_name in MAJORITY else 1500
        images = files_by_stem(BASE / group / class_name / "images")
        masks = files_by_stem(BASE / group / class_name / "masks")
        if len(images) != target or len(masks) != target or set(images) != set(masks):
            raise RuntimeError(f"final validation failed: {group}/{class_name}")
        log(f"FINAL {group}/{class_name} images={len(images)} masks={len(masks)}")

log("COMPLETE")
