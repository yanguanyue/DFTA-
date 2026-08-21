import gc
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

ROOT = Path("/root/autodl-tmp")
sys.path.insert(0, str(ROOT / "metirc/segmentation"))
from dataset import JointTransform, SegmentationDataset, resolve_roots
from models import build_segformer, build_unet

DEVICE = torch.device("cuda")
OUT = ROOT / "output/ablation_complete/evaluation/gpu_grid_benchmark.json"
RESULTS = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else []
DONE_KEYS = {
    (item["task"], item["model"], item["batch"], item["workers"])
    for item in RESULTS
    if item.get("status") in {"ok", "oom"}
}
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def record(task, model, batch, workers, status, elapsed=None, samples=None):
    item = {
        "task": task,
        "model": model,
        "batch": batch,
        "workers": workers,
        "status": status,
    }
    if elapsed is not None:
        item["samples_per_sec"] = round(samples / elapsed, 2)
        item["elapsed_sec"] = round(elapsed, 4)
        item["peak_mib"] = round(torch.cuda.max_memory_allocated() / 1024**2)
    RESULTS.append(item)
    DONE_KEYS.add((task, model, batch, workers))
    OUT.write_text(json.dumps(RESULTS, indent=2), encoding="utf-8")
    print(json.dumps(item), flush=True)


normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)
cls_dataset = datasets.ImageFolder(
    ROOT / "output/ablation_complete/evaluation/classifier/mixed/baseline/train",
    transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    ),
)

cls_builders = {
    "resnet18": models.resnet18,
    "resnet50": models.resnet50,
    "efficientnet_b0": models.efficientnet_b0,
}
for model_name, builder in cls_builders.items():
    for workers in (2, 4, 8, 12, 16):
        for batch_size in (32, 64, 128, 256):
            if ("classification", model_name, batch_size, workers) in DONE_KEYS:
                continue
            cleanup()
            try:
                loader = DataLoader(
                    cls_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=workers,
                    pin_memory=True,
                    persistent_workers=True,
                )
                model = builder(weights=None)
                if hasattr(model, "fc"):
                    model.fc = nn.Linear(model.fc.in_features, 7)
                else:
                    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 7)
                model.to(DEVICE).train()
                optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
                scaler = torch.amp.GradScaler("cuda")
                iterator = iter(loader)
                measured = 0
                elapsed = 0.0
                target_samples = 4096
                warmup = 10
                step = 0
                while measured < target_samples:
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    (images, targets), iterator = next_batch(iterator, loader)
                    images = images.to(DEVICE, non_blocking=True)
                    targets = targets.to(DEVICE, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda"):
                        logits = model(images)
                        loss = F.cross_entropy(logits, targets)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    torch.cuda.synchronize()
                    if step >= warmup:
                        elapsed += time.perf_counter() - started
                        measured += images.size(0)
                    step += 1
                record(
                    "classification",
                    model_name,
                    batch_size,
                    workers,
                    "ok",
                    elapsed,
                    measured,
                )
            except torch.OutOfMemoryError:
                record("classification", model_name, batch_size, workers, "oom")
            finally:
                try:
                    iterator._shutdown_workers()
                except Exception:
                    pass
                iterator = loader = model = optimizer = scaler = None
                cleanup()

image_root, mask_root = resolve_roots(
    ROOT / "data/HAM10000/input", "train", "ham10000"
)
seg_dataset = SegmentationDataset(
    image_root, mask_root, JointTransform(512, is_train=True)
)
for model_name in ("unet", "segformer"):
    for workers in (2, 4, 8, 12, 16):
        for batch_size in (2, 4, 8, 16):
            if ("segmentation", model_name, batch_size, workers) in DONE_KEYS:
                continue
            cleanup()
            try:
                loader = DataLoader(
                    seg_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=workers,
                    pin_memory=True,
                    persistent_workers=True,
                )
                model = (
                    build_unet(num_classes=2)
                    if model_name == "unet"
                    else build_segformer(num_classes=2, pretrained=False)
                )
                model.to(DEVICE).train()
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
                scaler = torch.amp.GradScaler("cuda")
                iterator = iter(loader)
                measured = 0
                elapsed = 0.0
                target_samples = 256
                warmup = 5
                step = 0
                while measured < target_samples:
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    batch_data, iterator = next_batch(iterator, loader)
                    images = batch_data["image"].to(DEVICE, non_blocking=True)
                    masks = batch_data["mask"].to(DEVICE, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda"):
                        output = model(images)
                        logits = output.logits if hasattr(output, "logits") else output
                        if logits.shape[-2:] != masks.shape[-2:]:
                            logits = F.interpolate(
                                logits,
                                size=masks.shape[-2:],
                                mode="bilinear",
                                align_corners=False,
                            )
                        loss = F.cross_entropy(logits, masks)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    torch.cuda.synchronize()
                    if step >= warmup:
                        elapsed += time.perf_counter() - started
                        measured += images.size(0)
                    step += 1
                record(
                    "segmentation",
                    model_name,
                    batch_size,
                    workers,
                    "ok",
                    elapsed,
                    measured,
                )
            except torch.OutOfMemoryError:
                record("segmentation", model_name, batch_size, workers, "oom")
            finally:
                try:
                    iterator._shutdown_workers()
                except Exception:
                    pass
                iterator = loader = model = optimizer = scaler = None
                cleanup()
