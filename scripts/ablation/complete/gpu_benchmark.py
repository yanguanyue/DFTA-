import gc
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import models

sys.path.insert(0, "/root/autodl-tmp/metirc/segmentation")
from models import build_segformer, build_unet

DEVICE = "cuda"
RESULTS = []


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def run_case(task, model_name, batch, amp):
    cleanup()
    try:
        if task == "classification":
            builders = {
                "resnet18": models.resnet18,
                "resnet50": models.resnet50,
                "efficientnet_b0": models.efficientnet_b0,
            }
            model = builders[model_name](weights=None).to(DEVICE).train()
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
            shape = (batch, 3, 224, 224)
            warmup, steps = 3, 8
        else:
            model = (
                build_unet(num_classes=2)
                if model_name == "unet"
                else build_segformer(num_classes=2, pretrained=False)
            ).to(DEVICE).train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            shape = (batch, 3, 512, 512)
            warmup, steps = 2, 5

        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        elapsed = 0.0
        for step in range(warmup + steps):
            x = torch.randn(shape, device=DEVICE)
            optimizer.zero_grad(set_to_none=True)
            context = torch.amp.autocast("cuda") if amp else nullcontext()
            torch.cuda.synchronize()
            start = time.perf_counter()
            with context:
                output = model(x)
                logits = output.logits if hasattr(output, "logits") else output
                if task == "classification":
                    target = torch.randint(0, 7, (batch,), device=DEVICE)
                    loss = F.cross_entropy(logits, target)
                else:
                    if logits.shape[-2:] != (512, 512):
                        logits = F.interpolate(
                            logits, size=(512, 512), mode="bilinear", align_corners=False
                        )
                    target = torch.randint(0, 2, (batch, 512, 512), device=DEVICE)
                    loss = F.cross_entropy(logits, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()
            if step >= warmup:
                elapsed += time.perf_counter() - start

        result = {
            "task": task,
            "model": model_name,
            "batch": batch,
            "amp": amp,
            "status": "ok",
            "peak_mib": round(torch.cuda.max_memory_allocated() / 1024**2),
            "samples_per_sec": round(batch * steps / elapsed, 2),
            "step_sec": round(elapsed / steps, 4),
        }
    except torch.OutOfMemoryError:
        result = {
            "task": task,
            "model": model_name,
            "batch": batch,
            "amp": amp,
            "status": "oom",
        }
    except Exception as exc:
        result = {
            "task": task,
            "model": model_name,
            "batch": batch,
            "amp": amp,
            "status": f"error: {type(exc).__name__}: {exc}",
        }
    RESULTS.append(result)
    print(json.dumps(result), flush=True)
    try:
        del model, optimizer
    except UnboundLocalError:
        pass
    cleanup()


for model_name in ("resnet18", "resnet50", "efficientnet_b0"):
    for amp in (False, True):
        for batch in (64, 128, 256, 384):
            run_case("classification", model_name, batch, amp)

for model_name in ("unet", "segformer"):
    for amp in (False, True):
        for batch in (4, 8, 12, 16):
            run_case("segmentation", model_name, batch, amp)

Path("/root/autodl-tmp/output/ablation_complete/evaluation/gpu_benchmark.json").write_text(
    json.dumps(RESULTS, indent=2), encoding="utf-8"
)
