import json
import subprocess
import time
from pathlib import Path

root = Path("/root/autodl-tmp")
out = Path("/tmp/ablation_bench_cls")
out.mkdir(parents=True, exist_ok=True)
command = [
    "/root/miniconda3/envs/flow/bin/python",
    "customdata.py",
    "-a", "resnet18",
    "-d", str(root / "output/ablation_complete/evaluation/classifier/mixed/baseline"),
    "--pretrained",
    "--epochs", "1",
    "--schedule", "1",
    "--gamma", "0.1",
    "--lr", "0.001",
    "--gpu-id", "0",
    "--manualSeed", "42",
    "-c", str(out),
    "-j", "8",
    "--amp",
    "--train-batch", "64",
    "--test-batch", "64",
]
started = time.perf_counter()
with Path("/tmp/ablation_bench_cls.training.log").open("w") as log:
    result = subprocess.run(
        command,
        cwd=root / "metirc/pytorch-classification-extended-master",
        stdout=log,
        stderr=subprocess.STDOUT,
    )
elapsed = time.perf_counter() - started
payload = {"returncode": result.returncode, "elapsed_sec": elapsed}
Path("/tmp/ablation_bench_cls.result.json").write_text(
    json.dumps(payload), encoding="utf-8"
)
print(json.dumps(payload), flush=True)
