import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from metrics import compute_dice_iou


def main() -> None:
    logits = torch.tensor([
        [
            [[2.0, -1.0], [2.0, -1.0]],
            [[-2.0, 1.0], [-2.0, 1.0]],
        ]
    ])
    target = torch.tensor([[0, 1], [0, 1]]).unsqueeze(0)
    dice, iou = compute_dice_iou(logits, target, num_classes=2)
    assert dice > 0.9, f"Unexpected dice: {dice}"
    assert iou > 0.8, f"Unexpected iou: {iou}"
    print("metrics sanity check passed")


if __name__ == "__main__":
    main()
