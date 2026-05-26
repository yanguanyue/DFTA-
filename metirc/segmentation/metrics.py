from dataclasses import dataclass
from typing import Dict, Tuple

import torch


def _compute_intersection_union(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = torch.zeros(num_classes, device=pred.device)
    union = torch.zeros(num_classes, device=pred.device)
    sums = torch.zeros(num_classes, device=pred.device)
    for cls in range(num_classes):
        pred_i = pred == cls
        target_i = target == cls
        inter = (pred_i & target_i).sum()
        union_i = (pred_i | target_i).sum()
        intersection[cls] = inter
        union[cls] = union_i
        sums[cls] = pred_i.sum() + target_i.sum()
    return intersection, union, sums


def compute_dice_iou(logits: torch.Tensor, target: torch.Tensor, num_classes: int = 2, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    pred = logits.argmax(dim=1)
    intersection, union, sums = _compute_intersection_union(pred, target, num_classes)
    dice = (2.0 * intersection + eps) / (sums + eps)
    iou = (intersection + eps) / (union + eps)
    return dice.mean(), iou.mean()


@dataclass
class MetricTracker:
    total_dice: float = 0.0
    total_iou: float = 0.0
    count: int = 0

    def update(self, dice: torch.Tensor, iou: torch.Tensor, n: int) -> None:
        self.total_dice += float(dice) * n
        self.total_iou += float(iou) * n
        self.count += n

    def compute(self) -> Dict[str, float]:
        if self.count == 0:
            return {"mDice": 0.0, "mIoU": 0.0}
        return {"mDice": self.total_dice / self.count, "mIoU": self.total_iou / self.count}
