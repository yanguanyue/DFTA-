import math
from typing import Iterable, Tuple

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, rank: int = 4, alpha: float = 4.0, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be > 0")
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.lora_down = nn.Linear(base_layer.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, base_layer.out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

        for p in self.base_layer.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_layer(x) + self.lora_up(self.lora_down(self.dropout(x))) * self.scale

    def lora_parameters(self) -> Iterable[torch.nn.Parameter]:
        return list(self.lora_down.parameters()) + list(self.lora_up.parameters())


class LoRAConvNd(nn.Module):
    def __init__(self, base_layer: nn.Module, rank: int = 4, alpha: float = 4.0, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be > 0")
        if not isinstance(base_layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            raise TypeError("base_layer must be a Conv1d/Conv2d/Conv3d")
        if base_layer.groups != 1:
            raise NotImplementedError("LoRAConvNd currently supports groups=1 only")

        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        conv_cls = base_layer.__class__
        self.lora_down = conv_cls(
            in_channels=base_layer.in_channels,
            out_channels=rank,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )
        self.lora_up = conv_cls(
            in_channels=rank,
            out_channels=base_layer.out_channels,
            kernel_size=base_layer.kernel_size,
            stride=base_layer.stride,
            padding=base_layer.padding,
            dilation=base_layer.dilation,
            groups=1,
            bias=False,
        )

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

        for p in self.base_layer.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_layer(x) + self.lora_up(self.lora_down(self.dropout(x))) * self.scale

    def lora_parameters(self) -> Iterable[torch.nn.Parameter]:
        return list(self.lora_down.parameters()) + list(self.lora_up.parameters())


def _replace_with_lora(parent: nn.Module, name: str, module: nn.Module, rank: int, alpha: float, dropout: float) -> bool:
    if isinstance(module, nn.Linear):
        setattr(parent, name, LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout))
        return True
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        setattr(parent, name, LoRAConvNd(module, rank=rank, alpha=alpha, dropout=dropout))
        return True
    return False


def apply_lora(model: nn.Module, rank: int = 4, alpha: float = 4.0, dropout: float = 0.0) -> None:
    for name, module in model.named_children():
        if _replace_with_lora(model, name, module, rank, alpha, dropout):
            continue
        apply_lora(module, rank=rank, alpha=alpha, dropout=dropout)


def mark_only_lora_as_trainable(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for module in model.modules():
        if isinstance(module, (LoRALinear, LoRAConvNd)):
            for param in module.lora_parameters():
                param.requires_grad = True
