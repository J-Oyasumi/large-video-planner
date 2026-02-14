import math
from typing import Iterable, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if rank < 1:
            raise ValueError(f"LoRA rank must be >= 1, got {rank}.")
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"LoRA expects nn.Linear, got {type(base_layer)}.")

        self.base = base_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()

        for parameter in self.base.parameters():
            parameter.requires_grad = False

        base_weight = self.base.weight
        self.lora_a = nn.Parameter(
            torch.empty(
                self.rank,
                self.base.in_features,
                dtype=base_weight.dtype,
                device=base_weight.device,
            )
        )
        self.lora_b = nn.Parameter(
            torch.zeros(
                self.base.out_features,
                self.rank,
                dtype=base_weight.dtype,
                device=base_weight.device,
            )
        )
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = F.linear(F.linear(self.dropout(x), self.lora_a), self.lora_b)
        return base_out + lora_out * self.scaling


def _matches_any_suffix(name: str, suffixes: Sequence[str]) -> bool:
    return any(name.endswith(suffix) for suffix in suffixes)


def inject_lora_adapters(
    model: nn.Module,
    target_modules: Sequence[str],
    rank: int,
    alpha: float,
    dropout: float,
) -> List[str]:
    if not target_modules:
        raise ValueError("target_modules must contain at least one suffix.")

    replacements = []
    for parent_name, parent_module in model.named_modules():
        for child_name, child_module in parent_module.named_children():
            full_name = (
                f"{parent_name}.{child_name}" if parent_name else child_name
            )
            if isinstance(child_module, nn.Linear) and _matches_any_suffix(
                full_name, target_modules
            ):
                replacements.append((parent_module, child_name, full_name, child_module))

    replaced_names = []
    for parent_module, child_name, full_name, child_module in replacements:
        setattr(
            parent_module,
            child_name,
            LoRALinear(
                base_layer=child_module,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            ),
        )
        replaced_names.append(full_name)

    return replaced_names


def count_trainable_parameters(parameters: Iterable[torch.nn.Parameter]) -> int:
    return sum(p.numel() for p in parameters if p.requires_grad)
