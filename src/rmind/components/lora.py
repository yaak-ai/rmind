import math
from typing import override

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Module


class LoRALinear(Module):
    """Low-rank adapter wrapping a frozen `nn.Linear`.

    https://arxiv.org/abs/2106.09685. `base` is expected to already have
    `requires_grad_(False)` applied by the caller; only `lora_A`/`lora_B` are
    trainable. `lora_B` is zero-initialized so the adapter contributes nothing
    at initialization (`base(x) + 0`).
    """

    def __init__(self, base: nn.Linear, *, rank: int = 32, alpha: float = 32.0) -> None:
        super().__init__()

        self.base = base
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scale = alpha / rank

    @override
    def forward(self, input: Tensor) -> Tensor:
        return self.base(input) + self.scale * F.linear(
            F.linear(input, self.lora_A), self.lora_B
        )


def apply_lora(
    module: Module, *, target_suffixes: tuple[str, ...], rank: int, alpha: float
) -> None:
    """In-place replace `nn.Linear` submodules whose dotted name ends with one
    of `target_suffixes` with a `LoRALinear` wrapper."""
    for name, child in list(module.named_modules()):
        if isinstance(child, nn.Linear) and name.endswith(target_suffixes):
            parent_name, _, attr = name.rpartition(".")
            parent = module.get_submodule(parent_name) if parent_name else module
            setattr(parent, attr, LoRALinear(child, rank=rank, alpha=alpha))
