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


def merge_lora(module: Module) -> int:
    """In-place fold every `LoRALinear` under `module` back into a plain
    `nn.Linear`, so the exported/served graph pays nothing for the adapters.
    Returns how many were merged.

    Computes `W += scale * B @ A` (and leaves `bias` untouched -- LoRA never
    touches it) directly on `base`, then replaces the `LoRALinear` wrapper with
    `base` itself at the parent. No-op (returns 0) if `module` has no
    `LoRALinear` submodules, which is what makes it safe to call
    unconditionally on every export path -- see `rmind.scripts.export_onnx` and
    `rmind.scripts.decoder_only_export`.

    Safe under `torch.inference_mode()` (both export entry points run inside
    it): the in-place update is on the frozen `base.weight`, and the `B @ A`
    product it adds is computed under `no_grad`.
    """
    merged = 0
    for name, child in list(module.named_modules()):
        if isinstance(child, LoRALinear):
            with torch.no_grad():
                child.base.weight += child.scale * (child.lora_B @ child.lora_A)
            parent_name, _, attr = name.rpartition(".")
            parent = module.get_submodule(parent_name) if parent_name else module
            setattr(parent, attr, child.base)
            merged += 1

    return merged
