from typing import final

import torch
from torch import Tensor
from torch.nn import Module

from rmind.components.norm import UniformBinner


@final
class UnsupportedActionTokenizerError(TypeError):
    pass


def categorical_expected_value(logits: Tensor, tokenizer: Module) -> Tensor:
    if not isinstance(tokenizer, UniformBinner):
        msg = f"expected UniformBinner, got {type(tokenizer).__name__}"
        raise UnsupportedActionTokenizerError(msg)

    probs = logits.softmax(dim=-1)
    centers = tokenizer.bin_centers.to(device=logits.device, dtype=logits.dtype)

    return (probs * centers).sum(dim=-1)


def categorical_std(logits: Tensor, tokenizer: Module) -> Tensor:
    if not isinstance(tokenizer, UniformBinner):
        msg = f"expected UniformBinner, got {type(tokenizer).__name__}"
        raise UnsupportedActionTokenizerError(msg)

    probs = logits.softmax(dim=-1)
    centers = tokenizer.bin_centers.to(device=logits.device, dtype=logits.dtype)
    mean = (probs * centers).sum(dim=-1, keepdim=True)
    var = (probs * (centers - mean).square()).sum(dim=-1)

    return torch.sqrt(var.clamp_min(0.0))
