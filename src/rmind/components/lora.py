import math
from typing import override

import torch
from torch import Tensor, nn

from rmind.components.transformer.attention import RotaryMultiheadAttention


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, *, r: int, alpha: int, dropout: float) -> None:
        super().__init__()
        self.base = base.requires_grad_(False)  # noqa: FBT003
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scaling = alpha / r

    @override
    def forward(self, x: Tensor) -> Tensor:
        delta = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return self.base(x) + self.scaling * delta


class IdentityRope(nn.Module):
    """No-op stand-in for a rope module, so `RotaryMultiheadAttention` can be
    reused as a plain (non-rotary) self-attention with explicit qkv/out_proj
    `nn.Linear`s -- the LoRA-wrappable equivalent of `nn.MultiheadAttention`.
    """

    @override
    def forward(self, x: Tensor) -> Tensor:
        return x


def convert_multihead_attention(mha: nn.MultiheadAttention) -> RotaryMultiheadAttention:
    """Losslessly convert `nn.MultiheadAttention` self-attention into a
    `RotaryMultiheadAttention` with an identity rope.

    Both use the same packed qkv weight layout
    (`in_proj_weight` = concat of q/k/v weights, row-chunked) and the same
    boolean mask convention before `F.scaled_dot_product_attention`, so this
    is numerically equivalent to the original module -- it just exposes
    `qkv`/`out_proj` as `nn.Linear` submodules instead of a fused
    `in_proj_weight` Parameter, so they can be LoRA-wrapped.

    Requires `embed_dim // num_heads` to be even (the `RotaryMultiheadAttention`
    constructor's rope-pairing constraint, unused here since the rope is a
    no-op, but still enforced).
    """
    packed = RotaryMultiheadAttention(
        embed_dim=mha.embed_dim,
        num_heads=mha.num_heads,
        rope=IdentityRope(),
        attn_dropout=mha.dropout,
    )
    with torch.no_grad():
        packed.qkv.weight.copy_(mha.in_proj_weight)
        packed.qkv.bias.copy_(mha.in_proj_bias)
        packed.out_proj.weight.copy_(mha.out_proj.weight)
        packed.out_proj.bias.copy_(mha.out_proj.bias)
    return packed
