from typing import override

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn
from torch.nn.modules.module import Module


class RotaryMultiheadAttention(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, rope: Module, attn_dropout: float = 0.1
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            msg = f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}"
            raise ValueError(msg)
        head_dim = embed_dim // num_heads
        if head_dim % 2 != 0:
            msg = f"head_dim={head_dim} must be even for RoPE"
            raise ValueError(msg)

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.attn_dropout = attn_dropout

        self.qkv: nn.Linear = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.out_proj: nn.Linear = nn.Linear(embed_dim, embed_dim, bias=True)
        self.rope: Module = rope

    @override
    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = rearrange(q, "b s (h d) -> b s h d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b s h d", h=self.num_heads)
        v = rearrange(v, "b s (h d) -> b s h d", h=self.num_heads)

        q = rearrange(self.rope(q), "b s h d -> b h s d")
        k = rearrange(self.rope(k), "b s h d -> b h s d")
        v = rearrange(v, "b s h d -> b h s d")

        attn_mask = (
            torch.zeros_like(mask, dtype=q.dtype).masked_fill(mask, -torch.inf)
            if mask.dtype == torch.bool
            else mask.to(q.dtype)
        )

        y = F.scaled_dot_product_attention(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            attn_mask=attn_mask,
            dropout_p=(self.attn_dropout if self.training else 0.0),
        )

        y = rearrange(y, "b h s d -> b s (h d)")
        return self.out_proj(y)


class MaskedSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_dropout: float = 0.1,
        rope: Module | None = None,
    ) -> None:
        super().__init__()
        self.attn: RotaryMultiheadAttention | nn.MultiheadAttention
        if rope is not None:
            self.attn = RotaryMultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                rope=rope,
                attn_dropout=attn_dropout,
            )
        else:
            self.attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=attn_dropout,
                batch_first=True,
            )

    @override
    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        if isinstance(self.attn, RotaryMultiheadAttention):
            return self.attn(x, mask)

        attn_out, _ = self.attn(
            query=x, key=x, value=x, attn_mask=mask, need_weights=False
        )
        return attn_out
