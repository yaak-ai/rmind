from typing import override

import torch
from einops import rearrange
from torch import Tensor, nn

from rmind.components.nn import Embedding


class RotaryPositionalEmbeddings(nn.Module):
    """
    inspirations:
    https://github.com/elefant-ai/open-p2p/blob/main/elefant/policy_model/pos_embed.py
    https://github.com/meta-llama/llama/blob/main/llama/model.py
    """

    def __init__(self, dim: int, max_seq_len: int = 4096, base: int = 10_000) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len

        theta = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self._build_rope_cache(self.max_seq_len)

    def _build_rope_cache(self, max_seq_len: int) -> None:
        seq_idx = torch.arange(
            end=max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        )  # ty:ignore[no-matching-overload]
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()
        cache = torch.polar(torch.ones_like(idx_theta), idx_theta)
        self.register_buffer("cache", cache, persistent=False)

    @override
    def forward(self, x: Tensor, *, input_pos: Tensor | None = None) -> Tensor:
        seq_len = x.size(1)
        if seq_len > self.cache.size(0):  # ty:ignore[call-non-callable]
            self._build_rope_cache(seq_len)

        freqs_cis = self.cache[:seq_len] if input_pos is None else self.cache[input_pos]  # ty:ignore[not-subscriptable]
        x_pairs = rearrange(x.float(), "b s h (d r) -> b s h d r", r=2)
        x_complex = torch.view_as_complex(x_pairs)
        match freqs_cis.ndim:
            case 2:
                freqs_cis = rearrange(freqs_cis, "s d -> 1 s 1 d")
            case 3:
                freqs_cis = rearrange(freqs_cis, "b s d -> b s 1 d")
            case _:
                msg = f"Invalid freqs_cis shape: {tuple(freqs_cis.shape)}"
                raise ValueError(msg)
        x_rotated = torch.view_as_real(x_complex * freqs_cis)
        return rearrange(x_rotated, "b s h d r -> b s h (d r)").type_as(x)


class PatchPositionEmbedding2D(nn.Module):
    def __init__(self, grid_size: tuple[int, int], embedding_dim: int) -> None:
        super().__init__()
        self.row_embed = Embedding(grid_size[0], embedding_dim)
        self.col_embed = Embedding(grid_size[1], embedding_dim)

    @override
    def forward(self, x: Tensor) -> Tensor:
        row_pos = self.row_embed.weight
        col_pos = self.col_embed.weight
        pos_embed = rearrange(row_pos, "h d -> h 1 d") + rearrange(
            col_pos, "w d -> 1 w d"
        )
        return x + rearrange(pos_embed, "h w d -> (h w) d")
