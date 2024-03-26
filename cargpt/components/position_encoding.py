import torch
from einops import einsum, rearrange, repeat
from jaxtyping import Float, Shaped
from torch import Tensor, nn


class PointPositionalEncoder3D(nn.Module):
    """3D point positional encoder.

    https://arxiv.org/abs/2211.14710 (Section 4.3)
    """

    def __init__(self, channels: int, temperature: float = 1e4) -> None:
        super().__init__()

        if channels % 2 != 0:
            msg = "channels must be divisible by 2"
            raise ValueError(msg)

        channels_pe = channels // 2

        self.mlp = nn.Sequential(
            nn.Linear(3 * channels_pe, 3 * channels_pe),
            nn.ReLU(),
            nn.Linear(3 * channels_pe, channels),
        )

        inv_freq = 1.0 / (
            temperature ** (torch.arange(0, channels_pe, 2) / channels_pe)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, points: Shaped[Tensor, "*b h w 3"]) -> Float[Tensor, "*b h w c"]:
        x = einsum(points, self.inv_freq, "... xyz, c -> ... xyz c")
        pe_sine = rearrange(
            [x.sin(), x.cos()],
            "sin_cos ... xyz c -> ... (xyz c sin_cos)",
        )
        return self.mlp(pe_sine)


class LearnablePositionalEmbedding1D(nn.Module):
    def __init__(self, seq_len, embedding_dim) -> None:
        super().__init__()

        self._emb = nn.Embedding(
            num_embeddings=seq_len,
            embedding_dim=embedding_dim,
        )

    @property
    def _seq_len(self):
        return self._emb.num_embeddings

    def forward(self, shape: torch.Size) -> Float[Tensor, "b s c"]:
        b, s, *_ = shape
        if s != self._seq_len:
            msg = f"expected sequence length {self._seq_len}, got {s}"
            raise ValueError(msg)

        return repeat(self._emb.weight, "s c -> b s c", b=b)
