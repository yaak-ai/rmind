from abc import ABC, abstractmethod

import torch
from einops import einsum, rearrange, repeat
from jaxtyping import Float, Int, Shaped
from torch import Tensor, nn
from torchvision.models import ResNet


class ResnetBackbone(torch.nn.Module):
    def __init__(self, resnet: ResNet, freeze: bool = True):
        super().__init__()

        self.resnet = resnet

        if freeze:
            self.requires_grad_(False)
            self.eval()

    def forward(self, x: Float[Tensor, "*b c1 h1 w1"]) -> Float[Tensor, "*b c2 h2 w2"]:
        *b, _, _, _ = x.shape
        x = rearrange(x, "... c h w -> (...) c h w")

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = x.view(*b, *x.shape[-3:])

        return x


class PointPositionalEncoder3D(torch.nn.Module):
    """3D point positional encoder

    https://arxiv.org/abs/2211.14710 (Section 4.3)
    """

    def __init__(self, channels: int, temperature: float = 1e4) -> None:
        super().__init__()

        if channels % 2 != 0:
            raise ValueError("channels must be divisible by 2")

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

    def forward(self, pos: Shaped[Tensor, "b h w 3"]) -> Float[Tensor, "b h w c"]:
        # NOTE: `pos` are the _actual positions_ being encoded
        x = einsum(pos, self.inv_freq, "b h w xyz, c -> b h w xyz c")
        pe_sine = rearrange(
            [x.sin(), x.cos()],  # type: ignore
            "sin_cos b h w xyz c -> b h w (xyz c sin_cos)",
        )
        pe = self.mlp(pe_sine)

        return pe


class LearnablePositionalEmbedding1D(torch.nn.Module):
    def __init__(self, seq_len, embedding_dim):
        super().__init__()

        self._emb = torch.nn.Embedding(
            num_embeddings=seq_len,
            embedding_dim=embedding_dim,
        )

    @property
    def _seq_len(self):
        return self._emb.num_embeddings

    def forward(self, shape: torch.Size) -> Float[Tensor, "b s c"]:
        b, s, *_ = shape
        if s != self._seq_len:
            raise ValueError(f"expected sequence length {self._seq_len}, got {s}")

        return repeat(self._emb.weight, "s c -> b s c", b=b)


class Invertible(ABC):
    @abstractmethod
    def invert(self, *args, **kwargs):
        pass


class MuLawCompressor(Invertible, torch.nn.Module):
    """Apply mu-law compression as in Gato paper.

    Appendix B, eq. 3. https://arxiv.org/abs/2205.06175
    """

    M: Tensor
    mu: Tensor

    def __init__(self, mu: int = 100, M: int = 256):
        super().__init__()
        self.register_buffer("mu", torch.tensor(mu, dtype=torch.int32))
        self.register_buffer("M", torch.tensor(M, dtype=torch.int32))

    def forward(self, x: Float[Tensor, "b v"]) -> Float[Tensor, "b v"]:
        out = torch.log(x.abs() * self.mu + 1.0)
        out = out / torch.log(self.M * self.mu + 1.0)
        out = torch.sign(x) * out
        return out

    def invert(self, x: Float[Tensor, "b v"]) -> Float[Tensor, "b v"]:
        non_zeros = x != 0
        abs_x = (
            torch.pow(torch.e, x * torch.log(self.M * self.mu + 1.0) / torch.sign(x))
            - 1.0
        ) / self.mu
        x = abs_x * torch.sign(x)
        x = torch.where(non_zeros, x, torch.zeros_like(x))
        return x


class Discretizer(Invertible, torch.nn.Module):
    """Discretize floating point values from a defined range into bins
    of uniform width.

    This is not a scaler, it uses range min and max to assume min and max
    values and clamp.

    The undo process is not precise, the error would be at most the bin width
    (range divided by the number of bins).

    Appendix B, as described after eq. 3. https://arxiv.org/abs/2205.06175
    """

    range_min: Tensor
    range_max: Tensor
    start_index: Tensor
    bins: Tensor

    def __init__(
        self,
        range_min: float = -1.0,
        range_max: float = 1.0,
        start_index: int = 0,
        bins: int = 1024,
    ):
        super().__init__()
        self.register_buffer("range_min", torch.tensor(range_min))
        self.register_buffer("range_max", torch.tensor(range_max))
        self.register_buffer(
            "start_index", torch.tensor(start_index, dtype=torch.int32)
        )
        self.register_buffer("bins", torch.tensor(bins, dtype=torch.int32))

    def forward(self, x: Float[Tensor, "b v"]) -> Int[Tensor, "b v"]:
        bin_width = torch.abs(self.range_max - self.range_min) / self.bins
        x = torch.clamp(x, self.range_min, self.range_max)
        x = x - self.range_min
        x = (x / bin_width).int()
        return x + self.start_index

    def invert(self, x: Int[Tensor, "b v"]) -> Float[Tensor, "b v"]:
        bin_width = torch.abs(self.range_max - self.range_min) / self.bins
        x = x - self.start_index
        x = x * bin_width + self.range_min
        return x
