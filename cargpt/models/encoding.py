from abc import ABC, abstractmethod
from functools import lru_cache
from math import prod
from typing import List, Tuple, cast

import torch
from dall_e import load_model
from deephouse.tools.camera import Camera
from einops import einsum, rearrange, reduce, repeat
from jaxtyping import Float, Int, Shaped
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import ResNet


class ResnetBackbone(nn.Module):
    def __init__(self, resnet: ResNet, freeze: bool = True):
        super().__init__()

        self.resnet = resnet.requires_grad_(not freeze).train(not freeze)

    def forward(self, x: Float[Tensor, "*b c1 h1 w1"]) -> Float[Tensor, "*b c2 h2 w2"]:
        *B, C, H, W = x.shape
        x = x.view(prod(B), C, H, W)

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        *_, C, H, W = x.shape
        x = x.view(*B, C, H, W)

        return x


class DepthFeatureEncoder(nn.Module):
    def __init__(self, disp_net: nn.Module, freeze: bool = True):
        super().__init__()

        encoder = cast(nn.Module, disp_net.encoder)
        self.encoder = encoder.requires_grad_(not freeze).train(not freeze)

    def forward(
        self, frames: Float[Tensor, "*b c1 h1 w1"]
    ) -> Float[Tensor, "*b c2 h2 w2"]:
        *B, C, H, W = frames.shape
        frames = frames.view(prod(B), C, H, W)

        disp = self.encoder(frames)[2]

        *_, C, H, W = disp.shape
        disp = disp.view(*B, C, H, W)

        return disp


class DepthEncoder(nn.Module):
    def __init__(self, disp_net: nn.Module, freeze: bool = True):
        super().__init__()

        self.disp_net = disp_net.requires_grad_(not freeze).train(not freeze)

    def forward(
        self,
        frames: Float[Tensor, "*b c1 h1 w1"],
        *,
        camera: Camera,
    ) -> Float[Tensor, "*b h2 w2 3"]:
        B, *_, H, W = frames.shape

        frames = rearrange(frames, "b t c h w -> (b t) c h w")
        disp = self.disp_net(frames)[0]
        depth = 1 / disp
        depth = rearrange(depth, "(b t) 1 h w -> b t h w 1", b=B)

        # compute 3d points (equivalent to Eq 6 [https://arxiv.org/abs/2211.14710])
        grid = self._generate_2d_grid(height=H, width=W).to(depth)
        pts = camera.unproject(grid, depth)  # type: ignore

        # normalize 3d points (Eq 7 [https://arxiv.org/abs/2211.14710])
        pts_min = reduce(pts, "b t h w c -> b t 1 1 c", "min")
        pts_max = reduce(pts, "b t h w c -> b t 1 1 c", "max")
        pts_norm = (pts - pts_min) / (pts_max - pts_min)

        return pts_norm

    @classmethod
    @lru_cache(maxsize=1)
    def _generate_2d_grid(cls, *, height: int, width: int) -> Shaped[Tensor, "h w 2"]:
        x_mesh, y_mesh = torch.meshgrid(
            torch.arange(width),
            torch.arange(height),
            indexing="xy",
        )

        grid = rearrange([x_mesh, y_mesh], "t h w -> h w t")

        return grid


class PointPositionalEncoder3D(nn.Module):
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

    def forward(self, points: Shaped[Tensor, "*b h w 3"]) -> Float[Tensor, "*b h w c"]:
        x = einsum(points, self.inv_freq, "... xyz, c -> ... xyz c")
        pe_sine = rearrange(
            [x.sin(), x.cos()],  # type: ignore
            "sin_cos ... xyz c -> ... (xyz c sin_cos)",
        )
        pe = self.mlp(pe_sine)

        return pe


class LearnablePositionalEmbedding1D(nn.Module):
    def __init__(self, seq_len, embedding_dim):
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
            raise ValueError(f"expected sequence length {self._seq_len}, got {s}")

        return repeat(self._emb.weight, "s c -> b s c", b=b)


class Invertible(ABC):
    @abstractmethod
    def invert(self, *args, **kwargs):
        pass


class MuLawCompressor(Invertible, nn.Module):
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


class Discretizer(Invertible, nn.Module):
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


class ResNetTokens(nn.Module):
    def __init__(self, tokens_mask=-1):
        super().__init__()
        self.tokens_mask = tokens_mask

    def forward(
        self,
        x: Float[Tensor, "b c1 h w"],
        token_shift: int,
        embeddings: nn.Module,
    ) -> Tuple[Float[Tensor, "b c2 h w"], Float[Tensor, "b h w"]]:
        BT, _, fH, fW = x.shape
        tokens = self.tokens_mask * torch.ones(  # type: ignore[union-attr]
            BT, fH, fW, device=x.device
        )
        tokens += token_shift

        return x, tokens


class DVAETokens(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        probs: Float[Tensor, "b c1 h w"],
        tokens_shift: int,
        embeddings: nn.Module,
    ) -> Tuple[Float[Tensor, "b c2 h w"], Int[Tensor, "b h w"]]:
        tokens = torch.argmax(probs.detach(), dim=1)
        tokens += tokens_shift
        x = embeddings(tokens)
        x = rearrange(x, "b h w d -> b d h w")

        return x, tokens


# https://github.com/openai/DALL-E/tree/master
class dalleDVAE(nn.Module):
    def __init__(self, enc_weights: str, freeze: bool = True):
        super().__init__()

        self.enc = load_model(enc_weights)

        if freeze:
            self.requires_grad_(False)
            self.enc.eval()

    def forward(
        self,
        x: Float[Tensor, "b c1 h1 w1"],
    ) -> Float[Tensor, "b c2 h2 w2"]:
        logits = self.enc(x)
        probs = F.softmax(logits, dim=1)

        return probs


class SenorDropout(nn.Module):
    def __init__(self, prob: float):
        super().__init__()
        self.prob = prob

    def forward(
        self, embeddings: List[Float[Tensor, "b t c d"]]
    ) -> List[Float[Tensor, "b t c d"]]:
        if len(embeddings) == 0:
            return []

        if self.prob == 0:
            return embeddings

        b, t, _, d = embeddings[0].shape
        num_samples_to_drop = 1 if b == 1 else int(b * self.prob)
        indices = torch.randperm(b)[:num_samples_to_drop]

        dropped_embeddings = []
        for embedding in embeddings:
            embedding = embedding.clone()
            # drop everything except last timestep
            embedding[indices, : t - 1] = 0
            dropped_embeddings.append(embedding)

        return dropped_embeddings
