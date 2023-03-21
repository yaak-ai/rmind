import torch
from einops import einsum, rearrange, repeat
from jaxtyping import Float, Shaped
from torch import Tensor, nn
from torchvision.models import ResNet


class ResnetBackbone(torch.nn.Module):
    def __init__(self, resnet: ResNet, freeze: bool = True):
        super().__init__()

        self.resnet = resnet

        if freeze:
            self.requires_grad_(False)
            self.eval()

    def forward(self, x: Float[Tensor, "b c1 h1 w1"]) -> Float[Tensor, "b c2 h2 w2"]:
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        layer_1 = self.resnet.layer1(x)
        layer_2 = self.resnet.layer2(layer_1)
        layer_3 = self.resnet.layer3(layer_2)
        layer_4 = self.resnet.layer4(layer_3)

        return layer_4


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

    def forward(self, shape: torch.Size) -> Float[Tensor, "b s c"]:
        b, s, *_ = shape
        if s != self._emb.num_embeddings:
            raise ValueError("invalid input sequence length")

        return repeat(self._emb.weight, "s c -> b s c", b=b)
