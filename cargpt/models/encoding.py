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
    def __init__(self, resnet: ResNet, freeze: bool = True) -> None:
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
        return x.view(*B, C, H, W)


class DepthFeatureEncoder(nn.Module):
    def __init__(self, disp_net: nn.Module, freeze: bool = True) -> None:
        super().__init__()

        encoder = cast(nn.Module, disp_net.encoder)
        self.encoder = encoder.requires_grad_(not freeze).train(not freeze)

    def forward(
        self,
        frames: Float[Tensor, "*b c1 h1 w1"],
    ) -> Float[Tensor, "*b c2 h2 w2"]:
        *B, C, H, W = frames.shape
        frames = frames.view(prod(B), C, H, W)

        disp = self.encoder(frames)[2]

        *_, C, H, W = disp.shape
        return disp.view(*B, C, H, W)


class DepthEncoder(nn.Module):
    def __init__(self, disp_net: nn.Module, freeze: bool = True) -> None:
        super().__init__()

        self.disp_net = disp_net.requires_grad_(not freeze).train(not freeze)

    def forward(
        self,
        frames: Float[Tensor, "*b c _h _w"],
        *,
        camera: Camera,
    ) -> Float[Tensor, "*b h w 3"]:
        B, *_, H, W = frames.shape

        frames = rearrange(frames, "b t c h w -> (b t) c h w")
        disp = self.disp_net(frames)[0]
        depth = 1 / disp
        depth = rearrange(depth, "(b t) 1 h w -> b t h w 1", b=B)

        # compute 3d points (equivalent to Eq 6 [https://arxiv.org/abs/2211.14710])
        grid = self._generate_2d_grid(height=H, width=W).to(depth)
        pts = camera.unproject(grid, depth)

        # normalize 3d points (Eq 7 [https://arxiv.org/abs/2211.14710])
        pts_min = reduce(pts, "b t h w c -> b t 1 1 c", "min")
        pts_max = reduce(pts, "b t h w c -> b t 1 1 c", "max")
        return (pts - pts_min) / (pts_max - pts_min)

    @classmethod
    @lru_cache(maxsize=1)
    def _generate_2d_grid(cls, *, height: int, width: int) -> Shaped[Tensor, "h w 2"]:
        x_mesh, y_mesh = torch.meshgrid(
            torch.arange(width),
            torch.arange(height),
            indexing="xy",
        )

        return rearrange([x_mesh, y_mesh], "t h w -> h w t")


class DinoEncoder(nn.Module):
    def __init__(self, dino: nn.Module, freeze: bool = True) -> None:
        super().__init__()

        self.dino = dino.requires_grad_(not freeze).train(not freeze)

    def forward(
        self,
        frames: Float[Tensor, "*b c1 h1 w1"],
    ) -> Float[Tensor, "*b h2 w2 c2"]:
        *B, _, H1, W1 = frames.shape

        frames = frames.view(prod(B), -1, H1, W1)
        feat = self.dino.forward_features(frames)  # pyright: ignore
        tokens = feat["x_norm_patchtokens"]

        H2 = H1 // self.dino.patch_size  # pyright: ignore
        W2 = W1 // self.dino.patch_size  # pyright: ignore
        return tokens.view(*B, H2, W2, -1)


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
            [x.sin(), x.cos()],  # type: ignore
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


class ResNetTokens(nn.Module):
    def __init__(self, tokens_mask=-1) -> None:
        super().__init__()
        self.tokens_mask = tokens_mask

    def forward(
        self,
        x: Float[Tensor, "... c"],
        token_shift: int,
    ) -> Tuple[Float[Tensor, "... c"], Int[Tensor, "..."]]:
        tokens = torch.full(x.shape[:-1], self.tokens_mask).to(x.device) + token_shift

        return x, tokens


class DVAETokens(nn.Module):
    def __init__(self) -> None:
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
    def __init__(self, enc_weights: str, freeze: bool = True) -> None:
        super().__init__()

        self.enc = load_model(enc_weights).requires_grad_(not freeze).train(not freeze)

    def forward(
        self,
        x: Float[Tensor, "b c1 h1 w1"],
    ) -> Float[Tensor, "b c2 h2 w2"]:
        logits = self.enc(x)
        return F.softmax(logits, dim=1)


class SenorDropout(nn.Module):
    def __init__(self, prob: float) -> None:
        super().__init__()
        self.prob = prob

    def forward(
        self,
        embeddings: List[Float[Tensor, "b t c d"]],
    ) -> List[Float[Tensor, "b t c d"]]:
        if len(embeddings) == 0:
            return []

        if self.prob == 0:
            return embeddings

        b, t, *_ = embeddings[0].shape
        num_samples_to_drop = 1 if b == 1 else int(b * self.prob)
        indices = torch.randperm(b)[:num_samples_to_drop]

        dropped_embeddings = []
        for _embedding in embeddings:
            embedding = _embedding.clone()
            # drop everything except last timestep
            embedding[indices, : t - 1] = 0
            dropped_embeddings.append(embedding)

        return dropped_embeddings


class PatchPositionEncoding(nn.Module):
    def __init__(self, num_rows: int, num_cols: int, embedding_dim: int):
        super().__init__()

        self.row_emb = nn.Embedding(
            num_embeddings=num_rows,
            embedding_dim=embedding_dim,
        )
        self.col_emb = nn.Embedding(
            num_embeddings=num_cols,
            embedding_dim=embedding_dim,
        )

    def forward(self, x: Float[Tensor, "*b h w c"]) -> Float[Tensor, "*b h w c"]:
        *_, h, w, _ = x.shape
        row_idxs = torch.arange(h, device=x.device)
        col_idxs = torch.arange(w, device=x.device)
        x += rearrange(self.row_emb(row_idxs), "h c -> h 1 c")
        x += rearrange(self.col_emb(col_idxs), "w c -> 1 w c")

        return x
