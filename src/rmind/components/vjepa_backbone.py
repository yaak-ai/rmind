from math import prod
from typing import override

import torch
from torch import Tensor, nn


class VjepaBackbone(nn.Module):
    """V-JEPA 2.1 ViT-L/16 image encoder.

    Wraps the vjepa2 VisionTransformer so it presents the same interface as
    TimmBackbone: accepts (*B, C, H, W) and returns (*B, N_patches, embed_dim).
    Weights are downloaded automatically via torch.hub on first use.
    """

    def __init__(self) -> None:
        super().__init__()
        encoder, _ = torch.hub.load(
            "facebookresearch/vjepa2",
            "vjepa2_1_vit_large_384",
            pretrained=True,
            trust_repo=True,
        )
        self.encoder: nn.Module = encoder

    @override
    def forward(self, x: Tensor) -> Tensor:
        # x: (*B, C, H, W) — same contract as TimmBackbone.
        # Unsqueeze temporal dim so VisionTransformer takes the img_temporal_dim_size=1
        # branch (patch_embed_img + img_mod_embed) instead of the video branch.
        *b, c, h, w = x.shape
        x = x.view(prod(b), c, 1, h, w)
        x = self.encoder(x)  # (prod(B), N_patches, 1024)
        return x.view(*b, x.shape[-2], x.shape[-1])  # (*B, N_patches, 1024)


class VjepaVideoBackbone(nn.Module):
    """V-JEPA 2.1 ViT-L/16 *video* encoder over a frame history.

    Unlike VjepaBackbone (per-frame image branch, T=1), this feeds the whole
    temporal window through the VisionTransformer video branch so the encoder
    can capture dynamics (closing speed, approaching stops) across the history
    rather than a single instant.

    Input:  (*B, T, C, H, W) — a clip of T frames (T must be even for the
            tubelet_size=2 temporal patch embedding).
    Output: (*B, N_tokens, embed_dim) with N_tokens = (T/2) * N_spatial.

    The same torch.hub checkpoint as VjepaBackbone is reused; only the forward
    path differs (video branch vs image branch), so an unfrozen pre-train can
    share one set of encoder weights across both the per-frame and video
    representations if desired.
    """

    def __init__(self) -> None:
        super().__init__()
        encoder, _ = torch.hub.load(
            "facebookresearch/vjepa2",
            "vjepa2_1_vit_large_384",
            pretrained=True,
            trust_repo=True,
        )
        self.encoder: nn.Module = encoder

    @override
    def forward(self, x: Tensor) -> Tensor:
        # x: (*B, T, C, H, W). VisionTransformer wants (B, C, T, H, W).
        *b, t, c, h, w = x.shape
        x = x.view(prod(b), t, c, h, w).transpose(1, 2)  # (B, C, T, H, W)
        x = self.encoder(x)  # (B, (T/2)*N_spatial, embed_dim)
        return x.view(*b, x.shape[-2], x.shape[-1])
