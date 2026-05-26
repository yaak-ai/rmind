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
