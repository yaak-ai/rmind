import sys
from math import prod
from pathlib import Path
from typing import override

import torch
from torch import Tensor, nn

_VJEPA2_ROOT = str(Path("/home/max/Code/vjepa2"))


def _ensure_vjepa2_on_path() -> None:
    if _VJEPA2_ROOT not in sys.path:
        sys.path.insert(0, _VJEPA2_ROOT)


def _clean_keys(state_dict: dict) -> dict:
    # Strip "module." and "backbone." prefixes written by vjepa2 DDP trainer.
    cleaned = {}
    for key, val in state_dict.items():
        key = key.replace("module.", "").replace("backbone.", "")  # noqa: PLW2901
        cleaned[key] = val
    return cleaned


class VjepaBackbone(nn.Module):
    """V-JEPA 2.1 ViT-L/16 image encoder.

    Wraps the vjepa2 VisionTransformer so it presents the same interface as
    TimmBackbone: accepts (*B, C, H, W) and returns (*B, N_patches, embed_dim).
    The encoder is fully trainable during rmind pre-training; the vjepa2 DINOv3
    assumption in CLAUDE.md was found to be incorrect — see Phase 1 report.
    """

    def __init__(
        self, checkpoint_path: str, img_size: tuple[int, int] = (256, 256)
    ) -> None:
        super().__init__()
        _ensure_vjepa2_on_path()
        from app.vjepa_2_1.models.vision_transformer import (  # noqa: PLC0415  # ty: ignore[unresolved-import]
            vit_large,  # type: ignore[import]
        )

        # Canonical vjepa2.1 ViT-L kwargs from src/hub/backbones.py:229-241.
        # img_temporal_dim_size=1 activates per-frame (image) mode when the
        # temporal dim of a 5-D input equals 1.
        self.encoder: nn.Module = vit_large(
            patch_size=16,
            img_size=img_size,
            num_frames=64,
            tubelet_size=2,
            use_sdpa=True,
            use_rope=True,
            img_temporal_dim_size=1,
            interpolate_rope=True,
        )

        raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        missing, unexpected = self.encoder.load_state_dict(
            _clean_keys(raw["ema_encoder"]), strict=True
        )
        if missing or unexpected:
            msg = f"Checkpoint key mismatch: missing={missing}, unexpected={unexpected}"
            raise RuntimeError(msg)

    @override
    def forward(self, x: Tensor) -> Tensor:
        # x: (*B, C, H, W) — same contract as TimmBackbone.
        # Unsqueeze temporal dim so VisionTransformer takes the img_temporal_dim_size=1
        # branch (patch_embed_img + img_mod_embed) instead of the video branch.
        *b, c, h, w = x.shape
        x = x.view(prod(b), c, 1, h, w)
        x = self.encoder(x)  # (prod(B), N_patches, 1024)
        return x.view(*b, x.shape[-2], x.shape[-1])  # (*B, N_patches, 1024)
