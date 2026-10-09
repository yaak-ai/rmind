"""The STRETCH image preprocessing (patch/paper): full field of view on a square grid.

`preprocess` maps a NATIVE camera frame (uint8 RGB, CHW) straight to the model
grid with ONE anisotropic torch bilinear antialias resize, then rounds to uint8
-- no letterbox, no crop. Used by the paper-faithful runs (Patch Policy, arXiv
2607.18236, real robot: DINOv2 ViT-S/14 at 224x224 = 16 x 16 patches/camera).

Why stretch and not the letterbox of `rmind.data.nero_image` at 224x224: the
cube rig's frames are 16:9 (base, 1920x1080) and 16:10 (sides, 1280x800). An
isotropic fit into 224x224 is 224x126 / 224x140 plus 98 / 84 black rows -- the
SAME pixels as the 140x224 grid, with ~35-45% of the 256 patches pure padding.
A centre crop keeps the scale but cuts 22% / 19% off each side, where the two
arms and the hand-overs live. The stretch keeps the full field of view and puts
real content in every one of the 256 patches, at 1.78x (base) / 1.6x (sides)
the vertical resolution of the letterbox; the cost is a non-square pixel aspect
DINOv2 was not pretrained on (frozen encoder, the trunk adapts).

SELF-CONTAINED ON PURPOSE (torch only), like `nero_image`: serving would vendor
this file and pin it by SHA256 (`PREPROCESSING_ID` + `preprocessing_sha256()`);
the frame cache manifest records both and refuses a mismatch. Serving does NOT
implement this mode yet (nutron-cli vendors only the letterbox `nero_image.py`).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

__all__ = ["PREPROCESSING_ID", "NeroImageStretch", "preprocess", "preprocessing_sha256"]

PREPROCESSING_ID = "torch_bilinear_antialias_stretch_v1"


def preprocess(frames: Tensor, input_hw: tuple[int, int]) -> Tensor:
    """`(..., 3, H, W)` uint8 RGB at native resolution -> `(..., 3, H_in, W_in)` uint8.

    Raises:
        TypeError: if `frames` is not uint8.
    """
    if frames.dtype != torch.uint8:
        msg = f"preprocess expects uint8 native RGB, got {frames.dtype}"
        raise TypeError(msg)
    *batch, c, h, w = frames.shape
    th, tw = int(input_hw[0]), int(input_hw[1])
    x = frames.reshape(-1, c, h, w).to(torch.float32)
    if (h, w) != (th, tw):
        x = F.interpolate(
            x, size=(th, tw), mode="bilinear", align_corners=False, antialias=True
        )
    x = x.round().clamp(0, 255).to(torch.uint8)
    return x.reshape(*batch, c, th, tw)


def preprocessing_sha256() -> str:
    """SHA256 of this file."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class NeroImageStretch(nn.Module):
    """`preprocess` as a module (for rbyte's `TransformedSource`)."""

    def __init__(self, input_hw: tuple[int, int] = (224, 224)) -> None:
        super().__init__()
        self.input_hw = (int(input_hw[0]), int(input_hw[1]))

    def forward(self, frames: Tensor) -> Tensor:
        return preprocess(frames, self.input_hw)
