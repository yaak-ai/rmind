"""THE image preprocessing of the nero patch family, train and serve (P10).

One function, `preprocess`, maps a NATIVE camera frame (uint8 RGB, CHW) to the
model grid: torch bilinear antialias resize straight to the isotropic fit, round
to uint8, then symmetric letterbox padding with `PAD_VALUE`. Training calls it
on the frames rbyte decodes at native resolution (`rbyte.streams.transformed.TransformedSource`
around a transform-free `TorchCodecVideoSource`); serving calls it on the relay
frame. Same function, same bytes: the only remaining train/serve difference is
the codec (mp4 vs relay JPEG), which no resize can remove and which is measured
separately.

Why uint8 out: rounding once, at the end of the resize, makes the output a
discrete value both sides reproduce exactly; feeding the float resize straight
to the model would make a 1-ulp difference in the resampler a different input.
The model then maps uint8 -> unit float (/255) and applies the ImageNet norm
in-graph, as the contract's `value_range: unit` / `normalize_in_graph: true`
says.

SELF-CONTAINED ON PURPOSE (torch only, no rmind imports): nutron-cli serving
vendors this file verbatim and pins it by SHA256 -- the contract's
`image.preprocessing_sha256` is the hash of THIS FILE and
`image.preprocessing_id` is `PREPROCESSING_ID`. Any edit (a comment included)
changes the hash; bump `PREPROCESSING_ID` on a semantic change.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

__all__ = [
    "PAD_VALUE",
    "PREPROCESSING_ID",
    "Geometry",
    "NeroImagePreprocess",
    "geometry",
    "preprocess",
    "preprocessing_sha256",
]

PREPROCESSING_ID = "torch_bilinear_antialias_letterbox_v1"
PAD_VALUE = 0


@dataclass(frozen=True)
class Geometry:
    """Where a native `(W, H)` frame lands on the `(H_in, W_in)` model grid."""

    native_wh: tuple[int, int]
    resize_wh: tuple[int, int]
    pad_ltrb: tuple[int, int, int, int]
    input_hw: tuple[int, int]

    def as_dict(self) -> dict[str, list[int]]:
        return {
            "native_wh": list(self.native_wh),
            "resize_wh": list(self.resize_wh),
            "pad_ltrb": list(self.pad_ltrb),
            "input_hw": list(self.input_hw),
        }


def geometry(native_wh: tuple[int, int], input_hw: tuple[int, int]) -> Geometry:
    """Isotropic fit of `native_wh` into `input_hw`, padding split symmetrically.

    1920x1080 -> 224x126 + 7 rows top and bottom = 140x224;
    1280x800 -> 224x140, no padding.
    """
    w, h = int(native_wh[0]), int(native_wh[1])
    th, tw = int(input_hw[0]), int(input_hw[1])
    scale = min(tw / w, th / h)
    rw, rh = round(w * scale), round(h * scale)
    rw, rh = min(rw, tw), min(rh, th)
    left = (tw - rw) // 2
    top = (th - rh) // 2
    return Geometry(
        native_wh=(w, h),
        resize_wh=(rw, rh),
        pad_ltrb=(left, top, tw - rw - left, th - rh - top),
        input_hw=(th, tw),
    )


def preprocess(frames: Tensor, input_hw: tuple[int, int]) -> Tensor:
    """`(..., 3, H, W)` uint8 RGB at native resolution -> `(..., 3, H_in, W_in)` uint8.

    Raises:
        TypeError: if `frames` is not uint8.
    """
    if frames.dtype != torch.uint8:
        msg = f"preprocess expects uint8 native RGB, got {frames.dtype}"
        raise TypeError(msg)
    *batch, c, h, w = frames.shape
    g = geometry((w, h), input_hw)
    x = frames.reshape(-1, c, h, w).to(torch.float32)
    if (g.resize_wh[0], g.resize_wh[1]) != (w, h):
        x = F.interpolate(
            x,
            size=(g.resize_wh[1], g.resize_wh[0]),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    x = x.round().clamp(0, 255).to(torch.uint8)
    left, top, right, bottom = g.pad_ltrb
    if any(g.pad_ltrb):
        x = F.pad(x, (left, right, top, bottom), value=PAD_VALUE)
    return x.reshape(*batch, c, *g.input_hw)


def preprocessing_sha256() -> str:
    """SHA256 of this file -- the contract's `image.preprocessing_sha256`."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class NeroImagePreprocess(nn.Module):
    """`preprocess` as a module (for rbyte's `TransformedSource`)."""

    def __init__(self, input_hw: tuple[int, int] = (140, 224)) -> None:
        super().__init__()
        self.input_hw = (int(input_hw[0]), int(input_hw[1]))

    def forward(self, frames: Tensor) -> Tensor:
        return preprocess(frames, self.input_hw)
