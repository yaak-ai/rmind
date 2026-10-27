"""The keypoint network: a full-resolution crop of `cam_fork` -> per-keypoint heatmaps.

Pipeline at runtime:

    1920x1080 frame -> pallet box (proposer) -> square crop, resized to
    `CROP_SIZE` -> heatmaps at stride 4 -> sub-pixel peaks + peak confidence ->
    back to full-frame pixels -> `pose.solve`

The crop matters: the policy's 256x144 input is 7.5x downscaled, which at 16 px
per DINOv3 ViT-S/16 patch puts a patch at ~120 full-res pixels - far too coarse for the
~1 px keypoints the pose accuracy budget assumes. A 384 px crop of a pallet that
spans 300-1400 px keeps it at 1-4 full-res pixels per crop pixel, and a
log-parabola peak fit on a stride-4 heatmap recovers well under a heatmap cell.

The proposer for the PoC is zero-shot Grounding DINO (`scripts/pallet_poc_zeroshot`).
Once the keypoint net exists it can propose its own crop from the previous
frame's keypoints, so the proposer only runs to (re)acquire.

Architecture: a timm backbone (`features_only`), a small top-down FPN to stride
4, a 3x3 conv head with one channel per keypoint. Targets are unnormalised
Gaussians; the loss is visibility-masked MSE. Deliberately plain - the keypoint
set and the label pipeline are the PoC's open questions, not the head.
"""

from typing import Final, NamedTuple

import numpy as np
import timm
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from rmind.pallet.geometry import KEYPOINT_NAMES

CROP_SIZE: Final = 384
STRIDE: Final = 4
# the crop is the proposer's box grown by this much on each side, then squared
CROP_MARGIN: Final = 0.15


class Crop(NamedTuple):
    """A square crop of the full frame: top-left corner and side, in pixels."""

    x0: float
    y0: float
    side: float

    @classmethod
    def around(
        cls, box: tuple[float, float, float, float], margin: float = CROP_MARGIN
    ) -> "Crop":
        """Square crop around an (x0, y0, x1, y1) box, grown by `margin`.

        The crop may extend past the frame; `extract` pads with zeros.
        """
        x0, y0, x1, y1 = box
        side = max(x1 - x0, y1 - y0) * (1 + 2 * margin)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        return cls(cx - side / 2, cy - side / 2, side)

    def to_crop(self, pixels: np.ndarray, size: int = CROP_SIZE) -> np.ndarray:
        return (pixels - np.array([self.x0, self.y0])) * (size / self.side)

    def to_frame(self, pixels: np.ndarray, size: int = CROP_SIZE) -> np.ndarray:
        return pixels * (self.side / size) + np.array([self.x0, self.y0])

    def extract(self, image: Tensor, size: int = CROP_SIZE) -> Tensor:
        """`(C, H, W)` float image -> `(C, size, size)` crop, bilinear, zero-padded."""
        _, h, w = image.shape
        # crop pixel j covers [j, j+1) in crop coordinates; sample at its centre.
        # Frame coordinates put pixel centres on integers, as cv2 and grid_sample
        # with align_corners=True do
        centres = (torch.arange(size, dtype=image.dtype) + 0.5) * (self.side / size)
        xs, ys = centres + self.x0, centres + self.y0
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([gx / (w - 1) * 2 - 1, gy / (h - 1) * 2 - 1], dim=-1)
        return F.grid_sample(
            image[None],
            grid[None].to(image.device),
            align_corners=True,
            padding_mode="zeros",
        )[0]


def _conv_bn_relu(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class PalletKeypointNet(nn.Module):
    """Heatmaps at stride 4 from a CNN pyramid or a plain ViT.

    CNN backbones (e.g. `resnet34`) give strides 4-32 and a top-down FPN merges
    them. ViTs (e.g. `vit_small_patch16_dinov3.lvd1689m`, the policy's encoder)
    give every layer at stride 16: four layers are summed, upsampled to stride 4,
    and joined with a small convolutional branch on the image itself, since a
    16 px patch grid alone cannot place a corner to a pixel.
    """

    def __init__(
        self,
        backbone: str = "resnet34",
        num_keypoints: int = len(KEYPOINT_NAMES),
        width: int = 128,
        *,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.vit = backbone.startswith(("vit", "deit", "eva"))
        if self.vit:
            kwargs = {"out_indices": (2, 5, 8, 11), "img_size": CROP_SIZE}
        else:
            # the stride 4-32 stages: indices 1-4 for a ResNet (which has a stride-2
            # stem level), 0-3 for a ConvNeXt
            reductions = timm.create_model(
                backbone, pretrained=False, features_only=True
            ).feature_info.reduction()  # ty: ignore[call-non-callable, unresolved-attribute]
            kwargs = {
                "out_indices": tuple(i for i, r in enumerate(reductions) if r >= STRIDE)
            }
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, features_only=True, **kwargs
        )
        channels: list[int] = self.backbone.feature_info.channels()  # ty: ignore[call-non-callable, unresolved-attribute]
        self.lateral = nn.ModuleList(nn.Conv2d(c, width, 1) for c in channels)
        if self.vit:
            self.upsample = nn.ModuleList([
                _conv_bn_relu(width, width),
                _conv_bn_relu(width, width),
            ])
            self.detail = nn.Sequential(
                _conv_bn_relu(3, 32, 2), _conv_bn_relu(32, width, 2)
            )
        self.smooth = _conv_bn_relu(width, width)
        self.head = nn.Conv2d(width, num_keypoints, 3, padding=1)

    def forward(self, images: Tensor) -> Tensor:
        """`(B, 3, S, S)` normalised crops -> `(B, K, S/4, S/4)` heatmaps."""
        features = self.backbone(images)
        if self.vit:
            x = sum(
                lateral(f) for lateral, f in zip(self.lateral, features, strict=True)
            )
            for block in self.upsample:  # stride 16 -> 8 -> 4
                x = block(
                    F.interpolate(
                        x, scale_factor=2, mode="bilinear", align_corners=False
                    )
                )
            x += self.detail(images)
        else:
            x = self.lateral[-1](features[-1])
            for lateral, feature in zip(
                reversed(self.lateral[:-1]), reversed(features[:-1]), strict=True
            ):
                x = F.interpolate(
                    x, size=feature.shape[-2:], mode="bilinear", align_corners=False
                )
                x += lateral(feature)
        return self.head(self.smooth(x))


def gaussian_targets(
    keypoints: Tensor,
    visible: Tensor,
    size: int = CROP_SIZE,
    stride: int = STRIDE,
    sigma: float = 2.0,
) -> Tensor:
    """`(B, K, 2)` crop pixels + `(B, K)` visibility -> `(B, K, S/4, S/4)` Gaussians.

    `sigma` is in heatmap cells. Invisible keypoints get an all-zero map, so the
    net learns to say "not here" rather than being left unsupervised.
    """
    n = size // stride
    grid = (
        torch.arange(n, dtype=keypoints.dtype, device=keypoints.device) + 0.5
    ) * stride
    # heatmap cell centres in crop pixels; distance in cells
    dx = (grid[None, None, None, :] - keypoints[..., 0, None, None]) / stride
    dy = (grid[None, None, :, None] - keypoints[..., 1, None, None]) / stride
    heat = torch.exp(-(dx**2 + dy**2) / (2 * sigma**2))
    return heat * visible[..., None, None].to(heat.dtype)


def heatmap_loss(heatmaps: Tensor, targets: Tensor, labelled: Tensor) -> Tensor:
    """MSE over keypoints that have a label at all (`labelled`, `(B, K)`).

    `labelled` is distinct from visibility: an out-of-frame keypoint IS labelled
    (its target is empty), a keypoint the tag pipeline could not resolve is not.
    """
    per_kp = ((heatmaps - targets) ** 2).mean(dim=(-2, -1))
    weight = labelled.to(per_kp.dtype)
    return (per_kp * weight).sum() / weight.sum().clamp(min=1)


def decode(heatmaps: Tensor, stride: int = STRIDE) -> tuple[Tensor, Tensor]:
    """Heatmaps -> `(B, K, 2)` crop pixels and `(B, K)` peak confidence.

    Argmax, then a parabola through the log of the peak and its two neighbours
    on each axis (DARK-style): exact for a Gaussian peak, so sub-cell precision
    is limited by the net, not the decoding. Confidence is the raw peak value
    (targets peak at 1).
    """
    b, k, h, w = heatmaps.shape
    flat = heatmaps.reshape(b, k, -1)
    confidence, index = flat.max(dim=-1)
    py, px = index // w, index % w
    log = flat.clamp(min=1e-6).log()

    def offset(i: Tensor, step: int, size: int, centre: Tensor) -> Tensor:
        lo = log.gather(-1, (index - step * (i > 0))[..., None])[..., 0]
        hi = log.gather(-1, (index + step * (i < size - 1))[..., None])[..., 0]
        curvature = lo - 2 * centre + hi
        # a flat or upturned neighbourhood gives no offset
        delta = 0.5 * (lo - hi) / torch.where(curvature < -1e-6, curvature, -torch.inf)  # ruff: ignore[magic-value-comparison]
        return delta.clamp(-0.5, 0.5)

    centre = log.gather(-1, index[..., None])[..., 0]
    x = px.to(heatmaps.dtype) + offset(px, 1, w, centre)
    y = py.to(heatmaps.dtype) + offset(py, w, h, centre)
    pixels = (torch.stack([x, y], dim=-1) + 0.5) * stride
    return pixels, confidence
