"""Few-shot keypoints by DINOv3 feature correspondence: a "visual prompt" from a few clicks.

A handful of support frames with the entry-face keypoints clicked (Label Studio)
define, per keypoint, a set of DINOv3 descriptors. On a new frame:

1. FEATURES. A square crop around the detected pallet box, resized to
   `CROP` px, through the frozen DINOv3 ViT-S/16 (the policy's encoder); the last
   two layers' patch tokens, L2-normalised, give a `CROP/16`-square descriptor grid.
2. MATCH. Per keypoint, cosine similarity of every patch to the support
   descriptors (max over supports), bilinearly upsampled to crop pixels; the
   peak is the match. The pallet's entry face is left/right symmetric and DINO
   features barely tell its left block from its right one, so the search is
   restricted to the matching side of the box ("left" as the truck sees it is
   image-left for a face towards the camera).
3. GEOMETRY. `pose.solve` (floor-constrained, RANSAC, free pitch) on the
   matches, weighted by similarity, rejects inconsistent ones and fills in
   missed points by projecting the solved pose.
4. SNAP. Each point moves to the strongest image corner within `SNAP_PX`
   (Shi-Tomasi + sub-pixel), since block and pocket corners are high-contrast and
   the geometry alone is off by the calibration and pallet-tolerance error.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
import torch
import torch.nn.functional as F

from rmind.pallet.geometry import CAM_FORK, EUR_KEYPOINTS
from rmind.pallet.keypoints import Crop
from rmind.pallet.pose import solve

MODEL: Final = "vit_small_patch16_dinov3.lvd1689m"
CROP: Final = 640
PATCH: Final = 16
SNAP_PX: Final = 14
# the entry-face points a person can click (the far deck corners are often hidden)
NAMES: Final = tuple(n for n in EUR_KEYPOINTS if not n.startswith("pallet_far"))
POINTS_3D: Final = np.array([EUR_KEYPOINTS[n] for n in NAMES])
# which side of the box each point lives on, as the truck sees it (None = centre)
_SIDE: Final = {
    n: (
        "left"
        if n.startswith("pocket_l_") or (n.endswith("_left") and n.startswith("pallet"))
        else "right"
        if n.startswith("pocket_r_")
        or (n.endswith("_right") and n.startswith("pallet"))
        else None
    )
    for n in NAMES
}
_MEAN: Final = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
_STD: Final = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


@dataclass(frozen=True)
class Matches:
    pixels: np.ndarray  # (K, 2) raw image pixels, NaN where nothing usable
    score: np.ndarray  # (K,) cosine similarity of the match (0 where projected/missing)
    solved: bool  # whether a floor-constrained pose backed the points


class DinoMatcher:
    def __init__(self, device: str = "cuda:0") -> None:
        import timm  # ruff: ignore[import-outside-top-level]

        self.device = device
        self.model = (
            timm.create_model(MODEL, pretrained=True, img_size=CROP).to(device).eval()
        )
        self.support: list[
            tuple[np.ndarray, np.ndarray]
        ] = []  # (descriptors (K, C), valid (K,))

    def _features(
        self, image: np.ndarray, box: tuple[float, float, float, float]
    ) -> tuple[torch.Tensor, Crop]:
        crop = Crop.around(box, margin=0.2)
        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255
        patch = (crop.extract(tensor, size=CROP) - _MEAN) / _STD
        with torch.no_grad():
            layers = self.model.forward_intermediates(  # ty: ignore[call-non-callable]
                patch[None].to(self.device),
                indices=2,
                output_fmt="NCHW",
                intermediates_only=True,
            )
        features = torch.cat(list(layers), dim=1)[0]  # (C, h, w)
        return F.normalize(features, dim=0), crop

    def add_support(
        self,
        image: np.ndarray,
        box: tuple[float, float, float, float],
        points: dict[str, tuple[float, float]],
    ) -> None:
        """One clicked frame: `points` maps keypoint name -> raw pixel (missing = not visible)."""
        features, crop = self._features(image, box)
        c, h, w = features.shape
        desc = np.zeros((len(NAMES), c), dtype=np.float32)
        valid = np.zeros(len(NAMES), dtype=bool)
        for k, name in enumerate(NAMES):
            if name not in points:
                continue
            local = (
                crop.to_crop(np.array([points[name]]), size=CROP)[0] / PATCH - 0.5
            )  # patch-grid coordinates
            grid = torch.tensor(
                [[[[2 * local[0] / (w - 1) - 1, 2 * local[1] / (h - 1) - 1]]]],
                dtype=features.dtype,
                device=features.device,
            )
            sample = F.grid_sample(features[None], grid, align_corners=True)[0, :, 0, 0]
            desc[k] = F.normalize(sample, dim=0).cpu().numpy()
            valid[k] = True
        self.support.append((desc, valid))

    def match(  # ruff: ignore[too-many-locals]
        self,
        image: np.ndarray,
        box: tuple[float, float, float, float],
        min_score: float = 0.35,
    ) -> Matches:
        features, crop = self._features(image, box)
        c, h, w = features.shape
        flat = features.reshape(c, -1)
        pixels = np.full((len(NAMES), 2), np.nan)
        score = np.zeros(len(NAMES))
        centre_u = crop.to_crop(np.array([[(box[0] + box[2]) / 2, 0.0]]), size=CROP)[
            0, 0
        ]
        us = np.arange(CROP)[None, :].repeat(CROP, 0)
        for k, name in enumerate(NAMES):
            descs = [torch.from_numpy(d[k]).to(flat) for d, v in self.support if v[k]]
            if not descs:
                continue
            sim = (
                torch
                .stack([d @ flat for d in descs])
                .max(dim=0)
                .values.reshape(1, 1, h, w)
            )
            sim = (
                F
                .interpolate(
                    sim, size=(CROP, CROP), mode="bilinear", align_corners=False
                )[0, 0]
                .cpu()
                .numpy()
            )
            side = _SIDE[name]
            if side == "left":
                sim = np.where(us < centre_u + 0.05 * CROP, sim, -1)
            elif side == "right":
                sim = np.where(us > centre_u - 0.05 * CROP, sim, -1)
            v, u = np.unravel_index(int(np.argmax(sim)), sim.shape)
            if sim[v, u] >= min_score:
                pixels[k] = crop.to_frame(np.array([[u + 0.5, v + 0.5]]), size=CROP)[0]
                score[k] = sim[v, u]
        found = np.isfinite(pixels).all(axis=1)
        estimate = (
            solve(
                CAM_FORK,
                np.nan_to_num(pixels),
                found,
                confidence=score,
                model=POINTS_3D,
                sigma_px=6.0,
            )
            if found.sum() >= 3  # ruff: ignore[magic-value-comparison]
            else None
        )
        if estimate is not None:
            projected, imaged = CAM_FORK.project(estimate.pose.apply(POINTS_3D))
            keep = estimate.inliers
            pixels = np.where(
                keep[:, None], pixels, np.where(imaged[:, None], projected, np.nan)
            )
            score = np.where(keep, score, 0.0)
        return Matches(pixels, score, estimate is not None)


def snap(image: np.ndarray, pixels: np.ndarray, radius: int = SNAP_PX) -> np.ndarray:
    """Move each point to the strongest corner within `radius` px (sub-pixel)."""
    import cv2  # ruff: ignore[import-outside-top-level]

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    out = pixels.copy()
    for k, (u, v) in enumerate(pixels):
        if not np.isfinite(u):
            continue
        x0, y0 = int(max(u - radius, 0)), int(max(v - radius, 0))
        x1, y1 = (
            int(min(u + radius + 1, gray.shape[1])),
            int(min(v + radius + 1, gray.shape[0])),
        )
        if x1 - x0 < 5 or y1 - y0 < 5:  # ruff: ignore[magic-value-comparison]
            continue
        corners = cv2.goodFeaturesToTrack(
            gray[y0:y1, x0:x1], maxCorners=5, qualityLevel=0.05, minDistance=3
        )
        if corners is None:
            continue
        corners = corners.reshape(-1, 2) + np.array([x0, y0])
        best = corners[np.argmin(np.linalg.norm(corners - [u, v], axis=1))]
        refined = cv2.cornerSubPix(
            gray,
            best.reshape(1, 1, 2).astype(np.float32),
            (4, 4),
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.05),
        )
        out[k] = refined.reshape(2)
    return out
