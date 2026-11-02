"""The pallet teacher: one crop net for keypoints, load, type and a floor-free metric pose.

Per pallet crop (`keypoints.Crop`, 384 px) the net predicts

    heatmaps      the 12 entry-face keypoints in the HAND-LABEL convention (real frames)
    heatmaps_syn  the same 12 in the SYNTHETIC renderer's convention (mesh corners; its
                  pocket corners sit 6-10 px from where people click them), so neither
                  source teaches the other a biased corner
    loaded, type  two logits (something on the deck; EUR vs other)
    pose          the entry-face centre (forward, left, up; m) and yaw, camera frame

POSE HEAD - scale-invariant translation plus allocentric yaw (as in CDPN / GDR-Net's SITE),
so no floor or mount height enters anywhere and stacked / rack pallets are first-class:

    centre   the face centre's image position, as one more heatmap (`heatmap_centre`, same
             stride-4 Gaussian and decoding as the keypoints): pixel-exact and, unlike a
             regression from pooled features, it knows WHERE in the crop it is. Back to
             frame pixels via the crop, then back-projected along its ray.
    depth    log(Z * side / fx): Z is the forward distance, side / fx the angle the crop
             spans, so the target is the metric width the crop covers at the pallet. It is
             what the crop's content shows (a 0.8 m face filling 70 % of the crop) and does
             not change when the same pallet is cropped looser or tighter.
    yaw      (sin, cos) of yaw minus the azimuth of the centre's viewing ray: a face seen
             square-on along its ray looks the same anywhere in the image, so the net
             predicts the viewpoint-relative angle and the ray puts it back.

Depth and yaw come from an MLP on pooled features plus a coarse 6x6 spatial map of the fused
features (the face's extent and perspective skew in the crop). The MLP also reads the crop's
geometry (crop-centre ray and angular size), so it knows where in the image it is looking
(lens effects, seeing a rack pallet from below), but the parametrisation, not that input, is
what carries position. The alternative - regress
(forward, left, up) directly from crop pixels - would have to infer where the crop sat in
the frame and how big it was from the pixels alone, which a crop destroys.

Camera: Veigel renders are pinhole with the D12 cam_forward intrinsics (x1.5 = `CAM_FORK`'s
fx, fy, cx, cy). Real `cam_fork` is strongly distorted; for real frames the same pinhole
formulas are used for the geometry input and `decode_pose(camera=...)` back-projects through
the fisheye. Depth normalisation by fx is only approximate off-axis on the fisheye.
"""

import math
from typing import Final

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import Dataset

from rmind.pallet.correspondence import NAMES
from rmind.pallet.entry import WIDTH, EntryKeypoints
from rmind.pallet.geometry import CAM_FORK, FisheyeCamera
from rmind.pallet.keypoints import (
    CROP_SIZE,
    PalletKeypointNet,
    decode,
    gaussian_targets,
    heatmap_loss,
)

TYPES: Final = ("eur", "other")
# the D12 cam_forward intrinsics at 1920x1080 (fx, fy, cx, cy): Veigel's pinhole and CAM_FORK's
DEFAULT_K: Final = (CAM_FORK.fx, CAM_FORK.fy, CAM_FORK.cx, CAM_FORK.cy)
SPATIAL: Final = 6  # the coarse spatial map the pose MLP reads, cells per side


class PalletTeacherNet(PalletKeypointNet):
    """`PalletKeypointNet` (same weights layout) plus a synthetic heatmap head and an MLP."""

    def __init__(
        self,
        backbone: str = "hrnet_w18.ms_aug_in1k",
        width: int = 128,
        hidden: int = 256,
        *,
        pretrained: bool = True,
    ) -> None:
        super().__init__(
            backbone, num_keypoints=len(NAMES), width=width, pretrained=pretrained
        )
        self.head_syn = nn.Conv2d(width, len(NAMES), 3, padding=1)
        self.head_centre = nn.Conv2d(width, 1, 3, padding=1)
        deep: int = self.backbone.feature_info.channels()[-1]  # ty: ignore[call-non-callable, unresolved-attribute]
        self.spatial = nn.Sequential(  # stride 4 -> 32, then SPATIAL x SPATIAL
            *(
                nn.Sequential(
                    nn.Conv2d(width if i == 0 else 64, 64, 3, 2, 1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                )
                for i in range(3)
            ),
            nn.AdaptiveAvgPool2d(SPATIAL),
            nn.Flatten(),
        )
        features = 2 * deep + 2 * width + 64 * SPATIAL**2
        self.trunk = nn.Sequential(
            nn.LayerNorm(features),
            nn.Linear(features, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.cls = nn.Linear(hidden, 2)  # loaded, type (other) logits
        self.pose = nn.Sequential(
            nn.Linear(hidden + 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3),  # log depth, sin, cos
        )

    def forward_all(self, images: Tensor, geometry: Tensor) -> dict[str, Tensor]:
        fused, features = self.fuse(images)
        pooled = torch.cat(
            [
                features[-1].mean(dim=(-2, -1)),
                features[-1].amax(dim=(-2, -1)),
                fused.mean(dim=(-2, -1)),
                fused.amax(dim=(-2, -1)),
                self.spatial(fused),
            ],
            dim=1,
        )
        h = self.trunk(pooled)
        logits = self.cls(h)
        pose = self.pose(torch.cat([h, geometry.to(h.dtype)], dim=1))
        return {
            "heatmaps": self.head(fused),
            "heatmaps_syn": self.head_syn(fused),
            "heatmap_centre": self.head_centre(fused),
            "loaded": logits[:, 0],
            "type": logits[:, 1],
            "log_depth": pose[:, 0],
            # (sin, cos), unnormalised: trained towards the unit target, normalised on decode
            "alpha": pose[:, 1:3],
        }


def crop_geometry(crop: Tensor, k: Tensor) -> Tensor:
    """`(B, 3)` crops (x0, y0, side) + `(B, 4)` (fx, fy, cx, cy) -> crop-centre ray and size."""
    centre = crop[:, :2] + crop[:, 2:3] / 2
    return torch.stack(
        [
            (centre[:, 0] - k[:, 2]) / k[:, 0],
            (centre[:, 1] - k[:, 3]) / k[:, 1],
            crop[:, 2] / k[:, 0],
        ],
        dim=1,
    ).float()


def decode_pose(
    out: dict[str, Tensor], crop: Tensor, k: Tensor, camera: FisheyeCamera | None = None
) -> Tensor:
    """Net outputs -> `(B, 4)` (forward, left, up in m; yaw in rad), camera frame.

    Pinhole with `k` unless `camera` is given, then its (fisheye) rays: a pixel's ray is
    scaled to the predicted forward distance.
    """
    crop, k = crop.double(), k.double()
    side = crop[:, 2]
    centre, _ = decode(out["heatmap_centre"].float())
    uv = crop[:, :2] + centre[:, 0].double() * (side / CROP_SIZE)[:, None]
    z = out["log_depth"].double().exp() * k[:, 0] / side
    if camera is None:
        x = (uv[:, 0] - k[:, 2]) / k[:, 0]
        y = (uv[:, 1] - k[:, 3]) / k[:, 1]
    else:  # optical-frame rays, normalised to unit forward
        rays = camera.rays(uv.cpu().numpy()) @ camera.rotation  # robot -> optical
        rays = torch.from_numpy(rays).to(uv)
        x, y = rays[:, 0] / rays[:, 2], rays[:, 1] / rays[:, 2]
    left, up = -x * z, -y * z
    alpha = out["alpha"].double()
    yaw = torch.atan2(alpha[:, 0], alpha[:, 1]) + torch.atan2(left, z)
    return torch.stack([z, left, up, wrap(yaw)], dim=1)


def wrap(angle: Tensor) -> Tensor:
    return torch.remainder(angle + math.pi, 2 * math.pi) - math.pi


class TeacherCrops(Dataset):
    """`EntryKeypoints` crops plus the teacher's targets from per-pallet metadata.

    `meta[name]` may carry `synthetic` (bool: which heatmap head), `loaded` / `type`
    (0/1, missing = unknown) and `pose` (a Veigel `.pose.json` record: `centre_m`,
    `yaw_deg`, `camera`). A horizontal flip mirrors the pose (left, yaw) and the camera
    (cx -> W - 1 - cx), so the flipped crop is a consistent view of a mirrored world.
    """

    def __init__(self, base: EntryKeypoints, meta: dict[str, dict]) -> None:
        self.base = base
        self.meta = meta

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int) -> dict[str, Tensor]:  # ruff: ignore[too-many-locals]
        sample = self.base[i]
        m = self.meta.get(self.base.items[i][0], {})
        flipped = bool(sample["flipped"])
        pose = m.get("pose")
        fx, fy, cx, cy = pose["camera"] if pose else DEFAULT_K
        if flipped:
            cx = WIDTH - 1 - cx
        k = torch.tensor([fx, fy, cx, cy], dtype=torch.float64)
        crop = sample["crop"]
        sample |= {
            "synthetic": torch.tensor(bool(m.get("synthetic"))),
            "intrinsics": k,
            "geometry": crop_geometry(crop[None], k[None])[0],
            "loaded": torch.tensor(float(m.get("loaded", -1))),
            "type": torch.tensor(float(m.get("type", -1))),
            "pose_known": torch.tensor(pose is not None),
            "pose": torch.zeros(4, dtype=torch.float64),
            "centre_target": torch.zeros(1, *sample["targets"].shape[-2:]),
            "log_depth": torch.tensor(0.0),
            "alpha": torch.tensor([0.0, 1.0]),
        }
        if pose is not None:
            fwd, left, up = pose["centre_m"]
            yaw = math.radians(pose["yaw_deg"])
            if flipped:
                left, yaw = -left, -yaw
            u, v = cx - fx * left / fwd, cy - fy * up / fwd
            x0, y0, side = map(float, crop)
            local = torch.tensor([[[u - x0, v - y0]]]) * (CROP_SIZE / side)
            inside = ((local >= 0) & (local < CROP_SIZE)).all(dim=-1)
            alpha = yaw - math.atan2(left, fwd)
            sample |= {
                "pose": torch.tensor([fwd, left, up, yaw], dtype=torch.float64),
                "centre_target": gaussian_targets(local, inside)[0],
                "log_depth": torch.tensor(math.log(fwd * side / fx)),
                "alpha": torch.tensor([math.sin(alpha), math.cos(alpha)]),
            }
        return sample


def _masked_mean(x: Tensor, mask: Tensor) -> Tensor:
    w = mask.to(x.dtype)
    return (x * w).sum() / w.sum().clamp(min=1)


def centre_loss(out: dict[str, Tensor], b: dict[str, Tensor]) -> Tensor:
    """Face-centre heatmap MSE on the crops with a known pose (an outside centre: empty map)."""
    return heatmap_loss(
        out["heatmap_centre"], b["centre_target"], b["pose_known"][:, None]
    )


def aux_losses(out: dict[str, Tensor], b: dict[str, Tensor]) -> dict[str, Tensor]:
    """Masked load / type BCE, depth and yaw L1, each a mean over the known ones."""
    pose = b["pose_known"]
    return {
        "loaded": _masked_mean(
            F.binary_cross_entropy_with_logits(
                out["loaded"], b["loaded"].clamp(min=0), reduction="none"
            ),
            b["loaded"] >= 0,
        ),
        "type": _masked_mean(
            F.binary_cross_entropy_with_logits(
                out["type"], b["type"].clamp(min=0), reduction="none"
            ),
            b["type"] >= 0,
        ),
        "depth": _masked_mean((out["log_depth"] - b["log_depth"]).abs(), pose),
        "yaw": _masked_mean((out["alpha"] - b["alpha"]).abs().sum(dim=1), pose),
    }


def layouts_from_poses(records: dict[str, dict]) -> dict[str, np.ndarray]:
    """Veigel `.pose.json` records -> model name -> `(12, 3)` keypoint layout, `NAMES` order.

    The pallet's own frame, origin at the entry-face centre: x inward, y left, z up.
    """
    out: dict[str, list] = {}
    for r in records.values():
        out.setdefault(r["model"], []).append([r["layout_m"][n] for n in NAMES])
    return {m: np.mean(v, axis=0) for m, v in out.items()}
