"""PalLoc6D (TU Hamburg, CC0) as keypoint training data for `PalletKeypointNet`.

PalLoc6D: 50k NDDS-rendered 640x480 images of one Euro pallet, domain-randomised,
each with the pallet's 6D pose and its 3D-box corners (`cuboid`, camera frame, cm)
and their projections (`projected_cuboid`). https://doi.org/10.15480/336.4470

The 8 box corners are the keypoints. They do not depend on which face the forks
enter, so they transfer to the D12 (which enters the 0.8 m face) whatever the
renderer considered "front". NDDS numbers them in the object's own frame, and the
pallet box looks the same turned by 180 deg, so the raw order is not learnable.
`canonical_corners` relabels them by viewpoint, the way the detector will see a
pallet it approaches:

    near/far     the long (1.2 m) axis is oriented AWAY from the camera, so "near"
                 is the 0.8 m face closer to the camera (the entry face on a D12
                 approach); ill-defined only when the pallet is seen side-on
    bottom/top   along the height axis, oriented to the camera's up
    left/right   up x away, i.e. +y of `geometry`'s pallet frame

so each label has a fixed pallet-frame point: `BOX_CORNERS_3D`, ready for
`pose.solve`.
"""

import json
from pathlib import Path
from typing import Final

import numpy as np
import torch
from torch.utils.data import Dataset

from rmind.pallet.geometry import EUR_HEIGHT, EUR_LENGTH, EUR_WIDTH
from rmind.pallet.keypoints import CROP_SIZE, Crop, gaussian_targets

BOX_CORNER_NAMES: Final = tuple(
    f"{end}_{level}_{side}"
    for end in ("near", "far")
    for level in ("bottom", "top")
    for side in ("left", "right")
)
# pallet frame (origin: near-face centre on the floor, +x away, +y left, +z up)
BOX_CORNERS_3D: Final = np.array([
    [
        0.0 if end == "near" else EUR_LENGTH,
        EUR_WIDTH / 2 if side == "left" else -EUR_WIDTH / 2,
        0.0 if level == "bottom" else EUR_HEIGHT,
    ]
    for end in ("near", "far")
    for level in ("bottom", "top")
    for side in ("left", "right")
])
BOX_FLIP_INDEX: Final = tuple(i ^ 1 for i in range(8))  # swap left/right

# NDDS cuboid dimensions, cm: (x, y, z) = (0.8 m width, 0.144 m height, 1.2 m length)
_DIMS_CM: Final = np.array([80.0, 14.4, 120.0])
_MEAN: Final = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
_STD: Final = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


def canonical_corners(cuboid_cm: np.ndarray, projected: np.ndarray) -> np.ndarray:
    """Reorder one NDDS object's projected corners into `BOX_CORNER_NAMES` order."""
    centre = cuboid_cm.mean(axis=0)
    rel = cuboid_cm - centre
    # recover the box axes from corner 0's three edges, matched by length
    edges = cuboid_cm - cuboid_cm[0]
    lengths = np.linalg.norm(edges, axis=1)
    axes = []
    for dim in _DIMS_CM:
        j = int(np.argmin(np.abs(lengths - dim) + (lengths < 1e-6) * 1e9))  # ruff: ignore[magic-value-comparison]
        axes.append(edges[j] / max(lengths[j], 1e-9))
    _width_axis, up, long_axis = axes
    if up @ np.array([0.0, -1.0, 0.0]) < 0:  # camera y is down
        up = -up
    if long_axis @ centre < 0:  # point away from the camera
        long_axis = -long_axis
    left = np.cross(up, long_axis)
    out = np.empty((8, 2))
    for k in range(8):
        near = rel[k] @ long_axis < 0
        top = rel[k] @ up > 0
        is_left = rel[k] @ left > 0
        index = (0 if near else 4) + (2 if top else 0) + (0 if is_left else 1)
        out[index] = projected[k]
    return out


def load_labels(json_path: Path) -> tuple[np.ndarray, float] | None:
    """Canonical 8 corners (pixels) of the image's pallet and its visibility."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    objects = [o for o in data["objects"] if o["class"] == "pallet"]
    if not objects:
        return None
    o = objects[0]
    return canonical_corners(
        np.array(o["cuboid"]), np.array(o["projected_cuboid"])
    ), float(o["visibility"])


class KeypointCrops(Dataset):
    """Square crops around a pallet with Gaussian heatmap targets for its 8 corners.

    Items are (image path, canonical corners (8, 2) in pixels, box or None). The
    crop is built around the box when given (a detector box, as at inference),
    else around the corners clipped to the image (what a detector would draw).
    Training adds: box jitter (scale and shift), a left/right flip with the label
    swap, colour jitter, and TRUNCATION - the image blanked past a random line
    through the pallet, bottom or side, with the box cut there too. That is the
    pallet at the fork tips running off the bottom of `cam_fork`, which the
    synthetic renders never show. Blanked corners keep their labels.
    """

    def __init__(
        self,
        items: list[tuple[Path, np.ndarray, np.ndarray | None]],
        *,
        train: bool,
        seed: int = 0,
    ) -> None:
        import torchvision.transforms.v2 as T  # ruff: ignore[import-outside-top-level]

        self.items = items
        self.train = train
        self.rng = np.random.default_rng(seed)
        self.color = T.Compose([
            T.ColorJitter(0.4, 0.4, 0.4, 0.05),
            T.RandomGrayscale(0.1),
            T.RandomApply([T.GaussianBlur(5, (0.1, 2.0))], p=0.3),
        ])

    def __len__(self) -> int:
        return len(self.items)

    def _truncate(self, image: torch.Tensor, box: np.ndarray) -> np.ndarray:
        """Blank the image past a random line through the box; return the cut box."""
        x0, y0, x1, y1 = box
        side = self.rng.choice(["bottom", "left", "right"], p=[0.6, 0.2, 0.2])
        t = self.rng.uniform(0.35, 0.9)
        if side == "bottom":
            cut = int(y0 + t * (y1 - y0))
            image[:, cut:, :] = 0
            return np.array([x0, y0, x1, cut])
        if side == "left":
            cut = int(x1 - t * (x1 - x0))
            image[:, :, :cut] = 0
            return np.array([cut, y0, x1, y1])
        cut = int(x0 + t * (x1 - x0))
        image[:, :, cut:] = 0
        return np.array([x0, y0, cut, y1])

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        from PIL import Image  # ruff: ignore[import-outside-top-level]

        path, corners, box = self.items[i]
        image = (
            torch
            .from_numpy(np.array(Image.open(path).convert("RGB")))
            .permute(2, 0, 1)
            .float()
            / 255
        )
        _, h, w = image.shape
        corners = corners.copy()
        if box is None:
            box = np.r_[corners.min(axis=0), corners.max(axis=0)]
        box = np.clip(
            np.asarray(box, dtype=np.float64),
            [0, 0, 0, 0],
            [w - 1, h - 1, w - 1, h - 1],
        )
        if self.train and self.rng.random() < 0.5:  # ruff: ignore[magic-value-comparison]
            image = image.flip(-1)
            corners[:, 0] = w - 1 - corners[:, 0]
            corners = corners[list(BOX_FLIP_INDEX)]
            box = np.array([w - 1 - box[2], box[1], w - 1 - box[0], box[3]])
        if self.train and self.rng.random() < 0.3:  # ruff: ignore[magic-value-comparison]
            box = self._truncate(image, box)
        crop = Crop.around(tuple(box))
        if self.train:
            scale = self.rng.uniform(0.85, 1.3)
            shift = self.rng.uniform(-0.1, 0.1, size=2) * crop.side
            side = crop.side * scale
            crop = Crop(
                crop.x0 + (crop.side - side) / 2 + shift[0],
                crop.y0 + (crop.side - side) / 2 + shift[1],
                side,
            )
        patch = crop.extract(image)
        if self.train:
            patch = self.color(patch)
        local = torch.from_numpy(crop.to_crop(corners)).float()
        visible = ((local >= 0) & (local < CROP_SIZE)).all(dim=-1)
        return {
            "image": (patch - _MEAN) / _STD,
            "keypoints": local,
            "visible": visible,
            "targets": gaussian_targets(local[None], visible[None])[0],
        }


def palloc6d_items(
    files: list[Path], min_visibility: float = 0.3
) -> list[tuple[Path, np.ndarray, None]]:
    items = []
    for f in files:
        labels = load_labels(f)
        if labels is not None and labels[1] > min_visibility:
            items.append((f.with_suffix(".png"), labels[0], None))
    return items


def pseudo_label_items(root: Path) -> list[tuple[Path, np.ndarray, np.ndarray]]:
    """Real `cam_fork` frames labelled by `scripts.pallet_pseudo_labels`."""
    import polars as pl  # ruff: ignore[import-outside-top-level]

    labels = pl.read_parquet(root / "labels.parquet")
    return [
        (
            root / "images" / f"{r['job']}_{r['frame']:06d}.jpg",
            np.array(r["corners"]),
            np.array(r["box"]),
        )
        for r in labels.iter_rows(named=True)
    ]
