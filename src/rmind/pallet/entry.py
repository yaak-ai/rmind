"""Entry-face keypoint data: hand-labelled real `cam_fork` frames, 12 points.

The 12 points are the pallet entry face as a person can click it
(`correspondence.NAMES`): the four face corners (`pallet_*`) and the four corners
of each fork pocket (`pocket_l_*`, `pocket_r_*`; a pocket's bottom corners are the
floor contacts of the blocks around it). Left/right are as the truck sees the face.

Labels: `{image name: {keypoint name: [u, v]}}` in raw 1920x1080 pixels; a missing
point means "not visible" and is excluded from the loss (not taught as absent).

Crops come from two sources so the net is robust to the box it gets at inference:
the Grounding DINO pallet box (covers the whole pallet incl. deck and load) and
the labelled points' own bounding box, both scale/shift-jittered in training.
"""

import json
from pathlib import Path
from typing import Final

import numpy as np
import torch
from torch.utils.data import Dataset

from rmind.pallet.camera_aug import camera_realism
from rmind.pallet.correspondence import NAMES, POINTS_3D
from rmind.pallet.keypoints import CROP_SIZE, Crop, RectCrop, gaussian_targets

FLIP_INDEX: Final = tuple(
    int(np.argmin(np.linalg.norm(POINTS_3D - p * np.array([1.0, -1.0, 1.0]), axis=1)))
    for p in POINTS_3D
)
MEAN: Final = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD: Final = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
WIDTH, HEIGHT = 1920, 1080


def load_labels(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """name -> (points (12, 2) with NaN for missing, labelled mask (12,))."""
    out = {}
    for name, pts in json.loads(path.read_text(encoding="utf-8")).items():
        xy = np.full((len(NAMES), 2), np.nan)
        for k, (u, v) in pts.items():
            xy[NAMES.index(k)] = (u, v)
        out[name] = (xy, np.isfinite(xy).all(axis=1))
    return out


def crop_for_box(
    box: np.ndarray, rng: np.random.Generator | None, *, wide: bool = False
) -> Crop:
    """Crop around a box; with `rng`, jittered in scale and position.

    `wide` jitters much more (scale 0.7-2.0, shift 25 %) so the net sees loose and
    off-centre crops, as a coarse proposer or a far pallet gives.
    """
    crop = Crop.around(tuple(box))
    if rng is None:
        return crop
    scale = rng.uniform(0.7, 2.0) if wide else rng.uniform(0.85, 1.35)
    side = crop.side * scale
    shift = (
        rng.uniform(-0.25 if wide else -0.12, 0.25 if wide else 0.12, size=2)
        * crop.side
    )
    return Crop(
        crop.x0 + (crop.side - side) / 2 + shift[0],
        crop.y0 + (crop.side - side) / 2 + shift[1],
        side,
    )


def rect_crop_for_box(
    box: np.ndarray,
    rng: np.random.Generator | None,
    *,
    margin: float,
    wide: bool = False,
) -> RectCrop:
    """Non-square crop around a box (`RectCrop.around`); with `rng`, jittered in scale
    and position independently per axis (`wide`: looser, as for `crop_for_box`)."""
    crop = RectCrop.around(tuple(box), margin)
    if rng is None:
        return crop
    lo, hi, shift = (0.8, 1.5, 0.2) if wide else (0.85, 1.25, 0.1)
    w, h = crop.w * rng.uniform(lo, hi), crop.h * rng.uniform(lo, hi)
    dx, dy = rng.uniform(-shift, shift, size=2) * np.array([crop.w, crop.h])
    return RectCrop(
        crop.x0 + (crop.w - w) / 2 + dx, crop.y0 + (crop.h - h) / 2 + dy, w, h
    )


class EntryKeypoints(Dataset):
    def __init__(  # ruff: ignore[too-many-arguments]
        self,
        root: Path,
        labels: dict[str, tuple[np.ndarray, np.ndarray]],
        detector_boxes: dict[str, list[float] | None],
        *,
        train: bool,
        seed: int = 0,
        quality_aug: bool = False,
        crop_shape: tuple[int, int] | None = None,
        crop_margin: float = 0.3,
        camera_aug_prefix: str | None = None,
        downsample_max: float = 5.0,
    ) -> None:
        import torchvision.transforms.v2 as T  # ruff: ignore[import-outside-top-level]

        self.root = root
        # crops of varied quality: loose / off-centre crops and low effective resolution
        self.quality_aug = quality_aug
        # None: square crops (`crop_for_box`); (W, H): box + `crop_margin` per side, resized
        self.crop_shape = crop_shape
        self.crop_margin = crop_margin
        # samples whose image path starts with this (renders) get real-camera degradations
        self.camera_aug_prefix = camera_aug_prefix
        # quality aug: simulate the pallet up to this many times further away (low resolution)
        self.downsample_max = downsample_max
        self.items = [(n, *labels[n], detector_boxes.get(n)) for n in sorted(labels)]
        self.train = train
        self.rng = np.random.default_rng(seed)
        self.color = T.Compose([
            T.ColorJitter(0.5, 0.5, 0.4, 0.05),
            T.RandomGrayscale(0.1),
            T.RandomApply([T.GaussianBlur(5, (0.1, 2.0))], p=0.3),
        ])

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:  # ruff: ignore[too-many-locals]
        from PIL import Image  # ruff: ignore[import-outside-top-level]

        name, xy, labelled, det_box = self.items[i]
        image = (
            torch
            # "frame.jpg#i" is the i-th labelled pallet of a frame with several
            .from_numpy(
                np.array(Image.open(self.root / name.split("#")[0]).convert("RGB"))
            )
            .permute(2, 0, 1)
            .float()
            / 255
        )
        xy, labelled = xy.copy(), labelled.copy()
        pts = xy[labelled]
        label_box = np.r_[pts.min(axis=0), pts.max(axis=0)]
        use_det = det_box is not None and (not self.train or self.rng.random() < 0.5)  # ruff: ignore[magic-value-comparison]
        box = np.asarray(det_box, dtype=np.float64) if use_det else label_box
        flipped = bool(self.train and self.rng.random() < 0.5)  # ruff: ignore[magic-value-comparison]
        if flipped:
            image = image.flip(-1)
            xy[:, 0] = WIDTH - 1 - xy[:, 0]
            xy, labelled = xy[list(FLIP_INDEX)], labelled[list(FLIP_INDEX)]
            box = np.array([WIDTH - 1 - box[2], box[1], WIDTH - 1 - box[0], box[3]])
        rng = self.rng if self.train else None
        if self.crop_shape is None:
            crop = crop_for_box(box, rng, wide=self.quality_aug)
            size = (CROP_SIZE, CROP_SIZE)
            patch = crop.extract(image)
            local = torch.from_numpy(crop.to_crop(np.nan_to_num(xy))).float()
            scale_xy = torch.tensor([crop.side / CROP_SIZE] * 2)
            geometry = [crop.x0, crop.y0, crop.side]
        else:
            crop = rect_crop_for_box(
                box, rng, margin=self.crop_margin, wide=self.quality_aug
            )
            size = self.crop_shape
            patch = crop.extract(image, size)
            local = torch.from_numpy(crop.to_crop(np.nan_to_num(xy), size)).float()
            scale_xy = torch.tensor([crop.w / size[0], crop.h / size[1]])
            geometry = [crop.x0, crop.y0, crop.w, crop.h]
        if self.train:
            if self.camera_aug_prefix and name.startswith(self.camera_aug_prefix):
                patch = camera_realism(patch, self.rng)
            patch = self.color(patch)
            if self.quality_aug and self.rng.random() < 0.5:  # ruff: ignore[magic-value-comparison]
                # as if the pallet were 1.5-5x further away: downsample, then back up
                factor = self.rng.uniform(
                    min(1.5, self.downsample_max), self.downsample_max
                )
                small = (max(16, int(size[1] / factor)), max(16, int(size[0] / factor)))
                patch = torch.nn.functional.interpolate(
                    torch.nn.functional.interpolate(
                        patch[None], size=small, mode="area"
                    ),
                    size=(size[1], size[0]),
                    mode="bilinear",
                    align_corners=False,
                )[0]
        inside = ((local >= 0) & (local < torch.tensor(size, dtype=local.dtype))).all(
            dim=-1
        )
        visible = inside & torch.from_numpy(labelled)
        return {
            "image": (patch - MEAN) / STD,
            "keypoints": local,
            "visible": visible,
            # a labelled point outside the crop is labelled "absent"; unlabelled is ignored
            "labelled": torch.from_numpy(labelled),
            "targets": gaussian_targets(local[None], visible[None], size)[0],
            # raw frame pixels per crop pixel; `scale_xy` per axis (non-square crops)
            "scale": scale_xy.mean(),
            "scale_xy": scale_xy,
            # the crop in (possibly flipped) frame pixels, for heads that need its geometry
            "crop": torch.tensor(geometry, dtype=torch.float64),
            "flipped": torch.tensor(flipped),
        }
