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

from rmind.pallet.correspondence import NAMES, POINTS_3D
from rmind.pallet.keypoints import CROP_SIZE, Crop, gaussian_targets

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


def crop_for_box(box: np.ndarray, rng: np.random.Generator | None) -> Crop:
    crop = Crop.around(tuple(box))
    if rng is None:
        return crop
    scale = rng.uniform(0.85, 1.35)
    side = crop.side * scale
    shift = rng.uniform(-0.12, 0.12, size=2) * crop.side
    return Crop(
        crop.x0 + (crop.side - side) / 2 + shift[0],
        crop.y0 + (crop.side - side) / 2 + shift[1],
        side,
    )


class EntryKeypoints(Dataset):
    def __init__(
        self,
        root: Path,
        labels: dict[str, tuple[np.ndarray, np.ndarray]],
        detector_boxes: dict[str, list[float] | None],
        *,
        train: bool,
        seed: int = 0,
    ) -> None:
        import torchvision.transforms.v2 as T  # ruff: ignore[import-outside-top-level]

        self.root = root
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

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        from PIL import Image  # ruff: ignore[import-outside-top-level]

        name, xy, labelled, det_box = self.items[i]
        image = (
            torch
            .from_numpy(np.array(Image.open(self.root / name).convert("RGB")))
            .permute(2, 0, 1)
            .float()
            / 255
        )
        xy, labelled = xy.copy(), labelled.copy()
        pts = xy[labelled]
        label_box = np.r_[pts.min(axis=0), pts.max(axis=0)]
        use_det = det_box is not None and (not self.train or self.rng.random() < 0.5)  # ruff: ignore[magic-value-comparison]
        box = np.asarray(det_box, dtype=np.float64) if use_det else label_box
        if self.train and self.rng.random() < 0.5:  # ruff: ignore[magic-value-comparison]
            image = image.flip(-1)
            xy[:, 0] = WIDTH - 1 - xy[:, 0]
            xy, labelled = xy[list(FLIP_INDEX)], labelled[list(FLIP_INDEX)]
            box = np.array([WIDTH - 1 - box[2], box[1], WIDTH - 1 - box[0], box[3]])
        crop = crop_for_box(box, self.rng if self.train else None)
        patch = crop.extract(image)
        if self.train:
            patch = self.color(patch)
        local = torch.from_numpy(crop.to_crop(np.nan_to_num(xy))).float()
        inside = ((local >= 0) & (local < CROP_SIZE)).all(dim=-1)
        visible = inside & torch.from_numpy(labelled)
        return {
            "image": (patch - MEAN) / STD,
            "keypoints": local,
            "visible": visible,
            # a labelled point outside the crop is labelled "absent"; unlabelled is ignored
            "labelled": torch.from_numpy(labelled),
            "targets": gaussian_targets(local[None], visible[None])[0],
            "scale": torch.tensor(crop.side / CROP_SIZE),
        }
