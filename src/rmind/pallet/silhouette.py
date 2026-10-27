"""Label-free pallet pose: fit the pallet's 3D box to its segmentation mask.

v0 of the pipeline, and the teacher for the keypoint net's pseudo-labels:

    box (Grounding DINO) -> mask (SAM2) -> floor-constrained (x, y, yaw) whose
    projected pallet silhouette best overlaps the mask

The pallet is rendered as its solid 1.2 x 0.8 x 0.144 m box: the pockets are
small next to the outline and SAM masks fill them inconsistently. Edges are
densified before projection because the lens bends them. The rendered silhouette
is clipped to the image exactly as the real one is, so a pallet cut off at the
bottom of the frame is still compared like for like.

Score is IoU inside a region of interest around the detection box. Search is a
coarse grid over yaw and the footprint, seeded by back-projecting the box's
bottom edge onto the floor, then a shrinking pattern search. Camera pitch is held
at the mount value here (the silhouette pins it poorly); the keypoint solver in
`pose` frees it.
"""

import itertools
import operator
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from rmind.pallet.geometry import (
    EUR_HEIGHT,
    EUR_LENGTH,
    EUR_WIDTH,
    Array,
    FisheyeCamera,
    Pose2D,
)

# region of interest around the detection box, as a fraction of its size
ROI_MARGIN: Final = 0.35
# work at a downscale that leaves the ROI about this wide (px), never upscaling;
# a far pallet is ~100 px across, so a fixed downscale would leave nothing to fit
ROI_WORK_WIDTH: Final = 320
# points per box edge, so the lens-curved edges render curved
EDGE_SAMPLES: Final = 24
# coarse yaw grid (deg), and footprint offsets (m). The full half-turn: early in an
# approach the pallet is often seen side-on, and the solid box looks the same
# turned by 180 deg, so +-90 covers every outline
YAW_GRID_DEG: Final = tuple(range(-90, 90, 6))
OFFSET_GRID_M: Final = (-0.45, -0.3, -0.15, 0.0, 0.15, 0.3)

# the pallet box in the pallet frame (origin: entry-face centre on the floor)
_CORNERS: Final = np.array([
    [x, y, z]
    for x in (0.0, EUR_LENGTH)
    for y in (-EUR_WIDTH / 2, EUR_WIDTH / 2)
    for z in (0.0, EUR_HEIGHT)
])
_EDGES: Final = [
    (i, j)
    for i, j in itertools.combinations(range(8), 2)
    if np.count_nonzero(_CORNERS[i] != _CORNERS[j]) == 1
]
_t = np.linspace(0.0, 1.0, EDGE_SAMPLES)[:, None]
BOX_POINTS: Final[Array] = np.concatenate([
    _CORNERS[i] * (1 - _t) + _CORNERS[j] * _t for i, j in _EDGES
])


@dataclass(frozen=True)
class SilhouetteFit:
    pose: Pose2D
    iou: float


def _roi(
    box: tuple[float, float, float, float], width: int, height: int
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    mx, my = (x1 - x0) * ROI_MARGIN, (y1 - y0) * ROI_MARGIN
    return (
        max(int(x0 - mx), 0),
        max(int(y0 - my), 0),
        min(int(x1 + mx), width),
        min(int(y1 + my), height),
    )


def silhouette_polygon(camera: FisheyeCamera, pose: Pose2D) -> Array | None:
    """The pallet's projected outline (convex hull of its densified box), pixels."""
    import cv2  # ruff: ignore[import-outside-top-level]

    pixels, _ = camera.project(pose.apply(BOX_POINTS))
    in_front = np.isfinite(pixels).all(axis=-1) & (
        camera.to_camera(pose.apply(BOX_POINTS))[:, 2] > 0.05  # ruff: ignore[magic-value-comparison]
    )
    if in_front.sum() < 3:  # ruff: ignore[magic-value-comparison]
        return None
    # beyond the lens model's valid field the projection is meaningless; such points
    # only occur far outside the frame, so clamp them well past the border
    pts = np.clip(pixels[in_front], -4 * camera.width, 5 * camera.width)
    return cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2)


class _Scorer:
    """IoU of a rendered silhouette against the mask, inside a downscaled ROI."""

    def __init__(
        self,
        camera: FisheyeCamera,
        mask: npt.NDArray[np.bool_],
        box: tuple[float, float, float, float],
    ) -> None:
        import cv2  # ruff: ignore[import-outside-top-level]

        self.cv2 = cv2
        self.camera = camera
        self.x0, self.y0, x1, y1 = _roi(box, camera.width, camera.height)
        crop = mask[self.y0 : y1, self.x0 : x1].astype(np.uint8)
        self.scale = min(1.0, ROI_WORK_WIDTH / max(x1 - self.x0, 1))
        self.size = (
            max(int((x1 - self.x0) * self.scale), 1),
            max(int((y1 - self.y0) * self.scale), 1),
        )
        self.mask = cv2.resize(crop, self.size, interpolation=cv2.INTER_AREA) > 0
        self.mask_area = int(self.mask.sum())

    def __call__(self, pose: Pose2D) -> float:
        polygon = silhouette_polygon(self.camera, pose)
        if polygon is None or self.mask_area == 0:
            return 0.0
        canvas = np.zeros(self.size[::-1], dtype=np.uint8)
        local = (polygon - np.array([self.x0, self.y0])) * self.scale
        self.cv2.fillPoly(canvas, [np.round(local).astype(np.int32)], 1)
        rendered = canvas > 0
        union = int((rendered | self.mask).sum())
        return float((rendered & self.mask).sum()) / union if union else 0.0


def _seed(
    camera: FisheyeCamera, box: tuple[float, float, float, float]
) -> Pose2D | None:
    """Entry-face centre from the box's bottom edge on the floor, facing the camera."""
    x0, _, x1, y1 = box
    point, ok = camera.intersect_plane(
        np.array([(x0 + x1) / 2, min(y1, camera.height - 2)]), 0.0
    )
    if not ok:
        return None
    return Pose2D(float(point[0]), float(point[1]), 0.0)


def fit(
    camera: FisheyeCamera,
    mask: npt.NDArray[np.bool_],
    box: tuple[float, float, float, float],
    init: Pose2D | None = None,
) -> SilhouetteFit | None:
    """Best floor pose for one mask; `init` (e.g. the previous frame) skips the grid."""
    score = _Scorer(camera, mask, box)
    if score.mask_area == 0:
        return None
    if init is None:
        seed = _seed(camera, box)
        if seed is None:
            return None
        candidates = [
            Pose2D(seed.x + dx, seed.y + dy, np.deg2rad(yaw))
            for yaw in YAW_GRID_DEG
            for dx in OFFSET_GRID_M
            for dy in OFFSET_GRID_M
        ]
        scored = [(score(p), p) for p in candidates]
        best_iou, best = max(scored, key=operator.itemgetter(0))
    else:
        best, best_iou = init, score(init)

    # shrinking pattern search over (x, y, yaw)
    steps = np.array([0.1, 0.1, np.deg2rad(4.0)])
    while steps[0] > 0.004:  # ruff: ignore[magic-value-comparison]
        improved = False
        for axis in range(3):
            for sign in (1.0, -1.0):
                p = np.array(best)
                p[axis] += sign * steps[axis]
                candidate = Pose2D(*map(float, p))
                iou = score(candidate)
                if iou > best_iou:
                    best, best_iou, improved = candidate, iou, True
        if not improved:
            steps /= 2
    return SilhouetteFit(best, best_iou)
