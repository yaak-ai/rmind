"""Floor-free pallet pose from keypoints: PnP against the pallet's own 3D keypoint layout.

`pose.solve` assumes the pallet stands on the floor in front of a known camera mount, which
is what breaks for stacked and rack pallets. Here nothing about the floor is assumed:

    6 DoF    generic PnP. The 12 entry-face keypoints are coplanar (all on the face), so
             IPPE (OpenCV's planar solver) gives the two candidate poses of a planar
             target; the lower-reprojection one is refined by Levenberg-Marquardt, points
             off by more than `max(4 px, 3 x median)` are dropped and it is solved again.
    upright  4 DoF (centre + yaw), started from the 6-DoF solution: the pallet is level
             and the camera is level (the D12 cam_fork's optical axis is horizontal), but
             its height is free. The gravity prior without the floor prior - it removes the
             planar target's tilt ambiguity, the main failure of the 6-DoF fit on a small,
             far face.

Everything is in the camera's forward / left / up frame (m); yaw is the heading of the
face's inward normal, + left. Pixels go to normalised image coordinates first (pinhole
`(fx, fy, cx, cy)` or a `FisheyeCamera`'s rays), so the same code serves Veigel's pinhole
renders and the real fisheye.
"""

from dataclasses import dataclass
from typing import Final

import cv2
import numpy as np

from rmind.pallet.geometry import Array, FisheyeCamera
from rmind.pallet.pose import levenberg_marquardt

MIN_POINTS: Final = 4
OUTLIER_PX: Final = 4.0


@dataclass(frozen=True)
class FreePose:
    centre: Array  # (3,) forward, left, up (m): the layout's origin
    yaw: float  # rad, inward normal heading, + left
    rms_px: float
    inliers: np.ndarray  # (K,) bool over the input keypoints


def normalise(
    pixels: Array, k: tuple[float, ...] | Array | None, camera: FisheyeCamera | None
) -> Array:
    """`(K, 2)` pixels -> `(K, 2)` normalised image coordinates (x/z, y/z), optical frame."""
    if camera is not None:
        rays = camera.rays(pixels) @ camera.rotation  # robot frame -> optical
        return rays[:, :2] / rays[:, 2:3]
    fx, fy, cx, cy = k  # ty: ignore[not-iterable]
    return np.stack([(pixels[:, 0] - cx) / fx, (pixels[:, 1] - cy) / fy], axis=1)


def _flu(p: Array) -> Array:
    """Optical (x right, y down, z forward) -> forward, left, up."""
    return np.array([p[2], -p[0], -p[1]])


def _ippe(obj: Array, img: Array) -> tuple[Array, Array] | None:
    """Best IPPE solution, LM-refined, in normalised coordinates: (rvec, tvec)."""
    ok, rvecs, tvecs, errs = cv2.solvePnPGeneric(
        obj, img, np.eye(3), None, flags=cv2.SOLVEPNP_IPPE
    )
    if not ok or not len(rvecs):
        return None
    best = int(np.argmin(np.asarray(errs).ravel()))
    rvec, tvec = cv2.solvePnPRefineLM(
        obj, img, np.eye(3), None, rvecs[best], tvecs[best]
    )
    return rvec, tvec


def _reprojection(obj: Array, img: Array, rvec: Array, tvec: Array) -> Array:
    projected, _ = cv2.projectPoints(obj, rvec, tvec, np.eye(3), None)
    return np.linalg.norm(projected.reshape(-1, 2) - img, axis=1)


def solve_6dof(
    pixels: Array,
    visible: np.ndarray,
    layout: Array,
    *,
    k: tuple[float, ...] | Array | None = None,
    camera: FisheyeCamera | None = None,
) -> FreePose | None:
    """Generic planar PnP; `layout` is `(K, 3)` in the pallet frame (x in, y left, z up)."""
    idx = np.flatnonzero(visible)
    if len(idx) < MIN_POINTS:
        return None
    focal = camera.fx if camera is not None else float(k[0])  # ty: ignore[not-subscriptable]
    img = normalise(pixels[idx], k, camera).astype(np.float64)
    obj = layout[idx].astype(np.float64)
    solution = _ippe(obj, img)
    if solution is None:
        return None
    keep = np.ones(len(idx), dtype=bool)
    err = _reprojection(obj, img, *solution) * focal
    gate = max(OUTLIER_PX, 3 * float(np.median(err)))
    if (err > gate).any() and (err <= gate).sum() >= MIN_POINTS:
        keep = err <= gate
        solution = _ippe(obj[keep], img[keep]) or solution
        err = _reprojection(obj, img, *solution) * focal
    rvec, tvec = solution
    if not (np.isfinite(rvec).all() and np.isfinite(tvec).all()) or tvec[2] <= 0:
        return None
    rotation, _ = cv2.Rodrigues(rvec)
    normal = _flu(rotation[:, 0])  # the layout's x axis: into the face
    inliers = np.zeros(len(visible), dtype=bool)
    inliers[idx[keep]] = True
    return FreePose(
        centre=_flu(tvec.ravel()),
        yaw=float(np.arctan2(normal[1], normal[0])),
        rms_px=float(np.sqrt(np.mean(err[keep] ** 2))),
        inliers=inliers,
    )


def solve_upright(  # ruff: ignore[too-many-arguments]
    pixels: Array,
    visible: np.ndarray,
    layout: Array,
    init: FreePose,
    *,
    k: tuple[float, ...] | Array | None = None,
    camera: FisheyeCamera | None = None,
) -> FreePose | None:
    """4-DoF (centre, yaw) LM on the 6-DoF fit's inliers: level pallet, level camera."""
    idx = np.flatnonzero(visible & init.inliers)
    if len(idx) < 3:  # ruff: ignore[magic-value-comparison]
        return None
    focal = camera.fx if camera is not None else float(k[0])  # ty: ignore[not-subscriptable]
    img = normalise(pixels[idx], k, camera)
    obj = layout[idx]

    def project(p: Array) -> Array:
        c, s = np.cos(p[3]), np.sin(p[3])
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        flu = obj @ rot.T + p[:3]
        return np.stack([-flu[:, 1] / flu[:, 0], -flu[:, 2] / flu[:, 0]], axis=1)

    p, _, r = levenberg_marquardt(
        lambda q: ((project(q) - img) * focal).ravel(),
        np.array([*init.centre, init.yaw]),
    )
    if not np.isfinite(p).all() or p[0] <= 0:
        return None
    inliers = np.zeros(len(visible), dtype=bool)
    inliers[idx] = True
    return FreePose(
        centre=p[:3],
        yaw=float(np.arctan2(np.sin(p[3]), np.cos(p[3]))),
        rms_px=float(np.sqrt(np.mean(r**2) * 2)),
        inliers=inliers,
    )
