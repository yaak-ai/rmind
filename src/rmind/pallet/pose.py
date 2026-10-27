"""Detected keypoints -> the pallet's (x, y, yaw) in the robot frame.

The pallet sits on the floor and the camera's mount is known, so instead of the
6 DoF of generic PnP only (x, y, yaw) - plus, by default, a small correction to
the camera's pitch - are free:

1. CLOSED FORM. Every keypoint has a known height on the pallet, hence a known
   height in the robot frame. Intersecting its viewing ray with that horizontal
   plane gives a metric floor-plane point, and a weighted 2D rigid fit (Kabsch)
   of model points to those points is the pose. Two keypoints suffice.
2. RANSAC over all keypoint pairs with that closed form, scored by reprojection
   error in pixels, to drop bad keypoints.
3. REFINE: Levenberg-Marquardt on the inliers' reprojection error (in the
   distorted image, where the keypoint noise actually lives).

WHY PITCH IS FREE BY DEFAULT. The synthetic study (`scripts/pallet_poc_synthetic`)
shows the fixed-mount solution is fragile: a 0.5 deg camera-pitch error alone
biases forward by ~7 cm and yaw by ~4 deg at 1 m, ~16 cm and ~14 deg at 2 m - a
pitch error rescales depth differently for floor-level and deck-level keypoints,
and the 3-DoF fit soaks that up in yaw. Pitch moves at runtime too (braking,
load, tyre compression, floor joints). The pallet's vertical structure (pockets
at 0.1 m, deck at 0.144 m) and its 1.2 m deck make a pitch offset observable, and
freeing it removes the bias at almost no cost in noise. Camera height and yaw are
NOT freed: they are fixed by the mount and the study shows them benign (1 cm of
height -> ~3 cm forward at 1 m; 0.5 deg of yaw -> 0.5 deg of yaw).

`solve_6dof` is the unconstrained baseline the synthetic study compares against.
"""

import dataclasses
import itertools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from rmind.pallet.geometry import (
    KEYPOINTS_3D,
    Array,
    FisheyeCamera,
    Pose2D,
    rot_x,
    rot_y,
    rot_z,
)

# a keypoint further than this from its reprojection is an outlier, at least
MIN_THRESHOLD_PX: Final = 4.0
# ... and at least this many keypoint-noise sigmas (2D: 3 sigma keeps ~99%)
THRESHOLD_SIGMAS: Final = 3.0
# RANSAC runs at the fixed mount, so with pitch free its gate is this much looser
COARSE_GATE: Final = 3.0
# below this many inliers, report nothing rather than a guess
MIN_INLIERS: Final = 3
# RANSAC pairs closer than this on the floor plane are skipped (m)
MIN_PAIR_SPAN: Final = 0.05


@dataclass(frozen=True)
class PalletEstimate:
    pose: Pose2D
    # over (x, y, yaw), marginal, from the refined Jacobian and inlier residual
    covariance: Array
    rms_px: float
    inliers: npt.NDArray[np.bool_]
    confidence: float
    # the camera-pitch correction the fit found (rad, + = more downward); a
    # persistent offset means the mount calibration is off
    pitch_offset: float = 0.0

    @property
    def std(self) -> Array:
        return np.sqrt(np.clip(np.diag(self.covariance), 0.0, None))


def fit_rigid_2d(model_xy: Array, measured_xy: Array, weights: Array) -> Pose2D:
    """Weighted least-squares rotation + translation taking model to measured."""
    w = weights / weights.sum()
    mu_m, mu_d = w @ model_xy, w @ measured_xy
    a, b = model_xy - mu_m, measured_xy - mu_d
    # 2D Kabsch: the angle that maximises sum w <R a, b>
    sxx = w @ (a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1])
    sxy = w @ (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
    yaw = float(np.arctan2(sxy, sxx))
    c, s = np.cos(yaw), np.sin(yaw)
    t = mu_d - np.array([[c, -s], [s, c]]) @ mu_m
    return Pose2D(float(t[0]), float(t[1]), yaw)


def closed_form(
    camera: FisheyeCamera, pixels: Array, model: Array, weights: Array | None = None
) -> Pose2D | None:
    """Pose from >= 2 keypoints by intersecting each ray with its own height plane."""
    points, ok = camera.intersect_plane(pixels, model[:, 2])
    if ok.sum() < 2:  # ruff: ignore[magic-value-comparison]
        return None
    w = np.ones(len(model)) if weights is None else weights
    return fit_rigid_2d(model[ok, :2], points[ok, :2], w[ok])


def reprojection_error(
    camera: FisheyeCamera, pose: Pose2D, pixels: Array, model: Array
) -> Array:
    projected, _ = camera.project(pose.apply(model))
    return np.linalg.norm(projected - pixels, axis=-1)


def _jacobian(fn: Callable[[Array], Array], p: Array, step: float = 1e-6) -> Array:
    columns = []
    for i in range(len(p)):
        d = np.zeros_like(p)
        d[i] = step
        columns.append((fn(p + d) - fn(p - d)) / (2 * step))
    return np.stack(columns, axis=-1)


def levenberg_marquardt(
    fn: Callable[[Array], Array], p0: Array, iterations: int = 50, tol: float = 1e-10
) -> tuple[Array, Array, Array]:
    """Minimise |fn(p)|^2; returns the solution, its Jacobian and residuals."""
    p = p0.astype(np.float64).copy()
    r = fn(p)
    cost = float(r @ r)
    lam = 1e-3
    jac = _jacobian(fn, p)
    for _ in range(iterations):
        hessian = jac.T @ jac
        gradient = jac.T @ r
        damped = hessian + lam * np.diag(np.diag(hessian) + 1e-12)
        step = np.linalg.solve(damped, -gradient)
        r_new = fn(p + step)
        cost_new = float(r_new @ r_new)
        if cost_new < cost:
            p, r = p + step, r_new
            converged = cost - cost_new < tol * max(cost, 1.0)
            cost = cost_new
            lam = max(lam * 0.3, 1e-12)
            jac = _jacobian(fn, p)
            if converged:
                break
        else:
            lam *= 10
            if lam > 1e10:  # ruff: ignore[magic-value-comparison]
                break
    return p, jac, r


def with_pitch(camera: FisheyeCamera, offset: float) -> FisheyeCamera:
    return (
        camera
        if offset == 0
        else dataclasses.replace(camera, pitch=camera.pitch + offset)
    )


def refine(  # ruff: ignore[too-many-arguments]
    camera: FisheyeCamera,
    pixels: Array,
    model: Array,
    weights: Array,
    init: Pose2D,
    *,
    pitch_offset: float | None = None,
) -> tuple[Pose2D, float | None, Array, Array]:
    """LM on (confidence-weighted) pixel reprojection error.

    3 DoF when `pitch_offset` is None, else 4 with the camera-pitch offset free,
    started at `pitch_offset`. Returns the pose, the offset, Jacobian, residuals.
    """
    sqrt_w = np.sqrt(weights)[:, None]

    def residuals(p: Array) -> Array:
        cam = camera if len(p) == 3 else with_pitch(camera, p[3])  # ruff: ignore[magic-value-comparison]
        projected, _ = cam.project(Pose2D(*p[:3]).apply(model))
        return ((projected - pixels) * sqrt_w).ravel()

    p0 = np.array(init) if pitch_offset is None else np.array([*init, pitch_offset])
    p, jac, r = levenberg_marquardt(residuals, p0)
    pose = Pose2D(*map(float, p[:3]))
    return pose, (None if pitch_offset is None else float(p[3])), jac, r


def _confidence(
    n_inliers: int, n_observed: int, rms_px: float, threshold_px: float
) -> float:
    """Placeholder score in [0, 1]; to be replaced by a calibrated one on real data."""
    return float(n_inliers / n_observed * np.exp(-((rms_px / threshold_px) ** 2)))


def _ransac(
    camera: FisheyeCamera, px: Array, pts: Array, threshold_px: float
) -> Pose2D | None:
    """Exhaustive over pairs (14 keypoints is 91), best by inlier count then error."""
    best: tuple[int, float, Pose2D] | None = None
    for i, j in itertools.combinations(range(len(px)), 2):
        # a pocket corner above a block corner fixes no yaw
        if np.linalg.norm(pts[i, :2] - pts[j, :2]) < MIN_PAIR_SPAN:
            continue
        pose = closed_form(camera, px[[i, j]], pts[[i, j]])
        if pose is None:
            continue
        err = reprojection_error(camera, pose, px, pts)
        score = (
            int((err < threshold_px).sum()),
            -float(np.sum(np.minimum(err, threshold_px))),
        )
        if best is None or score > best[:2]:
            best = (*score, pose)
    return None if best is None else best[2]


def solve(  # ruff: ignore[too-many-arguments, too-many-locals]
    camera: FisheyeCamera,
    pixels: Array,
    visible: npt.NDArray[np.bool_],
    *,
    confidence: Array | None = None,
    model: Array = KEYPOINTS_3D,
    sigma_px: float = 1.0,
    threshold_px: float | None = None,
    free_pitch: bool = True,
) -> PalletEstimate | None:
    """Floor-constrained pallet pose from detected keypoints.

    `pixels` and `visible` are per model keypoint; `confidence` optionally weights
    them; `sigma_px` is the expected keypoint noise. Returns None when fewer than
    `MIN_INLIERS` keypoints agree on a pose - the caller should then hold the last
    good pose (dead reckoning), not guess.
    """
    if threshold_px is None:
        threshold_px = max(MIN_THRESHOLD_PX, THRESHOLD_SIGMAS * sigma_px)
    weights = np.ones(len(model)) if confidence is None else np.asarray(confidence)
    idx = np.flatnonzero(visible & (weights > 0))
    if len(idx) < MIN_INLIERS:
        return None
    px, pts, w = pixels[idx], model[idx], weights[idx]

    gate = threshold_px * (COARSE_GATE if free_pitch else 1.0)
    pose = _ransac(camera, px, pts, gate)
    if pose is None:
        return None

    offset = 0.0 if free_pitch else None
    inliers = reprojection_error(camera, pose, px, pts) < gate
    for _ in range(3):  # refine, re-select inliers at the fitted pitch, repeat
        # pitch needs vertical extent: fewer than 4 points and it is not worth it
        free = offset is not None and inliers.sum() >= 4  # ruff: ignore[magic-value-comparison]
        if inliers.sum() < MIN_INLIERS:
            return None
        pose, fitted, jac, r = refine(
            camera,
            px[inliers],
            pts[inliers],
            w[inliers],
            pose,
            pitch_offset=offset if free else None,
        )
        if fitted is not None:
            offset = fitted
        cam = with_pitch(camera, offset or 0.0)
        inliers = reprojection_error(cam, pose, px, pts) < threshold_px
    n_inliers = int(inliers.sum())
    if n_inliers < MIN_INLIERS:
        return None

    dof = jac.shape[1]
    rms = float(np.sqrt(np.mean(r**2) * 2)) if len(r) else 0.0
    sigma2 = max(sigma_px**2, float(r @ r) / max(len(r) - dof, 1))
    covariance = np.asarray(sigma2 * np.linalg.pinv(jac.T @ jac), dtype=np.float64)[
        :3, :3
    ]
    mask = np.zeros(len(model), dtype=bool)
    mask[idx[inliers]] = True
    return PalletEstimate(
        pose=pose,
        covariance=covariance,
        rms_px=rms,
        inliers=mask,
        confidence=_confidence(n_inliers, len(idx), rms, threshold_px),
        pitch_offset=offset or 0.0,
    )


def pose_6dof_points(params: Array, model: Array) -> Array:
    """Model points under (x, y, z, yaw, pitch, roll) - the unconstrained pose."""
    x, y, z, yaw, pitch, roll = params
    rotation = rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)
    return model @ rotation.T + np.array([x, y, z])


def solve_6dof(
    camera: FisheyeCamera,
    pixels: Array,
    visible: npt.NDArray[np.bool_],
    init: Pose2D,
    model: Array = KEYPOINTS_3D,
) -> Array | None:
    """Generic-PnP baseline: LM over all 6 DoF, started at `init`. Same keypoints."""
    idx = np.flatnonzero(visible)
    if len(idx) < 4:  # ruff: ignore[magic-value-comparison]
        return None
    px, pts = pixels[idx], model[idx]

    def residuals(p: Array) -> Array:
        projected, _ = camera.project(pose_6dof_points(p, pts))
        return (projected - px).ravel()

    p0 = np.array([init.x, init.y, 0.0, init.yaw, 0.0, 0.0])
    p, _, _ = levenberg_marquardt(residuals, p0)
    return p
