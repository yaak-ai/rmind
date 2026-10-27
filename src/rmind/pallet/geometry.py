"""Frames, the pallet's 3D keypoints, and the `cam_fork` fisheye camera.

FRAMES - everything is metres, right-handed, z up:

    robot   origin on the floor at the fork base, centred between the forks;
            +x along the forks towards the tips, +y left. The UWB-derived
            `relative_ego_pos` already points +x at the pallet on approach.
    pallet  origin on the floor at the centre of the ENTRY face (the one the forks
            go into); +x into the pallet, +y left. A pallet squared up in front of
            the forks has yaw 0.
    camera  OpenCV optical frame: +z along the optical axis, +x right, +y down.

The pallet sits on the floor, so its pose in the robot frame is 3 DoF:
`Pose2D(x, y, yaw)`. Floor tilt and the robot's pitch/roll are ignored; the
residual of a full 6-DoF tag pose against that assumption is reported by
`labels.tag_pallet_pose` so the assumption can be checked on real data.

CAMERA MODEL - `cam_fork` is a wide-angle lens (~96 deg horizontal) with strong
barrel distortion, so it is never treated as plain pinhole. Two models:

    rational        OpenCV `CALIB_RATIONAL_MODEL` (k1 k2 p1 p2 k3 k4 k5 k6), what the
                    D12 calib.io calibration is in, and what `CAM_FORK` uses
    kannala_brandt  OpenCV `fisheye` / Coppernic's `InferFisheyeCalibration`:
                    theta_d = theta (1 + k1 theta^2 + ... + k4 theta^8)

`CAM_FORK` is still NOT a `cam_fork` calibration: there is none. It borrows the
measured D12 `cam_forward` intrinsics (same lens; undistorting real `cam_fork`
frames with it straightens building edges and floor joints) and takes the mount
from the D12 CAD (`recon/docs/cad_cams.json`, "Front center fork camera": 426 mm
up, optical axis horizontal, 52 mm behind the fork face). Calibrate `cam_fork`
itself before trusting metric numbers at the centimetre level.
"""

from dataclasses import dataclass, field
from typing import Final, NamedTuple

import numpy as np
import numpy.typing as npt

type Array = npt.NDArray[np.float64]


class Pose2D(NamedTuple):
    """The pallet frame in the robot frame: translation (m) and yaw (rad, CCW)."""

    x: float
    y: float
    yaw: float

    def apply(self, points: Array) -> Array:
        """Map `(..., 3)` pallet-frame points into the robot frame."""
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        return points @ rotation.T + np.array([self.x, self.y, 0.0])


# --- the pallet ------------------------------------------------------------------
#
# EUR pallet (EPAL 1, 1200 x 800 x 144) entered from the 800 mm side, which is how
# the D12 recordings pick it up (the five top boards run along the forks). The
# entry face reads, left to right as the truck sees it,
#
#     block 100 | pocket 227.5 | block 145 | pocket 227.5 | block 100   (= 800)
#
# and a pocket runs from the floor to the underside of the stringer board at
# 22 + 78 = 100 mm. The deck top is at 144 mm. VERIFY against the real pallets.
EUR_LENGTH: Final = 1.2  # along the forks
EUR_WIDTH: Final = 0.8  # across the entry face
EUR_HEIGHT: Final = 0.144
EUR_POCKET_HEIGHT: Final = 0.1

# lateral block edges on the entry face, +y left; outer blocks 100, centre 145
_EDGES: Final = (0.4, 0.3, 0.0725, -0.0725, -0.3, -0.4)

# name -> pallet-frame point. Entry-face points carry most of the signal on the
# approach; the two far deck corners are what is left in view once the entry face
# has dropped out under the image (it does so at about the fork tips).
EUR_KEYPOINTS: Final[dict[str, tuple[float, float, float]]] = {
    # left/right as the truck sees the entry face; `pocket_l` is the left opening.
    # A pocket runs from the floor to the stringer board's underside, so its bottom
    # corners are floor contacts (the blocks' inner bottom corners)
    "pallet_bottom_left": (0.0, _EDGES[0], 0.0),
    "pocket_l_bottom_left": (0.0, _EDGES[1], 0.0),
    "pocket_l_bottom_right": (0.0, _EDGES[2], 0.0),
    "pocket_r_bottom_left": (0.0, _EDGES[3], 0.0),
    "pocket_r_bottom_right": (0.0, _EDGES[4], 0.0),
    "pallet_bottom_right": (0.0, _EDGES[5], 0.0),
    "pocket_l_top_left": (0.0, _EDGES[1], EUR_POCKET_HEIGHT),
    "pocket_l_top_right": (0.0, _EDGES[2], EUR_POCKET_HEIGHT),
    "pocket_r_top_left": (0.0, _EDGES[3], EUR_POCKET_HEIGHT),
    "pocket_r_top_right": (0.0, _EDGES[4], EUR_POCKET_HEIGHT),
    # the entry face's top corners, and the far end of the deck
    "pallet_top_left": (0.0, _EDGES[0], EUR_HEIGHT),
    "pallet_top_right": (0.0, _EDGES[5], EUR_HEIGHT),
    "pallet_far_top_left": (EUR_LENGTH, _EDGES[0], EUR_HEIGHT),
    "pallet_far_top_right": (EUR_LENGTH, _EDGES[5], EUR_HEIGHT),
}
KEYPOINT_NAMES: Final = tuple(EUR_KEYPOINTS)
KEYPOINTS_3D: Final[Array] = np.array(list(EUR_KEYPOINTS.values()))
# left/right swap, for horizontal-flip augmentation: the keypoint at mirrored y
FLIP_INDEX: Final = tuple(
    int(
        np.argmin(
            np.linalg.norm(KEYPOINTS_3D - point * np.array([1.0, -1.0, 1.0]), axis=1)
        )
    )
    for point in KEYPOINTS_3D
)


# --- rotations -------------------------------------------------------------------


def rot_x(a: float) -> Array:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(a: float) -> Array:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(a: float) -> Array:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotvec_to_matrix(rotvec: Array) -> Array:
    """Rodrigues, without pulling in scipy or cv2."""
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:  # ruff: ignore[magic-value-comparison]
        return np.eye(3)
    k = rotvec / angle
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(angle) * kx + (1 - np.cos(angle)) * kx @ kx


# camera axes in the robot frame for a level camera looking along +x
_OPTICAL_TO_ROBOT: Final = np.array([
    [0.0, 0.0, 1.0],  # robot x <- camera z
    [-1.0, 0.0, 0.0],  # robot y <- -camera x
    [0.0, -1.0, 0.0],  # robot z <- -camera y
])


# --- the camera ------------------------------------------------------------------


@dataclass(frozen=True)
class FisheyeCamera:
    """Kannala-Brandt intrinsics plus the camera's mount on the robot.

    `pitch` is positive looking DOWN, `yaw` positive looking left, `roll` about
    the optical axis. `position` is the optical centre in the robot frame.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    # 4 coefficients for `kannala_brandt`, 8 (OpenCV order) for `rational`
    k: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
    model: str = "kannala_brandt"
    position: tuple[float, float, float] = (0.0, 0.0, 0.5)
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    # rays further than this from the axis are treated as not imaged
    max_theta: float = np.deg2rad(95.0)
    rotation: Array = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        # columns: camera axes in the robot frame
        rotation = (
            rot_z(self.yaw) @ rot_y(self.pitch) @ _OPTICAL_TO_ROBOT @ rot_z(self.roll)
        )
        object.__setattr__(self, "rotation", rotation)

    @property
    def origin(self) -> Array:
        return np.asarray(self.position, dtype=np.float64)

    def to_camera(self, points: Array) -> Array:
        """Robot-frame `(..., 3)` points into the optical frame."""
        return (points - self.origin) @ self.rotation

    def distort(self, theta: Array) -> Array:
        """Kannala-Brandt: incidence angle -> distorted radius (normalised)."""
        t2 = theta * theta
        k1, k2, k3, k4 = self.k
        return theta * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))

    def _distort_rational(self, a: Array, b: Array) -> tuple[Array, Array]:
        """OpenCV rational: undistorted normalised (a, b) -> distorted (x, y)."""
        k1, k2, p1, p2, k3, k4, k5, k6 = self.k
        r2 = a * a + b * b
        radial = (1 + r2 * (k1 + r2 * (k2 + r2 * k3))) / (
            1 + r2 * (k4 + r2 * (k5 + r2 * k6))
        )
        x = a * radial + 2 * p1 * a * b + p2 * (r2 + 2 * a * a)
        y = b * radial + p1 * (r2 + 2 * b * b) + 2 * p2 * a * b
        return x, y

    def project(self, points: Array) -> tuple[Array, npt.NDArray[np.bool_]]:
        """Robot-frame points -> pixels `(..., 2)` and an imaged-and-in-frame mask."""
        pc = self.to_camera(points)
        r = np.hypot(pc[..., 0], pc[..., 1])
        theta = np.arctan2(r, pc[..., 2])
        if self.model == "rational":
            z = np.where(pc[..., 2] > 1e-9, pc[..., 2], np.nan)  # ruff: ignore[magic-value-comparison]
            x, y = self._distort_rational(pc[..., 0] / z, pc[..., 1] / z)
        else:
            scale = np.where(r > 1e-12, self.distort(theta) / np.maximum(r, 1e-12), 0.0)  # ruff: ignore[magic-value-comparison]
            x, y = pc[..., 0] * scale, pc[..., 1] * scale
        u = self.fx * x + self.cx
        v = self.fy * y + self.cy
        pixels = np.stack([u, v], axis=-1)
        visible = (
            (theta < self.max_theta)
            & np.isfinite(u)
            & (u >= 0)
            & (u <= self.width - 1)
            & (v >= 0)
            & (v <= self.height - 1)
        )
        return pixels, visible

    def rays(self, pixels: Array, iterations: int = 20) -> Array:
        """Pixels `(..., 2)` -> unit viewing rays in the ROBOT frame."""
        mx = (pixels[..., 0] - self.cx) / self.fx
        my = (pixels[..., 1] - self.cy) / self.fy
        if self.model == "rational":
            rays_cam = self._undistort_rational(mx, my, iterations)
        else:
            theta_d = np.hypot(mx, my)
            theta = theta_d.copy()
            # Newton on distort(theta) = theta_d; the polynomial is monotone in range
            for _ in range(iterations):
                step = 1e-6
                f = self.distort(theta) - theta_d
                df = (self.distort(theta + step) - self.distort(theta - step)) / (
                    2 * step
                )
                theta -= f / np.where(np.abs(df) > 1e-12, df, 1.0)  # ruff: ignore[magic-value-comparison]
            scale = np.where(
                theta_d > 1e-12,  # ruff: ignore[magic-value-comparison]
                np.sin(theta) / np.maximum(theta_d, 1e-12),
                0.0,
            )
            rays_cam = np.stack([mx * scale, my * scale, np.cos(theta)], axis=-1)
        return rays_cam @ self.rotation.T

    def _undistort_rational(self, mx: Array, my: Array, iterations: int) -> Array:  # ruff: ignore[too-many-locals]
        """Invert `_distort_rational` by 2D Newton (numeric Jacobian); unit rays."""
        # start from the radial-only inverse: a lookup over the monotone part of the
        # radial curve (it folds over past the calibrated field of view), which is
        # exact when p1 = p2 = 0 and close otherwise; plain Newton from the pixel
        # itself diverges near the image corners
        ru = np.linspace(0.0, np.tan(self.max_theta), 4096)
        rd, _ = self._distort_rational(ru, np.zeros_like(ru))
        monotone = np.concatenate([[True], np.diff(rd) > 0]).cumprod().astype(bool)
        r_measured = np.hypot(mx, my)
        r_start = np.interp(r_measured, rd[monotone], ru[monotone])
        tiny = 1e-12
        scale = np.where(r_measured > tiny, r_start / np.maximum(r_measured, tiny), 1.0)
        a, b = mx * scale, my * scale
        step = 1e-7
        for _ in range(iterations):
            x, y = self._distort_rational(a, b)
            fx_, fy_ = x - mx, y - my
            xa, ya = self._distort_rational(a + step, b)
            xb, yb = self._distort_rational(a, b + step)
            j11, j21 = (xa - x) / step, (ya - y) / step
            j12, j22 = (xb - x) / step, (yb - y) / step
            det = j11 * j22 - j12 * j21
            det = np.where(np.abs(det) > 1e-12, det, 1e-12)  # ruff: ignore[magic-value-comparison]
            a -= (j22 * fx_ - j12 * fy_) / det
            b -= (-j21 * fx_ + j11 * fy_) / det
        rays = np.stack([a, b, np.ones_like(a)], axis=-1)
        # pixels past the fold have no valid preimage
        outside = r_measured > rd[monotone].max()
        rays = np.where(outside[..., None], np.nan, rays)
        return rays / np.linalg.norm(rays, axis=-1, keepdims=True)

    def intersect_plane(
        self, pixels: Array, z: Array | float
    ) -> tuple[Array, npt.NDArray[np.bool_]]:
        """Back-project pixels onto horizontal planes `z` (robot frame).

        This is the floor constraint: a keypoint whose height on the pallet is known
        has a known height in the robot frame too, so its ray pins down a metric
        point. Returns `(..., 3)` points and whether the plane is hit in front.
        """
        rays = self.rays(pixels)
        dz = np.where(np.abs(rays[..., 2]) > 1e-9, rays[..., 2], np.nan)  # ruff: ignore[magic-value-comparison]
        s = (np.asarray(z) - self.origin[2]) / dz
        ok = np.isfinite(s) & (s > 0)
        return self.origin + np.where(ok, s, np.nan)[..., None] * rays, ok


# D12 calib.io `cam_forward` intrinsics, borrowed for `cam_fork` (no calibration of
# its own yet): /nasa/drives/yaak/data/intrinsics/delta-ids/d12_cam_forward-calibio.json
# (sha256 3ddd6e1b...). Mount from the D12 CAD: 426 mm up, level, centred, 52 mm
# behind the fork face (the CAD's ego origin, which is this module's robot origin).
CAM_FORK: Final = FisheyeCamera(
    fx=1226.0564109240534,
    fy=1226.0564109240534,
    cx=974.0853192003578,
    cy=560.9774873311378,
    width=1920,
    height=1080,
    model="rational",
    k=(
        -0.4649558350085068,
        -0.24260913138579188,
        0.0,
        0.0,
        0.0181086359387106,
        -0.043414318688275284,
        -0.5364901888671187,
        0.0,
    ),
    # the rational denominator 1 + k4 r^2 + k5 r^4 + k6 r^6 reaches zero at ~49 deg
    # off-axis, so the model folds over past ~48.5 deg (~961 px from the centre):
    # only the extreme image corners, which are treated as not imaged
    max_theta=np.deg2rad(48.5),
    position=(-0.052, 0.0, 0.426),
)

# fork length on the D12 (fork base -> tips), for "how far out is the pallet"
FORK_LENGTH: Final = 1.15
