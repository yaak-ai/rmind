"""Keypoint labels and ground-truth poses from an AprilTag on the pallet.

During collection runs one AprilTag (36h11) is fixed to the pallet. Its pose on
the pallet, `TagMount`, is measured once. Per `cam_fork` frame, offline:

    detect tag corners -> 6-DoF tag pose (fisheye LM) -> pallet pose in the robot
    frame -> project `EUR_KEYPOINTS` -> keypoint labels + visibility

The same pallet pose is the ground truth the detector is scored against and the
aux-head target; it is never a policy input. The 6-DoF tag pose is also checked
against the floor assumption: `TagPallet.off_floor` (height, roll, pitch of the
pallet it implies) should be ~0, and is a direct test of the `cam_fork` mount
calibration.

The tag must not leak into training images - a keypoint net would learn to find
the tag instead of the pallet. Mount it away from the keypoints and blank it with
`mask_tag` before an image becomes a training sample.

cv2 is imported lazily: only detection and masking need it.
"""

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
    rotvec_to_matrix,
)
from rmind.pallet.pose import levenberg_marquardt

# corner order of `cv2.aruco`: top-left, top-right, bottom-right, bottom-left,
# in the tag's own frame (+x right, +y down on the printed tag, +z into it)
_CORNERS_UNIT: Final = np.array([
    [-0.5, -0.5, 0.0],
    [0.5, -0.5, 0.0],
    [0.5, 0.5, 0.0],
    [-0.5, 0.5, 0.0],
])


@dataclass(frozen=True)
class Rigid:
    """A 3D rigid transform `p -> rotation @ p + translation`."""

    rotation: Array
    translation: Array

    def apply(self, points: Array) -> Array:
        return points @ self.rotation.T + self.translation

    def compose(self, other: "Rigid") -> "Rigid":
        """`self ∘ other`."""
        return Rigid(
            self.rotation @ other.rotation,
            self.rotation @ other.translation + self.translation,
        )

    def inverse(self) -> "Rigid":
        return Rigid(self.rotation.T, -self.rotation.T @ self.translation)


@dataclass(frozen=True)
class TagMount:
    """Where the tag sits on the pallet: the tag frame in the pallet frame.

    `size` is the black-border edge length in metres. The default is a tag on the
    entry face's centre block, facing the truck - a PLACEHOLDER; the plan says to
    mount it away from the keypoints, e.g. on the deck.
    """

    tag_id: int
    size: float
    pose: Rigid


def facing_truck_mount(tag_id: int, size: float, y: float, z: float) -> TagMount:
    """A tag on the entry face (pallet x = 0), printed side towards the truck."""
    # tag +x (right as the truck sees it) = pallet -y, tag +y (down) = pallet -z,
    # tag +z (into the tag) = pallet +x
    rotation = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    return TagMount(tag_id, size, Rigid(rotation, np.array([0.0, y, z])))


@dataclass(frozen=True)
class TagPallet:
    """One frame's ground truth from the tag."""

    pose: Pose2D
    # the 6-DoF pallet pose implied by the tag, (z, roll, pitch) part: ~0 if the
    # floor assumption and the camera mount are right
    off_floor: tuple[float, float, float]
    reprojection_px: float


def tag_pose_robot(
    camera: FisheyeCamera, corners: Array, size: float, init: Rigid | None = None
) -> tuple[Rigid, float]:
    """6-DoF tag pose in the robot frame from its 4 detected corners.

    Initialised from the corner rays (the tag's centre ray at a range fixed by its
    apparent size, facing the camera) unless `init` is given, then LM on the
    fisheye reprojection error.
    """
    model = _CORNERS_UNIT * size
    if init is None:
        rays = camera.rays(corners)
        centre_ray = rays.mean(axis=0)
        centre_ray /= np.linalg.norm(centre_ray)
        # apparent edge angle -> range
        edge = np.mean([
            np.arccos(np.clip(rays[i] @ rays[(i + 1) % 4], -1.0, 1.0)) for i in range(4)
        ])
        distance = size / max(edge, 1e-6)
        # facing the camera: tag +z along the ray, tag +y roughly world-down
        z_axis = centre_ray
        y_axis = np.array([0.0, 0.0, -1.0]) - z_axis * -z_axis[2]
        y_axis /= np.linalg.norm(y_axis)
        rotation = np.stack([np.cross(y_axis, z_axis), y_axis, z_axis], axis=1)
        init = Rigid(rotation, camera.origin + distance * centre_ray)

    def residuals(p: Array) -> Array:
        rotation = init.rotation @ rotvec_to_matrix(p[:3])
        projected, _ = camera.project(model @ rotation.T + init.translation + p[3:])
        return (projected - corners).ravel()

    p, _, r = levenberg_marquardt(residuals, np.zeros(6))
    pose = Rigid(init.rotation @ rotvec_to_matrix(p[:3]), init.translation + p[3:])
    return pose, float(np.sqrt(np.mean(r**2) * 2))


def pallet_from_tag(robot_tag: Rigid, mount: TagMount) -> Rigid:
    """The pallet frame in the robot frame, from the tag's pose and its mount."""
    return robot_tag.compose(mount.pose.inverse())


def to_floor(robot_pallet: Rigid) -> tuple[Pose2D, tuple[float, float, float]]:
    """Split a 6-DoF pallet pose into (x, y, yaw) and its off-floor (z, roll, pitch)."""
    r = robot_pallet.rotation
    yaw = float(np.arctan2(r[1, 0], r[0, 0]))
    # residual rotation after removing yaw: R = Rz(yaw) Ry(pitch) Rx(roll)
    rest = rot_z(-yaw) @ r
    pitch = float(np.arctan2(-rest[2, 0], rest[0, 0]))
    roll = float(np.arctan2(rest[2, 1], rest[2, 2]))
    x, y, z = map(float, robot_pallet.translation)
    return Pose2D(x, y, yaw), (z, roll, pitch)


def tag_ground_truth(
    camera: FisheyeCamera, corners: Array, mount: TagMount
) -> TagPallet:
    tag, err = tag_pose_robot(camera, corners, mount.size)
    pose, off_floor = to_floor(pallet_from_tag(tag, mount))
    return TagPallet(pose, off_floor, err)


def keypoint_labels(
    camera: FisheyeCamera, pose: Pose2D, model: Array = KEYPOINTS_3D
) -> tuple[Array, npt.NDArray[np.bool_]]:
    """Project the pallet keypoints: pixels and in-image visibility.

    Visibility here is geometric only (in front, inside the frame). Occlusion by
    the forks or the pallet itself is not modelled yet; the deck's far corners and
    the entry face can both be in frame while one hides the other only at angles
    the approach does not reach, but the forks DO hide the block bottoms once
    inserted, so labels past the fork tips need an occlusion mask (TODO).
    """
    return camera.project(pose.apply(model))


def pallet_pose_6dof(
    pose: Pose2D, z: float = 0.0, roll: float = 0.0, pitch: float = 0.0
) -> Rigid:
    """A `Rigid` for a pallet pose, e.g. to render synthetic tags in tests."""
    rotation = rot_z(pose.yaw) @ rot_y(pitch) @ rot_x(roll)
    return Rigid(rotation, np.array([pose.x, pose.y, z]))


def tag_corners_robot(robot_pallet: Rigid, mount: TagMount) -> Array:
    """The tag's 4 corners in the robot frame, in `cv2.aruco` order."""
    return robot_pallet.compose(mount.pose).apply(_CORNERS_UNIT * mount.size)


# --- image side (cv2) ------------------------------------------------------------


def detect_tags(image: npt.NDArray[np.uint8]) -> dict[int, Array]:
    """AprilTag 36h11 corners per id, sub-pixel refined, in `cv2.aruco` order.

    Detection runs on the distorted image: a tag is small and near the image centre
    on the approach, where the fisheye is close to pinhole. Corners stay in the
    distorted image, which is what `tag_pose_robot` expects.
    """
    import cv2  # ruff: ignore[import-outside-top-level]

    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),
        cv2.aruco.DetectorParameters(),
    )
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image  # ruff: ignore[magic-value-comparison]
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return {}
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    out = {}
    for tag_id, c in zip(ids.ravel(), corners, strict=True):
        refined = cv2.cornerSubPix(
            gray, c.reshape(4, 1, 2).astype(np.float32), (5, 5), (-1, -1), criteria
        )
        out[int(tag_id)] = refined.reshape(4, 2).astype(np.float64)
    return out


def mask_tag(
    image: npt.NDArray[np.uint8], corners: Array, margin: float = 0.25
) -> npt.NDArray[np.uint8]:
    """Inpaint the tag (grown by `margin` of its size) out of a training image."""
    import cv2  # ruff: ignore[import-outside-top-level]

    centre = corners.mean(axis=0)
    grown = centre + (corners - centre) * (1 + margin)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(grown).astype(np.int32), 255)
    return np.asarray(cv2.inpaint(image, mask, 5, cv2.INPAINT_TELEA), dtype=np.uint8)
