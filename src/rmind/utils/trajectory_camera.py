"""Ego trajectories (x right, y forward, metres, origin = rear axle) drawn on the camera image."""

import numpy as np

# cam_front_left relative to the trajectory reference point (rear axle): metres / degrees
DEFAULT_CAMERA = {"forward": 1.7, "right": 0.0, "height": 1.3, "pitch_deg": 3.0}


def project(xy: np.ndarray, cam: dict[str, float], size: tuple[int, int]) -> np.ndarray:
    """Ground-plane points (n, 2) in the ego frame (x right, y forward, metres, origin = trajectory
    reference point) -> pixel coordinates (m, 2) of the points in front of the camera.

    Pinhole model of the defished cam_front_left: 90 deg horizontal field of view, so the
    focal length is half the image width.
    """
    w, h = size
    f, cx, cy = w / 2, w / 2, h / 2
    p = np.radians(cam["pitch_deg"])
    right = xy[:, 0] - cam["right"]
    fwd = xy[:, 1] - cam["forward"]
    up = -cam["height"]
    zc = fwd * np.cos(p) + up * np.sin(p)
    yc = fwd * np.sin(p) - up * np.cos(p)
    ok = zc > 0.5  # noqa: PLR2004  (near plane, m)
    return np.stack([cx + f * right[ok] / zc[ok], cy + f * yc[ok] / zc[ok]], -1)


def densify(poses: np.ndarray, step_m: float = 0.5) -> np.ndarray:
    """Polyline from the origin through the poses, resampled every `step_m` metres."""
    xy = np.vstack([[0.0, 0.0], poses[:, :2]])
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    if s[-1] < 1e-6:  # noqa: PLR2004
        return xy[:1]
    q = np.arange(0, s[-1] + 1e-9, step_m)
    return np.stack([np.interp(q, s, xy[:, 0]), np.interp(q, s, xy[:, 1])], -1)
