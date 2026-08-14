"""Pinhole projection of ego-frame ground-plane trajectories into
`cam_front_left` pixel space, for overlaying predicted/candidate/ground-truth
trajectories directly onto the camera image (e.g. via rerun `Points2D` logged
as children of the image entity) instead of a disconnected 3D view.

Two corrections vs. a naive "reuse the raw fisheye calibration" approach,
both sourced from `/home/alex/rsim/README.md` (which already had to solve
this exact problem to simulate the camera in CARLA) and
`/home/alex/data/dvc.yaml` (the real data pipeline):

1. **The dataset frames are already defished, not raw fisheye.**
   `dvc.yaml`'s pipeline is `stitch -> defish_video (cam-90-deg.json) ->
   pii_video -> extract_frames`, i.e. `cam_front_left.pii.mp4` (what
   `extract_frames` downsamples to the dataset's 576x324 frames) is the
   OUTPUT of `defish_video`, not the raw lens capture. So the model/overlay
   target is a rectilinear (undistorted) image, not a fisheye one -- applying
   the raw lens's Kannala-Brandt distortion (as an earlier version of this
   module did) is simply the wrong model for it. Per the README, the raw lens
   captures 90.2 deg HFOV x 48.4 deg VFOV; the defished output is a
   rectilinear 90 deg x 58.7 deg (hence the black top/bottom bands with no
   fisheye data, visible in real frames). We derive a plain pinhole K from
   that output FOV (at the calibration's native 1920x1080 resolution, scaled
   down to the dataset's 576x324) rather than reusing the raw K/D.

2. **Camera pose**: rather than reuse cam_front_center's calibrated pitch
   with an assumed height (which turned out to have the pitch sign backwards
   -- see below -- and used a too-tall height guess), use rsim's
   `carla_yaak_pov` mount pose, empirically tuned/measured for this exact
   camera and confirmed as the "empirically best closed-loop config" used at
   both training-data generation and inference:
     x=0.8m forward, y=-0.032m (3.2cm left), z=1.03m above road, pitch=4 deg.
   Notably rsim's pitch is documented as **nose-up** (CARLA convention:
   positive pitch = tilt up), whereas this module previously treated pitch as
   nose-down. The two sources agree in magnitude (rsim's empirically-tuned 4
   deg vs. the real `cam_front_center` calibration's ~2.5-5 deg range) but
   not in sign as originally assumed here -- so extrinsics pitch is now
   negated before use.
"""

import json
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

DATA_ROOT = Path("/nasa/drives/yaak/data")
FRAME_RES = (576, 324)  # (width, height) of the dataset's cam_front_left frames
NATIVE_RES = (1920, 1080)  # resolution the calibration files (and rsim's FOV figures) are defined at

# defished cam_front_left output FOV (rsim/README.md "Why fisheye simulation"):
# raw lens is 90.2x48.4 deg fisheye; `lsd defish_video` rectifies it to a
# 90x58.7 deg rectilinear (pinhole) image -- NOT the raw fisheye FOV/model.
DEFISH_HFOV_DEG = 90.0
DEFISH_VFOV_DEG = 58.7

# rsim carla_yaak_pov / carla_yaak_pov_defish mount pose (README.md
# "Camera positioning rationale" + "Camera alignment: training vs
# inference"), vehicle-frame (+X forward, +Y right, +Z up from ground):
CAMERA_FORWARD_OFFSET_M = 0.8  # m forward of vehicle bounding-box centre
CAMERA_RIGHT_OFFSET_M = -0.032  # m right (negative = 3.2cm left)
CAMERA_HEIGHT_M = 1.03  # m above road ("windshield height, ~1.0-1.1m on a Kia e-Niro")


def _pinhole_K(*, width: int, height: int) -> np.ndarray:
    fx = (width / 2) / np.tan(np.deg2rad(DEFISH_HFOV_DEG) / 2)
    fy = (height / 2) / np.tan(np.deg2rad(DEFISH_VFOV_DEG) / 2)
    return np.array(
        [[fx, 0.0, width / 2], [0.0, fy, height / 2], [0.0, 0.0, 1.0]], dtype=np.float64
    )


_K_NATIVE = _pinhole_K(width=NATIVE_RES[0], height=NATIVE_RES[1])
_SCALE = FRAME_RES[0] / NATIVE_RES[0]
K = _K_NATIVE.copy()
K[0, 0] *= _SCALE
K[1, 1] *= _SCALE
K[0, 2] *= _SCALE
K[1, 2] *= _SCALE


@lru_cache(maxsize=None)
def load_camera_pitch(vehicle: str) -> float:
    """Returns cam_front_left's pitch in radians for `vehicle`, nose-up
    positive. `cam_front_left` isn't individually self-calibrated (only
    `cam_front_center` is) -- per Samples.md's viewing-angle table they're
    co-located lenses of the same tri-focal housing sharing pitch, so we
    reuse cam_front_center's, negated to match rsim's nose-up convention
    (see module docstring).
    """
    with (DATA_ROOT / "extrinsics" / f"{vehicle}.json").open() as f:
        extrinsics = json.load(f)["extrinsics"]
    _fov, _yaw, pitch_deg = extrinsics["cam_front_center"]
    return -np.deg2rad(pitch_deg)


def project_ego_ground_points(xy_m: np.ndarray, *, pitch_rad: float) -> np.ndarray:
    """Project ego-frame ground-plane points `(..., x=right, y=forward)`,
    meters, into `cam_front_left` pixel coordinates `(..., u, v)`, using the
    rectilinear (defished) pinhole model -- see module docstring. Points
    behind (or at) the camera are set to NaN so downstream consumers (rerun,
    matplotlib) skip them instead of wrapping them around to bogus pixel
    locations.
    """
    x_t = xy_m[..., 0] - CAMERA_RIGHT_OFFSET_M
    y_t = xy_m[..., 1] - CAMERA_FORWARD_OFFSET_M

    # un-tilted camera-nominal axes (X=right, Y=down, Z=forward): ground point
    # is `CAMERA_HEIGHT_M` below the camera -> Y0 = +height (down-positive).
    X0 = x_t
    Y0 = np.full_like(x_t, CAMERA_HEIGHT_M)
    Z0 = y_t

    cos_p, sin_p = np.cos(pitch_rad), np.sin(pitch_rad)
    X = X0
    Y = Y0 * cos_p - Z0 * sin_p
    Z = Y0 * sin_p + Z0 * cos_p

    u = K[0, 0] * (X / Z) + K[0, 2]
    v = K[1, 1] * (Y / Z) + K[1, 2]

    uv = np.stack([u, v], axis=-1)
    uv[Z <= 0.3] = np.nan
    return uv


def project_trajectories_to_image(xy_m: Tensor, vehicles: Sequence[str]) -> Tensor:
    """Batched wrapper: `xy_m` is `(B, ..., >=2)` ego-frame meters (x, y are
    the first two trailing components), `vehicles` is length-`B` (one vehicle
    id per batch item, e.g. `"Niro115-HQ"`). Returns `(B, ..., 2)` pixel
    coordinates, NaN where a point falls behind the camera.
    """
    xy_np = xy_m[..., :2].detach().cpu().double().numpy()
    out = np.empty_like(xy_np)
    for i, vehicle in enumerate(vehicles):
        pitch_rad = load_camera_pitch(vehicle)
        out[i] = project_ego_ground_points(xy_np[i], pitch_rad=pitch_rad)
    return torch.from_numpy(out).to(xy_m.dtype)
