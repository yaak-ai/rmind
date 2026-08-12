"""Overlay DrivoR predicted trajectories (from a `just predict` parquet) onto
2D BEV plots and the raw cam_front_left frame.

Usage:
    nix develop --command uv run python visualize_drivor_predictions.py \
        outputs/.../predictions/yaak/alex-tmp/model-mdm5kmkg:v3.parquet \
        --out-dir outputs/drivor_viz --n-per-drive 4

Camera geometry
----------------
Extrinsics (pitch) and intrinsics (fisheye K/D) are read directly from Yaak's
real per-vehicle self-calibration files on the NAS:
    /nasa/drives/yaak/data/extrinsics/<vehicle>.json   -- [fov, yaw, pitch] per camera
    /nasa/drives/yaak/data/intrinsics/cam-<fov>-deg.json -- {"K": ..., "D": ...}
(these are the real, self-calibrated numbers; more accurate than the
"design spec, approximate" table in Samples.md).

`cam_front_left` itself is never individually self-calibrated in these files
(only `cam_front_center` is) -- per Samples.md's viewing-angle table, FC/FL/FR
share the same mount/pitch/yaw (0 deg yaw, ~+4 deg pitch design spec; only FOV
differs per lens: 110/90/57 deg respectively), so we reuse cam_front_center's
calibrated pitch for cam_front_left, with yaw=roll=0 and cam-90-deg.json
intrinsics (FL's FOV per Samples.md), scaled from the native 1920x1080
calibration resolution down to the dataset's 576x324 frame resolution.

Camera height above the ground plane is NOT present anywhere in the source
data (neither Samples.md's placement table -- FC/FL/FR are all listed at
(0,0,0), i.e. they define the vehicle-frame origin rather than giving a
ground-relative height -- nor the extrinsics json, which only has
[fov, yaw, pitch], no translation). CAMERA_HEIGHT_M below is therefore an
ASSUMPTION (typical windshield-mounted camera height for a compact car like
the Kia Niro fleet). It, and the pitch sign, were validated empirically
against real drive data before this script was written: a ground-truth
dead-reckoned trajectory projected with these parameters lands visually on
the road in the real frame (see conversation / commit notes).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw

DATA_ROOT = Path("/nasa/drives/yaak/data")
FRAME_RES = (576, 324)  # (width, height) of the dataset's cam_front_left frames
NATIVE_RES = (1920, 1080)  # resolution the calibration K matrices are defined at
CAM_FRONT_LEFT_FOV_DEG = 90  # per Samples.md viewing-angle table
CAMERA_HEIGHT_M = 1.4  # ASSUMPTION -- see module docstring


def load_camera_model(vehicle: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Returns (K, D, pitch_rad) for cam_front_left on `vehicle`."""
    with open(DATA_ROOT / "extrinsics" / f"{vehicle}.json") as f:
        extrinsics = json.load(f)["extrinsics"]
    _fov, _yaw, pitch_deg = extrinsics["cam_front_center"]

    with open(DATA_ROOT / "intrinsics" / f"cam-{CAM_FRONT_LEFT_FOV_DEG}-deg.json") as f:
        intr = json.load(f)
    K = np.array(intr["K"], dtype=np.float64)
    D = np.array(intr["D"], dtype=np.float64).flatten()

    scale = FRAME_RES[0] / NATIVE_RES[0]
    K = K.copy()
    K[0, 0] *= scale
    K[1, 1] *= scale
    K[0, 2] *= scale
    K[1, 2] *= scale

    return K, D, np.deg2rad(pitch_deg)


def project_ego_ground_points(
    xy_m: np.ndarray, *, height_m: float, pitch_rad: float, K: np.ndarray, D: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project ego-frame ground-plane points `(x=right, y=forward)`, meters,
    into fisheye image pixel coordinates `(u, v)`. Also returns forward depth
    `Z` (camera-frame), so callers can drop points behind/at the camera.

    Fisheye (OpenCV/Kannala-Brandt equidistant) distortion model, matching
    the `libCalib::CameraModelOpenCVFisheye` calibration format used by
    the intrinsics files.
    """
    x_t, y_t = xy_m[:, 0], xy_m[:, 1]

    # un-tilted camera-nominal axes (X=right, Y=down, Z=forward): ground point
    # is `height_m` below the camera -> Y0 = +height_m (down-positive).
    X0 = x_t
    Y0 = np.full_like(x_t, height_m)
    Z0 = y_t

    cos_p, sin_p = np.cos(pitch_rad), np.sin(pitch_rad)
    X = X0
    Y = Y0 * cos_p - Z0 * sin_p
    Z = Y0 * sin_p + Z0 * cos_p

    r = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(r, Z)
    theta2 = theta**2
    theta_d = theta * (
        1 + D[0] * theta2 + D[1] * theta2**2 + D[2] * theta2**3 + D[3] * theta2**4
    )
    scale = np.where(r > 1e-9, theta_d / np.where(r > 1e-9, r, 1.0), 1.0)
    u = K[0, 0] * (X * scale) + K[0, 2]
    v = K[1, 1] * (Y * scale) + K[1, 2]
    return u, v, Z


def frame_path(input_id: str, frame_idx: int) -> Path:
    return (
        DATA_ROOT
        / input_id
        / "frames"
        / "cam_front_left.pii.mp4"
        / f"{FRAME_RES[0]}x{FRAME_RES[1]}"
        / f"{frame_idx:09d}.jpg"
    )


def draw_trajectory(
    draw: ImageDraw.ImageDraw, u: np.ndarray, v: np.ndarray, z: np.ndarray, color: tuple
) -> None:
    pts = [(uu, vv) for uu, vv, zz in zip(u, v, z, strict=True) if zz > 0.3]
    for i, (uu, vv) in enumerate(pts):
        r = 4
        draw.ellipse([uu - r, vv - r, uu + r, vv + r], fill=color)
        if i > 0:
            draw.line([pts[i - 1], (uu, vv)], fill=color, width=2)


def plot_bev(ax, best_xy: np.ndarray, gt_xy: np.ndarray) -> None:
    ax.plot(gt_xy[:, 0], gt_xy[:, 1], "o-", color="tab:blue", label="ground truth (dead-reckoned)")
    ax.plot(best_xy[:, 0], best_xy[:, 1], "o-", color="tab:red", label="prediction (best)")
    ax.plot(0, 0, "k^", markersize=10, label="ego (t0)")
    ax.set_xlabel("x -- right (m)")
    ax.set_ylabel("y -- forward (m)")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/drivor_viz"))
    parser.add_argument("--n-per-drive", type=int, default=4)
    args = parser.parse_args()

    df = pl.read_parquet(args.parquet)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    camera_cache: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}

    for input_id in df["batch/meta/input_id"].unique().to_list():
        vehicle = input_id.split("/")[0]
        if vehicle not in camera_cache:
            camera_cache[vehicle] = load_camera_model(vehicle)
        K, D, pitch_rad = camera_cache[vehicle]

        sub = df.filter(pl.col("batch/meta/input_id") == input_id)
        n = min(args.n_per_drive, len(sub))
        # spread samples across the drive rather than just the first N
        idxs = np.linspace(0, len(sub) - 1, n, dtype=int)

        for row_i in idxs:
            row = sub[int(row_i)]
            frame_idx = int(row["batch/data/meta/ImageMetadata.cam_front_left/frame_idx"][0][0])
            best_xy = np.array(row["trajectory/best_prediction"][0].to_list())[:, :2] * 100.0
            gt_xy = np.array(row["trajectory/ground_truth_xy"][0].to_list()) * 100.0

            img_path = frame_path(input_id, frame_idx)
            if not img_path.exists():
                print(f"skip (missing frame): {img_path}")
                continue
            img = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(img)

            u_gt, v_gt, z_gt = project_ego_ground_points(
                gt_xy, height_m=CAMERA_HEIGHT_M, pitch_rad=pitch_rad, K=K, D=D
            )
            u_pred, v_pred, z_pred = project_ego_ground_points(
                best_xy, height_m=CAMERA_HEIGHT_M, pitch_rad=pitch_rad, K=K, D=D
            )
            draw_trajectory(draw, u_gt, v_gt, z_gt, color=(0, 120, 255))
            draw_trajectory(draw, u_pred, v_pred, z_pred, color=(255, 40, 40))

            stem = f"{vehicle}_{input_id.split('/')[1]}_{frame_idx:09d}"
            img.save(args.out_dir / f"{stem}_cam.png")

            fig, ax = plt.subplots(figsize=(4, 5))
            plot_bev(ax, best_xy, gt_xy)
            ax.set_title(stem, fontsize=8)
            fig.tight_layout()
            fig.savefig(args.out_dir / f"{stem}_bev.png", dpi=120)
            plt.close(fig)

            print(f"wrote {stem}")


if __name__ == "__main__":
    main()
