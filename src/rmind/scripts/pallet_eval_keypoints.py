"""Run a trained keypoint net on real `cam_fork` approaches and solve the pallet pose.

Reuses `pallet_pipeline`'s per-frame target boxes (Grounding DINO) and its
silhouette fits as the reference. Per frame:

    raw frame -> undistorted 90 deg pinhole view (what the synthetic training
    images are) -> crop around the box -> `PalletKeypointNet` -> 8 box corners
    -> back to raw pixels through `CAM_FORK` -> `pose.solve` (floor-constrained,
    free pitch) -> pallet pose

Outputs per job: `{job}_kp.parquet`, `{job}_kp.mp4` (predicted corners coloured
by confidence + the solved box, top-down map with both estimates) and a summary
against the high-IoU silhouette fits: corner pixel distance, centre and yaw
differences.
"""

import argparse
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
import torch

from rmind.pallet.geometry import CAM_FORK, EUR_HEIGHT, EUR_LENGTH, Pose2D
from rmind.pallet.keypoints import Crop, PalletKeypointNet, decode
from rmind.pallet.palloc6d import _MEAN, _STD, BOX_CORNERS_3D
from rmind.pallet.pose import solve
from rmind.scripts.pallet_pipeline import MAP_SIZE, VideoWriter, _map_px, draw_map
from rmind.scripts.pallet_poc_zeroshot import D12_ROOT, iter_frames

VW, VH = 1280, 720
KV: Final = np.array([[VW / 2, 0, VW / 2], [0, VW / 2, VH / 2], [0, 0, 1.0]])
MIN_CONFIDENCE: Final = 0.1
REFERENCE_MIN_IOU: Final = 0.8
EDGES: Final = [
    (0, 1),
    (2, 3),
    (0, 2),
    (1, 3),
    (4, 5),
    (6, 7),
    (4, 6),
    (5, 7),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
]


def virtual_rays(pixels: np.ndarray) -> np.ndarray:
    """Virtual pinhole pixels -> unit rays in the robot frame."""
    rays = np.c_[
        (pixels[:, 0] - KV[0, 2]) / KV[0, 0],
        (pixels[:, 1] - KV[1, 2]) / KV[1, 1],
        np.ones(len(pixels)),
    ]
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    return rays @ CAM_FORK.rotation.T


def canonical(pose: Pose2D) -> Pose2D:
    """The same box with its near (x = 0) end towards the camera, as the labels are.

    The silhouette fit cannot tell the box from itself turned by 180 deg.
    """
    c, s = np.cos(pose.yaw), np.sin(pose.yaw)
    turned = Pose2D(pose.x + EUR_LENGTH * c, pose.y + EUR_LENGTH * s, pose.yaw + np.pi)
    camera = CAM_FORK.origin[:2]
    return min(
        (pose, turned), key=lambda p: float(np.hypot(p.x - camera[0], p.y - camera[1]))
    )


def raw_to_virtual(pixels: np.ndarray) -> np.ndarray:
    rays = CAM_FORK.rays(pixels) @ CAM_FORK.rotation  # optical frame
    return np.c_[
        KV[0, 0] * rays[:, 0] / rays[:, 2] + KV[0, 2],
        KV[1, 1] * rays[:, 1] / rays[:, 2] + KV[1, 2],
    ]


def main() -> None:
    import cv2

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--pipeline", type=Path, required=True, help="pallet_pipeline output dir"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", nargs="+", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    net = PalletKeypointNet(ckpt["backbone"], num_keypoints=8, pretrained=False)
    net.load_state_dict(ckpt["state_dict"])
    net = net.to(args.device).eval()
    k = np.array([
        [CAM_FORK.fx, 0, CAM_FORK.cx],
        [0, CAM_FORK.fy, CAM_FORK.cy],
        [0, 0, 1],
    ])
    map1, map2 = cv2.initUndistortRectifyMap(
        k, np.array(CAM_FORK.k), None, KV, (VW, VH), cv2.CV_32FC1
    )

    for job in args.jobs:
        sil = pl.read_parquet(args.pipeline / f"{job}.parquet")
        frames = sil["cam_fork/frame_idx"].to_list()
        stride = frames[1] - frames[0]
        by_frame = {r["cam_fork/frame_idx"]: r for r in sil.iter_rows(named=True)}
        writer = VideoWriter(
            args.out / f"{job}_kp.mp4", 1280 + MAP_SIZE, 720, fps=30 / stride
        )
        rows, n = [], 0
        for images in iter_frames(
            D12_ROOT / job / "cam_fork" / "video.mp4", frames[0], frames[-1], stride, 8
        ):
            for image in images:
                f = frames[n]
                n += 1
                ref = by_frame[f]
                view = image[:, :, ::-1].copy()
                row: dict = {"cam_fork/frame_idx": f}
                estimate = None
                if ref["box"] is not None:
                    x0, y0, x1, y1 = ref["box"]
                    corners = np.array([
                        [x0, y0],
                        [x1, y0],
                        [x1, y1],
                        [x0, y1],
                        [(x0 + x1) / 2, y1],
                        [(x0 + x1) / 2, y0],
                    ])
                    valid = np.isfinite(CAM_FORK.rays(corners)).all(axis=1)
                    if valid.sum() >= 3:
                        vbox = raw_to_virtual(corners[valid])
                        crop = Crop.around((*vbox.min(axis=0), *vbox.max(axis=0)))
                        virtual = cv2.remap(image, map1, map2, cv2.INTER_LINEAR)
                        tensor = (
                            torch.from_numpy(virtual).permute(2, 0, 1).float() / 255
                        )
                        patch = (crop.extract(tensor) - _MEAN) / _STD
                        with torch.no_grad():
                            pixels, conf = decode(net(patch[None].to(args.device)))
                        vk = crop.to_frame(pixels[0].cpu().numpy())
                        conf = conf[0].cpu().numpy()
                        raw, imaged = CAM_FORK.project(
                            CAM_FORK.origin + virtual_rays(vk) * 3.0
                        )
                        estimate = solve(
                            CAM_FORK,
                            raw,
                            imaged & (conf > MIN_CONFIDENCE),
                            confidence=conf,
                            model=BOX_CORNERS_3D,
                            sigma_px=3.0,
                        )
                        row |= {"kp_raw": raw.tolist(), "kp_conf": conf.tolist()}
                        for (u, v), c in zip(raw, conf, strict=True):
                            if np.isfinite(u):
                                color = (
                                    0,
                                    int(255 * min(c, 1)),
                                    int(255 * (1 - min(c, 1))),
                                )
                                cv2.circle(view, (int(u), int(v)), 9, color, -1)
                        if (
                            ref["fit_iou"] is not None
                            and ref["fit_iou"] >= REFERENCE_MIN_IOU
                        ):
                            ref_px, _ = CAM_FORK.project(
                                canonical(
                                    Pose2D(
                                        ref["x"], ref["y"], np.deg2rad(ref["yaw_deg"])
                                    )
                                ).apply(BOX_CORNERS_3D)
                            )
                            row["corner_px_err_median"] = float(
                                np.nanmedian(np.linalg.norm(raw - ref_px, axis=1))
                            )
                if estimate is not None:
                    pose = estimate.pose
                    centre = pose.apply(
                        np.array([[EUR_LENGTH / 2, 0.0, EUR_HEIGHT / 2]])
                    )[0]
                    row |= {
                        "x": pose.x,
                        "y": pose.y,
                        "yaw_deg": float(np.rad2deg(pose.yaw)),
                        "centre_x": centre[0],
                        "centre_y": centre[1],
                        "inliers": int(estimate.inliers.sum()),
                        "rms_px": estimate.rms_px,
                        "pitch_offset_deg": float(np.rad2deg(estimate.pitch_offset)),
                    }
                    box_px, imaged = CAM_FORK.project(pose.apply(BOX_CORNERS_3D))
                    for a, b in EDGES:
                        if imaged[a] and imaged[b]:
                            cv2.line(
                                view,
                                tuple(map(int, box_px[a])),
                                tuple(map(int, box_px[b])),
                                (255, 60, 230),
                                3,
                            )
                if ref["centre_x"] is not None and estimate is not None:
                    row |= {
                        "sil_centre_x": ref["centre_x"],
                        "sil_centre_y": ref["centre_y"],
                        "sil_yaw": ref["yaw_deg"],
                        "sil_iou": ref["fit_iou"],
                    }
                rows.append(row)
                view = cv2.resize(view, (1280, 720), interpolation=cv2.INTER_AREA)
                text = f"{job} frame {f}  " + (
                    f"keypoint pose: face {estimate.pose.x - 1.15:+.2f} m ahead of tips, lat {estimate.pose.y:+.2f}, yaw {np.rad2deg(estimate.pose.yaw):+.1f}, inliers {int(estimate.inliers.sum())}/8"
                    if estimate
                    else "no keypoint pose"
                )
                cv2.putText(view, text, (14, 34), 0, 0.7, (0, 0, 0), 4)
                cv2.putText(view, text, (14, 34), 0, 0.7, (255, 255, 255), 2)
                cv2.putText(
                    view,
                    "dots: predicted corners (green = confident, red = not); magenta: solved box",
                    (14, 64),
                    0,
                    0.6,
                    (255, 255, 255),
                    1,
                )
                fake = (
                    type("F", (), {"pose": estimate.pose})()
                    if estimate is not None
                    else None
                )
                panel = draw_map(cv2, fake, (ref["uwb_fwd"], ref["uwb_lat"]), [])
                if ref["centre_x"] is not None:
                    c = _map_px(
                        np.array([ref["centre_x"]]), np.array([ref["centre_y"]])
                    )[0]
                    cv2.circle(panel, tuple(map(int, c)), 8, (255, 200, 40), 2)
                cv2.putText(
                    panel,
                    "keypoint pose (green) / silhouette centre (blue ring)",
                    (10, 62),
                    0,
                    0.5,
                    (255, 200, 40),
                    1,
                )
                writer.write(np.hstack([view, panel]))
        writer.close()
        out = pl.DataFrame(rows, infer_schema_length=None)
        out.write_parquet(args.out / f"{job}_kp.parquet")
        have = (
            out.filter(pl.col("x").is_not_null()) if "x" in out.columns else out.head(0)
        )
        cmp = (
            have.filter(pl.col("sil_iou") >= REFERENCE_MIN_IOU)
            if "sil_iou" in have.columns
            else have.head(0)
        )
        if len(cmp):
            cmp = cmp.with_columns(
                (pl.col("centre_x") - pl.col("sil_centre_x")).alias("dx"),
                (pl.col("centre_y") - pl.col("sil_centre_y")).alias("dy"),
                (((pl.col("yaw_deg") - pl.col("sil_yaw") + 90) % 180) - 90).alias(
                    "dyaw"
                ),
            )
        print(  # ruff: ignore[print]
            f"{job}: frames {len(out)}, keypoint poses {len(have)} ({100 * len(have) / max(len(out), 1):.0f}%)"
            + (
                f"; vs high-IoU silhouette ({len(cmp)} frames): corner px err median {cmp['corner_px_err_median'].median():.1f}, "
                f"|d centre x| {cmp['dx'].abs().median():.2f} m, |d centre y| {cmp['dy'].abs().median():.2f} m, |d yaw| {cmp['dyaw'].abs().median():.1f} deg"
                if len(cmp)
                else ""
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
