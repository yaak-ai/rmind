"""Real-frame pseudo-labels for the keypoint net from confident silhouette fits.

For every `pallet_pipeline` frame whose silhouette IoU is at least `--min-iou`:
the fitted pose, turned to the canonical "near end towards the camera"
convention, gives the 8 box corners (`palloc6d.BOX_CORNERS_3D`), which are
projected EXACTLY into the same undistorted 90 deg pinhole view the synthetic
training images and `pallet_eval_keypoints` use (no lens model in the way:
robot frame -> optical frame -> pinhole). Corners outside the view stay labelled;
the net learns to place them past the frame edge, which is what a truncated
pallet at the fork tips needs.

Writes `{out}/images/{job}_{frame:06d}.jpg` and `{out}/labels.parquet`
(job, frame, corners (8x2), box (detector box in view pixels), iou).
"""

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from rmind.pallet.geometry import CAM_FORK, Pose2D
from rmind.pallet.palloc6d import BOX_CORNERS_3D
from rmind.scripts.pallet_eval_keypoints import KV, VH, VW, canonical, raw_to_virtual
from rmind.scripts.pallet_poc_zeroshot import D12_ROOT, iter_frames


def project_virtual(points_robot: np.ndarray) -> np.ndarray:
    optical = CAM_FORK.to_camera(points_robot)
    return np.c_[
        KV[0, 0] * optical[:, 0] / optical[:, 2] + KV[0, 2],
        KV[1, 1] * optical[:, 1] / optical[:, 2] + KV[1, 2],
    ]


def main() -> None:
    import cv2

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-iou", type=float, default=0.85)
    args = parser.parse_args()
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    k = np.array([
        [CAM_FORK.fx, 0, CAM_FORK.cx],
        [0, CAM_FORK.fy, CAM_FORK.cy],
        [0, 0, 1],
    ])
    map1, map2 = cv2.initUndistortRectifyMap(
        k, np.array(CAM_FORK.k), None, KV, (VW, VH), cv2.CV_32FC1
    )

    rows = []
    for parquet in sorted(args.pipeline.glob("*.parquet")):
        job = parquet.stem
        fits = pl.read_parquet(parquet)
        frames = fits["cam_fork/frame_idx"].to_list()
        if len(frames) < 2:
            continue
        stride = frames[1] - frames[0]
        good = {
            r["cam_fork/frame_idx"]: r
            for r in fits.filter(pl.col("fit_iou") >= args.min_iou).iter_rows(
                named=True
            )
        }
        if not good:
            continue
        n = 0
        for images in iter_frames(
            D12_ROOT / job / "cam_fork" / "video.mp4", frames[0], frames[-1], stride, 8
        ):
            for image in images:
                f = frames[n]
                n += 1
                r = good.get(f)
                if r is None:
                    continue
                pose = canonical(Pose2D(r["x"], r["y"], np.deg2rad(r["yaw_deg"])))
                corners = project_virtual(pose.apply(BOX_CORNERS_3D))
                x0, y0, x1, y1 = r["box"]
                box_raw = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
                valid = np.isfinite(CAM_FORK.rays(box_raw)).all(axis=1)
                if valid.sum() < 3 or not np.isfinite(corners).all():
                    continue
                vb = raw_to_virtual(box_raw[valid])
                virtual = cv2.remap(image, map1, map2, cv2.INTER_LINEAR)
                cv2.imwrite(
                    str(args.out / "images" / f"{job}_{f:06d}.jpg"),
                    virtual[:, :, ::-1],
                    [cv2.IMWRITE_JPEG_QUALITY, 92],
                )
                rows.append({
                    "job": job,
                    "frame": f,
                    "corners": corners.tolist(),
                    "box": [*vb.min(axis=0), *vb.max(axis=0)],
                    "iou": r["fit_iou"],
                })
        print(job, len([x for x in rows if x["job"] == job]), flush=True)  # ruff: ignore[print]
    pl.DataFrame(rows).write_parquet(args.out / "labels.parquet")
    print("total", len(rows))  # ruff: ignore[print]


if __name__ == "__main__":
    main()
