"""Overlay DrivoR predicted trajectories (from a `just predict` parquet) onto
2D BEV plots and the raw cam_front_left frame.

Usage:
    nix develop --command uv run python visualize_drivor_predictions.py \
        outputs/.../predictions/yaak/alex-tmp/model-mdm5kmkg:v3.parquet \
        --out-dir outputs/drivor_viz --n-per-drive 4

Camera geometry: see `rmind.utils._camera_projection` (shared with the
`DrivoRRerunPredictionWriter` live-logging path) for the pinhole model and
mount pose this uses, and why -- both sourced from `rsim/README.md` and
`data/dvc.yaml` rather than assumed.
"""

import argparse
from pathlib import Path

import numpy as np
import polars as pl
import torch
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw

from rmind.utils._camera_projection import DATA_ROOT, FRAME_RES, project_trajectories_to_image


def frame_path(input_id: str, frame_idx: int) -> Path:
    return (
        DATA_ROOT
        / input_id
        / "frames"
        / "cam_front_left.pii.mp4"
        / f"{FRAME_RES[0]}x{FRAME_RES[1]}"
        / f"{frame_idx:09d}.jpg"
    )


def draw_trajectory(draw: ImageDraw.ImageDraw, uv: np.ndarray, color: tuple) -> None:
    pts = [(u, v) for u, v in uv if np.isfinite(u) and np.isfinite(v)]
    for i, (u, v) in enumerate(pts):
        r = 4
        draw.ellipse([u - r, v - r, u + r, v + r], fill=color)
        if i > 0:
            draw.line([pts[i - 1], (u, v)], fill=color, width=2)


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

    for input_id in df["batch/meta/input_id"].unique().to_list():
        vehicle = input_id.split("/")[0]

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

            gt_uv = project_trajectories_to_image(
                torch.from_numpy(gt_xy).unsqueeze(0), [vehicle]
            )[0].numpy()
            best_uv = project_trajectories_to_image(
                torch.from_numpy(best_xy).unsqueeze(0), [vehicle]
            )[0].numpy()
            draw_trajectory(draw, gt_uv, color=(0, 120, 255))
            draw_trajectory(draw, best_uv, color=(255, 40, 40))

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
