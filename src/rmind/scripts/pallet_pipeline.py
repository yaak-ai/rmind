"""End-to-end pallet pipeline v0 on real D12 approaches, with visualisations.

Per approach frame of `cam_fork` (1920x1080, fork down):

    1. boxes      Grounding DINO (tiny), fork tips dropped; the target is tracked
                  frame to frame by box overlap, seeded by the best-scoring box
    2. mask       SAM2 prompted with the target box
    3. pose v0    `silhouette.fit`: floor-constrained (x, y, yaw) of the EUR box
                  whose projected outline best overlaps the mask
    4. keypoints  the 14 model keypoints projected from that pose (pseudo-labels
                  for the keypoint net; not yet a detector of their own)

Outputs, per job, under `--out`:

    {job}.parquet      frame, box, mask IoU-fit, pose, keypoints, UWB pallet vector
    {job}.mp4          camera overlay | top-down map, side by side
    {job}_series.png   range / lateral / yaw against UWB over the approach

UWB is shown for orientation only: its ego reference is not the fork base and its
pallet tag sits on the deck centre, so expect an offset, and ±0.2 m of noise.

    CUDA_VISIBLE_DEVICES=1 uv run --extra pallet python -m rmind.scripts.pallet_pipeline \
        --out outputs/pallet_pipeline --jobs 2026-10-12--15-42-58
"""

import argparse
import operator
import os
import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import]
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import polars as pl
from structlog import get_logger

from rmind.data.d12 import row_table
from rmind.pallet.geometry import (
    CAM_FORK,
    EUR_HEIGHT,
    EUR_LENGTH,
    EUR_WIDTH,
    FORK_LENGTH,
    KEYPOINT_NAMES,
    KEYPOINTS_3D,
)
from rmind.pallet.silhouette import BOX_POINTS, SilhouetteFit, fit
from rmind.scripts.pallet_poc_zeroshot import (
    D12_ROOT,
    HEIGHT,
    WIDTH,
    Proposer,
    approach,
    iter_frames,
    with_pallet_vector,
)

logger = get_logger(__name__)

DETECTOR_ID: Final = "IDEA-Research/grounding-dino-tiny"
SEGMENTER_ID: Final = "facebook/sam2.1-hiera-large"
# re-run the full grid search instead of tracking when the last fit was this poor
TRACK_MIN_IOU: Final = 0.6
# a candidate box continues the track if it overlaps the last target this much
TRACK_MIN_BOX_IOU: Final = 0.2

# drawing: video panels and the top-down map's extent (robot frame, m)
CAM_W, CAM_H = 1280, 720
MAP_SIZE: Final = 720
MAP_X: Final = (-0.6, 6.0)  # forward
MAP_Y: Final = 3.3  # +- lateral
FORK_CENTRES_Y: Final = (0.19, -0.19)
FORK_WIDTH: Final = 0.16

GREEN, RED, ORANGE, CYAN, WHITE, GREY = (
    (80, 220, 80),
    (60, 60, 230),
    (40, 140, 250),
    (230, 200, 40),
    (240, 240, 240),
    (120, 120, 120),
)


class Segmenter:
    def __init__(self, device: str) -> None:
        import torch
        from transformers import Sam2Model, Sam2Processor

        self.torch = torch
        self.device = device
        self.processor = Sam2Processor.from_pretrained(SEGMENTER_ID)
        self.model = Sam2Model.from_pretrained(SEGMENTER_ID).to(device).eval()

    def __call__(
        self, images: list[np.ndarray], boxes: list[list[float]]
    ) -> list[npt.NDArray[np.bool_]]:
        inputs = self.processor(
            images=images, input_boxes=[[b] for b in boxes], return_tensors="pt"
        ).to(self.device)
        with self.torch.no_grad():
            outputs = self.model(**inputs, multimask_output=False)
        masks = self.processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"]
        )
        return [m[0, 0].numpy().astype(bool) for m in masks]


def _box_iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def pick_target(
    candidates: list[tuple[list[float], float]], previous: list[float] | None
) -> tuple[list[float], float] | None:
    """Continue the track if a box overlaps the last target, else the best score."""
    if not candidates:
        return None
    if previous is not None:
        overlap = max(candidates, key=lambda c: _box_iou(c[0], previous))
        if _box_iou(overlap[0], previous) >= TRACK_MIN_BOX_IOU:
            return overlap
    return max(candidates, key=operator.itemgetter(1))


# --- drawing -----------------------------------------------------------------------


def _map_px(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    scale = MAP_SIZE / (MAP_X[1] - MAP_X[0])
    u = MAP_SIZE / 2 - y * scale
    v = MAP_SIZE - (x - MAP_X[0]) * scale
    return np.stack([u, v], axis=-1)


def draw_map(
    cv2,
    fit_: SilhouetteFit | None,
    uwb: tuple[float, float],
    trail: list[tuple[float, float]],
) -> np.ndarray:
    canvas = np.full((MAP_SIZE, MAP_SIZE, 3), 25, dtype=np.uint8)
    for metres in range(1, 7):  # range rings from the camera
        centre = _map_px(np.array([CAM_FORK.position[0]]), np.array([0.0]))[0]
        radius = int(metres * MAP_SIZE / (MAP_X[1] - MAP_X[0]))
        cv2.circle(canvas, tuple(map(int, centre)), radius, (55, 55, 55), 1)
        cv2.putText(
            canvas,
            f"{metres} m",
            (int(centre[0]) + 4, int(centre[1] - radius) + 14),
            0,
            0.45,
            GREY,
            1,
        )
    for yc in FORK_CENTRES_Y:  # forks
        corners = np.array([
            [0, yc - FORK_WIDTH / 2],
            [FORK_LENGTH, yc - FORK_WIDTH / 2],
            [FORK_LENGTH, yc + FORK_WIDTH / 2],
            [0, yc + FORK_WIDTH / 2],
        ])
        cv2.fillPoly(
            canvas,
            [np.round(_map_px(corners[:, 0], corners[:, 1])).astype(np.int32)],
            (90, 90, 90),
        )
    cam = _map_px(np.array([CAM_FORK.position[0]]), np.array([0.0]))[0]
    cv2.circle(canvas, tuple(map(int, cam)), 5, WHITE, -1)
    if len(trail) > 1:
        pts = _map_px(np.array([t[0] for t in trail]), np.array([t[1] for t in trail]))
        cv2.polylines(
            canvas,
            [np.round(pts).astype(np.int32)],
            isClosed=False,
            color=(60, 120, 60),
            thickness=2,
        )
    if fit_ is not None:
        footprint = fit_.pose.apply(
            np.array([
                [0, -EUR_WIDTH / 2, 0],
                [EUR_LENGTH, -EUR_WIDTH / 2, 0],
                [EUR_LENGTH, EUR_WIDTH / 2, 0],
                [0, EUR_WIDTH / 2, 0],
            ])
        )
        poly = np.round(_map_px(footprint[:, 0], footprint[:, 1])).astype(np.int32)
        cv2.polylines(canvas, [poly], isClosed=True, color=GREEN, thickness=2)
        cv2.line(canvas, tuple(poly[3]), tuple(poly[0]), GREEN, 5)  # entry face
    if np.isfinite(uwb).all():
        p = _map_px(np.array([uwb[0]]), np.array([uwb[1]]))[0]
        cv2.drawMarker(canvas, tuple(map(int, p)), ORANGE, cv2.MARKER_CROSS, 18, 2)
    cv2.putText(canvas, "estimate (entry face thick)", (10, 22), 0, 0.5, GREEN, 1)
    cv2.putText(
        canvas, "UWB pallet tag (uncalibrated ref.)", (10, 42), 0, 0.5, ORANGE, 1
    )
    return canvas


def draw_camera(
    cv2,
    image: np.ndarray,
    box: list[float] | None,
    mask: np.ndarray | None,
    fit_: SilhouetteFit | None,
    keypoints: np.ndarray | None,
    text: list[str],
) -> np.ndarray:
    view = image[:, :, ::-1].copy()
    if mask is not None:
        view[mask] = (0.6 * view[mask] + 0.4 * np.array(CYAN)).astype(np.uint8)
    if box is not None:
        cv2.rectangle(
            view, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), RED, 3
        )
    if fit_ is not None:
        pixels, imaged = CAM_FORK.project(fit_.pose.apply(BOX_POINTS))
        for segment, ok in zip(
            np.split(pixels, len(pixels) // 24),
            np.split(imaged, len(imaged) // 24),
            strict=True,
        ):
            if ok.sum() > 1:
                cv2.polylines(
                    view,
                    [np.round(segment[ok]).astype(np.int32)],
                    isClosed=False,
                    color=GREEN,
                    thickness=2,
                )
    if keypoints is not None:
        for (u, v), name in zip(keypoints, KEYPOINT_NAMES, strict=True):
            if np.isfinite(u) and 0 <= u < WIDTH and 0 <= v < HEIGHT:
                color = (
                    ORANGE
                    if "pocket" in name
                    else (WHITE if "bottom" in name else (200, 120, 255))
                )
                cv2.circle(view, (int(u), int(v)), 7, color, -1)
    view = cv2.resize(view, (CAM_W, CAM_H), interpolation=cv2.INTER_AREA)
    for i, line in enumerate(text):
        cv2.putText(view, line, (14, 34 + 30 * i), 0, 0.8, (0, 0, 0), 4)
        cv2.putText(view, line, (14, 34 + 30 * i), 0, 0.8, WHITE, 2)
    return view


class VideoWriter:
    def __init__(self, path: Path, width: int, height: int, fps: float) -> None:
        env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
        self.process = subprocess.Popen(  # ruff: ignore[subprocess-without-shell-equals-true]
            [
                shutil.which("ffmpeg") or "ffmpeg",
                "-v",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "23",
                str(path),
            ],
            env=env,
            stdin=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        assert self.process.stdin is not None  # ruff: ignore[assert]
        self.process.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        assert self.process.stdin is not None  # ruff: ignore[assert]
        self.process.stdin.close()
        self.process.wait()


# --- the pipeline ------------------------------------------------------------------


def run_job(
    job: str,
    proposer: Proposer,
    segmenter: Segmenter,
    stride: int,
    batch: int,
    out: Path,
) -> pl.DataFrame:
    import cv2

    table = row_table(
        D12_ROOT / job,
        cameras=("cam_fork",),
        reference="cam_fork",
        traction_source="reported",
        max_skew_s=1.0,
        include_fork_height=True,
    )
    frames = approach(with_pallet_vector(table))
    frame_idx = frames["cam_fork/frame_idx"].to_numpy()
    first, last = int(frame_idx.min()), int(frame_idx.max())
    indices = list(range(first, last + 1, stride))
    uwb = {
        int(f): (fwd, lat)
        for f, fwd, lat in frames.select(
            "cam_fork/frame_idx", "p_fwd", "p_lat"
        ).iter_rows()
    }

    writer = VideoWriter(out / f"{job}.mp4", CAM_W + MAP_SIZE, CAM_H, fps=30 / stride)
    rows, trail = [], []
    previous_box: list[float] | None = None
    previous_fit: SilhouetteFit | None = None
    video = D12_ROOT / job / "cam_fork" / "video.mp4"
    for start, images in zip(
        range(0, len(indices), batch),
        iter_frames(video, first, last, stride, batch),
        strict=False,
    ):
        chunk = indices[start : start + len(images)]
        candidates = proposer.detect_all(images)
        targets = []
        for c in candidates:
            target = pick_target(c, previous_box)
            targets.append(target)
            previous_box = target[0] if target is not None else previous_box
        with_box = [i for i, t in enumerate(targets) if t is not None]
        masks: dict[int, np.ndarray] = {}
        if with_box:
            found = segmenter(
                [images[i] for i in with_box], [targets[i][0] for i in with_box]
            )  # ty: ignore[not-subscriptable]
            masks = dict(zip(with_box, found, strict=True))
        for i, (index, image) in enumerate(zip(chunk, images, strict=True)):
            target, mask = targets[i], masks.get(i)
            result = None
            if target is not None and mask is not None:
                init = (
                    previous_fit.pose
                    if previous_fit is not None and previous_fit.iou >= TRACK_MIN_IOU
                    else None
                )
                result = fit(CAM_FORK, mask, tuple(target[0]), init=init)  # ty: ignore[invalid-argument-type]
                if (
                    result is not None
                    and init is not None
                    and result.iou < TRACK_MIN_IOU
                ):
                    result = fit(CAM_FORK, mask, tuple(target[0]))  # ty: ignore[invalid-argument-type]
            previous_fit = result
            keypoints = (
                CAM_FORK.project(result.pose.apply(KEYPOINTS_3D))[0]
                if result is not None
                else None
            )
            fwd, lat = uwb.get(index, (np.nan, np.nan))
            centre = (
                result.pose.apply(np.array([[EUR_LENGTH / 2, 0.0, EUR_HEIGHT / 2]]))[0]
                if result is not None
                else None
            )
            if centre is not None:
                trail.append((float(centre[0]), float(centre[1])))
            rows.append({
                "cam_fork/frame_idx": index,
                "box": target[0] if target is not None else None,
                "box_score": target[1] if target is not None else None,
                "fit_iou": result.iou if result is not None else None,
                "x": result.pose.x if result is not None else None,
                "y": result.pose.y if result is not None else None,
                "yaw_deg": float(np.rad2deg(result.pose.yaw))
                if result is not None
                else None,
                "centre_x": float(centre[0]) if centre is not None else None,
                "centre_y": float(centre[1]) if centre is not None else None,
                "keypoints": keypoints.tolist() if keypoints is not None else None,
                "uwb_fwd": fwd,
                "uwb_lat": lat,
            })
            text = [f"{job}  frame {index}"]
            if result is not None:
                ahead = result.pose.x - FORK_LENGTH
                text += [
                    f"face {ahead:+.2f} m ahead of fork tips   lat {result.pose.y:+.2f} m   yaw {np.rad2deg(result.pose.yaw):+.1f} deg",
                    f"silhouette IoU {result.iou:.2f}   box {target[1]:.2f}",  # ty: ignore[not-subscriptable]
                ]
            else:
                text.append("no pallet")
            panel = draw_camera(
                cv2,
                image,
                target[0] if target is not None else None,
                mask,
                result,
                keypoints,
                text,
            )
            writer.write(
                np.hstack([panel, draw_map(cv2, result, (fwd, lat), trail[-60:])])
            )
        logger.info("batch", job=job, done=start + len(images), of=len(indices))
    writer.close()
    results = pl.DataFrame(
        rows,
        schema_overrides={"cam_fork/frame_idx": pl.Int32},
        infer_schema_length=None,
    ).with_columns(pl.lit(job).alias("job"))
    results.write_parquet(out / f"{job}.parquet")
    plot_series(results, out / f"{job}_series.png")
    return results


def plot_series(results: pl.DataFrame, path: Path) -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    blue, orange, muted, grid = "#2a78d6", "#eb6834", "#6b6a64", "#e4e3de"
    t = results["cam_fork/frame_idx"].to_numpy() / 30.0
    est_range = np.hypot(
        results["centre_x"].cast(pl.Float64).to_numpy(),
        results["centre_y"].cast(pl.Float64).to_numpy(),
    )
    uwb_range = np.hypot(
        results["uwb_fwd"].cast(pl.Float64).to_numpy(),
        results["uwb_lat"].cast(pl.Float64).to_numpy(),
    )
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    panels = [
        (axes[0], "Pallet centre range (m)", est_range, uwb_range),
        (
            axes[1],
            "Pallet centre lateral (m)",
            results["centre_y"].cast(pl.Float64).to_numpy(),
            results["uwb_lat"].cast(pl.Float64).to_numpy(),
        ),
        (
            axes[2],
            "Pallet yaw (deg)",
            results["yaw_deg"].cast(pl.Float64).to_numpy(),
            None,
        ),
        (
            axes[3],
            "Silhouette IoU",
            results["fit_iou"].cast(pl.Float64).to_numpy(),
            None,
        ),
    ]
    for ax, title, ours, theirs in panels:
        ax.plot(t, ours, color=blue, lw=2, label="vision v0")
        if theirs is not None:
            ax.plot(
                t, theirs, color=orange, lw=1.5, label="UWB (uncalibrated reference)"
            )
            ax.legend(frameon=False, fontsize=8)
        ax.set_title(title, loc="left", fontsize=10, color="#2b2b2a")
        ax.grid(color=grid, lw=0.8)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(colors=muted, labelsize=8)
    axes[-1].set_xlabel("time in job (s)", color=muted)
    fig.suptitle(str(results["job"][0]), x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", nargs="+", required=True)
    parser.add_argument(
        "--stride", type=int, default=2, help="every n-th approach frame"
    )
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    proposer = Proposer(args.device, DETECTOR_ID)
    segmenter = Segmenter(args.device)
    for job in args.jobs:
        run_job(job, proposer, segmenter, args.stride, args.batch, args.out)


if __name__ == "__main__":
    main()
