"""Zero-shot pallet proposals on real `cam_fork` approaches, checked against UWB.

What it answers, with no labels:

1. Does an off-the-shelf open-vocabulary detector (Grounding DINO) find the pallet
   over the approach, and from how far? Its boxes are the crop proposer for the
   keypoint net (`rmind.pallet.keypoints`) and the seed for AprilTag-free label
   bootstrapping.
2. Is `CAM_FORK` roughly to scale? The box's bottom-centre is ~the pallet's
   nearest floor contact; back-projected onto the floor it gives a forward
   distance that should track UWB's pallet-forward with slope ~1 (the UWB tag
   sits on the deck centre, ~0.6 m behind the entry face, and the UWB ego
   reference is not the fork base - both are an intercept). A slope far from 1
   means the focal length / mount guess is off by that factor.

The approach is every frame before the fork is first lifted (or, in jobs without
a fork-height signal, before UWB's closest pallet distance). Fork tips are a
fixed-position false positive and are dropped by `FORK_TIP_ROI`.

    CUDA_VISIBLE_DEVICES=1 uv run --with transformers python -m \
        rmind.scripts.pallet_poc_zeroshot --out outputs/pallet_poc/zeroshot
"""

import argparse
import operator
import os
import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import]
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from structlog import get_logger

from rmind.data.d12 import row_table
from rmind.pallet.geometry import CAM_FORK

logger = get_logger(__name__)

D12_ROOT: Final = Path("/nasa/data/d12/yaak")
MODEL_ID: Final = "IDEA-Research/grounding-dino-base"
PROMPT: Final = "a wooden pallet."
BOX_THRESHOLD: Final = 0.25
WIDTH, HEIGHT = 1920, 1080
# where the empty fork tips sit in every frame (x0, y0, x1, y1); a box mostly
# inside it and no taller than the tips is a fork tip, not a pallet
FORK_TIP_ROI: Final = (400, 820, 1400, 1080)
FORK_TIP_MAX_HEIGHT: Final = 260
LIFT_M: Final = 0.2
# UWB pallet-forward bins for the recall table (m)
BINS: Final = (0.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 8.0)


def approach(table: pl.DataFrame) -> pl.DataFrame:
    """Frames before the pickup: first fork lift, else the closest UWB distance."""
    lifted = table["fork_height"].list.get(0).to_numpy() * 2.3 > LIFT_M
    end = (
        int(np.argmax(lifted))
        if lifted.any()
        else int(np.argmin(table["p_dist"].to_numpy()))
    )
    return table.head(end)


def with_pallet_vector(table: pl.DataFrame) -> pl.DataFrame:
    """UWB pallet position in the robot frame (forward, lateral, distance)."""
    h = np.deg2rad(table["ego_heading"].to_numpy())
    dx = table["pallet_x"].to_numpy() - table["ego_x"].to_numpy()
    dy = table["pallet_y"].to_numpy() - table["ego_y"].to_numpy()
    return table.with_columns(
        pl.Series("p_fwd", np.cos(h) * dx + np.sin(h) * dy),
        pl.Series("p_lat", -np.sin(h) * dx + np.cos(h) * dy),
        pl.Series("p_dist", np.hypot(dx, dy)),
    )


def iter_frames(
    video: Path, first: int, last: int, stride: int, batch: int
) -> Iterator[np.ndarray]:
    """Decode encoded frames `first, first + stride, ... <= last` in one streaming pass.

    Yields:
        `(<=batch, H, W, 3)` uint8 arrays, in frame order.
    """
    select = f"between(n\\,{first}\\,{last})*not(mod(n-{first}\\,{stride}))"
    # the nix ffmpeg must not see the devshell's LD_LIBRARY_PATH (glibc clash)
    env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
    size = HEIGHT * WIDTH * 3
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    with subprocess.Popen(  # ruff: ignore[subprocess-without-shell-equals-true]
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(video),
            "-vf",
            # 720p drives are upscaled so pixel coordinates match the 1080p calibration
            f"select={select},scale={WIDTH}:{HEIGHT}",
            "-vsync",
            "0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        env=env,
        stdout=subprocess.PIPE,
    ) as process:
        assert process.stdout is not None  # ruff: ignore[assert]
        frames: list[np.ndarray] = []
        while raw := process.stdout.read(size):
            frames.append(np.frombuffer(raw, dtype=np.uint8).reshape(HEIGHT, WIDTH, 3))
            if len(frames) == batch:
                yield np.stack(frames)
                frames = []
        if frames:
            yield np.stack(frames)


def is_fork_tip(box: list[float]) -> bool:
    x0, y0, x1, y1 = box
    rx0, ry0, rx1, ry1 = FORK_TIP_ROI
    inter = max(0.0, min(x1, rx1) - max(x0, rx0)) * max(
        0.0, min(y1, ry1) - max(y0, ry0)
    )
    area = max((x1 - x0) * (y1 - y0), 1.0)
    return inter / area > 0.5 and (y1 - y0) < FORK_TIP_MAX_HEIGHT


class Proposer:
    def __init__(self, device: str, model_id: str = MODEL_ID) -> None:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.torch = torch
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = (
            AutoModelForZeroShotObjectDetection
            .from_pretrained(model_id)
            .to(device)
            .eval()
        )

    def detect_all(self, images: np.ndarray) -> list[list[tuple[list[float], float]]]:
        """Every non-fork-tip pallet box per image, as (x0, y0, x1, y1), score."""
        inputs = self.processor(
            images=list(images), text=[PROMPT] * len(images), return_tensors="pt"
        ).to(self.device)
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=BOX_THRESHOLD,
            text_threshold=BOX_THRESHOLD,
            target_sizes=[(HEIGHT, WIDTH)] * len(images),
        )
        return [
            [
                (b, s)
                for b, s in zip(r["boxes"].tolist(), r["scores"].tolist(), strict=True)
                if not is_fork_tip(b)
            ]
            for r in results
        ]

    def __call__(self, images: np.ndarray) -> list[tuple[list[float], float] | None]:
        """Best non-fork-tip pallet box per image, or None."""
        return [
            max(candidates, key=operator.itemgetter(1)) if candidates else None
            for candidates in self.detect_all(images)
        ]


def box_floor_point(box: list[float]) -> tuple[float, float]:
    """Robot-frame (x, y) of the box's bottom-centre, back-projected to the floor."""
    x0, _, x1, y1 = box
    point, ok = CAM_FORK.intersect_plane(np.array([(x0 + x1) / 2, y1]), 0.0)
    return (float(point[0]), float(point[1])) if ok else (np.nan, np.nan)


def run_job(
    job: str, proposer: Proposer, stride: int, batch: int, out: Path
) -> pl.DataFrame:
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
    rows = []
    video = D12_ROOT / job / "cam_fork" / "video.mp4"
    for start, images in zip(
        range(0, len(indices), batch),
        iter_frames(video, first, last, stride, batch),
        strict=False,
    ):
        chunk = indices[start : start + len(images)]
        found = proposer(images)
        for index, best in zip(chunk, found, strict=True):
            box, score = best if best is not None else ([np.nan] * 4, np.nan)
            fx, fy = box_floor_point(box) if best is not None else (np.nan, np.nan)
            rows.append({
                "cam_fork/frame_idx": index,
                "score": score,
                "x0": box[0],
                "y0": box[1],
                "x1": box[2],
                "y1": box[3],
                "box_floor_x": fx,
                "box_floor_y": fy,
            })
        if start == 0:
            _save_overlay(images[:4], found[:4], out / f"{job}_first.jpg")
    detections = pl.DataFrame(rows, schema_overrides={"cam_fork/frame_idx": pl.Int32})
    return (
        frames
        .select("cam_fork/frame_idx", "p_fwd", "p_lat", "p_dist", "speed")
        .join(detections, on="cam_fork/frame_idx")
        .with_columns(pl.lit(job).alias("job"))
    )


def _save_overlay(images: np.ndarray, found: list, path: Path) -> None:
    from PIL import Image, ImageDraw

    tiles = []
    for image, f in zip(images, found, strict=True):
        im = Image.fromarray(image)
        if f is not None:
            ImageDraw.Draw(im).rectangle(f[0], outline="red", width=6)
        tiles.append(im.resize((960, 540)))
    sheet = Image.new("RGB", (1920, 540 * ((len(tiles) + 1) // 2)))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % 2) * 960, (i // 2) * 540))
    sheet.save(path)


def summarize(results: pl.DataFrame) -> None:
    pl.Config.set_tbl_rows(50)
    pl.Config.set_float_precision(3)
    detected = pl.col("score").is_not_null() & pl.col("score").is_not_nan()
    recall = (
        results
        .with_columns(pl.col("p_fwd").cut(list(BINS)).alias("uwb_forward_m"))
        .group_by("uwb_forward_m")
        .agg(
            pl.len().alias("frames"),
            detected.mean().alias("detected"),
            pl.col("score").median(),
        )
        .sort("uwb_forward_m")
    )
    print("\n== proposer recall over the approach, by UWB pallet-forward")  # ruff: ignore[print]
    print(recall)  # ruff: ignore[print]

    ok = results.filter(
        detected
        & pl.col("box_floor_x").is_finite()
        & pl.col("p_fwd").is_between(1.0, 6.0)
    )
    x, y = ok["p_fwd"].to_numpy(), ok["box_floor_x"].to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    print(  # ruff: ignore[print]
        f"\n== box floor-x vs UWB forward, {len(ok)} frames: slope {slope:.3f}, "
        f"intercept {intercept:+.3f} m, residual std {resid.std():.3f} m "
        "(slope ~1 = CAM_FORK roughly to scale; the residual is mostly UWB)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", nargs="*", default=None, help="default: every job")
    parser.add_argument(
        "--stride", type=int, default=5, help="every n-th approach frame"
    )
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = args.jobs or sorted(
        p.name for p in D12_ROOT.iterdir() if p.name.startswith("20")
    )

    proposer = Proposer(args.device)
    results = []
    for job in jobs:
        try:
            results.append(run_job(job, proposer, args.stride, args.batch, args.out))
        except (ValueError, subprocess.CalledProcessError) as e:
            logger.warning("skip", job=job, error=str(e)[:200])
            continue
        logger.info("job", job=job, frames=len(results[-1]))
    combined = pl.concat(results)
    combined.write_parquet(args.out / "detections.parquet")
    summarize(combined)


if __name__ == "__main__":
    main()
