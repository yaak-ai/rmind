"""Sample approach frames for entry-face labelling: forks empty, a pallet ahead.

Per job, every `--step`-th frame is scored for "fork tips visible" (the camera
rides the fork carriage, so the empty tips sit in a fixed region at the bottom of
the frame; nothing on the forks and nothing covering the lens means they show).
The reference is the job's own fork-region edge map, seeded from the start of
the recording (forks empty) and refined on its most fork-like frames, so each
truck/mount calibrates itself. Clear frames are then
checked with Grounding DINO for a pallet that is AHEAD of the forks (box above the
fork region, not tiny), and `--per-job` of those are taken evenly over the job.

Frame timestamps are not needed, so drives that fail the clock-pairing guard
(e.g. 2026-10-22--09-50-09) can be sampled too. Frames are written at 1920x1080
(720p drives are upscaled; same field of view).

Writes `{out}/{images_dir}/{job}_{frame:06d}.jpg` and `{out}/{tasks_name}`
(Label Studio tasks, no predictions; `scripts.pallet_dino_prelabel` adds them).
"""

import argparse
import json
import os
import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import]
from pathlib import Path
from typing import Final

import numpy as np

from rmind.scripts.pallet_label_export import read_frames
from rmind.scripts.pallet_pipeline import DETECTOR_ID, pick_target
from rmind.scripts.pallet_poc_zeroshot import D12_ROOT, Proposer

W, H = 480, 270
# fork-tip region at 480x270 (full-res x 480-1380, y 880-1080)
ROI: Final = (120, 345, 220, 270)
CLEAR: Final = 0.35
# a pallet "ahead": box bottom above the fork region and at least this wide (px)
MAX_BOX_BOTTOM: Final = 880
MIN_BOX_WIDTH: Final = 120


def fork_scores(video: Path, step: int) -> tuple[np.ndarray, np.ndarray]:
    """Frame indices and their smoothed fork-visibility score."""
    import cv2

    env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
    proc = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [
            shutil.which("ffmpeg") or "ffmpeg",
            "-v",
            "error",
            "-i",
            str(video),
            "-vf",
            f"select=not(mod(n\\,{step})),scale={W}:{H}",
            "-vsync",
            "0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-",
        ],
        env=env,
        capture_output=True,
        check=True,
    )
    frames = np.frombuffer(proc.stdout, np.uint8).reshape(-1, H, W)
    x0, x1, y0, y1 = ROI
    rois, std = [], frames.reshape(len(frames), -1).std(axis=1)
    for g in frames:
        g = cv2.GaussianBlur(g.astype(np.float32), (3, 3), 0)  # ruff: ignore[redefined-loop-name]
        rois.append(
            cv2.magnitude(
                cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
            )[y0:y1, x0:x1]
        )
    rois = np.stack(rois)

    def ncc(a: np.ndarray, b: np.ndarray) -> float:
        a, b = a - a.mean(), b - b.mean()
        d = np.sqrt((a * a).sum() * (b * b).sum())
        return float((a * b).sum() / d) if d > 0 else 0.0

    # self-calibrating reference, seeded from the start of the recording: jobs begin
    # with empty forks, while in a short approach job the pallet sits on the forks for
    # most of it, so a whole-job median would learn "loaded" instead. Then re-estimated
    # from the frames that match it best
    ref = np.median(rois[: max(len(rois) // 10, 5)], axis=0)
    for _ in range(2):
        s = np.array([ncc(r, ref) for r in rois])
        ref = np.median(rois[s >= np.quantile(s, 0.7)], axis=0)
    s = np.convolve(np.array([ncc(r, ref) for r in rois]), np.ones(3) / 3, mode="same")
    s[std < 8] = 0
    return np.arange(len(frames)) * step, s


def main() -> None:
    import cv2

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--images-dir", default="entry")
    parser.add_argument("--tasks-name", default="tasks_entry.json")
    parser.add_argument("--per-job", type=int, default=8)
    parser.add_argument("--long-job", default="2026-10-22--09-50-09")
    parser.add_argument("--long-job-frames", type=int, default=40)
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument(
        "--exclude",
        type=Path,
        nargs="*",
        default=[],
        help="image folders whose frames to skip",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    (args.out / args.images_dir).mkdir(parents=True, exist_ok=True)
    excluded = {p.stem for d in args.exclude for p in d.glob("*.jpg")}
    rng = np.random.default_rng(args.seed)
    proposer = Proposer(args.device, DETECTOR_ID)

    tasks = []
    for job in sorted(p.name for p in D12_ROOT.iterdir() if p.name.startswith("20")):
        video = D12_ROOT / job / "cam_fork" / "video.mp4"
        if not video.is_file():
            continue
        frames, score = fork_scores(video, args.step)
        clear = frames[score >= CLEAR]
        want = args.long_job_frames if job == args.long_job else args.per_job
        if len(clear) < want:
            print(job, "clear frames", len(clear), "- skipped", flush=True)  # ruff: ignore[print]
            continue
        # check ~4x more than needed for a pallet ahead, evenly over the job
        probe = [
            int(f)
            for f in clear[
                np.linspace(0, len(clear) - 1, min(len(clear), 4 * want)).astype(int)
            ]
        ]
        probe = [f for f in probe if f"{job}_{f:06d}" not in excluded]
        images = np.concatenate([
            read_frames(video, probe[i : i + 16]) for i in range(0, len(probe), 16)
        ])
        ahead = []
        for f, image, boxes in zip(
            probe,
            images,
            [
                c
                for i in range(0, len(images), 8)
                for c in proposer.detect_all(images[i : i + 8])
            ],
            strict=True,
        ):
            target = pick_target(boxes, None)
            if (
                target is not None
                and target[0][3] < MAX_BOX_BOTTOM
                and target[0][2] - target[0][0] >= MIN_BOX_WIDTH
            ):
                ahead.append((f, image))
        if not ahead:
            print(job, "no pallet ahead in clear frames - skipped", flush=True)  # ruff: ignore[print]
            continue
        keep = np.unique(
            np.linspace(0, len(ahead) - 1, min(want, len(ahead))).round().astype(int)
        )
        for i in keep:
            f, image = ahead[i]
            name = f"{job}_{f:06d}.jpg"
            cv2.imwrite(
                str(args.out / args.images_dir / name),
                image[:, :, ::-1],
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )
            tasks.append({
                "data": {
                    "image": f"/data/local-files/?d={args.images_dir}/{name}",
                    "job": job,
                    "frame": f,
                }
            })
    rng.shuffle(tasks)
    (args.out / args.tasks_name).write_text(json.dumps(tasks, indent=1))
    print("tasks", len(tasks))  # ruff: ignore[print]


if __name__ == "__main__":
    main()
