"""Demo videos of the trained entry-face keypoint net on whole pallet approaches.

Per frame (every `--stride`-th, played back in real time): Grounding DINO pallet box
-> crop -> keypoint net -> 12 points -> floor-constrained pose (`pose.solve`, free
pitch). No tracking or smoothing: every frame is independent, so jitter is shown
as it is.

Left: the camera with the detector box, the 12 points coloured by confidence
(green >= 0.6, yellow >= 0.3, red below; a white ring = RANSAC inlier), the entry
face drawn between them, and the pallet box re-projected from the solved pose.
Right: top-down map with the estimated footprint and the UWB pallet tag
(uncalibrated reference, orientation only).

    uv run --extra pallet python -m rmind.scripts.pallet_demo_entry \
        --checkpoint .../best.pt --out .../demo --jobs 2026-09-10--15-17-38
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from structlog import get_logger

from rmind.data.d12 import row_table
from rmind.pallet.correspondence import NAMES, POINTS_3D
from rmind.pallet.entry import MEAN, STD, crop_for_box
from rmind.pallet.geometry import CAM_FORK, EUR_HEIGHT, EUR_LENGTH, FORK_LENGTH
from rmind.pallet.keypoints import PalletKeypointNet, decode
from rmind.pallet.pose import solve, with_pitch
from rmind.pallet.silhouette import BOX_POINTS
from rmind.scripts.pallet_pipeline import (
    CAM_H,
    CAM_W,
    DETECTOR_ID,
    MAP_SIZE,
    VideoWriter,
    draw_map,
    pick_target,
)
from rmind.scripts.pallet_poc_zeroshot import (
    D12_ROOT,
    Proposer,
    approach,
    iter_frames,
    with_pallet_vector,
)

logger = get_logger(__name__)

# the entry face as line segments between named points
FACE_EDGES = [
    ("pallet_top_left", "pallet_top_right"),
    ("pallet_top_right", "pallet_bottom_right"),
    ("pallet_bottom_right", "pallet_bottom_left"),
    ("pallet_bottom_left", "pallet_top_left"),
    *[
        (f"pocket_{s}_{a}", f"pocket_{s}_{b}")
        for s in "lr"
        for a, b in [
            ("top_left", "top_right"),
            ("top_right", "bottom_right"),
            ("bottom_right", "bottom_left"),
            ("bottom_left", "top_left"),
        ]
    ],
]
GREEN, YELLOW, RED, WHITE, CYAN, BOXRED = (
    (80, 220, 80),
    (40, 210, 240),
    (60, 60, 235),
    (255, 255, 255),
    (230, 200, 60),
    (40, 40, 200),
)


def draw(cv2, image, box, kp, conf, est, text):
    view = image[:, :, ::-1].copy()
    if box is not None:
        cv2.rectangle(
            view, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), BOXRED, 2
        )
    if est is not None:
        cam = with_pitch(CAM_FORK, est.pitch_offset)
        pixels, imaged = cam.project(est.pose.apply(BOX_POINTS))
        for seg, ok in zip(
            np.split(pixels, len(pixels) // 24),
            np.split(imaged, len(imaged) // 24),
            strict=True,
        ):
            if ok.sum() > 1:
                cv2.polylines(
                    view,
                    [np.round(seg[ok]).astype(np.int32)],
                    isClosed=False,
                    color=CYAN,
                    thickness=2,
                )
    if kp is not None:
        idx = {n: i for i, n in enumerate(NAMES)}
        for a, b in FACE_EDGES:
            i, j = idx[a], idx[b]
            if conf[i] >= 0.3 and conf[j] >= 0.3:
                cv2.line(
                    view,
                    tuple(np.round(kp[i]).astype(int)),
                    tuple(np.round(kp[j]).astype(int)),
                    WHITE,
                    2,
                )
        inliers = est.inliers if est is not None else np.zeros(len(NAMES), bool)
        for (u, v), c, inl in zip(kp, conf, inliers, strict=True):
            colour = GREEN if c >= 0.6 else YELLOW if c >= 0.3 else RED
            cv2.circle(view, (int(u), int(v)), 9, colour, -1)
            if inl:
                cv2.circle(view, (int(u), int(v)), 14, WHITE, 2)
    view = cv2.resize(view, (CAM_W, CAM_H), interpolation=cv2.INTER_AREA)
    for i, line in enumerate(text):
        cv2.putText(view, line, (14, 34 + 30 * i), 0, 0.8, (0, 0, 0), 4)
        cv2.putText(view, line, (14, 34 + 30 * i), 0, 0.8, WHITE, 2)
    return view


def run_job(
    job: str,
    model: PalletKeypointNet,
    proposer: Proposer,
    args: argparse.Namespace,
    label: str,
    span: tuple[int, int] | None = None,
) -> None:
    import cv2

    if span is None:  # the whole approach, with the UWB tag for reference
        table = row_table(
            D12_ROOT / job,
            cameras=("cam_fork",),
            reference="cam_fork",
            traction_source="reported",
            max_skew_s=1.0,
            include_fork_height=True,
        )
        frames = approach(with_pallet_vector(table))
        idx = frames["cam_fork/frame_idx"].to_numpy()
        first, last = int(idx.min()), int(idx.max())
        uwb = {
            int(f): (a, b)
            for f, a, b in frames.select(
                "cam_fork/frame_idx", "p_fwd", "p_lat"
            ).iter_rows()
        }
        name = job
    else:  # a frame range, no UWB (e.g. drives that fail the clock-skew guard)
        first, last = span
        uwb = {}
        name = f"{job}_{first:06d}-{last:06d}"
    indices = list(range(first, last + 1, args.stride))
    writer = VideoWriter(
        args.out / f"{name}.mp4", CAM_W + MAP_SIZE, CAM_H, fps=30 / args.stride
    )
    trail: list[tuple[float, float]] = []
    previous_box = None
    video = D12_ROOT / job / "cam_fork" / "video.mp4"
    for start, images in zip(
        range(0, len(indices), args.batch),
        iter_frames(video, first, last, args.stride, args.batch),
        strict=False,
    ):
        cands = proposer.detect_all(images)
        for index, image, c in zip(
            indices[start : start + len(images)], images, cands, strict=True
        ):
            target = pick_target(c, previous_box)
            previous_box = target[0] if target is not None else previous_box
            kp = conf = est = None
            text = [f"{job}  frame {index}   [{label}]"]
            if target is not None:
                crop = crop_for_box(np.asarray(target[0]), None)
                patch = (
                    crop.extract(torch.from_numpy(image).permute(2, 0, 1).float() / 255)
                    - MEAN
                ) / STD
                with torch.no_grad():
                    px, cf = decode(model(patch[None].to(args.device)))
                kp, conf = crop.to_frame(px[0].cpu().numpy()), cf[0].cpu().numpy()
                est = solve(
                    CAM_FORK,
                    kp,
                    conf > 0.3,
                    confidence=conf,
                    model=POINTS_3D,
                    sigma_px=3.0,
                )
            if est is not None:
                centre = est.pose.apply(
                    np.array([[EUR_LENGTH / 2, 0.0, EUR_HEIGHT / 2]])
                )[0]
                trail.append((float(centre[0]), float(centre[1])))
                text += [
                    f"face {est.pose.x - FORK_LENGTH:+.2f} m ahead of fork tips   lat {est.pose.y:+.2f} m   yaw {np.rad2deg(est.pose.yaw):+.1f} deg",
                    f"inliers {int(est.inliers.sum())}/12   reproj {est.rms_px:.1f} px   pitch corr {np.rad2deg(est.pitch_offset):+.1f} deg   conf {conf.mean():.2f}",
                ]
            else:
                text.append("no pose" if target is not None else "no pallet")
            panel = draw(
                cv2,
                image,
                target[0] if target is not None else None,
                kp,
                conf,
                est,
                text,
            )
            fit_like = SimpleNamespace(pose=est.pose) if est is not None else None
            writer.write(
                np.hstack([
                    panel,
                    draw_map(
                        cv2, fit_like, uwb.get(index, (np.nan, np.nan)), trail[-60:]
                    ),
                ])
            )  # ty: ignore[invalid-argument-type]
    writer.close()
    logger.info("done", video=name, frames=len(indices))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--jobs",
        nargs="+",
        required=True,
        help="job, or job:first:last for a frame range",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="training labels, to tag each job as labelled or not",
    )
    parser.add_argument(
        "--val-jobs",
        nargs="*",
        default=["2026-09-10--15-17-38", "2026-10-02--15-48-24"],
    )
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = PalletKeypointNet(
        ckpt["backbone"], num_keypoints=len(NAMES), pretrained=False
    )
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(args.device).eval()
    proposer = Proposer(args.device, DETECTOR_ID)
    for spec in args.jobs:
        job, *rest = spec.split(":")
        span = (int(rest[0]), int(rest[1])) if rest else None
        labelled = {n.rsplit("_", 1)[0] for n in json.loads(args.labels.read_text())}
        tag = (
            "held-out val recording"
            if job in args.val_jobs
            else "train recording"
            if job in labelled
            else "unlabelled recording"
        )
        run_job(job, model, proposer, args, tag, span)


if __name__ == "__main__":
    main()
