"""Sample `cam_fork` frames and export them, pre-labelled, as a Label Studio project.

Sampling is stratified, not uniform: every job (all sessions, including the indoor
2026-08-25 one) contributes `--per-job` frames spread evenly over its approach,
which also spreads them over distance - far, mid, and close/truncated at the fork
tips. A uniform draw would be mostly "pallet straight ahead at 3 m".

Pre-labels (Label Studio "predictions", editable):

    box corners     8 viewpoint-canonical corners of the pallet's 3D box
                    (`palloc6d.BOX_CORNER_NAMES`), from the keypoint net
    pocket corners  8 corners of the two fork pockets on the entry face, projected
                    from the floor-constrained pose solved from those box corners

Only points inside the image and above `--min-confidence` are pre-filled. The
labeller drags wrong points into place and DELETES points that are not visible
(hidden behind the pallet, outside the frame): a missing point means "not
visible", so no guessed positions end up in the labels.

Writes under `--out`: `images/{job}_{frame:06d}.jpg` (raw 1920x1080 frames),
`tasks.json`, `labeling_config.xml`, `README.md`.
"""

import argparse
import itertools
import json
import os
import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import]
from pathlib import Path
from typing import Final

import numpy as np
import torch

from rmind.data.d12 import row_table
from rmind.pallet.geometry import CAM_FORK, EUR_POCKET_HEIGHT
from rmind.pallet.keypoints import Crop, PalletKeypointNet, decode
from rmind.pallet.palloc6d import _MEAN, _STD, BOX_CORNER_NAMES, BOX_CORNERS_3D
from rmind.pallet.pose import solve
from rmind.scripts.pallet_eval_keypoints import KV, VH, VW, raw_to_virtual, virtual_rays
from rmind.scripts.pallet_pipeline import DETECTOR_ID, pick_target
from rmind.scripts.pallet_poc_zeroshot import (
    D12_ROOT,
    HEIGHT,
    WIDTH,
    Proposer,
    approach,
    with_pallet_vector,
)

# the two pockets on the entry face (pallet x = 0), seen from the truck: left = +y
_POCKETS: Final = {"pocket_l": (0.3, 0.0725), "pocket_r": (-0.0725, -0.3)}
POCKET_CORNER_NAMES: Final = tuple(
    f"{pocket}_{level}_{side}"
    for pocket in _POCKETS
    for level in ("top", "bottom")
    for side in ("left", "right")
)
POCKET_CORNERS_3D: Final = np.array([
    [
        0.0,
        edges[0] if side == "left" else edges[1],
        EUR_POCKET_HEIGHT if level == "top" else 0.0,
    ]
    for edges in _POCKETS.values()
    for level in ("top", "bottom")
    for side in ("left", "right")
])

BOX_COLOURS: Final = (
    "#e6194b",
    "#f58231",
    "#ffe119",
    "#bfef45",
    "#3cb44b",
    "#42d4f4",
    "#4363d8",
    "#911eb4",
)
POCKET_COLOURS: Final = (
    "#ff00aa",
    "#ff66cc",
    "#aa0066",
    "#cc3399",
    "#00ffaa",
    "#66ffcc",
    "#00aa66",
    "#33cc99",
)


def labeling_config() -> str:
    labels = [
        f'    <Label value="{n}" background="{c}"/>'
        for n, c in zip(BOX_CORNER_NAMES, BOX_COLOURS, strict=True)
    ]
    labels += [
        f'    <Label value="{n}" background="{c}"/>'
        for n, c in zip(POCKET_CORNER_NAMES, POCKET_COLOURS, strict=True)
    ]
    return "\n".join([
        "<View>",
        '  <Header value="Drag points onto the pallet. DELETE any point you cannot see (hidden or outside the frame)."/>',
        '  <KeyPointLabels name="kp" toName="image" strokeWidth="2">',
        *labels,
        "  </KeyPointLabels>",
        '  <Image name="image" value="$image" zoom="true" zoomControl="true" brightnessControl="true"/>',
        '  <TextArea name="notes" toName="image" placeholder="optional: why this frame is odd" rows="1"/>',
        "</View>",
        "",
    ])


def read_frames(video: Path, indices: list[int]) -> np.ndarray:
    select = "+".join(f"eq(n\\,{i})" for i in sorted(indices))
    env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
    raw = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        # some drives are recorded at 1280x720 (same field of view, downscaled); bring
        # every frame to the 1920x1080 the calibration is for
        [
            shutil.which("ffmpeg") or "ffmpeg",
            "-v",
            "error",
            "-i",
            str(video),
            "-vf",
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
        check=True,
        capture_output=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, HEIGHT, WIDTH, 3)


def _point(name: str, u: float, v: float) -> dict:
    return {
        "from_name": "kp",
        "to_name": "image",
        "type": "keypointlabels",
        "original_width": WIDTH,
        "original_height": HEIGHT,
        "value": {
            "x": 100 * u / WIDTH,
            "y": 100 * v / HEIGHT,
            "width": 0.4,
            "keypointlabels": [name],
        },
    }


def main() -> None:
    import cv2

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-job", type=int, default=8)
    parser.add_argument("--total", type=int, default=300)
    parser.add_argument("--min-confidence", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--job",
        default=None,
        help="export given frames of one job instead of sampling all jobs",
    )
    parser.add_argument(
        "--frames",
        type=Path,
        default=None,
        help="parquet with a `frame` column, for --job",
    )
    parser.add_argument(
        "--require-pallet",
        action="store_true",
        help="drop frames with no detected pallet",
    )
    parser.add_argument("--tasks-name", default="tasks.json")
    args = parser.parse_args()
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    net = PalletKeypointNet(ckpt["backbone"], num_keypoints=8, pretrained=False)
    net.load_state_dict(ckpt["state_dict"])
    net = net.to(args.device).eval()
    proposer = Proposer(args.device, DETECTOR_ID)
    k = np.array([
        [CAM_FORK.fx, 0, CAM_FORK.cx],
        [0, CAM_FORK.fy, CAM_FORK.cy],
        [0, 0, 1],
    ])
    map1, map2 = cv2.initUndistortRectifyMap(
        k, np.array(CAM_FORK.k), None, KV, (VW, VH), cv2.CV_32FC1
    )

    if args.job is not None:
        import polars as pl

        plan = [(args.job, sorted(pl.read_parquet(args.frames)["frame"].to_list()))]
    else:
        plan = []
        for job in sorted(
            p.name for p in D12_ROOT.iterdir() if p.name.startswith("20")
        ):
            try:
                table = row_table(
                    D12_ROOT / job,
                    cameras=("cam_fork",),
                    reference="cam_fork",
                    traction_source="reported",
                    max_skew_s=1.0,
                    include_fork_height=True,
                )
            except (ValueError, FileNotFoundError):  # incomplete or still-uploading job
                continue
            frames = approach(with_pallet_vector(table))[
                "cam_fork/frame_idx"
            ].to_numpy()
            if len(frames) < args.per_job:
                continue
            # evenly over the approach, jittered within each slot
            edges = np.linspace(0, len(frames), args.per_job + 1).astype(int)
            plan.append((
                job,
                sorted({
                    int(frames[rng.integers(a, max(b, a + 1))])
                    for a, b in itertools.pairwise(edges)
                }),
            ))
    tasks = []
    for job, picks in plan:
        images = np.concatenate([
            read_frames(D12_ROOT / job / "cam_fork" / "video.mp4", picks[i : i + 16])
            for i in range(0, len(picks), 16)
        ])
        candidates = [
            c
            for i in range(0, len(images), 8)
            for c in proposer.detect_all(images[i : i + 8])
        ]
        for frame, image, boxes in zip(picks, images, candidates, strict=True):
            target = pick_target(boxes, None)
            if target is None and args.require_pallet:
                continue
            name = f"{job}_{frame:06d}.jpg"
            cv2.imwrite(
                str(args.out / "images" / name),
                image[:, :, ::-1],
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )
            result = []
            if target is not None:
                x0, y0, x1, y1 = target[0]
                corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
                valid = np.isfinite(CAM_FORK.rays(corners)).all(axis=1)
                if valid.sum() >= 3:
                    vb = raw_to_virtual(corners[valid])
                    crop = Crop.around((*vb.min(axis=0), *vb.max(axis=0)))
                    virtual = (
                        torch
                        .from_numpy(cv2.remap(image, map1, map2, cv2.INTER_LINEAR))
                        .permute(2, 0, 1)
                        .float()
                        / 255
                    )
                    with torch.no_grad():
                        pixels, conf = decode(
                            net(
                                ((crop.extract(virtual) - _MEAN) / _STD)[None].to(
                                    args.device
                                )
                            )
                        )
                    conf = conf[0].cpu().numpy()
                    raw, imaged = CAM_FORK.project(
                        CAM_FORK.origin
                        + virtual_rays(crop.to_frame(pixels[0].cpu().numpy())) * 3.0
                    )
                    for n, (u, v), c, ok in zip(
                        BOX_CORNER_NAMES, raw, conf, imaged, strict=True
                    ):
                        if ok and c >= args.min_confidence:
                            result.append(_point(n, float(u), float(v)))
                    estimate = solve(
                        CAM_FORK,
                        raw,
                        imaged & (conf >= args.min_confidence),
                        confidence=conf,
                        model=BOX_CORNERS_3D,
                        sigma_px=3.0,
                    )
                    if estimate is not None:
                        pocket_px, pocket_ok = CAM_FORK.project(
                            estimate.pose.apply(POCKET_CORNERS_3D)
                        )
                        result += [
                            _point(n, float(u), float(v))
                            for n, (u, v), ok in zip(
                                POCKET_CORNER_NAMES, pocket_px, pocket_ok, strict=True
                            )
                            if ok
                        ]
            tasks.append({
                "data": {
                    "image": f"/data/local-files/?d=images/{name}",
                    "job": job,
                    "frame": frame,
                },
                "predictions": [
                    {"model_version": args.checkpoint.parent.name, "result": result}
                ],
            })
        print(job, len(picks), flush=True)  # ruff: ignore[print]

    if len(tasks) > args.total:
        keep = set(rng.choice(len(tasks), args.total, replace=False).tolist())
        for i, t in enumerate(tasks):
            if i not in keep:
                (
                    args.out / "images" / Path(t["data"]["image"].split("d=")[1]).name
                ).unlink(missing_ok=True)
        tasks = [t for i, t in enumerate(tasks) if i in keep]
    (args.out / args.tasks_name).write_text(json.dumps(tasks, indent=1))
    (args.out / "labeling_config.xml").write_text(labeling_config())
    print("tasks", len(tasks))  # ruff: ignore[print]


if __name__ == "__main__":
    main()
