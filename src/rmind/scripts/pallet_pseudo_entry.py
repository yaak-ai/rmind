"""Turn keypoints-on-every-candidate output into pseudo-labelled training pallets.

Input: `pallet_predict_candidates` parquet (one row per detector candidate with the 12
predicted points, confidences and the floor-pose check). Kept: candidates whose points
agree with one floor pose (>= `--min-inliers` RANSAC inliers, reprojection <= `--max-rms`
px), outside the held-out keypoint validation (`--val-jobs`, matched as name prefixes),
at most one per `--min-gap` frames per recording and position (near-duplicate frames).
Each kept pallet gets its points with confidence >= `--min-conf` (the others are
"not visible"), its crop box, and a loaded / empty tag from a detector head that predicts
it (`--load-head`, `pallet_proposer_multi` checkpoint): the head's box nearest the pallet.

Writes, next to `--out` (a labels JSON in the entry format, keys "frame.jpg#pN"):
`.boxes.json` (crop boxes), `.load.json` (loaded / empty) and the frames as JPEG to
`--images`. Keys hold the absolute image path ("/.../frame.jpg#pN"), so the training
images folder of the hand labels still works for both.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from structlog import get_logger

from rmind.components.timm_backbone import TimmBackbone
from rmind.pallet.correspondence import NAMES
from rmind.scripts.pallet_label_export import read_frames
from rmind.scripts.pallet_poc_zeroshot import D12_ROOT
from rmind.scripts.pallet_proposer_head import ENCODER, LAYER, MEAN, STD
from rmind.scripts.pallet_proposer_multi import decode, make_head

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--load-head", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--val-jobs",
        nargs="+",
        default=[
            "2026-09-10--15-17-38",
            "2026-10-02--15-48-24",
            "2026-10-22--09-50-09_03",
            "2026-10-22--09-50-09_04",
            "2026-10-22--09-50-09_05",
        ],
    )
    parser.add_argument("--min-inliers", type=int, default=10)
    parser.add_argument("--max-rms", type=float, default=4.0)
    parser.add_argument("--min-conf", type=float, default=0.5)
    parser.add_argument(
        "--min-gap",
        type=int,
        default=15,
        help="frames between kept pallets at the same place",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.images.mkdir(parents=True, exist_ok=True)

    d = pl.concat(
        [pl.read_parquet(p) for p in sorted(args.candidates.glob("*.parquet"))],
        how="diagonal_relaxed",
    ).filter(
        (pl.col("inliers") >= args.min_inliers) & (pl.col("rms_px") <= args.max_rms)
    )
    rows = []
    for r in d.sort("job", "frame").iter_rows(named=True):
        name = f"{r['job']}_{r['frame']:06d}.jpg"
        if name.startswith(tuple(args.val_jobs)):
            continue
        cx = (r["box"][0] + r["box"][2]) / 2
        if any(
            q["job"] == r["job"]
            and r["frame"] - q["frame"] < args.min_gap
            and abs(q["cx"] - cx) < 150
            for q in rows[-20:]
        ):
            continue
        rows.append(r | {"cx": cx, "name": name})
    logger.info("kept", pallets=len(rows), frames=len({q["name"] for q in rows}))

    ck = torch.load(args.load_head, map_location="cpu", weights_only=False)
    res = tuple(ck["res"])
    head = make_head(ck["state_dict"]).to(args.device).eval()
    enc = (
        TimmBackbone(ENCODER, out_indices=[LAYER], img_size=[res[1], res[0]])
        .to(args.device)
        .eval()
    )

    labels, boxes, load = {}, {}, {}
    by_job: dict[str, dict[int, list[dict]]] = {}
    for q in rows:
        by_job.setdefault(q["job"], {}).setdefault(q["frame"], []).append(q)
    for job, frames in by_job.items():
        idx = sorted(frames)
        for s in range(0, len(idx), 32):
            chunk = idx[s : s + 32]
            ims = read_frames(D12_ROOT / job / "cam_fork" / "video.mp4", chunk)
            x = torch.from_numpy(ims).permute(0, 3, 1, 2).float().to(args.device) / 255
            x = F.interpolate(
                x,
                size=(res[1], res[0]),
                mode="bilinear",
                antialias=True,
                align_corners=False,
            )
            with torch.no_grad():
                dets = decode(
                    head(enc((x - MEAN.to(args.device)) / STD.to(args.device))), 0.2
                )
            for f, im, det in zip(chunk, ims, dets, strict=True):
                name = f"{job}_{f:06d}.jpg"
                if not (args.images / name).exists():
                    cv2.imwrite(
                        str(args.images / name),
                        im[:, :, ::-1],
                        [cv2.IMWRITE_JPEG_QUALITY, 95],
                    )
                for n, q in enumerate(frames[f]):
                    key = f"{args.images / name}#p{n}"  # absolute: images live outside the hand-label folder
                    kp, conf = np.array(q["kp"]), np.array(q["conf"])
                    labels[key] = {
                        k: [round(float(u), 1), round(float(v), 1)]
                        for k, (u, v), c in zip(NAMES, kp, conf, strict=True)
                        if c >= args.min_conf
                    }
                    boxes[key] = q["box"]
                    # the detector box whose bottom centre is nearest the pallet's lowest points
                    base = np.array([q["cx"], kp[:, 1].max()])
                    near = min(
                        det,
                        key=lambda b: np.hypot(
                            (b[0] + b[2]) / 2 - base[0], b[3] - base[1]
                        ),
                        default=None,
                    )
                    load[key] = (
                        "loaded" if near is not None and near[5] >= 0.5 else "empty"
                    )
        logger.info("job", job=job, pallets=sum(len(v) for v in frames.values()))
    args.out.write_text(json.dumps(labels, indent=1))
    args.out.with_suffix(".boxes.json").write_text(json.dumps(boxes))
    args.out.with_suffix(".load.json").write_text(json.dumps(load, indent=1))
    logger.info(
        "done", pallets=len(labels), loaded=sum(v == "loaded" for v in load.values())
    )


if __name__ == "__main__":
    main()
