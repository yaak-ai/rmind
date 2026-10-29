"""Keypoints and floor pose for EVERY pallet candidate, not just the chosen target.

For every cached frame (`pallet_proposer_head cache`: every 10th frame, Grounding DINO
boxes down to its 0.25 threshold, fork tips removed) the keypoint net runs on each
candidate box with score >= `--min-score`, and `pose.solve` checks the 12 points
against one floor pose. Writes one parquet per job, one row per candidate: frame,
box, box score, points, confidences, pose, RANSAC inliers, reprojection RMS and the
entry-face box (bounds of the confident points).

Candidates that yield a consistent floor pose are entry faces the keypoint net can
read; they give per-pallet poses for multi-pallet scenes and keypoint-verified
entry-face boxes for training a proposer.
"""

import argparse
from pathlib import Path

import numpy as np
import polars as pl
import torch
from structlog import get_logger

from rmind.pallet.correspondence import NAMES, POINTS_3D
from rmind.pallet.entry import MEAN, STD, crop_for_box
from rmind.pallet.geometry import CAM_FORK
from rmind.pallet.keypoints import PalletKeypointNet, decode
from rmind.pallet.pose import solve
from rmind.scripts.pallet_poc_zeroshot import D12_ROOT
from rmind.scripts.pallet_predict_entry import frames

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("/nasa/data/d12/yaak/pallet-proposer-20261027-ab/cache"),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", nargs="*", default=None)
    parser.add_argument("--min-score", type=float, default=0.25)
    parser.add_argument("--stride", type=int, default=10, help="must match the cache")
    parser.add_argument(
        "--batch", type=int, default=32, help="crops per keypoint-net batch"
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    net = PalletKeypointNet(
        ckpt["backbone"], num_keypoints=len(NAMES), pretrained=False
    )
    net.load_state_dict(ckpt["state_dict"])
    net = net.to(args.device).eval()
    jobs = args.jobs or sorted(p.stem for p in args.cache.glob("*.pt"))
    for job in jobs:
        if (args.out / f"{job}.parquet").exists():
            continue
        cache = torch.load(
            args.cache / f"{job}.pt", weights_only=False, map_location="cpu"
        )
        cands = dict(zip(cache["frame"], cache["boxes"], strict=True))
        rows = []
        for idx, images in frames(
            D12_ROOT / job / "cam_fork" / "video.mp4", args.stride, 8
        ):
            items = [
                (f, im, b)
                for f, im in zip(idx, images, strict=True)
                for b in cands.get(f, [])
                if b[4] >= args.min_score
            ]
            for s in range(0, len(items), args.batch):
                chunk = items[s : s + args.batch]
                crops = [crop_for_box(np.asarray(b[:4]), None) for _, _, b in chunk]
                patches = torch.stack([
                    (
                        c.extract(torch.from_numpy(im).permute(2, 0, 1).float() / 255)
                        - MEAN
                    )
                    / STD
                    for c, (_, im, _) in zip(crops, chunk, strict=True)
                ])
                with torch.no_grad():
                    px, conf = decode(net(patches.to(args.device)))
                for c, (f, _, b), p, cf in zip(
                    crops, chunk, px.cpu().numpy(), conf.cpu().numpy(), strict=True
                ):
                    kp = c.to_frame(p)
                    keep = cf > 0.3
                    est = solve(
                        CAM_FORK, kp, keep, confidence=cf, model=POINTS_3D, sigma_px=3.0
                    )
                    face = (
                        [*kp[keep].min(0).tolist(), *kp[keep].max(0).tolist()]
                        if keep.sum() >= 4
                        else None
                    )
                    rows.append({
                        "job": job,
                        "frame": f,
                        "box": b[:4],
                        "box_score": b[4],
                        "kp": kp.tolist(),
                        "conf": cf.tolist(),
                        "conf_mean": float(cf.mean()),
                        "face": face,
                        "pose": None if est is None else list(est.pose),
                        "inliers": 0 if est is None else int(est.inliers.sum()),
                        "rms_px": None if est is None else est.rms_px,
                    })
        pl.DataFrame(rows, infer_schema_length=None).write_parquet(
            args.out / f"{job}.parquet"
        )
        ok = sum(r["inliers"] >= 10 for r in rows)
        logger.info("job", job=job, candidates=len(rows), consistent=ok)


if __name__ == "__main__":
    main()
