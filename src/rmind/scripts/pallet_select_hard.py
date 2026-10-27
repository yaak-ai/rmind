"""Pick the most diverse among the hardest predicted frames for manual relabelling.

Candidates: frames with a pallet box AHEAD of the forks (box bottom above the
fork region, box not tiny), not already labelled or skipped, at least
`--min-gap` frames away from any labelled frame of the same job.

Hardness, higher = worse:  (1 - mean keypoint confidence)
                           + 0.5 if no floor-constrained pose was found
                           + 0.05 * (12 - RANSAC inliers)
                           + min(reprojection RMS, 20) / 40

Diversity: k-means (`--k` = number of frames wanted) on the crop embeddings of
the `--pool` hardest candidates; per cluster the hardest member is taken, with at
most `--per-job` frames from one recording. Writes the frames as JPEG plus
`selected.json` (name -> predicted points, as a starting draft).
"""

import argparse
import json
import operator
from pathlib import Path

import numpy as np
import polars as pl

from rmind.pallet.correspondence import NAMES
from rmind.scripts.pallet_label_export import read_frames
from rmind.scripts.pallet_poc_zeroshot import D12_ROOT

MAX_BOX_BOTTOM = 880
MIN_BOX_WIDTH = 120


def kmeans(x: np.ndarray, k: int, iters: int = 50, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centres = x[rng.choice(len(x), k, replace=False)]
    for _ in range(iters):
        labels = np.argmin(((x[:, None] - centres[None]) ** 2).sum(-1), axis=1)
        for j in range(k):
            if (labels == j).any():
                centres[j] = x[labels == j].mean(0)
    return labels


def main() -> None:
    import cv2

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument(
        "--labelled",
        type=Path,
        nargs="+",
        required=True,
        help="label / skip json files (names)",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--pool", type=int, default=600)
    parser.add_argument("--per-job", type=int, default=4)
    parser.add_argument("--min-gap", type=int, default=30)
    parser.add_argument(
        "--min-box-score",
        type=float,
        default=0.45,
        help="Grounding DINO boxes below this are mostly not pallets (covered lens, carts, racks)",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    seen: dict[str, list[int]] = {}
    for f in args.labelled:
        data = json.loads(f.read_text())
        for name in data if isinstance(data, list) else data.keys():
            job, frame = name.rsplit(".", 1)[0].rsplit("_", 1)
            seen.setdefault(job, []).append(int(frame))

    d = pl.concat(
        [pl.read_parquet(p) for p in sorted(args.pred.glob("*.parquet"))],
        how="diagonal_relaxed",
    )
    d = d.filter(pl.col("has_box"))
    d = d.with_columns(
        pl.col("box").list.get(3).alias("box_bottom"),
        (pl.col("box").list.get(2) - pl.col("box").list.get(0)).alias("box_w"),
    ).filter(
        (pl.col("box_bottom") < MAX_BOX_BOTTOM)
        & (pl.col("box_w") >= MIN_BOX_WIDTH)
        & (pl.col("box_score") >= args.min_box_score)
    )
    far = [
        all(abs(f - s) >= args.min_gap for s in seen.get(j, []))
        for j, f in zip(d["job"].to_list(), d["frame"].to_list(), strict=True)
    ]
    d = d.filter(pl.Series(far))
    hard = (
        (1 - pl.col("conf_mean"))
        + pl.col("pose").is_null().cast(pl.Float64) * 0.5
        + (12 - pl.col("inliers")) * 0.05
        + pl.col("rms_px").fill_null(20).clip(upper_bound=20) / 40
    )
    # the pool is the hardest frames PER RECORDING, so the long drives (thousands of
    # frames) cannot crowd out the short approaches
    per = max(args.pool // max(d["job"].n_unique(), 1), args.per_job)
    d = (
        d
        .with_columns(hard.alias("hardness"))
        .sort("hardness", descending=True)
        .group_by("job", maintain_order=True)
        .head(per)
    )

    emb = np.array(d["emb"].to_list())
    labels = kmeans(emb, args.k)
    d = d.with_columns(pl.Series("cluster", labels))
    # round-robin over clusters, hardest first, until k frames; members of jobs
    # that reached --per-job are passed over
    picked, per_job = [], {}
    groups = [
        g.sort("hardness", descending=True).to_dicts() for _, g in d.group_by("cluster")
    ]
    groups.sort(key=lambda g: -g[0]["hardness"])
    while len(picked) < args.k and any(groups):
        for g in groups:
            while g and per_job.get(g[0]["job"], 0) >= args.per_job:
                g.pop(0)
            if g and len(picked) < args.k:
                r = g.pop(0)
                picked.append(r)
                per_job[r["job"]] = per_job.get(r["job"], 0) + 1
    print("picked", len(picked), "jobs", len(per_job))  # ruff: ignore[print]

    selected = {}
    by_job: dict[str, list[dict]] = {}
    for r in picked:
        by_job.setdefault(r["job"], []).append(r)
    for job, rows in by_job.items():
        images = read_frames(
            D12_ROOT / job / "cam_fork" / "video.mp4", [r["frame"] for r in rows]
        )
        for r, im in zip(
            sorted(rows, key=operator.itemgetter("frame")), images, strict=True
        ):
            name = f"{job}_{r['frame']:06d}.jpg"
            cv2.imwrite(
                str(args.out / name), im[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95]
            )
            selected[name] = {
                "kp": dict(zip(NAMES, r["kp"], strict=True)),
                "conf": r["conf"],
                "hardness": r["hardness"],
                "box": r["box"],
            }
    (args.out / "selected.json").write_text(json.dumps(selected, indent=1))


if __name__ == "__main__":
    main()
