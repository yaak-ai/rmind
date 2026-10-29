"""Sample frames for box labelling and pre-label them with a detector's boxes.

From `pallet_detect_all` output: per recording, `--per-job` frames spread evenly over the
frames where the detector found any pallet (frames with more boxes preferred within each
time slice), at least `--min-gap` frames apart and away from already-labelled frames.
Recordings matching `--test-prefix` get `--per-test-job` frames and are tagged as test.
Writes the full-res frames to `--images` and `--out` (draft labels, one entry per
frame: list of {"box", "score", "floor": true, "face": true}) plus `split.json`.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import polars as pl

from rmind.scripts.pallet_label_export import read_frames
from rmind.scripts.pallet_poc_zeroshot import D12_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--labelled", type=Path, nargs="*", default=[])
    parser.add_argument("--exclude-prefix", nargs="*", default=["2026-10-22"])
    parser.add_argument("--test-prefix", default="2026-10-12")
    parser.add_argument("--per-job", type=int, default=3)
    parser.add_argument("--per-test-job", type=int, default=5)
    parser.add_argument("--min-gap", type=int, default=60)
    parser.add_argument("--threshold", type=float, default=0.2)
    args = parser.parse_args()
    args.images.mkdir(parents=True, exist_ok=True)
    seen: dict[str, list[int]] = {}
    for f in args.labelled:
        for n in json.loads(f.read_text()):
            job, fr = n.split("#")[0][:-4].rsplit("_", 1)
            seen.setdefault(job, []).append(int(fr))
    draft, split = {}, {"train": [], "test": []}
    for p in sorted(args.detections.glob("*.parquet")):
        job = p.stem
        if job.startswith(tuple(args.exclude_prefix)):
            continue
        d = pl.read_parquet(p).with_columns(
            pl
            .col("boxes")
            .list.eval(pl.element().list.get(4) >= args.threshold)
            .list.sum()
            .alias("n")
        )
        d = d.filter(pl.col("n") > 0).sort("frame")
        if d.is_empty():
            continue
        test = job.startswith(args.test_prefix)
        k = args.per_test_job if test else args.per_job
        picked: list[int] = []
        for chunk in np.array_split(np.arange(len(d)), k):
            if not len(chunk):
                continue
            sl = d[chunk].sort("n", descending=True)
            for fr in sl["frame"].to_list():
                if all(abs(fr - q) >= args.min_gap for q in picked + seen.get(job, [])):
                    picked.append(fr)
                    break
        boxes = dict(zip(d["frame"].to_list(), d["boxes"].to_list(), strict=True))
        for fr, im in zip(
            sorted(picked),
            read_frames(D12_ROOT / job / "cam_fork" / "video.mp4", sorted(picked)),
            strict=True,
        ):
            name = f"{job}_{fr:06d}.jpg"
            cv2.imwrite(
                str(args.images / name), im[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95]
            )
            draft[name] = [
                {
                    "box": [round(v, 1) for v in b[:4]],
                    "score": round(b[4], 3),
                    "floor": True,
                    "face": True,
                }
                for b in boxes[fr]
                if b[4] >= args.threshold
            ]
            split["test" if test else "train"].append(name)
    args.out.write_text(json.dumps(draft, indent=1))
    (args.out.parent / (args.out.stem + ".split.json")).write_text(
        json.dumps(split, indent=1)
    )


if __name__ == "__main__":
    main()
