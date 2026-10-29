"""Evaluate target choice against the UWB pallet tag on the approach frames.

Per recording with a tag: frames up to the pickup (`approach`: first fork lift, else the
closest tag distance). The tag is at the pallet centre and the UWB ego reference is not
at the fork base, while our floor point is the entry face; one constant offset (dx, dy)
between them is calibrated as the median difference over all recordings (2 parameters).

Reports, for the tracker and for the per-frame choice:
    found      share of approach frames with any target
    on_tag     share of approach frames whose target is within 0.75 m of the (offset) tag
    jumps/min  target position jumps > 0.5 m between consecutive steps
also split by tag range (< 2 m, 2-4 m, > 4 m).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl


def approach_end(job: str) -> int | None:
    from rmind.data.d12 import row_table
    from rmind.scripts.pallet_poc_zeroshot import D12_ROOT, approach, with_pallet_vector

    try:
        t = row_table(
            D12_ROOT / job,
            cameras=("cam_fork",),
            reference="cam_fork",
            traction_source="reported",
            max_skew_s=1.0,
            include_fork_height=True,
        )
        return int(approach(with_pallet_vector(t))["cam_fork/frame_idx"].max())
    except Exception:  # ruff: ignore[blind-except]
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tracks", type=Path, required=True)
    args = parser.parse_args()
    data = []
    for p in sorted(args.tracks.glob("*.parquet")):
        d = pl.read_parquet(p)
        if d["tag_xy"].is_null().all():
            continue
        end = approach_end(d["job"][0])
        if end is not None:
            data.append(d.filter(pl.col("frame") <= end))
    allf = pl.concat(data, how="diagonal_relaxed")
    diffs = np.array([
        np.array(t) - np.array(n)
        for t, n in zip(allf["tag_xy"], allf["naive_xy"], strict=True)
        if t is not None and n is not None
    ])
    offset = np.median(diffs, axis=0)
    summary = f"recordings {len(data)}  approach frames {len(allf)}  tag-to-entry-face offset {offset.round(2).tolist()} m"
    print(summary)  # ruff: ignore[print]
    res = {}
    for name, col in (("tracker", "target_xy"), ("per_frame", "naive_xy")):
        rows = {"all": [], "<2 m": [], "2-4 m": [], ">4 m": []}
        jumps, minutes = 0, 0.0
        for d in data:
            prev = None
            for r in d.iter_rows(named=True):
                if r["tag_xy"] is None:
                    continue
                tag = np.array(r["tag_xy"]) - offset
                rng = float(np.hypot(*tag))
                if tag[0] < 0.3:
                    continue
                v = None if r[col] is None else np.array(r[col])
                hit = v is not None and np.linalg.norm(v - tag) < 0.75
                band = "<2 m" if rng < 2 else "2-4 m" if rng < 4 else ">4 m"
                for b in ("all", band):
                    rows[b].append((v is not None, hit))
                if (
                    v is not None
                    and prev is not None
                    and np.linalg.norm(v - prev) > 0.5
                ):
                    jumps += 1
                prev = v if v is not None else prev
            minutes += len(d) * 5 / 30 / 60
        res[name] = {
            b: {
                "n": len(x),
                "found": float(np.mean([a for a, _ in x])) if x else 0.0,
                "on_tag": float(np.mean([h for _, h in x])) if x else 0.0,
            }
            for b, x in rows.items()
        }
        res[name]["jumps_per_min"] = jumps / max(minutes, 1e-9)
    for name, r in res.items():
        line = "  ".join(
            f"{b}: n={v['n']} found {v['found']:.0%} on-tag {v['on_tag']:.0%}"
            for b, v in r.items()
            if isinstance(v, dict)
        )
        print(f"{name:10s} {line}  jumps/min {r['jumps_per_min']:.1f}")  # ruff: ignore[print]
    (args.tracks / "eval.json").write_text(
        json.dumps({"offset": offset.tolist(), **res}, indent=1)
    )


if __name__ == "__main__":
    main()
