"""Target-pallet labels without UWB: the pallet on the forks at pickup was the target.

Per recording: pickups are the frames where the forks start to lift (fork height above
`--lift-m`, held for `--hold` frames, after being down). For each pickup the target box is
the detection right in front of the forks just before the lift (centre in the middle half
of the image, largest box among the last detection steps); it is then followed BACKWARDS
through the detections (every `stride`-th frame) by box overlap / centre distance, until it
is lost for `--max-miss` steps or the previous pickup is reached. Recordings without fork
height use the fork-lever signal (`fork1`, % of travel).

Writes `<out>/<job>.json`: {"pickups": [{"frame": P, "carried": share of steps after the
lift with a pallet on the forks, "target": {frame: [x0, y0, x1, y1, score, ...]}}]}.
Works without UWB: recordings that lack `rtls/*` get no world positions, which we do not need.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from structlog import get_logger

from rmind.data import d12
from rmind.scripts.pallet_poc_zeroshot import D12_ROOT

logger = get_logger(__name__)
WIDTH, HEIGHT = 1920, 1080

_world_positions = d12._world_positions  # ruff: ignore[private-member-access]


def _world_positions_optional(
    table: pl.DataFrame, topics: dict, job_dir: Path
) -> pl.DataFrame:
    """UWB world positions when the recording has them; otherwise the table unchanged."""
    if d12.POSE_TOPIC not in topics or d12.PALLET_TOPIC not in topics:
        return table
    return _world_positions(table, topics, job_dir)


d12._world_positions = _world_positions_optional  # ruff: ignore[private-member-access]


def fork_signal(
    job: str, max_skew_s: float = 1.0
) -> tuple[np.ndarray, np.ndarray, str]:
    """(cam_fork frame index, fork height in metres or fork lever in [0, 1], which)."""
    job_dir = D12_ROOT / job
    d12.MCAP_FILE = (
        "data_for_training.mcap"
        if (job_dir / "data_for_training.mcap").is_file()
        else "data.mcap"
    )
    t = d12.row_table(
        job_dir,
        cameras=("cam_fork",),
        reference="cam_fork",
        traction_source="reported",
        max_skew_s=max_skew_s,
        include_fork_height=True,
    )
    idx = t["cam_fork/frame_idx"].to_numpy()
    height = (
        t["fork_height"].list.get(0).to_numpy() * 2.3
    )  # normalised -> metres, as in the policy config
    if np.nanmax(height) > 0.05:
        return idx, height, "height_m"
    return idx, np.clip(t["fork1"].to_numpy(), 0, None), "fork1"


def pickups(idx: np.ndarray, signal: np.ndarray, lift: float, hold: int) -> list[int]:
    up = signal > lift
    out, run = [], 0
    for i in range(1, len(up)):
        run = run + 1 if up[i] else 0
        if run == hold and not up[i - hold]:
            out.append(int(idx[i - hold + 1]))
    return out


def iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def backtrack(
    dets: dict[int, list],
    frames: list[int],
    pickup: int,
    stop: int,
    max_miss: int,
    lookback: int,
) -> dict[int, list]:
    before = [f for f in frames if stop < f < pickup][::-1]
    # the target: right in front of the forks in the last few steps before the lift
    start = None
    for f in before[:lookback]:
        # close and centred: a pallet about to be picked fills a good part of the width
        front = [
            b
            for b in dets[f]
            if WIDTH / 4 < (b[0] + b[2]) / 2 < 3 * WIDTH / 4
            and b[3] > HEIGHT / 2
            and b[2] - b[0] > WIDTH / 6
            and b[4] >= 0.3
        ]
        if front:
            start = (f, max(front, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])))
            break
    if start is None:
        return {}
    track, prev, miss = {start[0]: start[1]}, start[1], 0
    for f in before[before.index(start[0]) + 1 :]:
        w = prev[2] - prev[0]
        cand = [
            b
            for b in dets[f]
            if iou(b, prev) > 0.2
            or np.hypot((b[0] + b[2] - prev[0] - prev[2]) / 2, b[3] - prev[3]) < 0.3 * w
        ]
        if not cand:
            miss += 1
            if miss > max_miss:
                break
            continue
        prev = max(cand, key=lambda b: iou(b, prev))
        track[f], miss = prev, 0
    return dict(sorted(track.items()))


def carried(
    dets: dict[int, list], frames: list[int], pickup: int, window: int = 90
) -> float:
    """Share of detection steps in the `window` frames after the lift that show a pallet on
    the forks: a wide box reaching the bottom of the image (the detector learned "carried"
    pallets as such). Low values mean the lift was not a pickup (e.g. forks raised at a rack)."""
    after = [f for f in frames if pickup <= f < pickup + window]
    if not after:
        return 0.0
    on_forks = [
        any(b[3] > 0.85 * HEIGHT and b[2] - b[0] > 0.4 * WIDTH for b in dets[f])
        for f in after
    ]
    return sum(on_forks) / len(on_forks)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", nargs="*", default=None)
    parser.add_argument(
        "--lift-m",
        type=float,
        default=0.12,
        help="fork height (m) that counts as lifted",
    )
    parser.add_argument(
        "--lift-lever",
        type=float,
        default=0.3,
        help="fork lever (0-1) for recordings without height",
    )
    parser.add_argument(
        "--hold", type=int, default=15, help="frames the forks must stay up"
    )
    parser.add_argument("--max-miss", type=int, default=6)
    parser.add_argument(
        "--lookback",
        type=int,
        default=100,
        help="detection steps before the lift to find the target in",
    )
    parser.add_argument("--max-skew-s", type=float, default=1.0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = args.jobs or sorted(p.stem for p in args.detections.glob("*.parquet"))
    summary = []
    for job in jobs:
        try:
            idx, signal, kind = fork_signal(job, args.max_skew_s)
        except Exception as e:  # ruff: ignore[blind-except] - skip recordings the loader cannot pair
            logger.warning("skip", job=job, error=str(e)[:120])
            continue
        det = pl.read_parquet(args.detections / f"{job}.parquet")
        dets = dict(zip(det["frame"].to_list(), det["boxes"].to_list(), strict=True))
        frames = sorted(dets)
        found = pickups(
            idx,
            signal,
            args.lift_m if kind == "height_m" else args.lift_lever,
            args.hold,
        )
        out, prev = [], -1
        for p in found:
            track = backtrack(dets, frames, p, prev, args.max_miss, args.lookback)
            out.append({
                "frame": p,
                "signal": kind,
                "carried": carried(dets, frames, p),
                "target": {str(f): b for f, b in track.items()},
            })
            prev = p
        (args.out / f"{job}.json").write_text(json.dumps({"pickups": out}))
        n = sum(len(o["target"]) for o in out)
        summary.append({
            "job": job,
            "pickups": len(out),
            "target_frames": n,
            "signal": kind,
        })
        logger.info(
            "job",
            job=job,
            pickups=len(out),
            target_frames=n,
            signal=kind,
            spans=[
                (min(map(int, o["target"])) if o["target"] else None, o["frame"])
                for o in out
            ],
        )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
