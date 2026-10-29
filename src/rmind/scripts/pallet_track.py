"""Track detected pallets over time and choose the target pallet.

Input: `pallet_detect_all` detections (every `stride`-th frame, boxes with scores).

1. Floor position per box: the bottom-centre pixel is the pallet's floor contact; its ray
   meets the floor (z = 0) at a robot-frame point (x forward along the forks, y left).
2. World frame: with the recording's ego pose (UWB, metres / degrees) the point is moved
   into the world frame, so a standing pallet stays put while the truck moves. Without a
   pose (e.g. a drive that fails the clock-skew guard) tracking stays in the robot frame.
3. Tracks: greedy nearest matching on floor distance, gated at 0.4 m + 10 % of range (range
   error grows with distance); a track is confirmed after 3 hits and dropped after 6
   misses; its position is the score-weighted running mean of its matches.
4. Target: among confirmed tracks in front of the forks (x > 0.8 m, |y| < 1 m + 0.4 x),
   the one with the lowest cost range + 2 |y|; a different track takes over only after it
   has been cheaper by 0.3 m for 3 steps in a row (hysteresis).

Scores, on recordings with a UWB pallet tag (the pallet actually picked up): share of
approach frames whose target is within 1 m of the tag, and target switches per minute,
for the tracker and for a per-frame choice without tracking (lowest cost box per frame).
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import polars as pl
from structlog import get_logger

from rmind.pallet.geometry import CAM_FORK

logger = get_logger(__name__)
CONFIRM, MAX_MISS, SWITCH_STEPS, SWITCH_MARGIN = 3, 6, 3, 0.3


def floor_points(boxes: list[list[float]]) -> np.ndarray:
    """(N, 2) robot-frame floor points of box bottom centres; NaN where the ray misses the floor."""
    if not boxes:
        return np.zeros((0, 2))
    px = np.array([[(b[0] + b[2]) / 2, b[3]] for b in boxes])
    pts, ok = CAM_FORK.intersect_plane(px, 0.0)
    pts = pts[:, :2]
    pts[~ok] = np.nan
    return pts


def cost(p_robot: np.ndarray) -> float:
    x, y = p_robot
    if not (x > 0.8 and abs(y) < 1.0 + 0.4 * x):
        return np.inf
    return float(np.hypot(x, y) + 2 * abs(y))


class Tracker:
    def __init__(self) -> None:
        self.tracks: list[dict] = []
        self.next_id = 0
        self.target: int | None = None
        self.challenger: tuple[int, int] | None = None  # (track id, steps cheaper)

    def step(
        self, pts: np.ndarray, scores: np.ndarray, to_robot
    ) -> tuple[int | None, list[dict]]:
        keep = np.isfinite(pts).all(axis=1)
        pts, scores = pts[keep], scores[keep]
        live = self.tracks
        if live and len(pts):
            d = np.linalg.norm(
                np.array([t["p"] for t in live])[:, None] - pts[None], axis=-1
            )
            rng = np.linalg.norm(to_robot(pts), axis=-1)
            gate = 0.4 + 0.1 * rng[None]
            pairs, used_t, used_d = [], set(), set()
            for i, j in zip(
                *np.unravel_index(np.argsort(d, axis=None), d.shape), strict=True
            ):
                if i not in used_t and j not in used_d and d[i, j] < gate[0, j]:
                    pairs.append((int(i), int(j)))
                    used_t.add(i)
                    used_d.add(j)
        else:
            pairs = []
        matched_t, matched_d = {i for i, _ in pairs}, {j for _, j in pairs}
        for i, j in pairs:
            t = live[i]
            w = t["w"] + scores[j]
            t["p"] = (t["p"] * t["w"] + pts[j] * scores[j]) / w
            t["w"], t["hits"], t["miss"] = min(w, 5.0), t["hits"] + 1, 0
        for i, t in enumerate(live):
            if i not in matched_t:
                t["miss"] += 1
        for j in range(len(pts)):
            if j not in matched_d:
                live.append({
                    "id": self.next_id,
                    "p": pts[j].copy(),
                    "w": scores[j],
                    "hits": 1,
                    "miss": 0,
                })
                self.next_id += 1
        self.tracks = [t for t in live if t["miss"] <= MAX_MISS]
        confirmed = [t for t in self.tracks if t["hits"] >= CONFIRM]
        costs = {t["id"]: cost(to_robot(t["p"][None])[0]) for t in confirmed}
        costs = {k: v for k, v in costs.items() if np.isfinite(v)}
        if self.target not in costs:
            self.target = min(costs, key=costs.get) if costs else None
            self.challenger = None
        elif costs:
            best = min(costs, key=costs.get)
            if best != self.target and costs[best] < costs[self.target] - SWITCH_MARGIN:
                n = (
                    self.challenger[1] + 1
                    if self.challenger and self.challenger[0] == best
                    else 1
                )
                self.challenger = (best, n)
                if n >= SWITCH_STEPS:
                    self.target, self.challenger = best, None
            else:
                self.challenger = None
        return self.target, confirmed


def poses(job: str) -> pl.DataFrame | None:
    from rmind.data.d12 import row_table
    from rmind.scripts.pallet_poc_zeroshot import D12_ROOT

    try:
        t = row_table(
            D12_ROOT / job,
            cameras=("cam_fork",),
            reference="cam_fork",
            traction_source="reported",
            max_skew_s=1.0,
        )
    except Exception:  # ruff: ignore[blind-except] - no usable pose: track in the robot frame
        return None
    return t.select(
        "cam_fork/frame_idx", "ego_x", "ego_y", "ego_heading", "pallet_x", "pallet_y"
    ).drop_nulls(["ego_x", "ego_y", "ego_heading"])


def run_job(path: Path) -> tuple[pl.DataFrame, dict]:
    det = pl.read_parquet(path).sort("frame")
    job = det["job"][0]
    pose = poses(job)
    if pose is not None:
        idx = pose["cam_fork/frame_idx"].to_numpy()
        ex, ey, eh = (pose[c].to_numpy() for c in ("ego_x", "ego_y", "ego_heading"))
        px, py = pose["pallet_x"].to_numpy(), pose["pallet_y"].to_numpy()
    tracker, rows = Tracker(), []
    for frame, boxes in zip(
        det["frame"].to_list(), det["boxes"].to_list(), strict=True
    ):
        if pose is not None:
            k = int(np.clip(np.searchsorted(idx, frame), 0, len(idx) - 1))
            c, s, o = (
                np.cos(np.deg2rad(eh[k])),
                np.sin(np.deg2rad(eh[k])),
                np.array([ex[k], ey[k]]),
            )
            to_world = lambda p, c=c, s=s, o=o: p @ np.array([[c, s], [-s, c]]) + o  # ruff: ignore[lambda-assignment]
            to_robot = lambda p, c=c, s=s, o=o: (p - o) @ np.array([[c, -s], [s, c]])  # ruff: ignore[lambda-assignment]
            tag = np.array([px[k], py[k]]) if np.isfinite(px[k]) else None
        else:
            to_world = to_robot = lambda p: p
            tag = None
        pr = floor_points(boxes)
        scores = np.array([b[4] for b in boxes]) if boxes else np.zeros(0)
        target, confirmed = tracker.step(
            to_world(pr) if len(pr) else pr, scores, to_robot
        )
        costs = [cost(q) if np.isfinite(q).all() else np.inf for q in pr]
        naive = int(np.argmin(costs)) if costs and np.isfinite(min(costs)) else None
        tp = next((t["p"] for t in confirmed if t["id"] == target), None)
        rows.append({
            "job": job,
            "frame": frame,
            "n_boxes": len(boxes),
            "n_tracks": len(confirmed),
            "target": target,
            "target_xy": None if tp is None else to_robot(tp[None])[0].tolist(),
            "naive_xy": None if naive is None else pr[naive].tolist(),
            "tag_xy": None if tag is None else to_robot(tag[None])[0].tolist(),
            "tracks": [
                {"id": t["id"], "xy": to_robot(t["p"][None])[0].tolist()}
                for t in confirmed
            ],
        })
    out = pl.DataFrame(rows, infer_schema_length=None)
    return out, {"job": job, "pose": pose is not None}


def summarize(d: pl.DataFrame, stride: int) -> dict:
    """Agreement with the UWB tag and switch rate, tracker vs per-frame choice."""
    tagged = d.filter(pl.col("tag_xy").is_not_null())
    res = {}
    for name, col in (("tracker", "target_xy"), ("per_frame", "naive_xy")):
        ok, n = 0, 0
        for r in tagged.iter_rows(named=True):
            tag = np.array(r["tag_xy"])
            if not (0.8 < tag[0] < 6.0):
                continue
            n += 1
            ok += r[col] is not None and np.linalg.norm(np.array(r[col]) - tag) < 1.0
        res[f"{name}_on_tag"] = ok / max(n, 1)
        res["tag_frames"] = n
    sw = d.filter(pl.col("target").is_not_null())["target"].to_numpy()
    minutes = len(d) * stride / 30 / 60
    res["tracker_switches_per_min"] = (
        float((np.diff(sw) != 0).sum() / max(minutes, 1e-9)) if len(sw) > 1 else 0.0
    )
    nx = [np.array(v) if v is not None else None for v in d["naive_xy"].to_list()]
    jumps = sum(
        a is not None and b is not None and np.linalg.norm(a - b) > 0.5
        for a, b in itertools.pairwise(nx)
    )
    res["per_frame_jumps_per_min"] = jumps / max(minutes, 1e-9)
    return res


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=5)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    allres = []
    for p in sorted(args.detections.glob("*.parquet")):
        d, info = run_job(p)
        d.write_parquet(args.out / p.name)
        res = info | summarize(d, args.stride)
        allres.append(res)
        logger.info(
            "job",
            **{k: round(v, 3) if isinstance(v, float) else v for k, v in res.items()},
        )
    (args.out / "summary.json").write_text(json.dumps(allres, indent=1))
    tagged = [r for r in allres if r["tag_frames"] > 0]
    n = sum(r["tag_frames"] for r in tagged)
    for k in ("tracker_on_tag", "per_frame_on_tag"):
        logger.info(
            "overall",
            metric=k,
            value=round(sum(r[k] * r["tag_frames"] for r in tagged) / max(n, 1), 3),
            frames=n,
        )
    for k in ("tracker_switches_per_min", "per_frame_jumps_per_min"):
        logger.info(
            "overall", metric=k, value=round(float(np.mean([r[k] for r in allres])), 2)
        )


if __name__ == "__main__":
    main()
