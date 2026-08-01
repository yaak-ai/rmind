"""Demonstrator speed profiles around speed-limit transitions (map-GT sidecars).

For every sidecar drive, finds transitions of the asserted legal limit
(`max_speed_kmh`, NaN = unknown, -1 = explicitly unlimited) between
consecutive KNOWN values, debounced: a limit run shorter than
``--min-run-s`` (default 2 s) is treated as a flicker and dropped, and the
two runs around a transition must be separated by at most ``--max-gap-s``
(default 3 s) of unknown coverage so the "sign moment" t0 (first frame of
the new limit) is well defined.

For each transition it joins VehicleMotion speed (km/h, parsed from
metadata.log with the self-contained reader in ``yaak_metadata``) on a
0.5 s grid over t0 +- ``--window-s`` (default 10 s) and derives:

- direction (drop / raise; UNLIMITED counts as the top of the ladder) and
  magnitude bucket (<=20 / 21-40 / >40 km/h; unlimited-involved separate)
- v_pre: mean speed over [-10, -8] s (approach speed)
- t_comply (drops): first grid time with speed <= new_limit + 3 km/h --
  negative means adaptation finished BEFORE the sign
- t_adapt (drops): first grid time where speed falls below v_pre - 3 km/h
  and stays there >= 1 s (braking onset), only when v_pre was above the
  new limit + 3
- overshoot (drops): max(0, max speed over [0, +10] - new_limit); plus the
  fraction of the after-window spent > new_limit + 3
- t_headroom (raises): first grid time with speed >= old_limit + 3 (the
  demonstrator starts using the new headroom)
- the full speed profile at 1 s offsets (for group-median profiles)

Writes a markdown report (counts per transition type + per-group adaptation
stats) and a per-transition parquet next to it for downstream analysis.
This is the template for the headline compliance metric: the same windows,
cut around a POLICY rollout, become the policy-vs-demonstrator comparison.

Self-contained: polars + numpy + protobuf only, no torch.

Usage:
  python src/rmind/scripts/map_gt/transitions.py \
      [--sidecar-root caches/map_gt] [--data-root /nasa/drives/yaak/data] \
      [--out diag_results/map_probe/transitions.md] [--workers 8] [--limit N]
"""

from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from yaak_metadata import parse_metadata_log  # noqa: E402

GRID_STEP_S = 0.5
COMPLY_TOL_KMH = 3.0
ADAPT_DROP_KMH = 3.0
ADAPT_HOLD_S = 1.0
MOVING_PRE_KMH = 10.0
UNLIMITED = -1.0

# module-level for Pool workers
_ARGS: dict = {}


def _runs_of_known_limit(
    ts_us: np.ndarray, limit: np.ndarray, *, min_run_s: float, max_gap_s: float
) -> list[tuple[int, int, float]]:
    """Debounced constant-limit runs as (start_ts_us, end_ts_us, value).

    Runs are built over KNOWN frames only (NaN excluded), split when the
    value changes or when consecutive known frames are more than 1 s apart;
    runs shorter than ``min_run_s`` are dropped as flickers; surviving
    adjacent runs with the SAME value merge back when separated by at most
    ``max_gap_s``.
    """
    known = np.isfinite(limit)
    if not known.any():
        return []
    ts = ts_us[known]
    val = limit[known]

    new_run = np.empty(len(ts), dtype=bool)
    new_run[0] = True
    new_run[1:] = (val[1:] != val[:-1]) | (np.diff(ts) > 1_000_000)
    run_id = np.cumsum(new_run) - 1

    runs: list[tuple[int, int, float]] = []
    for r in range(run_id[-1] + 1):
        idx = np.flatnonzero(run_id == r)
        start, end = int(ts[idx[0]]), int(ts[idx[-1]])
        if (end - start) / 1e6 >= min_run_s:
            runs.append((start, end, float(val[idx[0]])))

    merged: list[tuple[int, int, float]] = []
    for run in runs:
        if (
            merged
            and merged[-1][2] == run[2]
            and (run[0] - merged[-1][1]) / 1e6 <= max_gap_s
        ):
            merged[-1] = (merged[-1][0], run[1], run[2])
        else:
            merged.append(run)
    return merged


def _first_grid_time(grid_s: np.ndarray, cond: np.ndarray) -> float:
    hits = np.flatnonzero(cond)
    return float(grid_s[hits[0]]) if hits.size else float("nan")


def _process_drive(sidecar_str: str) -> list[dict]:
    try:
        return _process_drive_inner(sidecar_str)
    except Exception as exc:  # noqa: BLE001  (skip unreadable drives, keep the pool alive)
        print(f"skip {sidecar_str}: {exc}")
        return [{"_error": sidecar_str}]


def _process_drive_inner(sidecar_str: str) -> list[dict]:
    sidecar = Path(sidecar_str)
    data_root = Path(_ARGS["data_root"])
    window_s = _ARGS["window_s"]
    vehicle, drive_id = sidecar.parent.name, sidecar.stem

    df = pl.read_parquet(
        sidecar, columns=["time_stamp_us", "max_speed_kmh", "env_class"]
    ).sort("time_stamp_us")
    ts_us = df["time_stamp_us"].to_numpy()
    limit = df["max_speed_kmh"].to_numpy().astype(np.float64)
    env = df["env_class"].to_numpy()

    runs = _runs_of_known_limit(
        ts_us, limit, min_run_s=_ARGS["min_run_s"], max_gap_s=_ARGS["max_gap_s"]
    )
    if len(runs) < 2:
        return []

    transitions = [
        (prev, nxt)
        for prev, nxt in zip(runs, runs[1:])
        if prev[2] != nxt[2] and (nxt[0] - prev[1]) / 1e6 <= _ARGS["max_gap_s"]
    ]
    if not transitions:
        return []

    motion = parse_metadata_log(data_root / vehicle / drive_id / "metadata.log")[
        "motion"
    ]
    mts = motion["time_stamp_us"].to_numpy().astype(np.float64)
    mv = motion["speed_kmh"].to_numpy().astype(np.float64)

    grid_s = np.arange(-window_s, window_s + GRID_STEP_S / 2, GRID_STEP_S)
    rows: list[dict] = []
    for prev, nxt in transitions:
        old, new = prev[2], nxt[2]
        t0_us = float(nxt[0])
        lo, hi = t0_us - window_s * 1e6, t0_us + window_s * 1e6
        # require motion coverage of the full window
        if mts.size == 0 or mts[0] > lo or mts[-1] < hi:
            continue
        v = np.interp(t0_us + grid_s * 1e6, mts, mv)

        pre = v[grid_s <= -window_s + 2.0]
        v_pre = float(pre.mean())
        after = v[grid_s >= 0.0]

        # direction: UNLIMITED sits at the top of the ladder
        old_eff = np.inf if old == UNLIMITED else old
        new_eff = np.inf if new == UNLIMITED else new
        direction = "drop" if new_eff < old_eff else "raise"

        if old == UNLIMITED or new == UNLIMITED:
            bucket = "from UNLIMITED" if old == UNLIMITED else "to UNLIMITED"
        else:
            delta = abs(new - old)
            bucket = "<=20" if delta <= 20 else ("21-40" if delta <= 40 else ">40")

        row = {
            "drive": f"{vehicle}/{drive_id}",
            "t0_us": int(t0_us),
            "old_kmh": old,
            "new_kmh": new,
            "direction": direction,
            "bucket": bucket,
            "env_class": str(env[np.searchsorted(ts_us, t0_us).clip(0, len(env) - 1)]),
            "v_pre_kmh": v_pre,
            "moving_pre": v_pre > MOVING_PRE_KMH,
            "t_comply_s": float("nan"),
            "t_adapt_s": float("nan"),
            "already_compliant": False,
            "overshoot_kmh": float("nan"),
            "frac_over_after": float("nan"),
            "t_headroom_s": float("nan"),
            "profile_kmh": [float(x) for x in v[:: int(1 / GRID_STEP_S)]],  # 1 s grid
        }

        if direction == "drop" and new != UNLIMITED:
            row["t_comply_s"] = _first_grid_time(grid_s, v <= new + COMPLY_TOL_KMH)
            row["already_compliant"] = v_pre <= new + COMPLY_TOL_KMH
            if not row["already_compliant"]:
                hold = max(1, int(ADAPT_HOLD_S / GRID_STEP_S))
                below = v < v_pre - ADAPT_DROP_KMH
                sustained = np.array([
                    below[i : i + hold].all() for i in range(len(below) - hold + 1)
                ])
                row["t_adapt_s"] = _first_grid_time(grid_s[: len(sustained)], sustained)
            row["overshoot_kmh"] = max(0.0, float(after.max()) - new)
            row["frac_over_after"] = float((after > new + COMPLY_TOL_KMH).mean())
        elif direction == "raise" and old != UNLIMITED:
            row["t_headroom_s"] = _first_grid_time(grid_s, v >= old + COMPLY_TOL_KMH)

        rows.append(row)
    return rows


def _fmt(x: float, nd: int = 1) -> str:
    return "-" if not np.isfinite(x) else f"{x:.{nd}f}"


def _group_stats(g: pl.DataFrame, window_s: float) -> dict[str, str]:
    out: dict[str, str] = {"n": str(len(g))}
    tc = g["t_comply_s"].to_numpy()
    ta = g["t_adapt_s"].to_numpy()
    out["comply<=t0"] = (
        f"{100 * float(np.mean(tc[np.isfinite(tc)] <= 0)):.0f}%"
        if np.isfinite(tc).any()
        else "-"
    )
    out["med t_comply"] = _fmt(float(np.median(tc[np.isfinite(tc)])) if np.isfinite(tc).any() else float("nan"))
    out["med t_adapt"] = _fmt(float(np.median(ta[np.isfinite(ta)])) if np.isfinite(ta).any() else float("nan"))
    ov = g["overshoot_kmh"].to_numpy()
    out["med overshoot"] = _fmt(float(np.median(ov[np.isfinite(ov)])) if np.isfinite(ov).any() else float("nan"))
    fo = g["frac_over_after"].to_numpy()
    out["mean frac>lim after"] = (
        f"{100 * float(np.nanmean(fo)):.0f}%" if np.isfinite(fo).any() else "-"
    )
    th = g["t_headroom_s"].to_numpy()
    out["med t_headroom"] = _fmt(float(np.median(th[np.isfinite(th)])) if np.isfinite(th).any() else float("nan"))

    profiles = np.array(g["profile_kmh"].to_list())
    offsets_s = np.arange(-window_s, window_s + 0.5)
    med = np.median(profiles, axis=0)
    picks = [-10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0]
    out["median profile (km/h)"] = " ".join(
        f"{t:+.0f}s:{med[np.argmin(np.abs(offsets_s - t))]:.0f}" for t in picks
    )
    return out


def _md_table(rows: list[dict[str, str]], first_col: str) -> str:
    cols = [first_col, *[k for k in rows[0] if k != first_col]]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r.get(c, "-")) for c in cols) + " |" for r in rows]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sidecar-root", type=Path, default=Path("caches/map_gt"))
    ap.add_argument("--data-root", type=Path, default=Path("/nasa/drives/yaak/data"))
    ap.add_argument(
        "--out", type=Path, default=Path("diag_results/map_probe/transitions.md")
    )
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="max drives (debug)")
    ap.add_argument("--window-s", type=float, default=10.0)
    ap.add_argument("--min-run-s", type=float, default=2.0)
    ap.add_argument("--max-gap-s", type=float, default=3.0)
    args = ap.parse_args()

    sidecars = sorted(args.sidecar_root.glob("*/*.parquet"))
    if args.limit:
        sidecars = sidecars[: args.limit]
    if not sidecars:
        raise SystemExit(f"no sidecar parquets under {args.sidecar_root}")

    global _ARGS  # noqa: PLW0603
    _ARGS = {
        "data_root": str(args.data_root),
        "window_s": args.window_s,
        "min_run_s": args.min_run_s,
        "max_gap_s": args.max_gap_s,
    }

    rows: list[dict] = []
    failures = 0
    with Pool(args.workers) as pool:
        for i, res in enumerate(
            pool.imap_unordered(_process_drive, map(str, sidecars))
        ):
            if res and "_error" in res[0]:
                failures += 1
            else:
                rows.extend(res)
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(sidecars)} drives, {len(rows)} transitions")
    df = pl.DataFrame(rows)
    print(f"{len(sidecars)} drives -> {len(df)} transitions ({failures} failures)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq = args.out.with_suffix(".parquet")
    df.write_parquet(pq)

    moving = df.filter(pl.col("moving_pre"))

    pair_counts = (
        df.group_by("old_kmh", "new_kmh", "direction")
        .agg(pl.len().alias("n"), pl.col("moving_pre").sum().alias("n_moving"))
        .sort("n", descending=True)
    )

    lines = [
        "# Demonstrator speed profiles around speed-limit transitions",
        "",
        f"Sidecars: {len(sidecars)} drives -> {len(df)} debounced transitions "
        f"(limit runs >= {args.min_run_s:.0f} s, unknown gap <= "
        f"{args.max_gap_s:.0f} s, full +-{args.window_s:.0f} s motion coverage); "
        f"{len(moving)} with a moving approach "
        f"(v_pre > {MOVING_PRE_KMH:.0f} km/h). -1 = explicitly UNLIMITED.",
        "",
        "t0 = first frame asserting the new limit. t_comply = first time speed "
        f"<= new limit + {COMPLY_TOL_KMH:.0f} km/h (negative: before the sign). "
        "t_adapt = braking onset (speed sustainably below approach speed - "
        f"{ADAPT_DROP_KMH:.0f} km/h), drops from above the new limit only. "
        "t_headroom = first time speed > old limit + "
        f"{COMPLY_TOL_KMH:.0f} km/h after a raise.",
        "",
        "## Counts per transition (old -> new km/h)",
        "",
    ]
    pc_rows = [
        {
            "old -> new": f"{r['old_kmh']:.0f} -> {r['new_kmh']:.0f}",
            "direction": r["direction"],
            "n": str(r["n"]),
            "n moving": str(r["n_moving"]),
        }
        for r in pair_counts.iter_rows(named=True)
    ]
    lines += [_md_table(pc_rows, "old -> new"), ""]

    for name, sub in (("all transitions", df), ("moving approaches only", moving)):
        lines += [f"## Adaptation by group ({name})", ""]
        grows = []
        for (direction, bucket), g in sorted(
            sub.group_by("direction", "bucket", maintain_order=True),
            key=lambda kv: (kv[0][0], str(kv[0][1])),
        ):
            grows.append(
                {"group": f"{direction} {bucket}"}
                | _group_stats(g, args.window_s)
            )
        lines += [_md_table(grows, "group") if grows else "_none_", ""]

    lines += [
        "## Notes",
        "",
        f"- Per-transition rows (incl. 1 s speed profiles): `{pq}`.",
        "- This windowing is the template for the headline compliance metric: "
        "the same +-10 s cuts around transitions, evaluated on POLICY rollouts "
        "vs these demonstrator profiles.",
        "",
    ]
    args.out.write_text("\n".join(lines))
    print(f"report -> {args.out}")


if __name__ == "__main__":
    main()
