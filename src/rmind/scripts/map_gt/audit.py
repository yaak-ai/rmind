"""Demonstrator compliance audit against the map-GT sidecars (BC ceiling).

Joins VehicleMotion.speed (km/h, from metadata.log) onto each sidecar parquet
(as-of, nearest fix within 300 ms), then reports:

  1. % of driving time over the asserted legal limit at tolerance bands
     (+0/+3/+5/+10 km/h), split by env_class — frames with a known finite
     limit only. Frames are ~10 Hz, so frame share ~= time share.
  2. Same restricted to moving frames (speed > 5 km/h) — standing still can
     never violate, so this is the sharper number.
  3. Unlimited (max_speed == -1) motorway frames vs the 130 km/h advisory.
  4. Stop behaviour near mapped traffic lights / stop signs: per approach
     (contiguous run of frames with dist-to-node <= 50 m), the minimum speed
     reached — distribution over approaches. No light-state GT yet, so red
     and green approaches are pooled.

Writes diag_results/map_gt/audit_report.md.

Usage:
  python src/rmind/scripts/map_gt/audit.py \
      [--sidecar-root caches/map_gt] [--data-root /nasa/drives/yaak/data] \
      [--out diag_results/map_gt/audit_report.md]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from yaak_metadata import parse_metadata_log  # noqa: E402

BANDS = (0.0, 3.0, 5.0, 10.0)
MOVING_KMH = 5.0
APPROACH_RADIUS_M = 50.0
ADVISORY_KMH = 130.0
JOIN_TOLERANCE_US = 300_000


def load_joined(sidecar: Path, data_root: Path) -> pl.DataFrame:
    vehicle = sidecar.parent.name
    drive_id = sidecar.stem
    df = pl.read_parquet(sidecar)
    meta = parse_metadata_log(data_root / vehicle / drive_id / "metadata.log")
    motion = meta["motion"].rename({"time_stamp_us": "ts_motion"})
    joined = (
        df.sort("time_stamp_us")
        .join_asof(
            motion.sort("ts_motion"),
            left_on="time_stamp_us",
            right_on="ts_motion",
            strategy="nearest",
            tolerance=JOIN_TOLERANCE_US,
        )
        .with_columns(pl.lit(f"{vehicle}/{drive_id}").alias("drive"))
    )
    return joined


def over_limit_table(df: pl.DataFrame, moving_only: bool) -> pl.DataFrame:
    base = df.filter(
        pl.col("max_speed_kmh").is_finite()
        & (pl.col("max_speed_kmh") > 0)
        & pl.col("speed_kmh").is_not_null()
    )
    if moving_only:
        base = base.filter(pl.col("speed_kmh") > MOVING_KMH)
    aggs = [pl.len().alias("frames")]
    for b in BANDS:
        aggs.append(
            (pl.col("speed_kmh") > pl.col("max_speed_kmh") + b)
            .mean()
            .alias(f"+{b:.0f}")
        )
    total = base.group_by(pl.lit("ALL").alias("env_class")).agg(aggs)
    per_env = base.group_by("env_class").agg(aggs).sort("env_class")
    return pl.concat([per_env, total])


def approach_events(df: pl.DataFrame, dist_col: str) -> list[float]:
    """Min speed (km/h) per contiguous approach with dist <= APPROACH_RADIUS_M."""
    mins: list[float] = []
    sub = df.select(
        pl.col(dist_col).alias("d"), pl.col("speed_kmh").alias("v")
    ).to_numpy()
    d, v = sub[:, 0], sub[:, 1]
    inside = np.isfinite(d) & (d <= APPROACH_RADIUS_M) & np.isfinite(v)
    if not inside.any():
        return mins
    # split contiguous runs; also split when dist jumps up by > 100 m (new node)
    idx = np.flatnonzero(inside)
    breaks = np.flatnonzero(
        (np.diff(idx) > 5) | (np.diff(d[idx]) > 100.0)
    )
    for run in np.split(idx, breaks + 1):
        if run.size >= 3:  # >= ~0.3 s inside the window
            mins.append(float(np.nanmin(v[run])))
    return mins


def fmt_pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def md_table(df: pl.DataFrame) -> str:
    cols = df.columns
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.iter_rows(named=True):
        cells = []
        for c in cols:
            val = row[c]
            cells.append(fmt_pct(val) if isinstance(val, float) else str(val))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sidecar-root", type=Path, default=Path("caches/map_gt"))
    ap.add_argument("--data-root", type=Path, default=Path("/nasa/drives/yaak/data"))
    ap.add_argument(
        "--out", type=Path, default=Path("diag_results/map_gt/audit_report.md")
    )
    args = ap.parse_args()

    sidecars = sorted(args.sidecar_root.glob("*/*.parquet"))
    if not sidecars:
        msg = f"no sidecar parquets under {args.sidecar_root}"
        raise SystemExit(msg)

    frames = []
    for sc in sidecars:
        try:
            frames.append(load_joined(sc, args.data_root))
        except Exception as exc:  # noqa: BLE001
            print(f"skip {sc}: {exc}")
    df = pl.concat(frames)
    n_drives = df["drive"].n_unique()

    coverage = (
        df.group_by("drive")
        .agg(
            pl.len().alias("frames"),
            pl.col("max_speed_kmh").is_finite().mean().alias("max_speed known"),
            (pl.col("road_class") != "unknown").mean().alias("road_class known"),
        )
        .sort("drive")
    )

    tbl_all = over_limit_table(df, moving_only=False)
    tbl_moving = over_limit_table(df, moving_only=True)

    unlimited = df.filter(
        (pl.col("max_speed_kmh") == -1.0) & pl.col("speed_kmh").is_not_null()
    )
    n_unl = len(unlimited)
    unl_line = "No unlimited-limit frames observed."
    if n_unl:
        over_adv = float((unlimited["speed_kmh"] > ADVISORY_KMH).mean())
        v95 = float(unlimited["speed_kmh"].quantile(0.95))
        unl_line = (
            f"{n_unl} frames ({n_unl / len(df):.1%} of all) on explicitly "
            f"unlimited roads; {fmt_pct(over_adv)} of them above the 130 km/h "
            f"advisory (95th-pct speed {v95:.0f} km/h)."
        )

    stop_stats = {}
    for name, col in (
        ("traffic light", "dist_to_next_traffic_light_m"),
        ("stop sign", "dist_to_next_stop_sign_m"),
    ):
        mins: list[float] = []
        for _, g in df.group_by("drive", maintain_order=True):
            mins.extend(approach_events(g.sort("time_stamp_us"), col))
        arr = np.asarray(mins)
        stop_stats[name] = arr

    lines = [
        "# Demonstrator compliance audit (map-GT sidecars)",
        "",
        f"Drives: {n_drives}, frames: {len(df)} (~10 Hz; frame share ~ time share).",
        f"Frames with known finite limit: "
        f"{len(df.filter(pl.col('max_speed_kmh').is_finite() & (pl.col('max_speed_kmh') > 0)))}"
        f" / {len(df)}.",
        "",
        "## Sidecar coverage per drive",
        "",
        md_table(coverage),
        "",
        "## % of time over the asserted legal limit (all frames with known limit)",
        "",
        md_table(tbl_all),
        "",
        "## Same, moving frames only (speed > 5 km/h)",
        "",
        md_table(tbl_moving),
        "",
        "## Unlimited (autobahn) segments",
        "",
        unl_line,
        "",
        "## Stop behaviour near mapped nodes (min speed per approach, within 50 m)",
        "",
        "No light-state GT yet — red and green approaches are pooled, so the",
        "distribution mixes 'had to stop' and 'rolled through a green'.",
        "",
    ]
    for name, arr in stop_stats.items():
        if arr.size == 0:
            lines.append(f"- {name}: no approaches found")
            continue
        qs = np.percentile(arr, [10, 25, 50, 75, 90])
        frac_stop = float((arr < 5.0).mean())
        frac_slow = float((arr < 20.0).mean())
        lines.append(
            f"- {name}: {arr.size} approaches; min-speed percentiles "
            f"p10/p25/p50/p75/p90 = {qs[0]:.1f}/{qs[1]:.1f}/{qs[2]:.1f}/"
            f"{qs[3]:.1f}/{qs[4]:.1f} km/h; {fmt_pct(frac_stop)} came (near-)to "
            f"a stop (<5 km/h), {fmt_pct(frac_slow)} slowed below 20 km/h."
        )

    lines += ["", "## Interpretation", "", "_(filled in by the analysis run)_", ""]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    print(f"report -> {args.out}")

    # also print the tables for the caller
    print(tbl_all)
    print(tbl_moving)
    print(unl_line)


if __name__ == "__main__":
    main()
