"""Validates the at-rest brake-hold window cap on real training data.

Runs the `samples` pipeline of the rendered config/dataset/yaak/train.yaml
(the same DuckDB SQL used at train time, including the Stage-8 at-rest cap
gated by ``atrest_keep_pct``) on a deterministic subset of training drives,
once with the cap disabled (100) and once enabled (e.g. 5), and compares:

  * total windows and the overall dropped fraction
  * at-rest windows (speed at the decision frame < 1 km/h) and, among them,
    gas-pressed / brake-pressed counts at the decision frame (the headline
    at-rest->gas fraction and brake:gas ratio)
  * launch-ish windows (at rest at decision, any gas press in the clip) —
    these MUST be identical across cap settings; the cap must never drop them
  * static brake-hold windows (the exact cap predicate) and the achieved
    keep ratio

This doubles as the end-to-end smoke test of the cap SQL (DuckDB 1-based
list indexing, ``//`` integer division).

Usage:
    uv run python -m rmind.scripts.atrest_window_stats \
        --cache-dir /tmp/rbyte_cache --num-drives 30 --keep-pct 5
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import polars as pl
from omegaconf import OmegaConf
from rbyte.config import PipelineHydraConfig
from rbyte.dataset import Dataset

CONFIG_PATH = Path(__file__).parents[3] / "config/dataset/yaak/train.yaml"
DATA_ROOT = Path("/nasa/drives/yaak/data")
FULL_TRAIN_DRIVES = 655

# Interpolation context matching experiment/yaak/control_transformer/finetune_throttle.yaml
CLIP_LENGTH = 11
EPISODE_LENGTH = 6  # decision frame = last observed timestep (1-based in SQL)
EPISODE_STEP = 10
EPISODE_STRIDE = 10
CLIP_PERIOD = f"{CLIP_LENGTH * EPISODE_STEP}i"

SPEED = "meta/VehicleMotion/speed"
GAS = "meta/VehicleMotion/gas_pedal_normalized"
BRAKE = "meta/VehicleMotion/brake_pedal_normalized"
FRAME_IDX = "meta/ImageMetadata.cam_front_left/frame_idx"

# "Pressed" thresholds for the headline stats (Notion-plan definition)
PRESS_THRESH = 0.02
# The cap predicate's own thresholds (must mirror the SQL exactly)
CAP_SPEED_DECISION = 1.0
CAP_BRAKE_DECISION = 0.1
CAP_GAS_MAX = 0.02
CAP_SPEED_MAX = 2.0


def _build_samples_config(
    keep_pct: int, cache_dir: Path, num_drives: int
) -> PipelineHydraConfig:
    """Load the rendered dataset config and extract a resolved samples config.

    The dataset yaml interpolates against keys normally provided by the
    experiment config (paths.*, clip_length, ..., atrest_keep_pct); nest it
    under a synthetic root providing exactly that context.

    Raises:
        TypeError: if the resolved ``dataset.samples`` is not a mapping.
    """
    root = OmegaConf.create({
        "paths": {"data": str(DATA_ROOT), "rbyte": {"cache": str(cache_dir)}},
        "clip_length": CLIP_LENGTH,
        "episode_length": EPISODE_LENGTH,
        "episode_step": EPISODE_STEP,
        "episode_stride": EPISODE_STRIDE,
        "clip_period": CLIP_PERIOD,
        "atrest_keep_pct": keep_pct,
        "dataset": OmegaConf.load(CONFIG_PATH),
    })
    OmegaConf.resolve(root)

    samples = OmegaConf.to_container(root.dataset.samples)
    if not isinstance(samples, dict):
        msg = f"expected dataset.samples to be a mapping, got {type(samples)}"
        raise TypeError(msg)

    # Deterministically truncate to ~num_drives spread across the full list
    inputs = samples["inputs"]
    n = len(inputs["input_id"])
    stride = max(1, round(n / num_drives))
    keep = range(0, n, stride)
    samples["inputs"] = {k: [v[i] for i in keep] for k, v in inputs.items()}

    return PipelineHydraConfig.model_validate(samples)


def _run_pipeline(
    keep_pct: int, cache_dir: Path, num_drives: int
) -> tuple[pl.DataFrame, float, int]:
    config = _build_samples_config(keep_pct, cache_dir, num_drives)
    n_drives = len(config.inputs["input_id"])

    print(f"\n[cap{keep_pct}] building samples for {n_drives} drives…")  # noqa: T201
    start = time.perf_counter()
    df = Dataset._build_samples(config)  # noqa: SLF001
    elapsed = time.perf_counter() - start
    print(f"[cap{keep_pct}] done in {elapsed:,.1f}s ({df.height:,} windows)")  # noqa: T201

    return df, elapsed, n_drives


def _flags(df: pl.DataFrame) -> pl.DataFrame:
    """Per-window boolean flags; SQL 1-based index N -> polars 0-based N-1."""
    decision = EPISODE_LENGTH - 1
    speed = pl.col(SPEED).cast(pl.List(pl.Float64))
    gas = pl.col(GAS).cast(pl.List(pl.Float64))
    brake = pl.col(BRAKE).cast(pl.List(pl.Float64))

    at_rest = speed.list.get(decision) < CAP_SPEED_DECISION
    return df.select(
        at_rest.alias("at_rest"),
        (at_rest & (gas.list.get(decision) > PRESS_THRESH)).alias("at_rest_gas"),
        (at_rest & (brake.list.get(decision) > PRESS_THRESH)).alias("at_rest_brake"),
        (at_rest & (gas.list.max() > PRESS_THRESH)).alias("launchish"),
        (
            at_rest
            & (brake.list.get(decision) > CAP_BRAKE_DECISION)
            & (gas.list.max() <= CAP_GAS_MAX)
            & (speed.list.max() < CAP_SPEED_MAX)
        ).alias("static_brake_hold"),
    )


def _launch_keys(df: pl.DataFrame) -> set[tuple[str, int]]:
    """(drive, first frame_idx) identity keys of launch-ish windows."""
    flagged = df.filter(_flags(df)["launchish"])
    first_frame = pl.col(FRAME_IDX).cast(pl.List(pl.Int64)).list.get(0)
    return set(
        flagged.select("input_id", first_frame.alias("frame")).iter_rows()  # ty:ignore[possibly-missing-attribute]
    )


def _print_stats(label: str, df: pl.DataFrame) -> dict[str, int]:
    flags = _flags(df)
    counts = {name: int(flags[name].sum()) for name in flags.columns}
    counts["total"] = df.height

    at_rest, gas, brake = (
        counts["at_rest"],
        counts["at_rest_gas"],
        counts["at_rest_brake"],
    )
    print(f"\n[{label}]")  # noqa: T201
    print(f"  Total windows                : {counts['total']:>9,}")  # noqa: T201
    print(  # noqa: T201
        f"  At-rest (speed[{EPISODE_LENGTH}] < 1)      : {at_rest:>9,}"
        f"  ({at_rest / max(counts['total'], 1):.2%} of total)"
    )
    print(  # noqa: T201
        f"  … gas-pressed at decision    : {gas:>9,}"
        f"  ({gas / max(at_rest, 1):.2%} of at-rest)"
    )
    print(f"  … brake-pressed at decision  : {brake:>9,}")  # noqa: T201
    print(f"  … brake:gas ratio            : {brake / max(gas, 1):>9.1f} : 1")  # noqa: T201
    print(f"  Launch-ish (any gas in clip) : {counts['launchish']:>9,}")  # noqa: T201
    print(f"  Static brake-hold (cap pred) : {counts['static_brake_hold']:>9,}")  # noqa: T201

    return counts


def main(keep_pct: int, cache_dir: Path, num_drives: int) -> None:
    results = {
        pct: _run_pipeline(pct, cache_dir, num_drives) for pct in (100, keep_pct)
    }

    print(f"\n{'=' * 70}")  # noqa: T201
    print("AT-REST BRAKE-HOLD WINDOW CAP STATISTICS")  # noqa: T201
    print(f"{'=' * 70}")  # noqa: T201

    counts: dict[int, dict[str, int]] = {}
    for pct, (df, _, _) in results.items():
        counts[pct] = _print_stats(f"cap{pct}", df)

    base, capped = counts[100], counts[keep_pct]

    print(f"\n[cap100 -> cap{keep_pct}]")  # noqa: T201
    dropped = base["total"] - capped["total"]
    print(  # noqa: T201
        f"  Windows dropped              : {dropped:>9,}"
        f"  ({dropped / max(base['total'], 1):.2%} of total)"
    )
    hold_base, hold_capped = (base["static_brake_hold"], capped["static_brake_hold"])
    print(  # noqa: T201
        f"  Static brake-hold kept       : {hold_capped:>9,} / {hold_base:,}"
        f"  ({hold_capped / max(hold_base, 1):.2%}; target ≈ {keep_pct}%)"
    )

    launch_base = _launch_keys(results[100][0])
    launch_capped = _launch_keys(results[keep_pct][0])
    identical = launch_base == launch_capped
    print(  # noqa: T201
        f"  Launch-ish identical         : {identical}"
        f"  ({len(launch_base):,} vs {len(launch_capped):,})"
    )
    if not identical:
        print(  # noqa: T201
            f"  !! cap dropped {len(launch_base - launch_capped)} launch windows"
            f" / hallucinated {len(launch_capped - launch_base)} — MUST NOT HAPPEN"
        )

    elapsed, n_drives = results[100][1], results[100][2]
    per_drive = elapsed / max(n_drives, 1)
    print("\n[timing, cold cap100 run]")  # noqa: T201
    print(f"  Wall-clock                   : {elapsed:>9,.1f}s for {n_drives} drives")  # noqa: T201
    print(f"  Per drive                    : {per_drive:>9,.1f}s")  # noqa: T201
    print(  # noqa: T201
        f"  Extrapolated {FULL_TRAIN_DRIVES} drives      :"
        f" {per_drive * FULL_TRAIN_DRIVES / 3600:>9,.2f}h"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep-pct",
        type=int,
        default=5,
        help="atrest_keep_pct for the capped run (compared against 100)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="rbyte cache root (paths.rbyte.cache) — use a scratch dir, "
        "never the repo .rbyte_cache",
    )
    parser.add_argument(
        "--num-drives", type=int, default=30, help="approx. number of drives"
    )
    args = parser.parse_args()

    main(args.keep_pct, args.cache_dir, args.num_drives)
