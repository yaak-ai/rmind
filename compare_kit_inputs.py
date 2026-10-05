#!/usr/bin/env python
"""Does the kit build the policy's non-image inputs the way rmind's d12 dataset does?

For every window the kit sent to the engine (`policy/inputs`), recompute its four aux
tensors with rmind's own definitions and diff them against what the kit fed:

    speed, fork_above_300   rmind.data.d12 SIGNALS, nearest-in-time to each frame
    relative_ego_pos        rmind.data.d12 poses + the dataset config's DuckDB query
    relative_dropoff_pos    not recomputable: the dropoff is set on the kit's console and
                            not recorded. Instead the kit's tokens are rotated back into
                            the world with rmind's ego pose; a fixed dropoff must come out
                            as one point, so the spread checks the convention.

Each frame is matched by the log time of its `cam_fork/frame` record (joined on pts).
The recording's pose topics are `rtls/pose` / `rtls/pallet_pose` (renamed from the
`qorvo/*` the training jobs carry; same fields).

    . env; python compare_kit_inputs.py /path/to/2026-09-29--14-36-16
"""
# ruff: noqa: T201, PLC0415, ANN001, ANN201, PLR0914, PLR0915

import argparse
from pathlib import Path

import hydra
import numpy as np
import polars as pl
import rmind  # noqa: F401  — registers the `eval:` omegaconf resolver the configs use
from rmind.data import d12

from compare_kit_predictions import CAMERA, numbered, read_kit

POSE_TOPICS = {"POSE_TOPIC": "rtls/pose", "PALLET_TOPIC": "rtls/pallet_pose"}


def rmind_topics(session: Path) -> dict[str, pl.DataFrame]:
    for name, topic in POSE_TOPICS.items():
        setattr(d12, name, topic)
    parts: dict[str, list[pl.DataFrame]] = {}
    for path in numbered(session.glob("sensor--*.mcap")):
        for topic, df in d12.read_mcap(path, (CAMERA,), "command").items():
            parts.setdefault(topic, []).append(df)
    return {t: pl.concat(dfs).sort("log_time") for t, dfs in parts.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session", type=Path)
    args = ap.parse_args()

    with hydra.initialize(version_base=None, config_path="config"):
        cfg = hydra.compose(config_name="train", overrides=["experiment=palletjack/patch_policy/d12"])
    query = next(
        f.func.query for f in cfg.datamodule.train.dataset.samples.pipeline.functions
        if f.output_name == "positioned"
    )

    windows, _, log_time = read_kit(args.session)
    steps = sorted(windows)
    rows = pl.DataFrame({
        "window": np.repeat(np.arange(len(steps)), 6),
        "slot": np.tile(np.arange(6), len(steps)),
        "log_time": np.concatenate([log_time[windows[s]["records"]] for s in steps]),
    }).with_columns(pl.col("log_time").cast(pl.Datetime("ns"))).sort("log_time")

    topics = rmind_topics(args.session)
    table = rows
    for name in ("speed", "fork_above_300"):
        signal = d12.resolve_signal(name, "command")
        cols = [(pl.col(signal.field).cast(pl.Float32) * signal.scale).alias(name)]
        if signal.valid_field is not None:
            cols.append(pl.col(signal.valid_field).alias(f"{name}/valid"))
        series = topics[signal.topic].sort("log_time").select("log_time", *cols)
        table = table.join_asof(series, on="log_time", strategy="nearest")
    ego = topics[d12.POSE_TOPIC].select(
        "log_time", pl.col("x_m").alias("ego_x"), pl.col("y_m").alias("ego_y"),
        pl.col("heading_deg").alias("ego_heading"),
    )
    pallet = topics[d12.PALLET_TOPIC].select(
        "log_time", pl.col("x_m").alias("pallet_x"), pl.col("y_m").alias("pallet_y")
    )
    table = (
        table.join_asof(ego, on="log_time", strategy="nearest")
        .join_asof(pallet, on="log_time", strategy="nearest")
        .with_columns(target_x=pl.lit(0.0, pl.Float32), target_y=pl.lit(0.0, pl.Float32))
    )
    keep = table.select("window", "slot", "ego_x", "ego_y", "ego_heading", "fork_above_300/valid")
    positioned = d12.DuckDBStage(query=query)(row_table=table)
    rm = positioned.join(keep, on=["window", "slot"], suffix="_k").sort("window", "slot")

    kit = {
        name: np.stack([windows[s]["aux"][name].reshape(6, -1) for s in steps]).reshape(len(rm), -1)
        for name in ("speed", "fork_above_300", "relative_ego_pos", "relative_dropoff_pos")
    }
    ours = {
        "speed": rm["speed"].to_numpy()[:, None],
        "fork_above_300": rm["fork_above_300"].to_numpy()[:, None],
        "relative_ego_pos": np.stack(rm["relative_ego_pos"].to_numpy()),
    }

    print(f"{len(steps)} windows · {len(rm)} frames\n")
    print(f"  {'input':<22}{'mean|err|':>11}{'p95|err|':>10}{'max|err|':>10}{'exact':>8}   kit range")
    for name, value in ours.items():
        err = np.abs(value - kit[name])
        print(f"  {name:<22}{err.mean():>11.5f}{np.percentile(err, 95):>10.5f}{err.max():>10.5f}"
              f"{(err < 1e-6).mean():>8.1%}"
              f"   [{kit[name].min():+.3f}, {kit[name].max():+.3f}]")
    invalid = (~rm["fork_above_300/valid"].fill_null(False)).sum()
    print(f"\n  frames rmind would drop (fork_above_300 not valid): {invalid}")

    # the kit's dropoff tokens, back into the world with rmind's ego pose (inverse of the query)
    t = np.radians(rm["ego_heading"].to_numpy())
    fwd, lat = kit["relative_dropoff_pos"][:, 0] * 10.0, kit["relative_dropoff_pos"][:, 1] * 10.0
    wx = rm["ego_x"].to_numpy() + np.cos(t) * fwd - np.sin(t) * lat
    wy = rm["ego_y"].to_numpy() + np.sin(t) * fwd + np.cos(t) * lat
    print(f"  kit dropoff, back in the world: x {np.median(wx):+.3f} m (sd {wx.std():.3f}), "
          f"y {np.median(wy):+.3f} m (sd {wy.std():.3f})")


if __name__ == "__main__":
    main()
