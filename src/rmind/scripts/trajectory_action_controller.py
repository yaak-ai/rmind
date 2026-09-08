"""Trajectory -> action-sequence inverse-dynamics controller, built entirely
from one predict parquet (no Hydra/rbyte/checkpoint access needed).

Works on the parquet written by `DataFramePredictionWriter` with the
`policy_allfields` inference config, e.g.:

    outputs/2026-07-16/09-34-08/predictions/yaak/alex-tmp/model-4vqatiom:v4.parquet

Each row is one 11-tick window (history_steps=5, action_horizon=6):
`FEAT_IDX=4` is the last history tick ("now"), `GT_IDX=5` is the first scored
horizon tick (matches `policy/prediction_value` and `score_l1`'s own
convention, see `[[project_rmind_training_gotchas]]`).

This is a **learned inverse-dynamics / trajectory-tracking controller**, not
full MPC -- see `rmind.components.controller` module docstring. Subcommands:

    episode  -- reconstruct a continuous per-tick action trace for one drive
    stitch   -- undo the 11-tick window packaging into a per-drive tick trace
                and enumerate anchors with a contiguous N-tick future, so the
                trajectory horizon becomes a free parameter (no re-predict --
                see `rmind.components.tick_trace`)
    nearstop -- split the held-out near-stop rows by whether their future is
                stationary at all, and compare each group's achievable L1
                floor against what trained controllers reach there
    horizon  -- sweep the trajectory horizon (how far ahead the plan the
                controller consumes reaches) and the control horizon (how many
                actions it emits per pass), training every configuration on
                one identical anchor set and drive split
    reckon   -- dead-reckon the 6-step future trajectory from heading+speed,
                and sanity-gate it against raw GNSS and the existing
                (documented-flawed, GPS-chain) policy/trajectory_gt
    train    -- fit TrajectoryToActionMLP: dead-reckoned trajectory -> the
                action that was actually taken at GT_IDX
    eval     -- apply the trained controller to (a) the dead-reckoned GT
                trajectory, (b) the checkpoint's own predicted trajectory
                (policy/trajectory_value -- the "use it as MPC" case), and
                compare both to (c) the checkpoint's own end-to-end action
                head (policy/prediction_value) and (d) real GT.
    modes    -- test whether the pedal -> realized-motion mapping is
                vehicle-dependent (hidden "speed modes": regen calibration,
                drive mode, sensor drift/wear) using matched-condition
                one-way ANOVA on the REALIZED one-tick speed change -- a
                direct physical test, independent of the trained controller.

Examples
--------
uv run python -m rmind.scripts.trajectory_action_controller episode \\
    --drive Niro096-HQ/2023-01-11--13-47-36 --out /tmp/episode.png

uv run python -m rmind.scripts.trajectory_action_controller stitch \\
    --n-drives 40 --seed 7 --horizon 30 --verify

uv run python -m rmind.scripts.trajectory_action_controller reckon --n-sample 20000

uv run python -m rmind.scripts.trajectory_action_controller train \\
    --limit 200000 --steps 20000 --out /tmp/controller.pt

uv run python -m rmind.scripts.trajectory_action_controller eval \\
    --ckpt /tmp/controller.pt --limit 200000

uv run python -m rmind.scripts.trajectory_action_controller horizon \\
    --horizons 6 15 30 --action-steps 1 6 --out-dir /tmp/horizon_sweep
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import matplotlib as mpl

mpl.use("Agg")
import operator

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from torch import Tensor
from torch.nn import functional as F

from rmind.components.controller import (
    TrajectoryToActionMLP,
    build_features,
    fit_lateral_dynamics,
    fit_longitudinal_dynamics,
)
from rmind.components.dead_reckoning import (
    dead_reckon_future_trajectory,
    gnss_anchor_drift_m,
)
from rmind.components.loss import GaussianNLLLoss
from rmind.components.tick_trace import (
    MAX_TICK_GAP_MS,
    TICK_STRIDE_FRAME_IDX,
    build_tick_table,
    horizon_windows,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

HISTORY = 5
HORIZON = 6
FEAT_IDX = HISTORY - 1  # 4 -> last history tick ("now")
GT_IDX = HISTORY  # 5 -> first scored horizon tick

DEFAULT_PARQUET = Path(
    "outputs/2026-07-16/09-34-08/predictions/yaak/alex-tmp/model-4vqatiom:v4.parquet"
)
DEFAULT_TICKS = DEFAULT_PARQUET.with_suffix(".ticks.parquet")
SPEED_BANDS = ((0, 5), (5, 10), (10, 20), (20, 35), (35, 60), (60, 130))
VERIFY_ATOL = 1e-6  # `stitch --verify`: float32 round-trip, so exact agreement
VAL_CHUNK_ROWS = 200_000  # validation forward pass, to bound peak memory
CONTINUOUS_RAW_COLS = {
    "gas_pedal": "batch/data/meta/VehicleMotion/gas_pedal_normalized",
    "brake_pedal": "batch/data/meta/VehicleMotion/brake_pedal_normalized",
    "steering_angle": "batch/data/meta/VehicleMotion/steering_angle_normalized",
}
TURN_SIGNAL_RAW_COL = "batch/data/meta/VehicleState/turn_signal"

INPUT_ID_COL = "batch/meta/input_id"
FRAME_IDX_COL = "batch/data/meta/ImageMetadata.cam_front_left/frame_idx"
TIME_STAMP_COL = "batch/data/meta/ImageMetadata.cam_front_left/time_stamp"

# Everything needed to rebuild the dataset at an arbitrary horizon, and
# nothing else: the trajectory GT is dead-reckoned from speed + heading, the
# actions are the targets, and Gnss/xy is the QA anchor (`gnss_anchor_drift_m`)
# -- which matters more here than in the 6-step case, since dead-reckoning
# error compounds over the horizon. Images, waypoints and every `policy/*`
# column are unused. See `rmind.components.tick_trace`.
TICK_COLUMNS = {
    "time_stamp": TIME_STAMP_COL,
    "speed": "batch/data/meta/VehicleMotion/speed",
    "heading": "batch/data/headings_denoised/heading",
    "gnss_xy": "batch/data/meta/Gnss/xy",
    "gas_pedal": CONTINUOUS_RAW_COLS["gas_pedal"],
    "brake_pedal": CONTINUOUS_RAW_COLS["brake_pedal"],
    "steering_angle": CONTINUOUS_RAW_COLS["steering_angle"],
    "turn_signal": TURN_SIGNAL_RAW_COL,
}

COLUMNS = [
    "batch/meta/input_id",
    "batch/data/meta/ImageMetadata.cam_front_left/frame_idx",
    "batch/data/meta/ImageMetadata.cam_front_left/time_stamp",
    "batch/data/meta/VehicleMotion/speed",
    "batch/data/headings_denoised/heading",
    "batch/data/meta/Gnss/xy",
    TURN_SIGNAL_RAW_COL,
    *CONTINUOUS_RAW_COLS.values(),
    "policy/trajectory_value/value/xy",
    "policy/trajectory_value/value/yaw",
    "policy/trajectory_gt/value/xy",
    "policy/trajectory_gt/value/yaw",
    "policy/prediction_value/value/continuous/gas_pedal",
    "policy/prediction_value/value/continuous/brake_pedal",
    "policy/prediction_value/value/continuous/steering_angle",
    "policy/prediction_value/value/discrete/turn_signal",
]


# --------------------------------------------------------------------------- io


def _arr(df: pl.DataFrame, name: str) -> np.ndarray:
    """Column -> dense np array with the row axis first (list/array-typed cols)."""
    return np.asarray(df[name].to_list(), dtype=float)


def load(
    path: Path, *, limit: int | None = None, n_drives: int | None = None, seed: int = 0
) -> pl.DataFrame:
    """`limit` takes the first N rows in file order -- fine for a quick
    smoke test, but the file is NOT drive-shuffled (rows for a given vehicle
    are contiguous), so a modest `limit` covers only the first few vehicles
    alphabetically. Pass `n_drives` instead for anything that stratifies by
    vehicle (e.g. `modes`, `train --ablate-vehicle-conditioning`): it
    randomly samples whole drives across the full file first, so the
    resulting rows span many vehicles.
    """
    lf = pl.scan_parquet(path).select(COLUMNS)
    if n_drives is not None:
        chosen = _sample_drives(path, n_drives=n_drives, seed=seed)
        lf = lf.filter(pl.col(INPUT_ID_COL).is_in(chosen))
    elif limit is not None:
        lf = lf.limit(limit)
    return lf.collect()


def _sample_drives(path: Path, *, n_drives: int, seed: int) -> list[str]:
    """Reproducibly sample whole drives from `path`.

    sorted(): polars' `.unique()` row order is NOT stable across calls
    (observed directly -- two `unique()` reads of the same file gave only
    132/300 overlapping `rng.choice` picks at an identical seed), so without
    sorting, "seed" would silently fail to reproduce.
    """
    drives = sorted(
        pl
        .scan_parquet(path)
        .select(INPUT_ID_COL)
        .unique()
        .collect()[INPUT_ID_COL]
        .to_list()
    )
    rng = np.random.default_rng(seed)
    return rng.choice(drives, size=min(n_drives, len(drives)), replace=False).tolist()


# -------------------------------------------------------------- stitched ticks


def load_ticks(
    path: Path, *, n_drives: int | None = None, seed: int = 0
) -> pl.DataFrame:
    """Read `path` and undo the 11-tick window packaging.

    Consecutive predict windows advance by exactly one tick, so the parquet
    already tiles each drive at tick resolution; this returns that trace, one
    row per `(drive, tick)`. `n_drives` samples whole drives on the same
    protocol `load()` uses, so the two stay comparable at a shared seed.
    """
    lf = pl.scan_parquet(path).select([
        INPUT_ID_COL,
        FRAME_IDX_COL,
        *TICK_COLUMNS.values(),
    ])
    if n_drives is not None:
        lf = lf.filter(
            pl.col(INPUT_ID_COL).is_in(
                _sample_drives(path, n_drives=n_drives, seed=seed)
            )
        )
    table = build_tick_table(
        lf.collect(),
        input_id_column=INPUT_ID_COL,
        frame_idx_column=FRAME_IDX_COL,
        value_columns=TICK_COLUMNS.values(),
    )
    return table.rename(
        {v: k for k, v in TICK_COLUMNS.items()}
        | {FRAME_IDX_COL: "frame_idx", INPUT_ID_COL: "input_id"}
    )


def read_ticks(
    path: Path, *, n_drives: int | None = None, seed: int = 0
) -> pl.DataFrame:
    """Read a cached tick table (the `stitch --out` artifact).

    `load_ticks()` rebuilds this from the 516MB source parquet in ~3.5s; this
    reads the 40MB derived one instead, and is what everything horizon-related
    should start from. `n_drives` samples whole drives on the same
    sorted-then-`rng.choice` protocol `_sample_drives` uses (see the
    reproducibility note there), so a shared seed picks the same drives from
    either file.
    """
    ticks = pl.read_parquet(path)
    if n_drives is None:
        return ticks
    drives = sorted(ticks["input_id"].unique().to_list())
    rng = np.random.default_rng(seed)
    chosen = rng.choice(drives, size=min(n_drives, len(drives)), replace=False)
    return ticks.filter(pl.col("input_id").is_in(chosen.tolist()))


def drive_split(
    input_id: np.ndarray, *, seed: int, val_frac: float
) -> tuple[Tensor, Tensor, int, int]:
    """Hold out whole drives, never rows.

    Anchors overlap far more heavily than windowed rows did (consecutive
    anchors share `horizon` of their `horizon + 1` ticks), so a row-level
    split would put near-duplicates of every val anchor in train and report a
    meaningless val number. `np.unique` sorts, which is what makes `seed`
    reproducible here -- see `_sample_drives` for the polars `.unique()`
    version of the same trap.

    Returns:
        `(train_idx, val_idx, num_drives, num_val_drives)`.
    """
    drives = np.unique(input_id)
    rng = np.random.default_rng(seed)
    rng.shuffle(drives)
    n_val_drives = max(1, round(len(drives) * val_frac))
    is_val = np.isin(input_id, drives[:n_val_drives])
    return (
        torch.from_numpy(np.flatnonzero(~is_val)),
        torch.from_numpy(np.flatnonzero(is_val)),
        len(drives),
        n_val_drives,
    )


def _iter_anchor_index(
    ticks: pl.DataFrame, *, horizon: int
) -> Iterator[tuple[str, pl.DataFrame, np.ndarray]]:
    """Walk the drives in `ticks`, skipping any with no valid anchor.

    Yields:
        `(drive, that drive's tick rows, (num_anchors, horizon + 1) index)`.
    """
    for (drive,), sub in ticks.group_by(["input_id"], maintain_order=True):
        index = horizon_windows(
            frame_idx=sub["frame_idx"].to_numpy().astype(np.int64),
            time_stamp_us=sub["time_stamp"].to_numpy().astype(np.int64),
            horizon=horizon,
        )
        if index.size:
            yield str(drive), sub, index


def count_anchors(ticks: pl.DataFrame, *, horizon: int) -> int:
    """How many anchors `stitch_horizon` would yield, without building them.

    The trajectory arrays are `(n, horizon + 1)`, so materializing them just
    to take a length costs tens of GB at the horizons worth surveying.
    """
    return sum(
        index.shape[0] for _, _, index in _iter_anchor_index(ticks, horizon=horizon)
    )


def stitch_horizon(
    ticks: pl.DataFrame, *, horizon: int, action_steps: int = 1, with_gnss: bool = True
) -> dict[str, np.ndarray]:
    """Enumerate every anchor with a contiguous `horizon`-tick future.

    The anchor is "now": it plays the role `FEAT_IDX` plays inside a window,
    so `horizon=HORIZON, action_steps=1` reproduces the windowed formulation
    exactly (the trajectory covers anchor+1..anchor+horizon and the action
    target sits at anchor+1, matching `GT_IDX`).

    Args:
        ticks: a `load_ticks()` table.
        horizon: future ticks per anchor; 1 tick = 0.333s.
        action_steps: how many consecutive actions to take as targets,
            starting at anchor+1 -- the control horizon of
            `TrajectoryToActionMLP`. Must not exceed `horizon`: an action at
            anchor+k is only supervised by a trajectory that actually reaches
            tick k.
        with_gnss: carry `gnss_xy` through. It is QA-only
            (`gnss_anchor_drift_m`) and, at `(n, horizon + 1, 2)` float64,
            the single largest array here -- drop it for training runs.

    Returns:
        Arrays keyed by name. `speed`/`heading`/`time_stamp` are
        `(n, horizon + 1)` and `gnss_xy` (when requested) is
        `(n, horizon + 1, 2)`, all indexed from the anchor; the four action
        fields are `(n, action_steps)`, or `(n,)` when `action_steps == 1`.

    Raises:
        ValueError: if `action_steps` is not in `1..horizon`.
        SystemExit: if no anchor in `ticks` survives at this horizon.
    """
    if not 1 <= action_steps <= horizon:
        msg = f"action_steps must be in 1..horizon={horizon}, got {action_steps}"
        raise ValueError(msg)

    collected: dict[str, list[np.ndarray]] = {}
    drives: list[np.ndarray] = []
    for drive, sub, index in _iter_anchor_index(ticks, horizon=horizon):
        drives.append(np.full(index.shape[0], drive))
        collected.setdefault("frame_idx", []).append(
            sub["frame_idx"].to_numpy().astype(np.int64)[index[:, 0]]
        )
        for name in ("time_stamp", "speed", "heading"):
            collected.setdefault(name, []).append(sub[name].to_numpy()[index])
        if with_gnss:
            collected.setdefault("gnss_xy", []).append(
                np.asarray(sub["gnss_xy"].to_list(), dtype=float)[index]
            )
        # the actions realized over anchor+1..anchor+action_steps -- the
        # windowed GT_IDX and, beyond it, the rest of the control sequence.
        action_index = index[:, 1 : action_steps + 1]
        if action_steps == 1:
            action_index = action_index[:, 0]
        for name in (*CONTINUOUS_RAW_COLS, "turn_signal"):
            collected.setdefault(name, []).append(sub[name].to_numpy()[action_index])

    if not drives:
        msg = f"no anchor has a contiguous {horizon}-tick future"
        raise SystemExit(msg)
    out = {name: np.concatenate(parts) for name, parts in collected.items()}
    out["input_id"] = np.concatenate(drives)
    return out


def dead_reckon_stitched(anchors: dict[str, np.ndarray]) -> tuple[Tensor, Tensor]:
    """Dead-reckon a `stitch_horizon()` batch, anchored at its tick 0."""
    return _dead_reckon(
        speed_kmh=anchors["speed"],
        heading_deg=anchors["heading"],
        time_stamp_us=anchors["time_stamp"],
        reference_index=0,
    )


# ------------------------------------------------------------------ trajectory


def _dead_reckon(
    *,
    speed_kmh: np.ndarray,
    heading_deg: np.ndarray,
    time_stamp_us: np.ndarray,
    reference_index: int,
) -> tuple[Tensor, Tensor]:
    position, heading = dead_reckon_future_trajectory(
        speed_kmh=torch.from_numpy(np.asarray(speed_kmh, dtype=float)).float(),
        heading_deg=torch.from_numpy(np.asarray(heading_deg, dtype=float)).float(),
        # float64, NOT float32: these are Unix-epoch microseconds (~1.67e15)
        # -- float32's ~7 significant digits can't hold sub-second resolution
        # at that magnitude (its absolute precision there is >100s), which
        # silently collapses every `dt` to 0 and zeroes the whole
        # dead-reckoned position.
        time_stamp_s=torch.from_numpy(np.asarray(time_stamp_us, dtype=float)).double()
        / 1e6,
        reference_index=reference_index,
    )
    # internal math needs float64 (see time_stamp_s above); cast back down
    # for consistency with the rest of the pipeline (float32 throughout).
    return position.float(), heading.float()


def dead_reckon(df: pl.DataFrame) -> tuple[Tensor, Tensor]:
    """Dead-reckon the 6-step future trajectory (anchored at FEAT_IDX) for
    every windowed row -- see `rmind.components.dead_reckoning`."""
    return _dead_reckon(
        speed_kmh=_arr(df, "batch/data/meta/VehicleMotion/speed"),
        heading_deg=_arr(df, "batch/data/headings_denoised/heading"),
        time_stamp_us=_arr(df, TIME_STAMP_COL),
        reference_index=FEAT_IDX,
    )


def _integrate_relative(dxy: np.ndarray, dyaw: np.ndarray) -> np.ndarray:
    """Chain per-step relative (dx, dy, dyaw) into absolute ego-local
    positions, raw meters. Ported from (untracked)
    scripts/viz_policy_predictions.py::_integrate -- the same axis-labeling
    caveat applies (component 0 is lateral, component 1 is forward, per
    `rmind.components.dead_reckoning`'s convention note); for the small
    per-step yaw values seen in practice the rotation composition is
    insensitive to that ordering, so no swap is applied here, matching how
    the existing viz script already uses this function.
    """
    th = np.concatenate(
        [np.zeros_like(dyaw[:, :1]), np.cumsum(dyaw, axis=1)[:, :-1]], axis=1
    )
    c, s = np.cos(th), np.sin(th)
    gx = dxy[..., 0] * c - dxy[..., 1] * s
    gy = dxy[..., 0] * s + dxy[..., 1] * c
    return np.stack([np.cumsum(gx, axis=1), np.cumsum(gy, axis=1)], axis=-1)


def predicted_trajectory_as_dead_reckoned(df: pl.DataFrame) -> tuple[Tensor, Tensor]:
    """Convert `policy/trajectory_value` (per-step relative deltas, raw
    meters) into the same representation `dead_reckon_future_trajectory`
    produces (absolute position anchored at FEAT_IDX, /100-scaled; heading
    relative to the FEAT_IDX anchor, radians) so it can be fed to the same
    controller.
    """
    xy = _arr(df, "policy/trajectory_value/value/xy")
    yaw = _arr(df, "policy/trajectory_value/value/yaw")
    position = _integrate_relative(xy, yaw) / 100.0
    heading = np.cumsum(yaw, axis=1)
    return torch.from_numpy(position).float(), torch.from_numpy(heading).float()


# ------------------------------------------------------------------- reporting


def _print_speed_bands(speed_kmh: np.ndarray, values: np.ndarray) -> None:
    for lo, hi in SPEED_BANDS:
        mask = (speed_kmh >= lo) & (speed_kmh < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        v = values[mask]
        print(  # noqa: T201
            f"    [{lo:3d},{hi:3d}) n={n:8d}  mean={v.mean():8.4f}  "
            f"median={np.median(v):8.4f}  p90={np.percentile(v, 90):8.4f}"
        )


def _report(
    out: dict[str, Any], *, gt: dict[str, Tensor], turn_gt: Tensor, speed: np.ndarray
) -> None:
    for field, target in gt.items():
        mean = out["continuous"][field][..., 0]
        l1 = (mean - target).abs().numpy()
        print(f"  {field} L1:")  # noqa: T201
        _print_speed_bands(speed, l1)
    pred_turn = out["turn_signal"].argmax(-1)
    correct = (pred_turn == turn_gt).float().numpy()
    print("  turn_signal accuracy:")  # noqa: T201
    _print_speed_bands(speed, correct)


def _report_direct(
    pred: dict[str, Tensor],
    *,
    gt: dict[str, Tensor],
    turn_pred: Tensor,
    turn_gt: Tensor,
    speed: np.ndarray,
) -> None:
    for field, target in gt.items():
        l1 = (pred[field] - target).abs().numpy()
        print(f"  {field} L1:")  # noqa: T201
        _print_speed_bands(speed, l1)
    correct = (turn_pred == turn_gt).float().numpy()
    print("  turn_signal accuracy:")  # noqa: T201
    _print_speed_bands(speed, correct)


def _load_ground_truth(df: pl.DataFrame) -> tuple[dict[str, Tensor], Tensor]:
    gt = {
        field: torch.from_numpy(_arr(df, col)[:, GT_IDX]).float()
        for field, col in CONTINUOUS_RAW_COLS.items()
    }
    turn_gt = torch.from_numpy(_arr(df, TURN_SIGNAL_RAW_COL)[:, GT_IDX]).long()
    return gt, turn_gt


# ---------------------------------------------------- hidden-mode diagnostics


def _vehicle_id(df: pl.DataFrame) -> np.ndarray:
    """`batch/meta/input_id` is `<vehicle>/<drive-timestamp>`; the vehicle
    prefix names a real physical car (27 distinct "Niro*-HQ" units in this
    parquet). It's the natural covariate to test for hidden per-vehicle
    modes (regen calibration, sensor drift, wear) that would make the same
    trajectory realizable by different actions, or the same action realize
    different trajectories -- exactly the non-identifiability a
    trajectory<->action controller silently assumes away. The dataset
    pipeline itself already documents one instance of this
    (`config/_templates/dataset/yaak/train.yaml`: 7 named vehicles have "a
    constant brake sensor offset ... sensor noise, not real braking"), so
    per-vehicle idiosyncrasy here is not speculative.
    """
    return np.array([i.split("/")[0] for i in df["batch/meta/input_id"].to_list()])


def _one_way_anova(groups: list[np.ndarray]) -> tuple[float, float]:
    """One-way ANOVA F-statistic and effect size (eta^2) for whether group
    means differ more than within-group noise predicts. No p-value (scipy
    isn't a dependency here) -- eta^2 (fraction of total variance explained
    by group membership) is the practical read: near 0 means groups agree up
    to sampling noise, close to 1 means group identity dominates.
    """
    all_values = np.concatenate(groups)
    grand_mean = all_values.mean()
    n_total = len(all_values)
    k = len(groups)

    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_within = sum(((g - g.mean()) ** 2).sum() for g in groups)

    df_between, df_within = k - 1, n_total - k
    f_stat = (ss_between / df_between) / (ss_within / df_within)
    eta_squared = ss_between / (ss_between + ss_within)
    return f_stat, eta_squared


def _report_matched_condition(
    label: str,
    *,
    vehicle: np.ndarray,
    value: np.ndarray,
    value_name: str,
    min_count: int,
) -> None:
    print(f"\n[modes] {label}")  # noqa: T201
    print(f"  {'vehicle':<14}{'n':>8}{'mean ' + value_name:>16}{'std':>10}")  # noqa: T201
    groups = []
    for v in np.unique(vehicle):
        g = value[vehicle == v]
        if len(g) < min_count:
            continue
        groups.append(g)
        print(f"  {v:<14}{len(g):>8}{g.mean():>16.4f}{g.std():>10.4f}")  # noqa: T201
    if len(groups) < 2:  # noqa: PLR2004
        print("  (not enough vehicles with sufficient samples)")  # noqa: T201
        return
    f_stat, eta_squared = _one_way_anova(groups)
    within_vehicle_std = float(np.mean([g.std() for g in groups]))
    print(  # noqa: T201
        f"  -> {len(groups)} vehicles, one-way ANOVA F={f_stat:.1f}, "
        f"eta^2={eta_squared:.3f} (fraction of variance in {value_name} explained "
        f"by vehicle identity alone); typical within-vehicle std={within_vehicle_std:.4f}"
    )


# ---------------------------------------------------------------- subcommands


def cmd_episode(args: argparse.Namespace) -> None:
    df = load(args.parquet)
    sub = df.filter(pl.col("batch/meta/input_id") == args.drive)
    if sub.height == 0:
        msg = f"no rows for drive {args.drive!r}"
        raise SystemExit(msg)

    frame0 = _arr(sub, "batch/data/meta/ImageMetadata.cam_front_left/frame_idx")[:, 0]
    order = np.argsort(frame0)
    sub = sub[order.tolist()]
    frame0 = frame0[order]

    # rows tile the drive at 1-tick resolution when frame_idx advances by
    # exactly 10 between consecutive rows (empirically ~99.4%); larger jumps
    # are clip/pause boundaries -- break the plotted line rather than stitch.
    gap = np.concatenate([[False], np.diff(frame0) != TICK_STRIDE_FRAME_IDX])
    n_breaks = int(gap.sum())

    series = {
        "speed (km/h)": _arr(sub, "batch/data/meta/VehicleMotion/speed")[:, FEAT_IDX],
        "gas": _arr(sub, CONTINUOUS_RAW_COLS["gas_pedal"])[:, FEAT_IDX],
        "brake": _arr(sub, CONTINUOUS_RAW_COLS["brake_pedal"])[:, FEAT_IDX],
        "steering": _arr(sub, CONTINUOUS_RAW_COLS["steering_angle"])[:, FEAT_IDX],
        "turn_signal": _arr(sub, TURN_SIGNAL_RAW_COL)[:, FEAT_IDX],
    }
    for values in series.values():
        values[gap] = np.nan

    fig, axes = plt.subplots(len(series), 1, figsize=(12, 2 * len(series)), sharex=True)
    for ax, (name, values) in zip(axes, series.items(), strict=True):
        ax.plot(frame0, values, lw=1)
        ax.set_ylabel(name, fontsize=9)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("frame_idx")
    fig.suptitle(f"{args.drive} ({sub.height} ticks, {n_breaks} clip/pause breaks)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"[episode] {sub.height} ticks, {n_breaks} clip/pause breaks -> {args.out}")  # noqa: T201


def _verify_against_windows(args: argparse.Namespace, ticks: pl.DataFrame) -> None:
    """Assert the stitched path reproduces the windowed one at HORIZON.

    The stitched anchors are a superset of the windowed rows' FEAT_IDX ticks,
    so on the shared anchors the two dead-reckoned trajectories must agree
    exactly -- if they don't, the stitch has broken an index convention.

    Raises:
        SystemExit: if the two paths disagree by more than `VERIFY_ATOL`.
    """
    windowed = load(args.parquet, n_drives=args.n_drives, seed=args.seed)
    position, heading = dead_reckon(windowed)
    key = (
        windowed[INPUT_ID_COL].to_numpy().astype(str),
        _arr(windowed, FRAME_IDX_COL)[:, FEAT_IDX].astype(np.int64),
    )
    anchors = stitch_horizon(ticks, horizon=HORIZON)
    stitched_position, stitched_heading = dead_reckon_stitched(anchors)

    lookup = {
        (drive, frame): row
        for row, (drive, frame) in enumerate(
            zip(anchors["input_id"], anchors["frame_idx"], strict=True)
        )
    }
    rows = np.array([lookup.get((d, f), -1) for d, f in zip(*key, strict=True)])
    shared = rows >= 0
    print(  # noqa: T201
        f"[stitch] verify: {int(shared.sum()):,}/{windowed.height:,} windowed rows "
        f"matched to a stitched anchor ({len(lookup):,} anchors total)"
    )
    for name, got, want in (
        ("position", stitched_position[rows[shared]], position[shared]),
        ("heading", stitched_heading[rows[shared]], heading[shared]),
    ):
        max_abs = (got - want).abs().max().item()
        print(f"[stitch] verify: max |stitched - windowed| {name} = {max_abs:.3e}")  # noqa: T201
        if max_abs > VERIFY_ATOL:
            msg = f"stitched {name} disagrees with the windowed path ({max_abs:.3e})"
            raise SystemExit(msg)


def cmd_stitch(args: argparse.Namespace) -> None:
    t0 = time.time()
    ticks = load_ticks(args.parquet, n_drives=args.n_drives, seed=args.seed)
    print(  # noqa: T201
        f"[stitch] {ticks.height:,} ticks over "
        f"{ticks['input_id'].n_unique():,} drives in {time.time() - t0:.1f}s"
    )

    print(  # noqa: T201
        f"\n[stitch] anchors by horizon:\n"
        f"{'ticks':>7} {'sec':>6} {'anchors':>12} {'%':>7}"
    )
    for horizon in args.horizons:
        n = count_anchors(ticks, horizon=horizon)
        print(  # noqa: T201
            f"{horizon:>7} {horizon / 3:>6.1f} {n:>12,} {100 * n / ticks.height:>6.1f}%"
        )

    anchors = stitch_horizon(ticks, horizon=args.horizon)
    position, _ = dead_reckon_stitched(anchors)
    drift = gnss_anchor_drift_m(
        dead_reckoned_position_normalized=position,
        gnss_xy=torch.from_numpy(anchors["gnss_xy"]).float(),
        heading_deg=torch.from_numpy(anchors["heading"]).float(),
        reference_index=0,
    )
    print(  # noqa: T201
        f"\n[stitch] horizon={args.horizon} ({args.horizon / 3:.1f}s), "
        f"n={anchors['frame_idx'].size:,} -- dead-reckoned-vs-raw-GNSS drift (m) "
        f"at the final pose, by speed at the anchor:"
    )
    _print_speed_bands(anchors["speed"][:, 0], drift.numpy())

    if args.verify:
        print()  # noqa: T201
        _verify_against_windows(args, ticks)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        ticks.write_parquet(args.out)
        print(f"\n[stitch] cached tick table -> {args.out}")  # noqa: T201


def cmd_reckon(args: argparse.Namespace) -> None:
    df = load(args.parquet, limit=args.n_sample)
    position, heading = dead_reckon(df)
    speed = _arr(df, "batch/data/meta/VehicleMotion/speed")[:, FEAT_IDX]

    gnss_xy = torch.from_numpy(_arr(df, "batch/data/meta/Gnss/xy")).float()
    heading_deg = torch.from_numpy(
        _arr(df, "batch/data/headings_denoised/heading")
    ).float()
    drift = gnss_anchor_drift_m(
        dead_reckoned_position_normalized=position,
        gnss_xy=gnss_xy,
        heading_deg=heading_deg,
        reference_index=FEAT_IDX,
    )
    print(f"[reckon] n={df.height} dead-reckoned-vs-raw-GNSS anchor drift (m):")  # noqa: T201
    _print_speed_bands(speed, drift.numpy())

    traj_gt_xy = _arr(df, "policy/trajectory_gt/value/xy")
    traj_gt_yaw = _arr(df, "policy/trajectory_gt/value/yaw")
    gt_abs = _integrate_relative(traj_gt_xy, traj_gt_yaw) / 100.0
    disagreement = (
        np.linalg.norm(gt_abs[:, -1] - position[:, -1].numpy(), axis=-1) * 100.0
    )
    print(  # noqa: T201
        "\n[reckon] dead-reckoned vs existing (documented-flawed, GPS-chain) "
        "policy/trajectory_gt, final-pose disagreement (m):"
    )
    _print_speed_bands(speed, disagreement)

    if args.out is not None:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"position": position, "heading": heading, "feat_idx": FEAT_IDX}, out
        )
        print(f"[reckon] cached dead-reckoned trajectory -> {out}")  # noqa: T201


def _cross_entropy(logits: Tensor, target: Tensor) -> Tensor:
    """`F.cross_entropy` over a control sequence.

    At `action_steps == 1` the shapes are `(n, C)` / `(n,)` and this is plain
    cross-entropy; beyond it they are `(n, K, C)` / `(n, K)`, and torch wants
    the class axis second, so the step axis is moved to the end.
    """
    if logits.ndim == target.ndim + 1 and logits.ndim > 2:  # noqa: PLR2004
        logits = logits.transpose(-1, -2)
    return F.cross_entropy(logits, target)


class FitResult(NamedTuple):
    """`_fit` outcome. Per-step entries are length-`action_steps` lists over
    the control horizon; the scalars average over it (identical to the
    per-step value at `action_steps == 1`)."""

    model: TrajectoryToActionMLP
    val_l1: dict[str, float]
    val_turn_acc: float
    loss_history: list[dict[str, float]]
    val_l1_per_step: dict[str, list[float]]
    val_turn_acc_per_step: list[float]
    val_l1_rows: dict[str, np.ndarray]
    """Per-row |error| on the FIRST action of the sequence -- the one a
    receding-horizon controller would actually apply, and the only step every
    configuration has in common. Kept for stratified reporting."""


def _fit(  # noqa: PLR0913, PLR0914
    *,
    features: Tensor,
    targets: dict[str, Tensor],
    turn_gt: Tensor,
    train_idx: Tensor,
    val_idx: Tensor,
    hidden_size: int,
    steps: int,
    batch_size: int,
    lr: float,
    log_every: int,
    seed: int,
    label: str,
    action_steps: int = 1,
    device: str = "cpu",
) -> FitResult:
    """One train+val fit; factored out so `cmd_train --ablate-vehicle-conditioning`
    can run it twice (with/without a vehicle one-hot appended to `features`) on
    the identical split and compare, and so `cmd_horizon` can run it once per
    (trajectory horizon, control horizon) configuration on identical rows."""
    # seeds the INIT, not just the sampler below: without this the weights
    # come from the process-global RNG, so a fit's result depends on how many
    # fits preceded it in the same process and no single run is reproducible.
    # Deltas within a sweep were the casualty -- see bug #5.
    torch.manual_seed(seed)
    model = TrajectoryToActionMLP(
        in_features=features.shape[-1],
        hidden_size=hidden_size,
        action_steps=action_steps,
    ).to(device)
    features = features.to(device)
    targets = {field: target.to(device) for field, target in targets.items()}
    turn_gt = turn_gt.to(device)
    train_idx, val_idx = train_idx.to(device), val_idx.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    gaussian_nll = GaussianNLLLoss(reduction="mean")
    generator = torch.Generator(device=device).manual_seed(seed)

    loss_history: list[dict[str, float]] = []
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        idx = train_idx[
            torch.randint(
                0, len(train_idx), (batch_size,), generator=generator, device=device
            )
        ]
        out = model(features[idx])
        loss = torch.stack([
            gaussian_nll(out["continuous"][field], target[idx])
            for field, target in targets.items()
        ]).sum() + _cross_entropy(out["turn_signal"], turn_gt[idx])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step == 1 or step % log_every == 0 or step == steps:
            rate = step / (time.perf_counter() - t0)
            loss_value = float(loss.detach())
            loss_history.append({"step": step, "loss": loss_value})
            print(  # noqa: T201
                f"[train:{label}] step {step}/{steps} loss={loss_value:.5f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e} ({rate:.2f} steps/s)"
            )

    model.eval()
    # chunked: the val split can be hundreds of thousands of rows wide, and a
    # single forward over all of them at once is the peak allocation of the run.
    l1_rows: dict[str, list[Tensor]] = {field: [] for field in targets}
    turn_correct: list[Tensor] = []
    with torch.no_grad():
        for chunk in val_idx.split(VAL_CHUNK_ROWS):
            out = model(features[chunk])
            for field, target in targets.items():
                l1_rows[field].append(
                    (out["continuous"][field][..., 0] - target[chunk]).abs().cpu()
                )
            turn_correct.append(
                (out["turn_signal"].argmax(-1) == turn_gt[chunk]).float().cpu()
            )
    errors = {field: torch.cat(parts) for field, parts in l1_rows.items()}
    correct = torch.cat(turn_correct)

    val_l1_per_step = {
        field: e.reshape(e.shape[0], -1).mean(0).tolist() for field, e in errors.items()
    }
    val_l1 = {field: float(e.mean()) for field, e in errors.items()}
    val_turn_acc_per_step = correct.reshape(correct.shape[0], -1).mean(0).tolist()
    val_turn_acc = float(correct.mean())
    val_l1_rows = {
        field: (e if action_steps == 1 else e[:, 0]).numpy()
        for field, e in errors.items()
    }
    print(f"[train:{label}] val L1: {val_l1}  turn_signal acc: {val_turn_acc:.4f}")  # noqa: T201
    return FitResult(
        model=model.cpu(),
        val_l1=val_l1,
        val_turn_acc=val_turn_acc,
        loss_history=loss_history,
        val_l1_per_step=val_l1_per_step,
        val_turn_acc_per_step=val_turn_acc_per_step,
        val_l1_rows=val_l1_rows,
    )


def cmd_train(args: argparse.Namespace) -> None:  # noqa: PLR0914
    df = load(args.parquet, limit=args.limit, n_drives=args.n_drives, seed=args.seed)
    position, heading = dead_reckon(df)
    speed_now = torch.from_numpy(
        _arr(df, "batch/data/meta/VehicleMotion/speed")[:, FEAT_IDX]
    ).float()
    features = build_features(position=position, heading=heading, speed=speed_now)

    targets, turn_gt = _load_ground_truth(df)

    input_id = df["batch/meta/input_id"].to_numpy()
    drives = np.unique(input_id)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(drives)
    n_val_drives = max(1, round(len(drives) * args.val_frac))
    val_drives = set(drives[:n_val_drives].tolist())
    is_val = np.isin(input_id, list(val_drives))
    train_idx = torch.from_numpy(np.flatnonzero(~is_val))
    val_idx = torch.from_numpy(np.flatnonzero(is_val))
    print(  # noqa: T201
        f"[train] {df.height} rows, {len(drives)} drives -> "
        f"{len(drives) - n_val_drives} train / {n_val_drives} val drives "
        f"({len(train_idx)} train rows / {len(val_idx)} val rows)"
    )

    fit_kwargs: dict[str, Any] = {
        "targets": targets,
        "turn_gt": turn_gt,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "hidden_size": args.hidden_size,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "log_every": args.log_every,
        "seed": args.seed,
    }

    if args.ablate_vehicle_conditioning:
        # Does conditioning on vehicle identity reduce error? Direct test of
        # whether hidden per-vehicle modes (see `modes` subcommand) actually
        # hurt THIS controller, not just whether they exist physically.
        vehicle = _vehicle_id(df)
        vehicles_sorted = sorted(np.unique(vehicle).tolist())
        vehicle_idx = {v: i for i, v in enumerate(vehicles_sorted)}
        vehicle_in_both = len(
            set(vehicle[train_idx.numpy()]) & set(vehicle[val_idx.numpy()])
        )
        print(  # noqa: T201
            f"[train] vehicle-conditioning ablation: {len(vehicles_sorted)} vehicles, "
            f"{vehicle_in_both} appear in both train and val splits"
        )
        onehot = torch.zeros(len(vehicle), len(vehicles_sorted))
        onehot[torch.arange(len(vehicle)), [vehicle_idx[v] for v in vehicle]] = 1.0
        conditioned_features = torch.cat([features, onehot], dim=-1)

        # `model` stays the BASELINE fit (unconditioned in_features) so `--out`
        # remains loadable by `eval` unchanged; the conditioned model here is
        # diagnostic-only, not persisted.
        baseline = _fit(features=features, label="baseline", **fit_kwargs)
        model, val_l1, val_turn_acc, loss_history = (
            baseline.model,
            baseline.val_l1,
            baseline.val_turn_acc,
            baseline.loss_history,
        )
        conditioned = _fit(
            features=conditioned_features, label="+vehicle", **fit_kwargs
        )
        cond_l1, cond_acc = conditioned.val_l1, conditioned.val_turn_acc
        print("\n[train] vehicle-conditioning ablation (val L1, lower is better):")  # noqa: T201
        print(f"  {'field':<16}{'baseline':>12}{'+vehicle':>12}{'delta %':>10}")  # noqa: T201
        for field in targets:
            delta_pct = 100.0 * (cond_l1[field] - val_l1[field]) / val_l1[field]
            print(  # noqa: T201
                f"  {field:<16}{val_l1[field]:>12.4f}{cond_l1[field]:>12.4f}{delta_pct:>9.1f}%"
            )
        print(  # noqa: T201
            f"  {'turn_signal acc':<16}{val_turn_acc:>12.4f}{cond_acc:>12.4f}"
            f"{100.0 * (cond_acc - val_turn_acc) / val_turn_acc:>9.1f}%"
        )
    else:
        base = _fit(features=features, label="base", **fit_kwargs)
        model, val_l1, val_turn_acc, loss_history = (
            base.model,
            base.val_l1,
            base.val_turn_acc,
            base.loss_history,
        )

    config = {
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "hidden_size": args.hidden_size,
        "in_features": features.shape[-1],
        "seed": args.seed,
        "val_frac": args.val_frac,
        "parquet": str(args.parquet),
        "n_rows": df.height,
        "n_drives": len(drives),
        "val_l1": val_l1,
        "val_turn_signal_acc": val_turn_acc,
        "loss_history": loss_history,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    state_dict = {
        key: value.detach().cpu() for key, value in model.state_dict().items()
    }
    torch.save({"state_dict": state_dict, "config": config}, out)
    out.with_suffix(".json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[train] saved -> {out} (config: {out.with_suffix('.json')})")  # noqa: T201


def _band_l1(speed_kmh: np.ndarray, values: np.ndarray) -> dict[str, float]:
    """Mean `values` per `SPEED_BANDS` entry, keyed `"lo-hi"`; bands with no
    rows are omitted."""
    out: dict[str, float] = {}
    for lo, hi in SPEED_BANDS:
        mask = (speed_kmh >= lo) & (speed_kmh < hi)
        if mask.any():
            out[f"{lo}-{hi}"] = float(values[mask].mean())
    return out


def _horizon_features(
    *, position: Tensor, heading: Tensor, speed_now: Tensor, horizon: int
) -> Tensor:
    """Controller input for a `horizon`-tick trajectory, sliced out of a
    longer dead-reckoned one.

    This is exact, not an approximation: `dead_reckon_future_trajectory`
    anchors at tick 0 and integrates forward with `cumsum`, so the first
    `horizon` poses of a longer roll-out ARE the shorter roll-out, bit for
    bit (`test_dead_reckon_prefix_matches_shorter_horizon`). That is what
    lets a horizon sweep train every configuration on one identical anchor
    set -- rebuilding the anchors per horizon would instead hand the longer
    horizons a strictly easier, more contiguous subset of the drive and
    confound the comparison with survivorship.
    """
    return build_features(
        position=position[:, :horizon], heading=heading[:, :horizon], speed=speed_now
    )


def cmd_horizon(args: argparse.Namespace) -> None:  # noqa: PLR0914
    """Sweep the trajectory (prediction) horizon and the action (control)
    horizon of the controller on one shared anchor set and split."""
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    max_horizon, max_action_steps = max(args.horizons), max(args.action_steps)

    t0 = time.time()
    ticks = read_ticks(args.ticks, n_drives=args.n_drives, seed=args.seed)
    print(  # noqa: T201
        f"[horizon] {ticks.height:,} ticks over {ticks['input_id'].n_unique():,} "
        f"drives in {time.time() - t0:.1f}s (device={device})"
    )

    anchors = stitch_horizon(
        ticks,
        horizon=max_horizon,
        action_steps=max_action_steps,
        with_gnss=False,  # QA-only, and the largest array here
    )
    del ticks
    position, heading = dead_reckon_stitched(anchors)
    speed_now = torch.from_numpy(anchors["speed"][:, 0]).float()
    all_targets = {
        field: torch.from_numpy(anchors[field]).float() for field in CONTINUOUS_RAW_COLS
    }
    all_turn_gt = torch.from_numpy(anchors["turn_signal"]).long()

    train_idx, val_idx, n_drives, n_val_drives = drive_split(
        anchors["input_id"], seed=args.seed, val_frac=args.val_frac
    )
    val_speed = anchors["speed"][val_idx.numpy(), 0]
    print(  # noqa: T201
        f"[horizon] {len(anchors['frame_idx']):,} anchors at horizon="
        f"{max_horizon} ({max_horizon / 3:.1f}s), {n_drives} drives -> "
        f"{n_drives - n_val_drives} train / {n_val_drives} val "
        f"({len(train_idx):,} / {len(val_idx):,} rows). Every configuration "
        f"below trains and validates on exactly these rows."
    )
    near_stop = int(((val_speed >= 0) & (val_speed < SPEED_BANDS[0][1])).sum())
    print(  # noqa: T201
        f"[horizon] val near-stop band [0,{SPEED_BANDS[0][1]}) km/h: "
        f"{near_stop:,} rows ({100 * near_stop / len(val_speed):.1f}%)"
    )

    results: list[dict[str, Any]] = []
    for action_steps in sorted(args.action_steps):
        targets = {
            field: target if max_action_steps == 1 else target[:, :action_steps]
            for field, target in all_targets.items()
        }
        turn_gt = (
            all_turn_gt if max_action_steps == 1 else all_turn_gt[:, :action_steps]
        )
        if action_steps == 1 and max_action_steps > 1:
            targets = {field: target[:, 0] for field, target in targets.items()}
            turn_gt = turn_gt[:, 0]

        for horizon in sorted(args.horizons):
            label = f"h{horizon}k{action_steps}"
            fit = _fit(
                features=_horizon_features(
                    position=position,
                    heading=heading,
                    speed_now=speed_now,
                    horizon=horizon,
                ),
                targets=targets,
                turn_gt=turn_gt,
                train_idx=train_idx,
                val_idx=val_idx,
                hidden_size=args.hidden_size,
                steps=args.steps,
                batch_size=args.batch_size,
                lr=args.lr,
                log_every=args.log_every,
                seed=args.seed,
                label=label,
                action_steps=action_steps,
                device=device,
            )
            results.append({
                "horizon": horizon,
                "horizon_s": horizon / 3,
                "action_steps": action_steps,
                "in_features": horizon * 3 + 1,
                "val_l1": fit.val_l1,
                "val_turn_signal_acc": fit.val_turn_acc,
                "val_l1_per_step": fit.val_l1_per_step,
                "val_turn_signal_acc_per_step": fit.val_turn_acc_per_step,
                "val_l1_by_speed_band": {
                    field: _band_l1(val_speed, rows)
                    for field, rows in fit.val_l1_rows.items()
                },
                "loss_history": fit.loss_history,
            })
            if args.out_dir is not None:
                args.out_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "state_dict": {
                            key: value.detach().cpu()
                            for key, value in fit.model.state_dict().items()
                        },
                        "config": results[-1]
                        | {
                            "hidden_size": args.hidden_size,
                            "seed": args.seed,
                            "ticks": str(args.ticks),
                        },
                    },
                    args.out_dir / f"controller_{label}.pt",
                )

    _report_horizon_sweep(results)

    if args.out_dir is not None:
        summary = args.out_dir / "horizon_sweep.json"
        summary.write_text(
            json.dumps(
                {
                    "ticks": str(args.ticks),
                    "n_drives": n_drives,
                    "n_val_drives": n_val_drives,
                    "n_train_rows": len(train_idx),
                    "n_val_rows": len(val_idx),
                    "max_horizon": max_horizon,
                    "steps": args.steps,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "hidden_size": args.hidden_size,
                    "seed": args.seed,
                    "results": results,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\n[horizon] wrote {summary} (+ per-config checkpoints)")  # noqa: T201


DIVERGED_L1 = 1.0
"""val L1 above which a head is reported as diverged rather than as a result.

Every action field is normalized to `[0, 1]` (pedals) or `[-1, 1]`
(steering), so predicting the target's own mean caps mean |error| well below
1.0; only a blown-up head exceeds it. Undertrained fits do occasionally get
there -- `GaussianNLLLoss` scales the mean's gradient by `1/var`, so a
collapsing logvar can run away -- and a diverged cell must never be read as
"this configuration is worse".
"""


def _report_horizon_sweep(results: list[dict[str, Any]]) -> None:
    """Print the sweep as one table per continuous field, aggregate and
    near-stop side by side, with each row's delta against the shortest
    trajectory horizon at the same control horizon."""
    near_stop_band = f"{SPEED_BANDS[0][0]}-{SPEED_BANDS[0][1]}"
    diverged = [
        (r["horizon"], r["action_steps"], field, value)
        for r in results
        for field, value in r["val_l1"].items()
        if value > DIVERGED_L1
    ]
    if diverged:
        print(  # noqa: T201
            f"\n[horizon] WARNING: {len(diverged)} head(s) diverged (val L1 > "
            f"{DIVERGED_L1}, impossible for a normalized action) -- treat these "
            f"cells as failed fits, not as results:"
        )
        for horizon, action_steps, field, value in diverged:
            print(f"    traj={horizon} ctrl={action_steps} {field}: {value:.3f}")  # noqa: T201
    for field in CONTINUOUS_RAW_COLS:
        print(f"\n[horizon] {field} -- val L1 on the applied action (step 0)")  # noqa: T201
        print(  # noqa: T201
            f"  {'traj':>6}{'sec':>6}{'ctrl':>6}{'all bands':>12}{'vs h0':>8}"
            f"{'  [0,5) km/h':>14}{'vs h0':>8}{'ratio':>8}"
        )
        for action_steps in sorted({r["action_steps"] for r in results}):
            rows = sorted(
                (r for r in results if r["action_steps"] == action_steps),
                key=operator.itemgetter("horizon"),
            )
            base_all = rows[0]["val_l1"][field]
            base_near = rows[0]["val_l1_by_speed_band"][field][near_stop_band]
            for r in rows:
                aggregate = r["val_l1"][field]
                near = r["val_l1_by_speed_band"][field][near_stop_band]
                print(  # noqa: T201
                    f"  {r['horizon']:>6}{r['horizon_s']:>6.1f}{action_steps:>6}"
                    f"{aggregate:>12.4f}{100 * (aggregate / base_all - 1):>7.1f}%"
                    f"{near:>14.4f}{100 * (near / base_near - 1):>7.1f}%"
                    f"{near / aggregate:>7.1f}x"
                )

    print("\n[horizon] turn_signal accuracy on the applied action (step 0)")  # noqa: T201
    print(f"  {'traj':>6}{'sec':>6}{'ctrl':>6}{'accuracy':>12}")  # noqa: T201
    for r in sorted(results, key=operator.itemgetter("action_steps", "horizon")):
        print(  # noqa: T201
            f"  {r['horizon']:>6}{r['horizon_s']:>6.1f}{r['action_steps']:>6}"
            f"{r['val_turn_signal_acc_per_step'][0]:>12.4f}"
        )

    multi_step = [r for r in results if r["action_steps"] > 1]
    if multi_step:
        print(  # noqa: T201
            "\n[horizon] control-sequence decay -- val L1 per step ahead "
            "(step 0 is the action a receding-horizon controller applies)"
        )
        for r in sorted(multi_step, key=operator.itemgetter("action_steps", "horizon")):
            print(f"  traj={r['horizon']} ctrl={r['action_steps']}:")  # noqa: T201
            for field in CONTINUOUS_RAW_COLS:
                per_step = r["val_l1_per_step"][field]
                shown = " ".join(f"{v:.4f}" for v in per_step[: min(8, len(per_step))])
                tail = " ..." if len(per_step) > 8 else ""  # noqa: PLR2004
                print(f"    {field:<16}{shown}{tail}")  # noqa: T201


def _path_length_m(speed_kmh: np.ndarray, time_stamp_us: np.ndarray) -> np.ndarray:
    """Distance travelled over each anchor's whole horizon, meters."""
    dt_s = np.diff(time_stamp_us.astype(float), axis=1) / 1e6
    return (speed_kmh[:, :-1] / 3.6 * dt_s).sum(axis=1)


def _conditional_l1_floor(
    *, values: np.ndarray, speed_kmh: np.ndarray, speed_bin: float
) -> tuple[float, int]:
    """Best mean L1 any predictor conditioning only on `speed_kmh` can reach.

    The L1-optimal constant per group is that group's median, so binning by
    speed and averaging |value - median| over bins is exactly the Bayes floor
    for a predictor that sees nothing else. Applied to rows whose future
    trajectory is identically the zero trajectory, it is therefore the floor
    for a *trajectory*-conditioned controller too, at any horizon: those rows
    are indistinguishable to it apart from their current speed.

    Returns:
        `(floor, num_bins)`.
    """
    bins = np.floor(speed_kmh / speed_bin).astype(np.int64)
    total, n_bins = 0.0, 0
    for b in np.unique(bins):
        group = values[bins == b]
        total += float(np.abs(group - np.median(group)).sum())
        n_bins += 1
    return total / len(values), n_bins


def _score_checkpoint(
    path: Path, *, position: Tensor, heading: Tensor, speed_now: Tensor, rows: Tensor
) -> tuple[int, dict[str, Tensor]]:
    """Per-row |error| on `rows` for a `horizon`-subcommand checkpoint.

    Returns:
        `(the checkpoint's trajectory horizon, per-field errors)`.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config = payload["config"]
    model = TrajectoryToActionMLP(
        in_features=config["in_features"],
        hidden_size=config["hidden_size"],
        action_steps=config["action_steps"],
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()

    horizon = config["horizon"]
    with torch.no_grad():
        out = model(
            _horizon_features(
                position=position[rows],
                heading=heading[rows],
                speed_now=speed_now[rows],
                horizon=horizon,
            )
        )
    first = {
        field: value[..., 0] if config["action_steps"] == 1 else value[:, 0, 0]
        for field, value in out["continuous"].items()
    }
    return horizon, first


def cmd_nearstop(args: argparse.Namespace) -> None:  # noqa: PLR0914
    """Why the near-stop band resists a longer horizon.

    Splits the held-out near-stop rows into those whose entire future is
    stationary and those that go on to move, and compares each group's
    achievable L1 floor (`_conditional_l1_floor`) against what trained
    controllers actually reach there. If the floor and the models agree, the
    residual error is information the trajectory does not contain, and no
    horizon can recover it.
    """
    ticks = read_ticks(args.ticks, n_drives=args.n_drives, seed=args.seed)
    anchors = stitch_horizon(ticks, horizon=args.horizon, with_gnss=False)
    del ticks
    position, heading = dead_reckon_stitched(anchors)
    speed_now_np = anchors["speed"][:, 0]
    speed_now = torch.from_numpy(speed_now_np).float()

    _, val_idx, _, _ = drive_split(
        anchors["input_id"], seed=args.seed, val_frac=args.val_frac
    )
    val = val_idx.numpy()
    near = val[speed_now_np[val] < args.band_max]
    path_m = _path_length_m(anchors["speed"][near], anchors["time_stamp"][near])
    is_static = path_m < args.static_path_m

    print(  # noqa: T201
        f"[nearstop] horizon={args.horizon} ({args.horizon / 3:.1f}s), "
        f"{len(val):,} val rows, {len(near):,} below {args.band_max} km/h "
        f"({100 * len(near) / len(val):.1f}%)"
    )
    print(  # noqa: T201
        f"[nearstop] of those, {int(is_static.sum()):,} "
        f"({100 * is_static.mean():.1f}%) travel < {args.static_path_m}m over the "
        f"whole horizon -- their future trajectory is the zero trajectory, "
        f"identical no matter how far ahead it is drawn"
    )

    errors = {}
    for ckpt in args.ckpt:
        horizon, predictions = _score_checkpoint(
            ckpt,
            position=position,
            heading=heading,
            speed_now=speed_now,
            rows=torch.from_numpy(near),
        )
        errors[horizon] = {
            field: (predictions[field] - torch.from_numpy(anchors[field][near]).float())
            .abs()
            .numpy()
            for field in CONTINUOUS_RAW_COLS
        }

    groups = (
        # The speed-only oracle IS the trajectory-conditioned floor for the
        # stationary group (every row there shows the controller the same zero
        # trajectory, so speed is all that distinguishes them) and is merely a
        # baseline for the moving group, where the trajectory does carry
        # information the oracle ignores and a good model should beat it.
        ("stationary future", is_static, "floor"),
        ("moves", ~is_static, "speed-only baseline"),
    )
    for group_name, mask, oracle_name in groups:
        if not mask.any():
            continue
        print(f"\n[nearstop] {group_name}: n={int(mask.sum()):,}")  # noqa: T201
        for field in CONTINUOUS_RAW_COLS:
            target = anchors[field][near][mask]
            oracle, n_bins = _conditional_l1_floor(
                values=target,
                speed_kmh=speed_now_np[near][mask],
                speed_bin=args.speed_bin,
            )
            reached = "  ".join(
                f"h{horizon}={e[field][mask].mean():.4f}"
                for horizon, e in sorted(errors.items())
            )
            print(  # noqa: T201
                f"  {field:<16} target median={np.median(target):.4f} "
                f"p90={np.percentile(target, 90):.4f} | {oracle_name}="
                f"{oracle:.4f} ({n_bins} speed bins) | {reached}"
            )


def cmd_eval(args: argparse.Namespace) -> None:  # noqa: PLR0914
    payload = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    config = payload["config"]
    model = TrajectoryToActionMLP(
        in_features=config["in_features"], hidden_size=config["hidden_size"]
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()

    df = load(args.parquet, limit=args.limit)
    speed_now_t = torch.from_numpy(
        _arr(df, "batch/data/meta/VehicleMotion/speed")[:, FEAT_IDX]
    ).float()
    speed_now = speed_now_t.numpy()

    position_dr, heading_dr = dead_reckon(df)
    position_pred, heading_pred = predicted_trajectory_as_dead_reckoned(df)

    with torch.no_grad():
        out_dr = model(
            build_features(position=position_dr, heading=heading_dr, speed=speed_now_t)
        )
        out_pred = model(
            build_features(
                position=position_pred, heading=heading_pred, speed=speed_now_t
            )
        )

    gt, turn_gt = _load_ground_truth(df)

    own_pred = {
        field: torch.from_numpy(
            _arr(df, f"policy/prediction_value/value/continuous/{field}")[:, 0]
        ).float()
        for field in CONTINUOUS_RAW_COLS
    }
    own_turn = torch.from_numpy(
        _arr(df, "policy/prediction_value/value/discrete/turn_signal")[:, 0]
    ).long()

    print(f"[eval] n={df.height}\n")  # noqa: T201
    print("=== (a) controller on dead-reckoned GT trajectory (upper bound) ===")  # noqa: T201
    _report(out_dr, gt=gt, turn_gt=turn_gt, speed=speed_now)

    print(  # noqa: T201
        "\n=== (b) controller on policy/trajectory_value (predicted trajectory "
        "-- the 'MPC-lite' use case). NOTE: this WIP trajectory head's output "
        "is not present as importable code anywhere on disk; its scale/axis "
        "match to our own dead-reckoning convention is inferred "
        "(_integrate_relative), not verified against original source. ==="
    )
    _report(out_pred, gt=gt, turn_gt=turn_gt, speed=speed_now)

    _report_direct(
        own_pred, gt=gt, turn_pred=own_turn, turn_gt=turn_gt, speed=speed_now
    )


def cmd_modes(args: argparse.Namespace) -> None:  # noqa: PLR0914
    """Test whether the pedal<->motion mapping is vehicle-dependent -- a
    stand-in for hidden "speed modes" (regen calibration, drive mode, sensor
    drift/wear) that would break the assumption a trajectory<->action
    controller needs (that a trajectory pins down an action, and vice
    versa). Two directions, both direct physical measurements (not model
    error):

    (1)/(2) FORWARD (action -> outcome): hold pedal input near-fixed, compare
    the REALIZED one-tick speed change `dv` across vehicles.
    (3) INVERSE (outcome -> action): hold the REALIZED trajectory outcome
    near-fixed, compare the ACTION used to produce it across vehicles -- the
    literal non-identifiability the controller depends on not existing.

    If the mapping were vehicle-invariant, per-vehicle means at matched
    conditions should coincide up to sampling noise in all three.
    """
    df = load(args.parquet, limit=args.limit, n_drives=args.n_drives, seed=args.seed)
    speed = _arr(df, "batch/data/meta/VehicleMotion/speed")
    gas = _arr(df, CONTINUOUS_RAW_COLS["gas_pedal"])
    brake = _arr(df, CONTINUOUS_RAW_COLS["brake_pedal"])
    vehicle = _vehicle_id(df)

    speed_now = speed[:, FEAT_IDX]
    gas_now = gas[:, FEAT_IDX]
    brake_now = brake[:, FEAT_IDX]
    dv = speed[:, FEAT_IDX + 1] - speed_now  # km/h over one ~0.33s tick

    print(f"[modes] n={df.height}, {len(np.unique(vehicle))} vehicles")  # noqa: T201

    # (1) coast: foot fully off both pedals -- any deceleration beyond
    # aerodynamic drag (~vehicle-model-invariant across one fleet) must come
    # from regen/engine braking, whose calibration is exactly the kind of
    # "mode" that could differ per vehicle or be driver-selectable.
    coast = (gas_now < 0.02) & (brake_now < 0.02)  # noqa: PLR2004
    for lo, hi in ((20, 35), (35, 60), (60, 130)):
        mask = coast & (speed_now >= lo) & (speed_now < hi)
        _report_matched_condition(
            f"(1) FORWARD coast (gas<0.02, brake<0.02), speed [{lo},{hi})",
            vehicle=vehicle[mask],
            value=dv[mask],
            value_name="dv",
            min_count=args.min_count,
        )

    # (2) matched part-throttle, no braking -- isolates the gas-pedal ->
    # acceleration transfer function itself.
    part_throttle = (gas_now >= 0.15) & (gas_now < 0.25) & (brake_now < 0.02)  # noqa: PLR2004
    for lo, hi in ((10, 20), (20, 35), (35, 60)):
        mask = part_throttle & (speed_now >= lo) & (speed_now < hi)
        _report_matched_condition(
            f"(2) FORWARD gas in [0.15,0.25), brake<0.02, speed [{lo},{hi})",
            vehicle=vehicle[mask],
            value=dv[mask],
            value_name="dv",
            min_count=args.min_count,
        )

    # (3) INVERSE: hold the REALIZED 6-step dead-reckoned forward outcome
    # near-fixed (middle quintile of forward distance within a speed band)
    # and compare the gas pedal used to produce it -- this is the direction
    # the controller actually learns, not (1)/(2)'s forward direction.
    position, _ = dead_reckon(df)
    forward_m = position[:, -1, 1].numpy() * 100.0  # comp1=forward, /100-scale undone
    for lo, hi in ((20, 35), (35, 60)):
        speed_mask = (speed_now >= lo) & (speed_now < hi)
        if speed_mask.sum() < 500:  # noqa: PLR2004
            continue
        q40, q60 = np.quantile(forward_m[speed_mask], [0.4, 0.6])
        mask = speed_mask & (forward_m >= q40) & (forward_m < q60)
        _report_matched_condition(
            f"(3) INVERSE forward outcome in [{q40:.1f},{q60:.1f})m/6-tick, "
            f"speed [{lo},{hi}) -- action=gas_pedal",
            vehicle=vehicle[mask],
            value=gas_now[mask],
            value_name="gas_pedal",
            min_count=args.min_count,
        )


# --------------------------------------------------------------------------- CLI


# ------------------------------------------------------- per-drive calibration

CALIBRATION_MIN_PAIRS = 200
"""Minimum consecutive-tick pairs before a per-drive dynamics fit is trusted.

`fit_longitudinal_dynamics` solves for 4 parameters, so this is two orders of
magnitude of headroom -- the gate is about the ESTIMATE's stability across
halves of a drive, not about the solve succeeding.
"""

BRAKE_PRESSED = 0.05
"""`brake_pedal_normalized` above which a tick counts as actually braking."""

MOVING_KMH = 5.0
"""Speed above which a tick is out of the near-stop band (`SPEED_BANDS[0]`).

Below it `dv` is clamped by the stop itself rather than by the pedal, so a
fit pooled over these rows attributes to `brake_gain` an effect the brake did
not have -- finding #10, seen from the calibration side.
"""

HARD_BRAKING_KMH = 15.0
"""Speed floor for the strictest `brake_gain` gate."""

TURNING_STEERING = 0.02
"""`steering_angle_normalized` above which a tick counts as actually turning."""

CALIBRATION_OUTLIER_PERCENTILE = 1.0
"""Percent trimmed from each tail before correlating the two halves.

A handful of drives produce a wild fit (a half spent almost entirely
stationary, say), and an untrimmed Pearson r over those is a statement about
two outliers rather than about the fleet.
"""


def _tick_pairs(sub: pl.DataFrame, index: np.ndarray) -> dict[str, np.ndarray] | None:
    """Consecutive `(t, t+1)` tick pairs within `index`, for one drive.

    Gated exactly like `horizon_windows`: same phase grid, forward in wall
    clock, no gap beyond `MAX_TICK_GAP_MS` (bug #3's backwards clock reaches
    this parquet, and an unguarded `dv` integrates it).

    Returns:
        Per-pair arrays taken at the FIRST tick of each pair, plus the `dv` /
        `dheading` realized over it; `None` if `index` is too short.
    """
    frame_idx = sub["frame_idx"].to_numpy().astype(np.int64)
    time_stamp_us = sub["time_stamp"].to_numpy().astype(np.float64)
    if index.size < 2:  # noqa: PLR2004
        return None
    left, right = index[:-1], index[1:]
    dt_us = time_stamp_us[right] - time_stamp_us[left]
    contiguous = (
        (frame_idx[right] - frame_idx[left] == TICK_STRIDE_FRAME_IDX)
        & (dt_us > 0)
        & (dt_us <= MAX_TICK_GAP_MS * 1000.0)
    )
    left, right = left[contiguous], right[contiguous]
    speed = sub["speed"].to_numpy().astype(np.float64)
    heading = sub["heading"].to_numpy().astype(np.float64)
    return {
        "gas": sub["gas_pedal"].to_numpy().astype(np.float64)[left],
        "brake": sub["brake_pedal"].to_numpy().astype(np.float64)[left],
        "steering": sub["steering_angle"].to_numpy().astype(np.float64)[left],
        "speed": speed[left],
        "dv": speed[right] - speed[left],
        # compass degrees -> signed radians, wrapped: an unwrapped 359->1
        # step reads as -358 degrees of yaw in one tick.
        "dheading": np.deg2rad(
            (heading[right] - heading[left] + 180.0) % 360.0 - 180.0
        ),
    }


def _longitudinal_gain(
    pairs: dict[str, np.ndarray], mask: np.ndarray, *, index: int
) -> float:
    """`fit_longitudinal_dynamics` on `mask`ed pairs -> one coefficient."""
    if mask.sum() < CALIBRATION_MIN_PAIRS:
        return float("nan")
    fit = fit_longitudinal_dynamics(**{
        name: torch.from_numpy(pairs[name][mask])
        for name in ("gas", "brake", "speed", "dv")
    })
    return float(fit[index])


def _brake_only_gain(
    pairs: dict[str, np.ndarray], mask: np.ndarray, *, min_pairs: int = 100
) -> float:
    """`dv ~= -brake_gain*brake - drag*speed - offset`, with NO gas column.

    Gating to rows that are actually braking drives the gas column to nearly
    all zeros, which makes `fit_longitudinal_dynamics`' 4-column design
    rank-deficient; `lstsq` then returns a least-norm solution whose
    `brake_gain` is not the thing it is named after. Dropping the column is
    the honest way to ask whether `brake_gain` is identifiable where braking
    actually happens, and `min_pairs` is lower than
    `CALIBRATION_MIN_PAIRS` because that gate leaves few rows by
    construction (see `cmd_calibrate`'s coverage line).
    """
    if mask.sum() < min_pairs:
        return float("nan")
    count = int(mask.sum())
    design = np.stack(
        [-pairs["brake"][mask], -pairs["speed"][mask], -np.ones(count)], axis=-1
    )
    solution, _, rank, _ = np.linalg.lstsq(design, pairs["dv"][mask], rcond=None)
    return float(solution[0]) if rank == design.shape[1] else float("nan")


def _lateral_gain(pairs: dict[str, np.ndarray], mask: np.ndarray) -> float:
    """`fit_lateral_dynamics` on `mask`ed pairs -> `steer_gain`."""
    if mask.sum() < CALIBRATION_MIN_PAIRS:
        return float("nan")
    return float(
        fit_lateral_dynamics(
            steering_angle=torch.from_numpy(pairs["steering"][mask]),
            speed=torch.from_numpy(pairs["speed"][mask]),
            dheading=torch.from_numpy(pairs["dheading"][mask]),
        ).steer_gain
    )


CALIBRATION_ESTIMATORS: dict[str, Callable[[dict[str, np.ndarray]], float]] = {
    "gas_gain / all rows": lambda p: _longitudinal_gain(
        p, np.ones(p["gas"].size, dtype=bool), index=0
    ),
    "brake_gain / all rows": lambda p: _longitudinal_gain(
        p, np.ones(p["gas"].size, dtype=bool), index=1
    ),
    "brake_gain / v>5": lambda p: _longitudinal_gain(
        p, p["speed"] > MOVING_KMH, index=1
    ),
    "brake_gain / braking only": lambda p: _brake_only_gain(
        p, (p["speed"] > MOVING_KMH) & (p["brake"] > BRAKE_PRESSED)
    ),
    "brake_gain / hard braking": lambda p: _brake_only_gain(
        p, (p["speed"] > HARD_BRAKING_KMH) & (p["brake"] > 2 * BRAKE_PRESSED)
    ),
    "steer_gain / all rows": lambda p: _lateral_gain(
        p, np.ones(p["gas"].size, dtype=bool)
    ),
    "steer_gain / v>5": lambda p: _lateral_gain(p, p["speed"] > MOVING_KMH),
    "steer_gain / v>5, turning": lambda p: _lateral_gain(
        p, (p["speed"] > MOVING_KMH) & (np.abs(p["steering"]) > TURNING_STEERING)
    ),
}
"""Per-drive gain estimators, each under a different row gate.

The gates matter more than the estimators: the plain joint fit pools rows
from the near-stop regime where `dv` is clamped regardless of pedal (finding
#10), so it can "estimate" a `brake_gain` from rows that carry no information
about one.
"""


def drive_half_split(frame_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split one drive's ticks in half by `frame_idx`.

    Returns:
        `(calibration_index, evaluation_index)` -- the same first-half /
        second-half protocol the per-drive dynamics fits use throughout, so a
        gain fitted on the first is never scored on a row it saw.
    """
    split = int(np.median(frame_idx))
    return np.flatnonzero(frame_idx <= split), np.flatnonzero(frame_idx > split)


def calibrate_drives(
    ticks: pl.DataFrame, *, min_pairs: int = CALIBRATION_MIN_PAIRS
) -> tuple[dict[str, tuple[float, ...]], dict[str, int]]:
    """Fit `LongitudinalDynamicsParams` per drive on that drive's FIRST half.

    Returns:
        `(drive -> the 4 fitted params, drive -> its split frame_idx)`. A
        drive with too few usable calibration pairs is absent from the first
        dict but present in the second.
    """
    params: dict[str, tuple[float, ...]] = {}
    splits: dict[str, int] = {}
    for (drive,), sub in ticks.group_by(["input_id"], maintain_order=True):
        frame_idx = sub["frame_idx"].to_numpy().astype(np.int64)
        calibration, _ = drive_half_split(frame_idx)
        splits[str(drive)] = int(np.median(frame_idx))
        pairs = _tick_pairs(sub, calibration)
        if pairs is None or pairs["gas"].size < min_pairs:
            continue
        fit = fit_longitudinal_dynamics(**{
            name: torch.from_numpy(pairs[name])
            for name in ("gas", "brake", "speed", "dv")
        })
        if np.all(np.isfinite(fit)):
            params[str(drive)] = tuple(fit)
    return params, splits


def cmd_calibrate(args: argparse.Namespace) -> None:  # noqa: PLR0914
    """Split-half reliability of every per-drive gain.

    A per-drive scalar can only help a downstream model if the value fitted
    on one half of a drive predicts the value fitted on the other half better
    than it predicts a random other drive's. This measures that directly, and
    so bounds every conditioning scheme at once -- if the number is not
    there, no architecture can exploit it.
    """
    ticks = read_ticks(args.ticks, n_drives=args.n_drives, seed=args.seed)
    first: dict[str, list[float]] = {name: [] for name in CALIBRATION_ESTIMATORS}
    second: dict[str, list[float]] = {name: [] for name in CALIBRATION_ESTIMATORS}
    vehicles: list[str] = []
    braking_share: list[float] = []
    moving_braking_share: list[float] = []

    for (drive,), sub in ticks.group_by(["input_id"], maintain_order=True):
        frame_idx = sub["frame_idx"].to_numpy().astype(np.int64)
        calibration, evaluation = drive_half_split(frame_idx)
        a = _tick_pairs(sub, calibration)
        b = _tick_pairs(sub, evaluation)
        if a is None or b is None:
            continue
        if min(a["gas"].size, b["gas"].size) < CALIBRATION_MIN_PAIRS:
            continue
        vehicles.append(str(drive).split("/")[0])
        for name, estimator in CALIBRATION_ESTIMATORS.items():
            first[name].append(estimator(a))
            second[name].append(estimator(b))
        braking_share.append(float((a["brake"] > BRAKE_PRESSED).mean()))
        moving_braking_share.append(
            float(((a["brake"] > BRAKE_PRESSED) & (a["speed"] > MOVING_KMH)).mean())
        )

    vehicle = np.array(vehicles)
    print(f"[calibrate] {len(vehicle)} drives with both halves usable\n")  # noqa: T201
    for label, share in (
        (f"brake>{BRAKE_PRESSED}", braking_share),
        (f"brake>{BRAKE_PRESSED} & v>{MOVING_KMH:g}", moving_braking_share),
    ):
        values = np.array(share)
        print(  # noqa: T201
            f"[calibrate] calibration-half rows with {label}: median "
            f"{100 * np.median(values):.1f}% (p10 "
            f"{100 * np.percentile(values, 10):.1f}%, p90 "
            f"{100 * np.percentile(values, 90):.1f}%)"
        )

    header = (
        f"\n{'estimator':<28}{'n':>5}{'p5':>10}{'p50':>10}{'p95':>10}"
        f"{'half-half r':>13}{'reliability':>13}{'eta2(vehicle)':>15}"
    )
    print(header)  # noqa: T201
    print("-" * (len(header) - 1))  # noqa: T201
    for name in CALIBRATION_ESTIMATORS:
        a, b = np.array(first[name]), np.array(second[name])
        keep = np.isfinite(a) & np.isfinite(b)
        if keep.sum() < args.min_drives:
            print(f"{name:<28}{keep.sum():>5}   (too few drives fittable)")  # noqa: T201
            continue
        low, high = np.percentile(
            np.concatenate([a[keep], b[keep]]),
            [CALIBRATION_OUTLIER_PERCENTILE, 100 - CALIBRATION_OUTLIER_PERCENTILE],
        )
        keep &= (a > low) & (a < high) & (b > low) & (b < high)
        if keep.sum() < args.min_drives:
            print(f"{name:<28}{keep.sum():>5}   (degenerate after outlier trim)")  # noqa: T201
            continue
        a, b, vehicle_kept = a[keep], b[keep], vehicle[keep]
        # reliability: the share of the between-drive spread that survives
        # re-estimation. var(a-b)/2 is the per-half noise variance under the
        # usual assumption that the two halves are equally noisy.
        midpoint = (a + b) / 2
        between = float(np.var(midpoint, ddof=1))
        within = float(np.var(a - b, ddof=1)) / 2
        groups = [
            midpoint[vehicle_kept == name_]
            for name_ in np.unique(vehicle_kept)
            if (vehicle_kept == name_).sum() >= 3  # noqa: PLR2004
        ]
        grand = midpoint.mean()
        ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
        ss_within = sum(((g - g.mean()) ** 2).sum() for g in groups)
        print(  # noqa: T201
            f"{name:<28}{a.size:>5}{np.percentile(a, 5):>10.4f}"
            f"{np.median(a):>10.4f}{np.percentile(a, 95):>10.4f}"
            f"{np.corrcoef(a, b)[0, 1]:>+13.3f}"
            f"{max(0.0, (between - within) / between):>13.3f}"
            f"{ss_between / (ss_between + ss_within):>15.3f}"
        )


def _robust_z(values: np.ndarray, *, clip: float = 5.0) -> np.ndarray:
    """Median/MAD standardization, clipped.

    Mean/std would let the handful of drives with a wild fit set the scale
    for all 644; the point of the feature is the ordering of the bulk.
    """
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = 1.4826 * mad if mad > 0 else (float(values.std()) or 1.0)
    return np.clip((values - median) / scale, -clip, clip)


def cmd_gaincond(args: argparse.Namespace) -> None:  # noqa: PLR0914
    """Does conditioning the MLP on a per-drive calibration scalar help?

    Finding #3 showed a per-drive `LongitudinalDynamicsParams` fit transforms
    the PARAMETRIC controller; this asks whether handing the same calibration
    to the MLP as extra input features does the same for it.

    Protocol -- calibrate on each drive's first half, train and score only on
    anchors in its second half, hold out whole drives. Every arm sees
    identical rows; only the feature vector differs. Each arm is refit under
    every `--fit-seeds` value, because a single fit's run-to-run spread is the
    same order as the effect being measured.

    The shuffled arms are the load-bearing controls: the same scalars,
    permuted across drives. An arm that beats `base` by no more than its own
    shuffled twin has measured capacity or noise, not calibration.
    """
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ticks = read_ticks(args.ticks, n_drives=args.n_drives, seed=args.seed)
    print(  # noqa: T201
        f"[gaincond] {ticks.height:,} ticks over {ticks['input_id'].n_unique():,} "
        f"drives (device={device})"
    )

    params, splits = calibrate_drives(ticks)
    gains = np.array([value[0] for value in params.values()])
    print(  # noqa: T201
        f"[gaincond] calibrated {len(params)}/{len(splits)} drives on their "
        f"first half; gas_gain p5/p50/p95 = {np.percentile(gains, 5):.4f} / "
        f"{np.median(gains):.4f} / {np.percentile(gains, 95):.4f}"
    )

    anchors = stitch_horizon(
        ticks, horizon=args.horizon, action_steps=1, with_gnss=False
    )
    del ticks
    calibrated = np.array(sorted(params))
    keep = anchors["frame_idx"] > np.array([
        splits[drive] for drive in anchors["input_id"]
    ])
    keep &= np.isin(anchors["input_id"], calibrated)
    print(  # noqa: T201
        f"[gaincond] {keep.sum():,}/{keep.size:,} anchors survive the "
        f"second-half + calibrated-drive gate"
    )
    anchors = {name: value[keep] for name, value in anchors.items()}

    position, heading = dead_reckon_stitched(anchors)
    speed_now = torch.from_numpy(anchors["speed"][:, 0]).float()
    base = _horizon_features(
        position=position, heading=heading, speed_now=speed_now, horizon=args.horizon
    )
    targets = {
        field: torch.from_numpy(anchors[field]).float() for field in CONTINUOUS_RAW_COLS
    }
    turn_gt = torch.from_numpy(anchors["turn_signal"]).long()

    train_idx, val_idx, n_drives, n_val_drives = drive_split(
        anchors["input_id"], seed=args.seed, val_frac=args.val_frac
    )
    val_speed = anchors["speed"][val_idx.numpy(), 0]
    print(  # noqa: T201
        f"[gaincond] {n_drives} drives -> {n_drives - n_val_drives} train / "
        f"{n_val_drives} val ({len(train_idx):,} / {len(val_idx):,} rows)"
    )

    fitted = np.array([params[drive] for drive in calibrated])
    standardized = np.stack(
        [_robust_z(fitted[:, k]) for k in range(fitted.shape[1])], axis=1
    )
    permuted = standardized[
        np.random.default_rng(args.seed).permutation(len(calibrated))
    ]
    row_of = {drive: i for i, drive in enumerate(calibrated)}
    rows = np.array([row_of[drive] for drive in anchors["input_id"]])

    def feature(matrix: np.ndarray, columns: slice | int) -> Tensor:
        block = np.atleast_2d(matrix[rows][:, columns].T).T
        return torch.from_numpy(np.ascontiguousarray(block)).float()

    arms = {
        "base": base,
        "gasgain": torch.cat([base, feature(standardized, 0)], dim=-1),
        "gasgain_shuf": torch.cat([base, feature(permuted, 0)], dim=-1),
        "allparams": torch.cat([base, feature(standardized, slice(None))], dim=-1),
        "allparams_shuf": torch.cat([base, feature(permuted, slice(None))], dim=-1),
    }

    results: dict[str, dict[str, Any]] = {}
    for label, features in arms.items():
        runs = [
            _fit(
                features=features,
                targets=targets,
                turn_gt=turn_gt,
                train_idx=train_idx,
                val_idx=val_idx,
                hidden_size=args.hidden_size,
                steps=args.steps,
                batch_size=args.batch_size,
                lr=args.lr,
                log_every=args.log_every,
                seed=fit_seed,
                label=f"{label}:s{fit_seed}",
                action_steps=1,
                device=device,
            )
            for fit_seed in args.fit_seeds
        ]
        results[label] = {
            "in_features": int(features.shape[-1]),
            "fit_seeds": args.fit_seeds,
            "val_l1": {
                field: float(np.mean([run.val_l1[field] for run in runs]))
                for field in CONTINUOUS_RAW_COLS
            },
            "val_l1_sd": {
                field: float(np.std([run.val_l1[field] for run in runs]))
                for field in CONTINUOUS_RAW_COLS
            },
            "val_turn_signal_acc": float(np.mean([r.val_turn_acc for r in runs])),
            "val_turn_signal_acc_sd": float(np.std([r.val_turn_acc for r in runs])),
            "val_l1_by_speed_band": {
                field: _band_l1(
                    val_speed, np.mean([run.val_l1_rows[field] for run in runs], axis=0)
                )
                for field in CONTINUOUS_RAW_COLS
            },
        }

    _report_gaincond(results, val_speed=val_speed, n_seeds=len(args.fit_seeds))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "ticks": str(args.ticks),
                    "horizon": args.horizon,
                    "steps": args.steps,
                    "seed": args.seed,
                    "fit_seeds": args.fit_seeds,
                    "n_drives": n_drives,
                    "n_val_drives": n_val_drives,
                    "n_train_rows": len(train_idx),
                    "n_val_rows": len(val_idx),
                    "gas_gain_by_drive": {
                        drive: params[drive][0] for drive in calibrated
                    },
                    "results": results,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\n[gaincond] wrote {args.out}")  # noqa: T201


def _report_gaincond(
    results: dict[str, dict[str, Any]], *, val_speed: np.ndarray, n_seeds: int
) -> None:
    """Print each arm against `base`, with the seed spread beside the delta.

    A delta smaller than the spread of the two arms it compares is not a
    result; the table prints both so it cannot be read as one.
    """
    del val_speed
    base = results["base"]
    print(  # noqa: T201
        f"\n=== val L1 on the applied action, mean +/- sd over {n_seeds} fit "
        f"seeds (lower is better) ==="
    )
    fields = [*CONTINUOUS_RAW_COLS, "turn_signal acc"]
    print(f"{'arm':<16}" + "".join(f"{field:>30}" for field in fields))  # noqa: T201
    for label, result in results.items():
        cells = [
            f"{result['val_l1'][field]:.4f}+-{result['val_l1_sd'][field]:.4f}"
            f" ({100 * (result['val_l1'][field] / base['val_l1'][field] - 1):+.1f}%)"
            for field in CONTINUOUS_RAW_COLS
        ]
        cells.append(
            f"{result['val_turn_signal_acc']:.4f}"
            f"+-{result['val_turn_signal_acc_sd']:.4f} "
            f"({100 * (result['val_turn_signal_acc'] - base['val_turn_signal_acc']):+.2f}pp)"
        )
        print(f"{label:<16}" + "".join(f"{cell:>30}" for cell in cells))  # noqa: T201

    for field in CONTINUOUS_RAW_COLS:
        print(f"\n--- {field} by speed band ---")  # noqa: T201
        bands = list(base["val_l1_by_speed_band"][field])
        print(f"{'arm':<16}" + "".join(f"{band:>12}" for band in bands))  # noqa: T201
        for label, result in results.items():
            row = result["val_l1_by_speed_band"][field]
            print(f"{label:<16}" + "".join(f"{row[b]:>12.4f}" for b in bands))  # noqa: T201


def _add_calibrate_parser(subparsers: argparse._SubParsersAction) -> None:
    """`calibrate`: split-half reliability of the per-drive gains."""
    calibrate = subparsers.add_parser(
        "calibrate",
        help="split-half reliability of every per-drive gain: is the number there?",
    )
    calibrate.add_argument("--ticks", type=Path, default=DEFAULT_TICKS)
    calibrate.add_argument("--n-drives", type=int, default=None)
    calibrate.add_argument("--seed", type=int, default=7)
    calibrate.add_argument(
        "--min-drives",
        type=int,
        default=20,
        help="drives an estimator must fit on both halves before it is reported",
    )
    calibrate.set_defaults(func=cmd_calibrate)


def _add_gaincond_parser(subparsers: argparse._SubParsersAction) -> None:
    """`gaincond`: per-drive calibration as an MLP input, vs controls."""
    gaincond = subparsers.add_parser(
        "gaincond",
        help="condition the MLP on a per-drive calibration scalar, vs shuffled controls",
    )
    gaincond.add_argument("--ticks", type=Path, default=DEFAULT_TICKS)
    gaincond.add_argument("--horizon", type=int, default=30)
    gaincond.add_argument("--n-drives", type=int, default=None)
    gaincond.add_argument("--steps", type=int, default=20_000)
    gaincond.add_argument("--batch-size", type=int, default=1024)
    gaincond.add_argument("--lr", type=float, default=1e-3)
    gaincond.add_argument("--hidden-size", type=int, default=128)
    gaincond.add_argument("--val-frac", type=float, default=0.1)
    gaincond.add_argument("--seed", type=int, default=7)
    gaincond.add_argument(
        "--fit-seeds",
        type=int,
        nargs="+",
        default=[7, 11, 13],
        help="refit every arm once per seed; a single fit's spread is the "
        "same order as the effect being measured",
    )
    gaincond.add_argument("--log-every", type=int, default=10_000)
    gaincond.add_argument("--device", default=None)
    gaincond.add_argument("--out", type=Path, default=None)
    gaincond.set_defaults(func=cmd_gaincond)


def _add_horizon_parser(subparsers: argparse._SubParsersAction) -> None:
    """`horizon`: sweep the trajectory and control horizons."""
    horizon = subparsers.add_parser(
        "horizon", help="sweep the trajectory and control horizons on one shared split"
    )
    horizon.add_argument(
        "--ticks",
        type=Path,
        default=DEFAULT_TICKS,
        help="cached tick table from `stitch --out` (NOT the source parquet)",
    )
    horizon.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[6, 15, 30],
        help="trajectory horizons to compare, in ticks (1 tick = 0.333s); "
        "6 reproduces the windowed formulation and is the sweep's baseline",
    )
    horizon.add_argument(
        "--action-steps",
        type=int,
        nargs="+",
        default=[1],
        help="control horizons to compare: how many consecutive actions the "
        "model emits per forward pass, of which a receding-horizon controller "
        "applies the first. Every value must be <= min(--horizons)",
    )
    horizon.add_argument(
        "--n-drives", type=int, default=None, help="sample this many drives"
    )
    horizon.add_argument("--steps", type=int, default=20_000)
    horizon.add_argument("--batch-size", type=int, default=4096)
    horizon.add_argument("--lr", type=float, default=1e-3)
    horizon.add_argument("--hidden-size", type=int, default=128)
    horizon.add_argument(
        "--val-frac", type=float, default=0.1, help="fraction of DRIVES held out"
    )
    horizon.add_argument("--seed", type=int, default=7)
    horizon.add_argument("--log-every", type=int, default=2_000)
    horizon.add_argument(
        "--device", default=None, help="default: cuda when available, else cpu"
    )
    horizon.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="write horizon_sweep.json + one checkpoint per configuration",
    )
    horizon.set_defaults(func=cmd_horizon)


def _add_nearstop_parser(subparsers: argparse._SubParsersAction) -> None:
    """`nearstop`: near-stop information-floor diagnostic."""
    nearstop = subparsers.add_parser(
        "nearstop",
        help="is the near-stop band at its information floor, or undertrained?",
    )
    nearstop.add_argument("--ticks", type=Path, default=DEFAULT_TICKS)
    nearstop.add_argument("--horizon", type=int, default=30)
    nearstop.add_argument("--n-drives", type=int, default=None)
    nearstop.add_argument("--val-frac", type=float, default=0.1)
    nearstop.add_argument("--seed", type=int, default=7)
    nearstop.add_argument(
        "--band-max",
        type=float,
        default=float(SPEED_BANDS[0][1]),
        help="upper edge of the near-stop band, km/h",
    )
    nearstop.add_argument(
        "--static-path-m",
        type=float,
        default=0.5,
        help="distance below which a whole horizon counts as stationary",
    )
    nearstop.add_argument(
        "--speed-bin",
        type=float,
        default=0.5,
        help="speed resolution the L1 floor credits a model with, km/h",
    )
    nearstop.add_argument(
        "--ckpt",
        type=Path,
        nargs="*",
        default=[],
        help="`horizon`-subcommand checkpoints to score against the floor",
    )
    nearstop.set_defaults(func=cmd_nearstop)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="trajectory_action_controller",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    subparsers = parser.add_subparsers(dest="command", required=True)

    episode = subparsers.add_parser(
        "episode", help="continuous per-tick action trace for one drive"
    )
    episode.add_argument("--drive", required=True, help="batch/meta/input_id value")
    episode.add_argument("--out", type=Path, default=Path("/tmp/episode.png"))  # noqa: S108
    episode.set_defaults(func=cmd_episode)

    stitch = subparsers.add_parser(
        "stitch", help="rebuild the per-tick trace and report horizon coverage"
    )
    stitch.add_argument(
        "--horizon",
        type=int,
        default=30,
        help="future ticks per anchor (1 tick = 0.333s)",
    )
    stitch.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[6, 15, 30, 60, 90, 150],
        help="horizons to report anchor coverage for",
    )
    stitch.add_argument(
        "--n-drives",
        type=int,
        default=None,
        help="sample this many drives (see `train`)",
    )
    stitch.add_argument("--seed", type=int, default=0)
    stitch.add_argument(
        "--verify",
        action="store_true",
        help="assert the stitched path reproduces the windowed one at HORIZON",
    )
    stitch.add_argument("--out", type=Path, default=None, help="cache the tick table")
    stitch.set_defaults(func=cmd_stitch)

    _add_horizon_parser(subparsers)
    _add_nearstop_parser(subparsers)

    _add_calibrate_parser(subparsers)
    _add_gaincond_parser(subparsers)

    reckon = subparsers.add_parser(
        "reckon", help="dead-reckon + sanity-gate the trajectory"
    )
    reckon.add_argument("--n-sample", type=int, default=20_000)
    reckon.add_argument(
        "--out", type=Path, default=None, help="cache dead-reckoned tensor"
    )
    reckon.set_defaults(func=cmd_reckon)

    train = subparsers.add_parser("train", help="fit the trajectory->action controller")
    train.add_argument(
        "--limit", type=int, default=None, help="rows to load (default: all)"
    )
    train.add_argument(
        "--n-drives",
        type=int,
        default=None,
        help="sample this many whole drives across the full file instead of "
        "--limit's first-N-rows (spans many vehicles; use for "
        "--ablate-vehicle-conditioning)",
    )
    train.add_argument("--steps", type=int, default=20_000)
    train.add_argument("--batch-size", type=int, default=4096)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--hidden-size", type=int, default=128)
    train.add_argument(
        "--val-frac", type=float, default=0.1, help="fraction of DRIVES held out"
    )
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--log-every", type=int, default=500)
    train.add_argument(
        "--ablate-vehicle-conditioning",
        action="store_true",
        help="also fit a variant with a vehicle one-hot appended to features, "
        "on the identical split, and print a val-L1 comparison (diagnostic "
        "only -- --out always saves the unconditioned model)",
    )
    train.add_argument("--out", type=Path, required=True)
    train.set_defaults(func=cmd_train)

    evaluate = subparsers.add_parser(
        "eval", help="controller vs the checkpoint's own action head"
    )
    evaluate.add_argument("--ckpt", type=Path, required=True)
    evaluate.add_argument("--limit", type=int, default=None)
    evaluate.set_defaults(func=cmd_eval)

    modes = subparsers.add_parser(
        "modes", help="test for hidden per-vehicle pedal->motion modes"
    )
    modes.add_argument("--limit", type=int, default=None)
    modes.add_argument(
        "--n-drives",
        type=int,
        default=None,
        help="sample this many drives (see `train`)",
    )
    modes.add_argument("--seed", type=int, default=0)
    modes.add_argument(
        "--min-count", type=int, default=200, help="min rows/vehicle/cell"
    )
    modes.set_defaults(func=cmd_modes)

    args = parser.parse_args()
    if args.command == "horizon" and max(args.action_steps) > min(args.horizons):
        parser.error(
            f"--action-steps {max(args.action_steps)} exceeds the shortest "
            f"--horizons entry {min(args.horizons)}: an action at anchor+k is "
            f"only supervised by a trajectory that reaches tick k"
        )
    args.func(args)


if __name__ == "__main__":
    main()
