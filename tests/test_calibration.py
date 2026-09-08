import numpy as np
import polars as pl
import pytest
import torch

from rmind.components.controller import fit_longitudinal_dynamics
from rmind.components.tick_trace import TICK_STRIDE_FRAME_IDX
from rmind.scripts.trajectory_action_controller import (
    _brake_only_gain,  # noqa: PLC2701
    _robust_z,  # noqa: PLC2701
    _tick_pairs,  # noqa: PLC2701
    calibrate_drives,
    drive_half_split,
)

STRIDE = TICK_STRIDE_FRAME_IDX
TICK_US = 333_333


def _drive(
    n: int,
    *,
    heading: np.ndarray | None = None,
    time_stamp: np.ndarray | None = None,
    frame_idx: np.ndarray | None = None,
    seed: int = 0,
) -> pl.DataFrame:
    """A synthetic single-drive tick table on the nominal ~333ms grid."""
    rng = np.random.default_rng(seed)
    frames = np.arange(n, dtype=np.int64) * STRIDE if frame_idx is None else frame_idx
    stamps = (
        np.arange(n, dtype=np.int64) * TICK_US if time_stamp is None else time_stamp
    )
    return pl.DataFrame({
        "input_id": ["d"] * n,
        "frame_idx": frames,
        "time_stamp": stamps,
        "speed": rng.uniform(0, 60, n),
        "heading": rng.uniform(0, 360, n) if heading is None else heading,
        "gas_pedal": rng.uniform(0, 1, n),
        "brake_pedal": rng.uniform(0, 1, n),
        "steering_angle": rng.uniform(-1, 1, n),
    })


def test_drive_half_split_is_a_disjoint_cover() -> None:
    frame_idx = np.arange(11, dtype=np.int64) * STRIDE
    calibration, evaluation = drive_half_split(frame_idx)

    assert not set(calibration.tolist()) & set(evaluation.tolist())
    assert sorted([*calibration.tolist(), *evaluation.tolist()]) == list(range(11))
    # the split is by frame_idx value, so calibration is strictly earlier
    assert frame_idx[calibration].max() < frame_idx[evaluation].min()


def test_tick_pairs_rejects_a_backwards_clock_step() -> None:
    """Bug #3: 11/655 drives step backwards in wall clock, and an unguarded
    `dv` integrates that step as if it were forward time."""
    stamps = np.arange(6, dtype=np.int64) * TICK_US
    stamps[3] = stamps[2] - 1_000_000  # -1s, as `Niro111-HQ/2023-03-20--10-49-39` does
    df = _drive(6, time_stamp=stamps)

    pairs = _tick_pairs(df, np.arange(6))

    assert pairs is not None
    # pairs (2,3) and (3,4) both straddle the bad stamp and must be dropped
    assert pairs["gas"].size == 3  # noqa: PLR2004


def test_tick_pairs_rejects_a_phase_break() -> None:
    frames = np.array([0, 10, 20, 37, 47], dtype=np.int64)  # 20 -> 37 is off-grid
    df = _drive(5, frame_idx=frames)

    pairs = _tick_pairs(df, np.arange(5))

    assert pairs is not None
    assert pairs["gas"].size == 3  # noqa: PLR2004


def test_tick_pairs_wraps_heading_across_north() -> None:
    """A 359 -> 1 step is +2 degrees of yaw, not -358."""
    df = _drive(2, heading=np.array([359.0, 1.0]))

    pairs = _tick_pairs(df, np.arange(2))

    assert pairs is not None
    assert pairs["dheading"] == pytest.approx(np.deg2rad([2.0]))


def test_brake_only_gain_recovers_a_known_coefficient() -> None:
    rng = np.random.default_rng(0)
    n = 500
    brake = rng.uniform(0, 1, n)
    speed = rng.uniform(5, 60, n)
    pairs = {
        "brake": brake,
        "speed": speed,
        "gas": np.zeros(n),
        "dv": -3.5 * brake - 0.02 * speed - 0.1,
    }

    gain = _brake_only_gain(pairs, np.ones(n, dtype=bool))

    assert gain == pytest.approx(3.5, rel=1e-4)


def test_brake_only_gain_drops_the_degenerate_gas_column() -> None:
    """Gating to braking rows zeroes the gas column, which makes the joint
    4-column design rank-deficient; the brake-only fit must not inherit that."""
    rng = np.random.default_rng(1)
    n = 400
    brake = rng.uniform(0.1, 1, n)
    speed = rng.uniform(5, 60, n)
    gas = np.zeros(n)
    dv = -3.5 * brake - 0.02 * speed
    mask = np.ones(n, dtype=bool)

    joint = fit_longitudinal_dynamics(
        gas=torch.from_numpy(gas),
        brake=torch.from_numpy(brake),
        speed=torch.from_numpy(speed),
        dv=torch.from_numpy(dv),
    )

    assert _brake_only_gain({"brake": brake, "speed": speed, "dv": dv}, mask) == (
        pytest.approx(3.5, rel=1e-4)
    )
    assert np.isfinite(joint.gas_gain)  # least-norm, not an error -- and meaningless


def test_brake_only_gain_is_nan_below_min_pairs() -> None:
    n = 10
    pairs = {
        "brake": np.linspace(0, 1, n),
        "speed": np.linspace(5, 60, n),
        "dv": np.zeros(n),
    }

    assert np.isnan(_brake_only_gain(pairs, np.ones(n, dtype=bool), min_pairs=100))


def test_robust_z_ignores_a_wild_outlier() -> None:
    values = np.array([*np.linspace(4.0, 8.0, 99), 1e6])

    standardized = _robust_z(values)
    naive = (values - values.mean()) / values.std()

    # the bulk keeps its spread instead of being crushed toward zero by a
    # mean/std scale that one bad fit would dominate
    assert standardized[:99].std() > 1000 * naive[:99].std()
    assert standardized.max() == pytest.approx(5.0)


def test_calibrate_drives_skips_a_drive_with_too_few_pairs() -> None:
    short = _drive(50, seed=1).with_columns(pl.lit("short").alias("input_id"))
    long = _drive(1000, seed=2).with_columns(pl.lit("long").alias("input_id"))

    params, splits = calibrate_drives(pl.concat([short, long]))

    assert set(splits) == {"short", "long"}
    assert set(params) == {"long"}
    assert len(params["long"]) == 4  # noqa: PLR2004
