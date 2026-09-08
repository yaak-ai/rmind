import re

import numpy as np
import polars as pl
import pytest

from rmind.components.tick_trace import (
    MAX_TICK_GAP_MS,
    TICK_STRIDE_FRAME_IDX,
    build_tick_table,
    horizon_windows,
)
from rmind.scripts.trajectory_action_controller import drive_split, stitch_horizon

STRIDE = TICK_STRIDE_FRAME_IDX
TICK_US = 333_333


def _ticks(frame_idx: list[int], *, t0: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Frame indices plus timestamps at the nominal ~333ms tick spacing."""
    frames = np.asarray(frame_idx, dtype=np.int64)
    stamps = t0 + (frames - frames[0]) // STRIDE * TICK_US
    return frames, stamps.astype(np.int64)


def test_build_tick_table_deduplicates_overlapping_windows() -> None:
    # two windows sharing ticks 10..30, as consecutive predict rows do
    df = pl.DataFrame(
        {
            "input_id": ["a", "a"],
            "frame_idx": [[0, 10, 20, 30], [10, 20, 30, 40]],
            "speed": [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]],
        },
        schema={
            "input_id": pl.String,
            "frame_idx": pl.Array(pl.Int32, 4),
            "speed": pl.Array(pl.Float32, 4),
        },
    )

    table = build_tick_table(
        df,
        input_id_column="input_id",
        frame_idx_column="frame_idx",
        value_columns=["speed"],
    )

    assert table["frame_idx"].to_list() == [0, 10, 20, 30, 40]
    assert table["speed"].to_list() == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_build_tick_table_keeps_drives_separate() -> None:
    df = pl.DataFrame(
        {
            "input_id": ["b", "a"],
            "frame_idx": [[0, 10], [0, 10]],
            "speed": [[9.0, 9.0], [1.0, 1.0]],
        },
        schema={
            "input_id": pl.String,
            "frame_idx": pl.Array(pl.Int32, 2),
            "speed": pl.Array(pl.Float32, 2),
        },
    )

    table = build_tick_table(
        df,
        input_id_column="input_id",
        frame_idx_column="frame_idx",
        value_columns=["speed"],
    )

    # sorted by drive, and the shared frame_idx values do NOT collide
    assert table["input_id"].to_list() == ["a", "a", "b", "b"]
    assert table["speed"].to_list() == [1.0, 1.0, 9.0, 9.0]


def test_horizon_windows_contiguous_run() -> None:
    frames, stamps = _ticks([0, 10, 20, 30, 40])

    windows = horizon_windows(frame_idx=frames, time_stamp_us=stamps, horizon=2)

    assert windows.tolist() == [[0, 1, 2], [1, 2, 3], [2, 3, 4]]


def test_horizon_windows_breaks_on_frame_gap() -> None:
    # 20 -> 40 skips a tick, so no window may span it
    frames, stamps = _ticks([0, 10, 20, 40, 50, 60])
    stamps = np.array([0, TICK_US, 2 * TICK_US, 3 * TICK_US, 4 * TICK_US, 5 * TICK_US])

    windows = horizon_windows(frame_idx=frames, time_stamp_us=stamps, horizon=2)

    assert windows.tolist() == [[0, 1, 2], [3, 4, 5]]


def test_horizon_windows_breaks_on_recording_pause() -> None:
    """A pause keeps `frame_idx` marching, so only the clock can see it."""
    frames = np.array([0, 10, 20, 30, 40, 50])
    pause_us = int(MAX_TICK_GAP_MS * 1000) + 1
    stamps = np.array([
        0,
        TICK_US,
        2 * TICK_US,
        2 * TICK_US + pause_us,
        3 * TICK_US + pause_us,
        4 * TICK_US + pause_us,
    ])

    windows = horizon_windows(frame_idx=frames, time_stamp_us=stamps, horizon=2)

    assert windows.tolist() == [[0, 1, 2], [3, 4, 5]]


def test_horizon_windows_separates_interleaved_phase_grids() -> None:
    """Two phase grids interleave under a global sort; each is contiguous."""
    frames = np.array([0, 3, 10, 13, 20, 23, 30, 33])
    stamps = np.array([
        0,
        0,
        TICK_US,
        TICK_US,
        2 * TICK_US,
        2 * TICK_US,
        3 * TICK_US,
        3 * TICK_US,
    ])

    windows = horizon_windows(frame_idx=frames, time_stamp_us=stamps, horizon=3)

    # phase 0 -> rows 0,2,4,6; phase 3 -> rows 1,3,5,7. A phase-blind walk
    # would see diffs of 3/7 everywhere and return nothing.
    assert windows.tolist() == [[0, 2, 4, 6], [1, 3, 5, 7]]


def test_horizon_windows_too_short_returns_empty() -> None:
    frames, stamps = _ticks([0, 10, 20])

    windows = horizon_windows(frame_idx=frames, time_stamp_us=stamps, horizon=5)

    assert windows.shape == (0, 6)


def test_horizon_windows_rejects_bad_horizon() -> None:
    frames, stamps = _ticks([0, 10, 20])

    with pytest.raises(ValueError, match="horizon must be >= 1"):
        horizon_windows(frame_idx=frames, time_stamp_us=stamps, horizon=0)


def test_horizon_windows_rejects_shape_mismatch() -> None:
    frames, _ = _ticks([0, 10, 20])

    with pytest.raises(ValueError, match="shape mismatch"):
        horizon_windows(
            frame_idx=frames, time_stamp_us=np.zeros(2, dtype=np.int64), horizon=1
        )


# --------------------------------------------- stitch_horizon / drive_split


def _tick_table(n_drives: int = 3, n_ticks: int = 40) -> pl.DataFrame:
    """A synthetic `load_ticks()` table: contiguous ticks, one phase grid."""
    rows = []
    for d in range(n_drives):
        rows.extend(
            {
                "input_id": f"Niro{100 + d}-HQ/2023-01-0{d + 1}--00-00-00",
                "frame_idx": i * STRIDE,
                "time_stamp": 1_670_000_000_000_000 + i * TICK_US,
                "speed": 10.0 + i,
                "heading": float(i),
                "gnss_xy": [float(i), float(-i)],
                "gas_pedal": i / 100.0,
                "brake_pedal": i / 200.0,
                "steering_angle": i / 400.0,
                "turn_signal": i % 3,
            }
            for i in range(n_ticks)
        )
    return pl.DataFrame(rows)


def test_stitch_horizon_action_steps_takes_a_contiguous_action_sequence() -> None:
    """The K action targets must be the actions at anchor+1..anchor+K, in
    order -- the control sequence, not K copies of the next action."""
    ticks = _tick_table(n_drives=1, n_ticks=20)

    anchors = stitch_horizon(ticks, horizon=6, action_steps=4)

    assert anchors["gas_pedal"].shape == (20 - 6, 4)
    # anchor 0 sits at tick 0, so its sequence is ticks 1..4 -> i/100
    np.testing.assert_allclose(
        anchors["gas_pedal"][0], [0.01, 0.02, 0.03, 0.04], rtol=1e-6
    )
    assert anchors["turn_signal"][0].tolist() == [1, 2, 0, 1]


def test_stitch_horizon_action_steps_1_keeps_the_flat_shape() -> None:
    """The single-action case must stay `(n,)`, matching the windowed
    formulation and everything already built on it."""
    ticks = _tick_table(n_drives=1, n_ticks=20)

    anchors = stitch_horizon(ticks, horizon=6, action_steps=1)

    assert anchors["gas_pedal"].shape == (14,)
    np.testing.assert_allclose(
        anchors["gas_pedal"], stitch_horizon(ticks, horizon=6)["gas_pedal"]
    )


def test_stitch_horizon_action_steps_first_step_matches_single_action() -> None:
    """Step 0 of a K-step sequence is exactly the single-action target, so
    the two configurations are directly comparable on the applied action."""
    ticks = _tick_table()

    single = stitch_horizon(ticks, horizon=15, action_steps=1)
    sequence = stitch_horizon(ticks, horizon=15, action_steps=6)

    np.testing.assert_array_equal(single["frame_idx"], sequence["frame_idx"])
    for field in ("gas_pedal", "brake_pedal", "steering_angle", "turn_signal"):
        np.testing.assert_array_equal(single[field], sequence[field][:, 0])


def test_stitch_horizon_rejects_action_steps_past_the_horizon() -> None:
    """An action at anchor+k is only supervised by a trajectory that reaches
    tick k, so K > horizon is a configuration error, not a silent clip."""
    ticks = _tick_table()

    with pytest.raises(
        ValueError, match=re.escape("action_steps must be in 1..horizon")
    ):
        stitch_horizon(ticks, horizon=6, action_steps=7)


def test_stitch_horizon_without_gnss_omits_only_gnss() -> None:
    """`gnss_xy` is QA-only and the largest array here; dropping it must not
    change any other output."""
    ticks = _tick_table()

    with_gnss = stitch_horizon(ticks, horizon=6)
    without = stitch_horizon(ticks, horizon=6, with_gnss=False)

    assert "gnss_xy" in with_gnss
    assert "gnss_xy" not in without
    assert set(with_gnss) - set(without) == {"gnss_xy"}
    for key in without:
        np.testing.assert_array_equal(with_gnss[key], without[key])


def testdrive_split_holds_out_whole_drives() -> None:
    """Consecutive anchors share all but one tick, so a row-level split would
    leak near-duplicates into val. No drive may appear on both sides."""
    input_id = np.repeat([f"drive-{i}" for i in range(10)], 5)

    train_idx, val_idx, n_drives, n_val_drives = drive_split(
        input_id, seed=7, val_frac=0.2
    )

    assert (n_drives, n_val_drives) == (10, 2)
    assert len(train_idx) + len(val_idx) == len(input_id)
    assert not set(input_id[train_idx.numpy()]) & set(input_id[val_idx.numpy()])


def testdrive_split_is_reproducible_at_a_fixed_seed() -> None:
    """`np.unique` sorts before the shuffle, which is what makes `seed`
    actually reproduce -- the polars `.unique()` version of this silently
    did not (see `_sample_drives`)."""
    input_id = np.repeat([f"drive-{i}" for i in range(20)], 3)

    _, first_val, _, _ = drive_split(input_id, seed=7, val_frac=0.1)
    _, second_val, _, _ = drive_split(input_id, seed=7, val_frac=0.1)
    _, other_val, _, _ = drive_split(input_id, seed=8, val_frac=0.1)

    np.testing.assert_array_equal(first_val, second_val)
    assert not np.array_equal(first_val, other_val)
