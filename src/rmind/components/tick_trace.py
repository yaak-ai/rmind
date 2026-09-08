"""Stitch overlapping predict windows back into a per-drive tick trace.

The predict dataset is built with `episode_stride == episode_step == 10`
(`config/_templates/dataset/yaak/train.yaml`, resolved to `every: 10i` /
`gather_every: 10`), so consecutive windows advance by exactly **one tick**
and overlap by 10 of their 11 ticks. A parquet written from it therefore
already carries a drive-complete trace at tick resolution -- the 11-tick clip
is a packaging artifact, not a horizon limit.

That matters because the trajectory ground truth is dead-reckoned from
`speed` + `headings_denoised/heading` (see
`rmind.components.dead_reckoning`), both of which are raw per-tick columns.
Undo the packaging and the horizon becomes a free parameter, with no re-run
of `predict` and no GPU: `build_tick_table` deduplicates the overlap, and
`horizon_windows` enumerates the anchors whose future is genuinely
contiguous.

Verified on `model-4vqatiom:v4.parquet`: across 8 drives / 167k tick
observations at 8.7x redundancy, overlapping windows agree exactly on every
value column, so the deduplication is lossless.
"""

from collections.abc import Iterable

import numpy as np
import polars as pl

TICK_STRIDE_FRAME_IDX = 10
"""`frame_idx` advance between adjacent ticks (`episode_step`)."""

MAX_TICK_GAP_MS = 1000.0
"""Wall-clock gap above which two adjacent ticks are NOT continuous.

Mirrors the dataset's own gate (`max_frame_gap_ms`, default 1000). Needed
because a recording pause keeps incrementing `frame_idx` by 1 across the gap
(see commit 66529d8), so `frame_idx` arithmetic alone cannot see it. Nominal
spacing is ~333ms.
"""


def build_tick_table(
    df: pl.DataFrame,
    *,
    input_id_column: str,
    frame_idx_column: str,
    value_columns: Iterable[str],
) -> pl.DataFrame:
    """Explode windowed rows into one row per `(drive, tick)`, deduplicated.

    Args:
        df: windowed rows; every listed column must be a fixed-size `Array`
            over the clip's time axis.
        input_id_column: the per-drive key.
        frame_idx_column: the tick index, unique within a drive.
        value_columns: array columns to carry through, `frame_idx_column`
            excluded.

    Returns:
        One row per distinct `(input_id, frame_idx)`, sorted by both. Which
        duplicate survives is immaterial -- overlapping windows agree (see
        the module docstring).
    """
    exploded = df.select([input_id_column, frame_idx_column, *value_columns]).explode([
        frame_idx_column,
        *value_columns,
    ])
    return exploded.unique(
        subset=[input_id_column, frame_idx_column], keep="first"
    ).sort([input_id_column, frame_idx_column])


def horizon_windows(
    *,
    frame_idx: np.ndarray,
    time_stamp_us: np.ndarray,
    horizon: int,
    stride: int = TICK_STRIDE_FRAME_IDX,
    max_gap_ms: float = MAX_TICK_GAP_MS,
) -> np.ndarray:
    """Index every anchor in one drive whose next `horizon` ticks are contiguous.

    Filtering drops raw frames, which shifts a clip's start off the `every:
    10i` grid, so a single drive can carry several interleaved phase grids
    (1.67 on average, up to 8, in `model-4vqatiom:v4`). Sorting by
    `frame_idx` interleaves them, and the resulting non-`stride` diffs would
    hide runs that are perfectly contiguous within their own phase -- so each
    phase is walked separately. Ignoring this costs ~70% of the mean run
    length (74.7 vs 245.2 ticks).

    Args:
        frame_idx: `(T,)` tick indices for ONE drive, sorted ascending.
        time_stamp_us: `(T,)` matching timestamps, Unix-epoch microseconds.
        horizon: number of future ticks required after the anchor.
        stride: expected `frame_idx` advance per tick.
        max_gap_ms: reject a step whose wall-clock gap exceeds this.

    Returns:
        `(num_anchors, horizon + 1)` indices into `frame_idx` /
        `time_stamp_us`, column 0 being the anchor ("now") and columns
        `1..horizon` its future. Sorted by anchor.

    Raises:
        ValueError: if `horizon < 1`, or the inputs disagree in shape.
    """
    if horizon < 1:
        msg = f"horizon must be >= 1, got {horizon}"
        raise ValueError(msg)
    if frame_idx.shape != time_stamp_us.shape:
        msg = f"shape mismatch: {frame_idx.shape} vs {time_stamp_us.shape}"
        raise ValueError(msg)

    max_gap_us = max_gap_ms * 1000.0
    offsets = np.arange(horizon + 1)
    found = []
    for phase in np.unique(frame_idx % stride):
        where = np.flatnonzero(frame_idx % stride == phase)
        if where.size <= horizon:
            continue
        d_frame = np.diff(frame_idx[where])
        d_time = np.diff(time_stamp_us[where].astype(np.float64))
        step_ok = (d_frame == stride) & (d_time > 0) & (d_time <= max_gap_us)
        # a run of `horizon` consecutive good steps starting at `s` means
        # ticks s..s+horizon are all reachable; count them with a prefix sum.
        cumulative = np.concatenate([[0], np.cumsum(step_ok)])
        starts = np.arange(where.size - horizon)
        starts = starts[cumulative[starts + horizon] - cumulative[starts] == horizon]
        if starts.size:
            found.append(where[starts[:, None] + offsets])

    if not found:
        return np.empty((0, horizon + 1), dtype=np.intp)
    windows = np.concatenate(found)
    return windows[np.argsort(windows[:, 0], kind="stable")]
