"""Wall-clock gap detection in DuckDB clip-window filtering.

`config/_templates/dataset/yaak/*.yaml` builds clip windows with
`rbyte.io.DataFrameGroupByDynamic` -> `rbyte.io.DuckDBDataFrameQuery`
(`samples_cast`). The WHERE clause historically only checked `frame_idx`
contiguity (`len(...) == clip_length` and
`list_last(frame_idx) - list_first(frame_idx) == (clip_length - 1) * step`),
which cannot detect a wall-clock recording pause: on
Niro102-HQ/2023-04-29--10-02-20, `frame_idx` steps 267->268 normally while
`time_stamp` jumps 173.8s (a pause keeps incrementing `frame_idx` by 1 per
raw frame even while paused). This test isolates the new gap-check condition
added to those WHERE clauses -- the `AND list_max(list_transform(...))`
block in e.g. `config/_templates/dataset/yaak/train_3cam.yaml` -- and proves
it drops exactly the clip containing an oversized time_stamp gap while
keeping a clean clip, invoking `rbyte.io.DuckDBDataFrameQuery` the same way
the production templates do (a `pl.DataFrame` registered as `samples`,
`FROM samples WHERE ...`).

`DataFrameGroupByDynamic` itself is not exercised here: its output is just a
`pl.DataFrame` where every column has been turned into a `List(...)` column
via `.group_by_dynamic(...).agg(pl.all())`, so a hand-built two-row
`pl.DataFrame` with the right List-typed schema is a faithful, much cheaper
stand-in for what `samples_cast`'s WHERE clause actually sees.
"""

from datetime import datetime, timedelta

import polars as pl
import pytest
from rbyte.io import DuckDBDataFrameQuery

CLIP_LENGTH = 6
FRAME_PERIOD = timedelta(milliseconds=333)  # ~30fps nominal camera cadence
RECORDING_PAUSE = timedelta(seconds=173.8)  # observed on Niro102-HQ/2023-04-29--10-02-20
MAX_FRAME_GAP_MS = 1000  # matches the templates' `oc.select:max_frame_gap_ms,1000` default

# Isolated gap-check condition, copied verbatim from the `AND list_max(...)`
# block added to samples_cast's WHERE clause (see e.g. train_3cam.yaml) --
# deliberately excludes the frame_idx-stride and gas/brake checks so this
# test only exercises the new logic.
GAP_CHECK_QUERY = f"""
    SELECT
        *
    FROM
        samples
    WHERE
        list_max(
            list_transform(
                list_zip(
                    "meta/ImageMetadata.cam_front_left/time_stamp",
                    "meta/ImageMetadata.cam_front_left/time_stamp"[2:]
                ),
                x -> date_diff('millisecond', x[1], x[2])
            )
        ) <= {MAX_FRAME_GAP_MS}
"""

FRAME_IDX_SCHEMA = pl.List(pl.Int32)
TIME_STAMP_SCHEMA = pl.List(pl.Datetime("us"))


def _timestamps(*, gap_after: int | None) -> list[datetime]:
    """Build CLIP_LENGTH nominally-spaced timestamps, optionally inserting a
    RECORDING_PAUSE-sized gap after index `gap_after` (0-based) instead of
    one FRAME_PERIOD step.
    """
    start = datetime(2023, 4, 29, 10, 2, 20)
    timestamps = []
    elapsed = timedelta(0)
    for i in range(CLIP_LENGTH):
        if i > 0:
            elapsed += RECORDING_PAUSE if i - 1 == gap_after else FRAME_PERIOD
        timestamps.append(start + elapsed)
    return timestamps


def test_clean_clip_survives_and_gapped_clip_is_dropped() -> None:
    clean = _timestamps(gap_after=None)
    gapped = _timestamps(gap_after=2)  # pause inserted between samples 2 and 3

    # first frame_idx of each clip stands in for a row identifier: unique per
    # row, untouched by the gap-check column, and mirrors what a real clip's
    # frame_idx would carry -- there is no natural "input_id" at this stage
    # of the pipeline, so asserting on the surviving frame_idx values is the
    # right identifying check here.
    samples = pl.DataFrame(
        {
            "meta/ImageMetadata.cam_front_left/frame_idx": [
                list(range(CLIP_LENGTH)),
                list(range(100, 100 + CLIP_LENGTH)),
            ],
            "meta/ImageMetadata.cam_front_left/time_stamp": [clean, gapped],
        },
        schema={
            "meta/ImageMetadata.cam_front_left/frame_idx": FRAME_IDX_SCHEMA,
            "meta/ImageMetadata.cam_front_left/time_stamp": TIME_STAMP_SCHEMA,
        },
    )

    query = DuckDBDataFrameQuery(query=GAP_CHECK_QUERY)
    result = query(samples=samples)

    assert result.height == 1
    surviving_frame_idx = result["meta/ImageMetadata.cam_front_left/frame_idx"][
        0
    ].to_list()
    assert surviving_frame_idx == list(range(CLIP_LENGTH))


@pytest.mark.parametrize("gap_position", range(CLIP_LENGTH - 1))
def test_gap_check_is_insensitive_to_gap_position(gap_position: int) -> None:
    """The gap-check must catch a pause regardless of where in the window it
    falls, not just adjacent to a convenient boundary.
    """
    gapped = _timestamps(gap_after=gap_position)
    samples = pl.DataFrame(
        {
            "meta/ImageMetadata.cam_front_left/frame_idx": [list(range(CLIP_LENGTH))],
            "meta/ImageMetadata.cam_front_left/time_stamp": [gapped],
        },
        schema={
            "meta/ImageMetadata.cam_front_left/frame_idx": FRAME_IDX_SCHEMA,
            "meta/ImageMetadata.cam_front_left/time_stamp": TIME_STAMP_SCHEMA,
        },
    )

    result = DuckDBDataFrameQuery(query=GAP_CHECK_QUERY)(samples=samples)
    assert result.height == 0
