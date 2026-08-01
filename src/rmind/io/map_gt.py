"""Data-pipeline reader for map-GT sidecar parquets (Traffic rules, Arm M).

Sidecars live at ``<map_gt_root>/<Vehicle>/<drive>.parquet`` (see
``rmind.scripts.map_gt.build_sidecar``) and carry per-frame map ground truth
keyed on the ``cam_front_left`` frame index. This reader is the rbyte/pipefunc
entry point that joins them into the training samples: it is deliberately
TOLERANT — a missing sidecar file yields an empty (schema-correct) dataframe,
so the downstream LEFT JOIN produces NULLs which are coalesced to NaN, which
the model's `MaxSpeedTokenizer` maps to its UNKNOWN token.

Contract for ``max_speed_kmh`` (must match ``caches/map_gt`` and
``rmind.components.map_context``): Float32 km/h, NaN = unknown,
-1.0 = explicitly unlimited (German autobahn ``maxspeed=none``).
"""

from pathlib import Path
from typing import final

import polars as pl
from structlog import get_logger

logger = get_logger(__name__)

SCHEMA: dict[str, pl.DataType] = {
    "frame_idx": pl.Int32(),
    "max_speed_kmh": pl.Float32(),
}


@final
class MapGTSidecarReader:
    """Read one per-drive map-GT sidecar parquet for the rbyte sample pipeline.

    Returns a dataframe with columns ``frame_idx`` (Int32, cam_front_left
    index space) and ``max_speed_kmh`` (Float32; NaN unknown, -1 unlimited).
    A missing file returns an EMPTY dataframe with the same schema (drives
    without map GT train with the all-UNKNOWN token, by design).
    """

    __name__ = __qualname__

    def __call__(self, *, map_gt_path: str) -> pl.DataFrame:
        path = Path(map_gt_path)
        if not path.is_file():
            logger.warning(
                "map-GT sidecar missing; max_speed will be NaN (UNKNOWN)",
                path=map_gt_path,
            )
            return pl.DataFrame(schema=SCHEMA)

        return (
            pl.read_parquet(path, columns=list(SCHEMA))
            .cast(SCHEMA)  # ty: ignore[invalid-argument-type]
            # frame_idx must be unique for the LEFT JOIN not to fan out
            .unique(subset="frame_idx", keep="first", maintain_order=True)
        )
