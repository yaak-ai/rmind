"""Precompute a D12 job's dropoff target and write it beside `data.mcap`.

The intention the policy is conditioned on has two parts (see `rmind.data.d12`):
the live pallet position, which is in the mcap already, and the DROPOFF - where the
pallet is meant to end up. The dropoff is not a sensor reading, so it is computed
once, here, and stored rather than derived on the fly at load time. Keeping it in a
file also means it can be edited or made time-varying without touching code.

Output, next to the recording:

    {job-dir}/dropoff.parquet     log_time (ns) | x_m (f32) | y_m (f32)

world-frame pozyx metres, keyed by time. `rmind.data.d12` asof-joins it onto the
frame timeline with a BACKWARD strategy, so a row applies from its `log_time`
forward until the next row: one row is a constant goal for the whole job, several
rows are a goal that changes partway. The dataset config rotates the target into
the vehicle frame at load time.

The default rule computed here is the one in the brief - the dropoff is the pallet's
LAST recorded position - stamped at the job's first pallet fix so it covers every
frame. To use a different or time-varying target, overwrite this file with your own
(same schema); nothing else needs to change.

    uv run python -m rmind.scripts.prepare_dropoff --job-dir /nasa/team-space/nikita/data/d12/<job-id>
"""

import argparse
from pathlib import Path
from typing import Final

import polars as pl
from structlog import get_logger

logger = get_logger(__name__)

TAG_TOPIC: Final = "pozyx/tag"
PALLET_LABEL: Final = "pallet"
DROPOFF_FILE: Final = "dropoff.parquet"


def pallet_fixes(mcap_path: Path) -> pl.DataFrame:
    """Every pallet tag fix in the job, world-frame metres, sorted by time."""
    from rbyte.samples.mcap import McapReader, ProtobufDecoderFactory  # noqa: PLC0415

    fields: dict[str, dict[str, pl.DataType | None]] = {
        TAG_TOPIC: {
            "log_time": pl.Datetime("ns"),
            "label": pl.String(),
            "x_mm": pl.Int64(),
            "y_mm": pl.Int64(),
        }
    }
    topics = McapReader(decoder_factories=[ProtobufDecoderFactory], fields=fields)(
        mcap_path
    )

    return (
        topics[TAG_TOPIC]
        .filter(pl.col("label") == PALLET_LABEL)
        .sort("log_time")
        .select(
            "log_time",
            (pl.col("x_mm") / 1000).cast(pl.Float32).alias("x_m"),
            (pl.col("y_mm") / 1000).cast(pl.Float32).alias("y_m"),
        )
    )


def dropoff(fixes: pl.DataFrame) -> pl.DataFrame:
    """The default target: the pallet's last position, stamped at the first fix.

    Stamping at the first fix (rather than the last) is what makes the single row
    apply to the whole job under the backward-asof join the loader uses.

    Raises:
        ValueError: if there are no pallet fixes to define a dropoff from.
    """
    if fixes.is_empty():
        msg = "no pallet tag fixes: cannot define a dropoff"
        raise ValueError(msg)

    last = fixes.tail(1)

    return pl.DataFrame(
        {"log_time": fixes["log_time"][:1], "x_m": last["x_m"], "y_m": last["y_m"]},
        schema={"log_time": pl.Datetime("ns"), "x_m": pl.Float32, "y_m": pl.Float32},
    )


def main(job_dir: Path) -> None:
    fixes = pallet_fixes(job_dir / "data.mcap")
    target = dropoff(fixes)
    out = job_dir / DROPOFF_FILE
    target.write_parquet(out)

    logger.info(
        "wrote dropoff",
        job=job_dir.name,
        path=out.resolve().as_posix(),
        pallet_fixes=len(fixes),
        target=target.select("x_m", "y_m").row(0),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path, required=True)
    args = parser.parse_args()

    main(args.job_dir)
