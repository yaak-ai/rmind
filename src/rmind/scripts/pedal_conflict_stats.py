"""Counts episodes with simultaneous non-zero brake + gas pedal signals.

A frame where both gas_pedal_normalized > GAS_THRESH and
brake_pedal_normalized > BRAKE_THRESH simultaneously is physically
implausible and indicates sensor noise. Any 6-frame episode/sample
that contains such a frame should be dropped from training.

Proposed Stage-8 filter to add in all dataset configs:

    AND len(list_filter(
        list_zip(
            "meta/VehicleMotion/gas_pedal_normalized",
            "meta/VehicleMotion/brake_pedal_normalized"
        ),
        x -> x[1] > (1.0 / 255 + 0.001)
             AND x[2] > (1.0 / 164 + 0.001)
    )) = 0

Usage:
    uv run python -m rmind.scripts.pedal_conflict_stats           # all train+val drives
    uv run python -m rmind.scripts.pedal_conflict_stats --debug   # 3-drive debug set
"""

from __future__ import annotations

import argparse
import concurrent.futures
import re
from pathlib import Path

import polars as pl
from rbyte.io import YaakMetadataDataFrameBuilder
from rbyte.io.yaak.proto import can_pb2, sensor_pb2

DATA_ROOT = Path("/nasa/drives/yaak/data")
CONFIG_ROOT = Path(__file__).parents[4] / "rmind-main/config/_templates/dataset/yaak"

# One quantisation step above zero — matches existing idle-filter thresholds
GAS_THRESH: float = 1.0 / 255 + 0.001  # ≈ 0.0049
BRAKE_THRESH: float = 1.0 / 164 + 0.001  # ≈ 0.0071

BUILDER = YaakMetadataDataFrameBuilder(
    fields={  # ty:ignore[invalid-argument-type]
        can_pb2.VehicleMotion: {
            "time_stamp": pl.Datetime(time_unit="us"),
            "speed": pl.Float32(),
            "gas_pedal_normalized": pl.Float32(),
            "brake_pedal_normalized": pl.Float32(),
            "gear": pl.Enum(["0", "1", "2", "3"]),
        },
        sensor_pb2.ImageMetadata: {
            "time_stamp": pl.Datetime(time_unit="us"),
            "frame_idx": pl.Int32(),
            "camera_name": pl.Enum([
                "cam_front_left",
                "cam_front_right",
                "cam_front_center",
                "cam_left_forward",
                "cam_right_forward",
                "cam_left_backward",
                "cam_right_backward",
                "cam_rear",
            ]),
        },
    }
)


def _parse_drive_list(config_path: Path) -> list[str]:
    text = config_path.read_text(encoding="utf-8")
    section = re.search(r"#@ drives = \[(.*?)#@ \]", text, re.DOTALL)
    if section is None:
        return []
    return re.findall(
        r"'([A-Za-z0-9_-]+/\d{4}-\d{2}-\d{2}--\d{2}-\d{2}-\d{2})'", section.group(1)
    )


def _load_drive(drive_id: str) -> pl.DataFrame | None:
    path = DATA_ROOT / drive_id / "metadata.log"
    if not path.exists():
        return None
    dfs = BUILDER(path)

    vm = dfs.get("VehicleMotion")
    im = dfs.get("ImageMetadata.cam_front_left")
    if vm is None or vm.is_empty() or im is None or im.is_empty():
        return None

    # Align VehicleMotion to camera-frame timestamps via nearest interpolation,
    # matching Stage-4 DataFrameAligner behaviour closely enough for counting.
    vm_sorted = vm.sort("time_stamp")
    im_sorted = im.sort("time_stamp")

    aligned = im_sorted.join_asof(
        vm_sorted.select(
            "time_stamp",
            "speed",
            "gas_pedal_normalized",
            "brake_pedal_normalized",
            "gear",
        ),
        on="time_stamp",
        strategy="nearest",
        tolerance=200_000,  # 200 ms in microseconds
    ).drop_nulls()

    return aligned.with_columns(pl.lit(drive_id).alias("drive_id"))


def _quality_filter(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(
        (pl.col("gear") == "3")
        & pl.col("speed").is_between(0.0, 130.0)
        & pl.col("gas_pedal_normalized").is_between(0.0, 1.0)
        & pl.col("brake_pedal_normalized").is_between(0.0, 1.0)
    )


def _mark_conflicts(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        (
            (pl.col("gas_pedal_normalized") > GAS_THRESH)
            & (pl.col("brake_pedal_normalized") > BRAKE_THRESH)
        ).alias("conflict")
    )


def _episode_conflict_rate(df: pl.DataFrame, window: int = 6) -> tuple[int, int]:
    """Estimate how many sliding 6-frame episodes contain at least one conflict frame.

    The rbyte Stage-7 grouping creates windows of `window` consecutive camera
    frames. A window is flagged if any of its frames has a conflict.
    Returns (total_windows, flagged_windows).
    """
    conflicts = df.sort(["drive_id", "time_stamp"])["conflict"].to_list()
    total = max(0, len(conflicts) - window + 1)
    flagged = sum(any(conflicts[i : i + window]) for i in range(total))
    return total, flagged


def _process_drive(args: tuple[str, str]) -> dict | None:
    split, drive_id = args
    df = _load_drive(drive_id)
    if df is None:
        return None
    df = _quality_filter(df)
    df = _mark_conflicts(df)
    total_frames = len(df)
    conflict_frames = int(df["conflict"].sum())
    total_ep, conflict_ep = _episode_conflict_rate(df)
    return {
        "split": split,
        "drive_id": drive_id,
        "frames": total_frames,
        "conflict_frames": conflict_frames,
        "has_conflict": conflict_frames > 0,
        "estimated_episodes": total_ep,
        "estimated_conflict_episodes": conflict_ep,
    }


def main(splits: dict[str, list[str]], workers: int = 8) -> None:
    rows: list[dict] = []
    missing: list[str] = []

    all_tasks = [
        (split, drive_id) for split, drives in splits.items() for drive_id in drives
    ]
    total = len(all_tasks)

    print(f"\nProcessing {total} drives with {workers} workers…")  # noqa: T201
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (args, result) in enumerate(
            zip(
                all_tasks,
                pool.map(_process_drive, all_tasks, chunksize=4),
                strict=False,
            ),
            start=1,
        ):
            split, drive_id = args
            if result is None:
                missing.append(drive_id)
            else:
                rows.append(result)
            if i % 50 == 0 or i == total:
                print(f"  {i}/{total} processed, {len(missing)} missing so far")  # noqa: T201

    stats = pl.DataFrame(rows)

    print(f"\n{'=' * 70}")  # noqa: T201
    print("PEDAL CONFLICT STATISTICS")  # noqa: T201
    print(f"  Gas threshold  : > {GAS_THRESH:.4f}  (1 bin above zero in 255-bin space)")  # noqa: T201
    print(f"{'=' * 70}")  # noqa: T201

    for split in splits:
        s = stats.filter(pl.col("split") == split)
        n_drives = s.height
        n_conflict_drives = int(s["has_conflict"].sum())
        total_frames = int(s["frames"].sum())
        int(s["conflict_frames"].sum())
        total_ep = int(s["estimated_episodes"].sum())
        int(s["estimated_conflict_episodes"].sum())

        print(f"\n[{split}]")  # noqa: T201
        print(f"  Drives with any conflict : {n_conflict_drives:>5} / {n_drives}")  # noqa: T201
        print(f"  Frames (quality-filtered): {total_frames:>10,}")  # noqa: T201
        print(f"  Estimated episodes       : {total_ep:>10,}")  # noqa: T201

    print(f"\n{'Per-drive breakdown (conflict drives only)'!s}")  # noqa: T201
    conflicting = (
        stats
        .filter(pl.col("has_conflict"))
        .select(
            "split",
            "drive_id",
            "conflict_frames",
            "frames",
            "estimated_conflict_episodes",
            "estimated_episodes",
        )
        .sort(["split", "conflict_frames"], descending=[False, True])
    )
    print(conflicting)  # noqa: T201


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--debug", action="store_true", help="Run on 3-drive debug set only"
    )
    args = parser.parse_args()

    if args.debug:
        splits = {"debug": _parse_drive_list(CONFIG_ROOT / "train_debug.yaml")}
    else:
        splits = {
            "train": _parse_drive_list(CONFIG_ROOT / "train.yaml"),
            "val": _parse_drive_list(CONFIG_ROOT / "val.yaml"),
        }

    main(splits)
