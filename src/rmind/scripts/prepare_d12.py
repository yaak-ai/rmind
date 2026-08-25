"""Build an rbyte sample table from a D12 job: `data.mcap` + per-camera directories.

Job layout (`/nasa/team-space/nikita/data/d12/{job-id}/`):

    data.mcap                            protobuf + zstd, palleter.* schemas
    job-id.txt                           the recording's UUID
    {camera}/video.mp4                   h264 1920x1080, one per camera
    {camera}/frame_info_{camera}.txt     "w,h,codec" then one ns stamp per frame
    {camera}/frames/{W}x{H}/%06d.jpg     what training reads, from
                                         `scripts/extract_d12_frames.sh`

Job directories are named by datetime (`2026-08-25--13-44-11`), which is what the
sample table records as `input_id`.

Actuations come from the mcap and are already in convenient units:

    traction   linde/traction.traction_pct            percent -> /100 -> [-1, 1]
    steering   lindelot/adc.steering_angle_normalised  already [-1, 1]
    fork1      linde/fork.fork1_pct                    percent -> /100 -> [-1, 1]
    speed      lindelot/vehicle_state.speed            m/s, for the speed token

CLOCKS -- the one thing to understand here. Three timebases are in play:

  * mcap `log_time`, which every signal shares. This is the join clock.
  * `frame_info` stamps, which are a DIFFERENT epoch (~160 s offset on the job
    this was written against) AND synthesized at exactly nominal fps, so they
    carry no real per-frame timing.
  * the video's own PTS, likewise rewritten to exactly nominal fps by the muxer.

So neither the video nor `frame_info` can say which real capture an encoded frame
is. The bridge is the mcap's own `cam_{camera}/frame` topic, which pairs a real
`log_time` with each published frame -- but there are typically MORE of those than
encoded frames (13 more, on the reference job), because captures get published and
then dropped before encoding. Anchoring at the first frame and at the last
disagreed by exactly those 13 frames, so the drops are not all at one end.

`--pairing index` therefore pairs encoded frame i with the i-th published frame,
which is exact iff every drop is at the tail. `_pairing_skew` measures and reports
the disagreement so a malformed job is visible rather than silently misaligned;
anything beyond `--max-skew-s` is refused. The real fix belongs upstream: have the
recorder stamp encoded frames from the capture clock, and this whole problem goes
away.
"""

import argparse
import json
from pathlib import Path
from typing import Final

import polars as pl
from structlog import get_logger

logger = get_logger(__name__)

CAMERAS: Final = ("cam_fork", "cam_left_forward", "cam_right_forward")

# a skew needs at least two frames to be measurable
MIN_FRAMES_FOR_SKEW: Final = 2

# actuation name -> (mcap topic, field, scale). Scale brings percent to [-1, 1].
SIGNALS: Final = {
    "traction": ("linde/traction", "traction_pct", 0.01),
    "steering": ("lindelot/adc", "steering_angle_normalised", 1.0),
    "fork1": ("linde/fork", "fork1_pct", 0.01),
    "speed": ("lindelot/vehicle_state", "speed", 1.0),
}

# alternative traction sources, selectable with --traction-source
TRACTION_SOURCES: Final = {
    "reported": ("linde/traction", "traction_pct"),
    "applied": ("lindelot/applied", "traction_pct"),
    "command": ("lindelot/vehicle_state", "traction_command_pct"),
}


def encoded_frame_count(camera_dir: Path, camera: str) -> int:
    """Number of encoded frames, from `frame_info` (one stamp per encoded frame)."""
    info = (camera_dir / f"frame_info_{camera}.txt").read_text().split()

    return len(info) - 1  # first token is the "w,h,codec" header


def read_mcap(
    path: Path, cameras: tuple[str, ...], traction_source: str
) -> dict[str, pl.DataFrame]:
    """Decode the signal and per-camera frame topics into polars frames."""
    from rbyte.samples.mcap import McapReader, ProtobufDecoderFactory  # noqa: PLC0415

    topic, field = TRACTION_SOURCES[traction_source]
    fields: dict[str, dict[str, pl.DataType | None]] = {
        f"cam_{c.removeprefix('cam_')}/frame": {
            "log_time": pl.Datetime("ns"),
            "pts_ns": pl.Int64(),
        }
        for c in cameras
    }
    for name, (sig_topic, sig_field, _) in SIGNALS.items():
        chosen = (topic, field) if name == "traction" else (sig_topic, sig_field)
        fields.setdefault(chosen[0], {"log_time": pl.Datetime("ns")})[chosen[1]] = (
            pl.Float32()
        )

    return McapReader(decoder_factories=[ProtobufDecoderFactory], fields=fields)(path)


def _pairing_skew(frames: pl.DataFrame, encoded: int) -> float:
    """Seconds of disagreement between anchoring at the first vs the last frame.

    Zero when the published and encoded frame counts match. Otherwise it is the
    duration of the surplus published frames, i.e. the worst-case misalignment
    `--pairing index` can introduce.
    """
    published = len(frames)
    if published <= encoded or encoded < MIN_FRAMES_FOR_SKEW:
        return 0.0

    pts = frames["pts_ns"].to_list()

    return (pts[published - 1] - pts[encoded - 1]) / 1e9


def build(
    job_dir: Path,
    *,
    cameras: tuple[str, ...],
    reference: str,
    traction_source: str,
    max_skew_s: float,
) -> pl.DataFrame:
    """One row per reference-camera frame: a frame index per camera plus signals.

    Raises:
        ValueError: if a camera is missing, or its pairing skew exceeds `max_skew_s`.
    """
    topics = read_mcap(job_dir / "data.mcap", cameras, traction_source)

    per_camera: dict[str, pl.DataFrame] = {}
    for camera in cameras:
        camera_dir = job_dir / camera
        if not (camera_dir / "video.mp4").is_file():
            msg = f"{camera}: no video.mp4 in {camera_dir}"
            raise ValueError(msg)

        frames = topics[f"cam_{camera.removeprefix('cam_')}/frame"].sort("log_time")
        encoded = encoded_frame_count(camera_dir, camera)
        skew = _pairing_skew(frames, encoded)
        logger.info(
            "camera",
            camera=camera,
            published=len(frames),
            encoded=encoded,
            pairing_skew_s=round(skew, 4),
        )
        if abs(skew) > max_skew_s:
            msg = (
                f"{camera}: pairing skew {skew:.3f}s exceeds --max-skew-s "
                f"{max_skew_s}; {len(frames)} published vs {encoded} encoded frames, "
                "so frame-to-timestamp pairing is not trustworthy for this job"
            )
            raise ValueError(msg)

        # pair encoded frame i with the i-th published frame (see module docstring)
        per_camera[camera] = (
            frames
            .head(encoded)
            .with_row_index(f"{camera}/frame_idx")
            .select(pl.col(f"{camera}/frame_idx").cast(pl.Int32), pl.col("log_time"))
        )

    table = per_camera[reference].rename({f"{reference}/frame_idx": "frame_idx"})

    # nearest-in-time frame of every other camera, and the signals
    for camera, frames in per_camera.items():
        if camera != reference:
            table = table.join_asof(frames, on="log_time", strategy="nearest")

    for name, (sig_topic, sig_field, scale) in SIGNALS.items():
        topic, field = (
            TRACTION_SOURCES[traction_source]
            if name == "traction"
            else (sig_topic, sig_field)
        )
        series = (
            topics[topic]
            .sort("log_time")
            .select("log_time", pl.col(field).alias(name) * scale)
        )
        table = table.join_asof(series, on="log_time", strategy="nearest")

    return table.rename({"frame_idx": f"{reference}/frame_idx"})


def episodes(  # noqa: PLR0913
    table: pl.DataFrame,
    *,
    job_id: str,
    cameras: tuple[str, ...],
    episode_length: int,
    stride: int,
    every_nth: int,
) -> pl.DataFrame:
    """Sliding windows over the (decimated) reference timeline.

    Raises:
        ValueError: if the job is too short for one episode.
    """
    rows = table.gather_every(every_nth)
    if len(rows) < episode_length:
        msg = f"{len(rows)} usable frames < episode_length {episode_length}"
        raise ValueError(msg)

    columns = [f"{c}/frame_idx" for c in cameras] + list(SIGNALS)
    starts = range(0, len(rows) - episode_length + 1, stride)
    windows = {
        column: [
            rows[column][start : start + episode_length].to_list() for start in starts
        ]
        for column in columns
    }

    return pl.DataFrame(
        {"input_id": [job_id] * len(starts)} | windows,
        schema={"input_id": pl.String}
        | {
            column: pl.Array(
                pl.Int32 if column.endswith("frame_idx") else pl.Float32, episode_length
            )
            for column in columns
        },
    )


def main(  # noqa: PLR0913
    job_dir: Path,
    output_dir: Path,
    *,
    cameras: tuple[str, ...],
    reference: str,
    traction_source: str,
    episode_length: int,
    stride: int,
    every_nth: int,
    max_skew_s: float,
) -> None:
    table = build(
        job_dir,
        cameras=cameras,
        reference=reference,
        traction_source=traction_source,
        max_skew_s=max_skew_s,
    )
    samples = episodes(
        table,
        job_id=job_dir.name,
        cameras=cameras,
        episode_length=episode_length,
        stride=stride,
        every_nth=every_nth,
    )

    # one directory per job, which is how the dataset config addresses them
    job_out = output_dir / job_dir.name
    job_out.mkdir(parents=True, exist_ok=True)
    samples.write_parquet(job_out / "samples.parquet")
    (job_out / "job.json").write_text(
        json.dumps(
            {
                "job_dir": job_dir.as_posix(),
                "cameras": list(cameras),
                "reference": reference,
                "traction_source": traction_source,
                "episode_length": episode_length,
            },
            indent=2,
        )
    )

    described = table.select(list(SIGNALS)).describe()
    logger.info(
        "prepared d12 job",
        output_dir=job_out.resolve().as_posix(),
        frames=len(table),
        episodes=len(samples),
    )
    logger.info("signal ranges", stats=described.to_dicts())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/palletjack/d12"))
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=None,
        help="default: every subdirectory of the job holding a video.mp4",
    )
    parser.add_argument(
        "--reference",
        default="cam_fork",
        help="camera whose frames define the episode timeline",
    )
    parser.add_argument(
        "--traction-source", choices=sorted(TRACTION_SOURCES), default="command"
    )
    parser.add_argument("--episode-length", type=int, default=6)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--every-nth",
        type=int,
        default=3,
        help="reference-frame decimation (30 -> 10Hz)",
    )
    parser.add_argument(
        "--max-skew-s",
        type=float,
        default=1.0,
        help="refuse the job if frame pairing could be off by more than this",
    )
    args = parser.parse_args()

    cameras = (
        tuple(args.cameras)
        if args.cameras
        # jobs do not all carry the same camera set
        else tuple(sorted(p.parent.name for p in args.job_dir.glob("*/video.mp4")))
    )

    main(
        args.job_dir,
        args.output_dir,
        cameras=cameras,
        reference=args.reference,
        traction_source=args.traction_source,
        episode_length=args.episode_length,
        stride=args.stride,
        every_nth=args.every_nth,
        max_skew_s=args.max_skew_s,
    )
