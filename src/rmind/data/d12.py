"""Build an rbyte sample table from a D12 job: `data.mcap` + per-camera directories.

`D12SampleBuilder` is called from `config/_templates/dataset/palletjack/d12.yaml`
as a pipefunc stage: one call per job directory returns that job's episodes, and
the pipeline concatenates them. There is no parquet on disk - the samples are
built at load time, so adding a signal is a config change and nothing can go
stale. The clock reasoning below is the whole reason this is Python and not a
DuckDB query in the template.

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

Observations (model inputs, not targets):

    speed                 lindelot/vehicle_state.speed             m/s
    fork_above_300        lindelot/vehicle_state.fork_above_300    bool -> 0.0 / 1.0
    relative_ego_pos      pozyx pose + pallet tag       ego-frame [fwd, lat] metres
    relative_dropoff_pos  pozyx pose + last pallet tag  ego-frame [fwd, lat] metres

The two `relative_*` columns are the intention (see the constants block for the
frame and why they are metres, not normalized).

`fork_above_300` is a height switch on the mast: whether the fork is raised past
300 mm. It is recorded together with a `_valid` companion, which is False for the
first second or so of a job before the ECU has answered - those rows are dropped
rather than passed off as a confident False, since "not lifted" and "don't know
yet" are different states and the model would learn the wrong thing from the
conflation. `fork_above_1300` exists in the same message but is constant on the
jobs recorded so far, so it is not carried.

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

from pathlib import Path
from typing import Final, NamedTuple, final

import polars as pl
from pydantic import validate_call
from structlog import get_logger

logger = get_logger(__name__)

CAMERAS: Final = ("cam_fork", "cam_left_forward", "cam_right_forward")

# a skew needs at least two frames to be measurable
MIN_FRAMES_FOR_SKEW: Final = 2


class Signal(NamedTuple):
    """One scalar column of the sample table, and where it comes from.

    Everything lands as float32 whatever the wire type, because that is what the
    model's continuous inputs and Gaussian heads take.
    """

    topic: str
    field: str
    scale: float = 1.0
    dtype: pl.DataType = pl.Float32()
    # companion boolean field that says the reading is meaningful; rows where it
    # is False are dropped rather than trusted
    valid_field: str | None = None


# column name -> source. Scale brings percent to [-1, 1].
SIGNALS: Final = {
    "traction": Signal("linde/traction", "traction_pct", 0.01),
    "steering": Signal("lindelot/adc", "steering_angle_normalised"),
    "fork1": Signal("linde/fork", "fork1_pct", 0.01),
    "speed": Signal("lindelot/vehicle_state", "speed"),
    "fork_above_300": Signal(
        "lindelot/vehicle_state",
        "fork_above_300",
        dtype=pl.Boolean(),
        valid_field="fork_above_300_valid",
    ),
}

# alternative traction sources, selectable with --traction-source
TRACTION_SOURCES: Final = {
    "reported": ("linde/traction", "traction_pct"),
    "applied": ("lindelot/applied", "traction_pct"),
    "command": ("lindelot/vehicle_state", "traction_command_pct"),
}

# --- intention: two ego-frame position tokens -----------------------------------
#
# The pozyx indoor system carries both the vehicle pose and the pallet position in
# ONE world frame (gnss is all-zero indoors and unused). `relative_ego_pos` points
# the model at the pallet to pick up; `relative_dropoff_pos` at where to leave it -
# the dropoff, defined as the LAST pallet position in the job. Both are expressed
# in the vehicle's own frame (translate by -ego, rotate by -heading), so they read
# as "target is X metres ahead, Y metres left" independent of world orientation,
# and both are held in METRES here: the /scale + clamp that maps them to ~[-1, 1]
# lives in the model embedding, exactly as speed's binner does, so the transform is
# one graph and cannot drift between training and the kit.
#
# These are inputs, never targets - the analogue of the car model's waypoints.
POSE_TOPIC: Final = "pozyx/pose"
TAG_TOPIC: Final = "pozyx/tag"
PALLET_LABEL: Final = "pallet"
POSITION_DIM: Final = 2  # (forward, lateral); z/height is `fork_above_300`'s job

# column name -> the point it is measured to, relative to the vehicle
POSITIONS: Final = ("relative_ego_pos", "relative_dropoff_pos")


def resolve_signal(name: str, traction_source: str) -> Signal:
    """`SIGNALS[name]`, with traction redirected to the selected source."""
    signal = SIGNALS[name]
    if name != "traction":
        return signal

    topic, field = TRACTION_SOURCES[traction_source]

    return signal._replace(topic=topic, field=field)


def encoded_frame_count(camera_dir: Path, camera: str) -> int:
    """Number of encoded frames, from `frame_info` (one stamp per encoded frame)."""
    info = (camera_dir / f"frame_info_{camera}.txt").read_text().split()

    return len(info) - 1  # first token is the "w,h,codec" header


def read_mcap(
    path: Path, cameras: tuple[str, ...], traction_source: str
) -> dict[str, pl.DataFrame]:
    """Decode the signal and per-camera frame topics into polars frames."""
    from rbyte.samples.mcap import McapReader, ProtobufDecoderFactory  # noqa: PLC0415

    fields: dict[str, dict[str, pl.DataType | None]] = {
        f"cam_{c.removeprefix('cam_')}/frame": {
            "log_time": pl.Datetime("ns"),
            "pts_ns": pl.Int64(),
        }
        for c in cameras
    }
    for name in SIGNALS:
        signal = resolve_signal(name, traction_source)
        topic = fields.setdefault(signal.topic, {"log_time": pl.Datetime("ns")})
        topic[signal.field] = signal.dtype
        if signal.valid_field is not None:
            topic[signal.valid_field] = pl.Boolean()

    # ego pose and every tag fix (filtered to the pallet later), for the
    # intention tokens
    fields[POSE_TOPIC] = {
        "log_time": pl.Datetime("ns"),
        "center_x_m": pl.Float32(),
        "center_y_m": pl.Float32(),
        "heading_deg": pl.Float32(),
    }
    fields[TAG_TOPIC] = {
        "log_time": pl.Datetime("ns"),
        "label": pl.String(),
        "x_mm": pl.Int64(),
        "y_mm": pl.Int64(),
    }

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

    validity: list[str] = []
    for name in SIGNALS:
        signal = resolve_signal(name, traction_source)
        columns = [(pl.col(signal.field).cast(pl.Float32) * signal.scale).alias(name)]
        if signal.valid_field is not None:
            validity.append(valid := f"{name}/valid")
            columns.append(pl.col(signal.valid_field).alias(valid))

        series = topics[signal.topic].sort("log_time").select("log_time", *columns)
        table = table.join_asof(series, on="log_time", strategy="nearest")

    table = _relative_positions(table, topics)

    return _drop_invalid(table, validity).rename({
        "frame_idx": f"{reference}/frame_idx"
    })


def _ego_frame(target_x: pl.Expr, target_y: pl.Expr, ego: str) -> pl.Expr:
    """`target - ego`, rotated into the ego frame, as a fixed-size `[fwd, lat]`.

    `fwd` is the component along the vehicle's heading, `lat` the one 90 degrees to
    its left; the exact sign convention does not matter to the model as long as it
    is the same at training and inference, which it is because this is the only
    place it is computed. Held in metres - the model embedding scales and clamps.
    """
    theta = pl.col(f"{ego}_heading").radians()
    dx = target_x - pl.col(f"{ego}_x")
    dy = target_y - pl.col(f"{ego}_y")
    fwd = theta.cos() * dx + theta.sin() * dy
    lat = -theta.sin() * dx + theta.cos() * dy

    return pl.concat_list(fwd, lat).cast(pl.Array(pl.Float32, POSITION_DIM))


def _relative_positions(
    table: pl.DataFrame, topics: dict[str, pl.DataFrame]
) -> pl.DataFrame:
    """Add the two ego-frame intention columns to the reference-timeline table.

    Raises:
        ValueError: if the job carries no pallet tag fix (so no dropoff exists).
    """
    ego = (
        topics[POSE_TOPIC]
        .sort("log_time")
        .select(
            "log_time",
            pl.col("center_x_m").alias("ego_x"),
            pl.col("center_y_m").alias("ego_y"),
            pl.col("heading_deg").alias("ego_heading"),
        )
    )
    pallet = (
        topics[TAG_TOPIC]
        .filter(pl.col("label") == PALLET_LABEL)
        .sort("log_time")
        .select(
            "log_time",
            (pl.col("x_mm") / 1000).alias("pallet_x"),
            (pl.col("y_mm") / 1000).alias("pallet_y"),
        )
    )
    if pallet.is_empty():
        msg = f"no {PALLET_LABEL!r} tag fixes in the job: cannot define a dropoff"
        raise ValueError(msg)

    # the dropoff is where the pallet ends up, over the WHOLE job - taken before any
    # decimation or windowing so it is the true last fix, not the last kept row
    dropoff_x, dropoff_y = pallet.select("pallet_x", "pallet_y").row(-1)

    table = (
        table
        .join_asof(ego, on="log_time", strategy="nearest")
        .join_asof(pallet, on="log_time", strategy="nearest")
        .with_columns(
            _ego_frame(pl.col("pallet_x"), pl.col("pallet_y"), "ego").alias(
                "relative_ego_pos"
            ),
            _ego_frame(pl.lit(dropoff_x), pl.lit(dropoff_y), "ego").alias(
                "relative_dropoff_pos"
            ),
        )
    )

    return table.drop("ego_x", "ego_y", "ego_heading", "pallet_x", "pallet_y")


def _drop_invalid(table: pl.DataFrame, validity: list[str]) -> pl.DataFrame:
    """Drop rows any `_valid` companion marks as not-yet-known, and the columns.

    These are a prefix in practice - the ECU has not answered for the first
    second or so of a job - and dropping a prefix leaves the timeline contiguous.
    A gap in the middle would instead make a sliding episode window silently span
    a time discontinuity, so that case is warned about rather than hidden.
    """
    if not validity:
        return table

    keep = pl.all_horizontal(validity)
    kept = table.filter(keep).drop(validity)
    dropped = len(table) - len(kept)
    if dropped:
        invalid = table.with_row_index("_row").filter(~keep)["_row"].to_list()
        contiguous_prefix = invalid == list(range(dropped))
        logger.info(
            "dropped rows with invalid readings",
            rows=dropped,
            of=len(table),
            columns=validity,
            contiguous_prefix=contiguous_prefix,
        )
        if not contiguous_prefix:
            logger.warning(
                "invalid rows are not a leading run: episode windows may span the "
                "resulting gap",
                first=invalid[0],
                last=invalid[-1],
            )

    return kept


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

    # scalar columns window to Array(dtype, L); the position columns are already
    # Array(_, 2) per row, so they window to Array(_, (L, 2))
    scalar_columns = [f"{c}/frame_idx" for c in cameras] + list(SIGNALS)
    starts = range(0, len(rows) - episode_length + 1, stride)

    windows = {
        column: [
            rows[column][start : start + episode_length].to_list() for start in starts
        ]
        for column in [*scalar_columns, *POSITIONS]
    }

    schema: dict[str, pl.DataType] = {"input_id": pl.String()}
    for column in scalar_columns:
        dtype = pl.Int32() if column.endswith("frame_idx") else pl.Float32()
        schema[column] = pl.Array(dtype, episode_length)
    for column in POSITIONS:
        schema[column] = pl.Array(pl.Float32(), (episode_length, POSITION_DIM))

    return pl.DataFrame({"input_id": [job_id] * len(starts)} | windows, schema=schema)


@final
class D12SampleBuilder:
    """A pipefunc stage: one D12 job directory -> its episode rows.

    Data-layout parameters (which cameras, which timeline, how strict the pairing
    guard) are fixed per dataset and set here; the training parameters that shape
    the episodes (`episode_length`, `stride`, `every_nth`) are also constructor
    arguments so the dataset config can wire them to the experiment's values. The
    job directory is the one per-call input, so the pipeline maps this over jobs.
    """

    @validate_call
    def __init__(  # noqa: PLR0913
        self,
        *,
        cameras: tuple[str, ...],
        reference: str = "cam_fork",
        traction_source: str = "command",
        episode_length: int = 6,
        stride: int = 1,
        every_nth: int = 3,
        max_skew_s: float = 1.0,
    ) -> None:
        if reference not in cameras:
            msg = f"reference {reference!r} not in cameras {cameras}"
            raise ValueError(msg)
        if traction_source not in TRACTION_SOURCES:
            msg = f"unknown traction source {traction_source!r}"
            raise ValueError(msg)

        self._cameras = cameras
        self._reference = reference
        self._traction_source = traction_source
        self._episode_length = episode_length
        self._stride = stride
        self._every_nth = every_nth
        self._max_skew_s = max_skew_s

    @validate_call
    def __call__(self, *, job_dir: Path) -> pl.DataFrame:
        table = build(
            job_dir,
            cameras=self._cameras,
            reference=self._reference,
            traction_source=self._traction_source,
            max_skew_s=self._max_skew_s,
        )
        samples = episodes(
            table,
            job_id=job_dir.name,
            cameras=self._cameras,
            episode_length=self._episode_length,
            stride=self._stride,
            every_nth=self._every_nth,
        )
        logger.info(
            "built d12 samples",
            job=job_dir.name,
            frames=len(table),
            episodes=len(samples),
            signal_ranges=table.select(list(SIGNALS)).describe().to_dicts(),
            # metres, so their magnitude sanity-checks the scale in the model
            position_ranges=table
            .select(
                pl.col(p).arr.get(i).alias(f"{p}[{i}]")
                for p in POSITIONS
                for i in range(POSITION_DIM)
            )
            .describe()
            .to_dicts(),
        )

        return samples
