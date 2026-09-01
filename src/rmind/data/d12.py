"""Build an rbyte sample table from a D12 job: `data.mcap` + per-camera directories.

`D12RowTableBuilder` and `D12EpisodeWindower` are pipefunc stages driven from
`config/_templates/dataset/palletjack/d12.yaml`: per job, the builder decodes the
mcap into a per-frame row table, a DuckDB stage in the config turns the world-frame
positions into the ego-frame intention tokens, the windower slices sliding
episodes, and the pipeline concatenates jobs. There is no sample parquet on disk -
samples are built at load time, so adding a signal is a config change and nothing
can go stale. The clock reasoning below is the whole reason the row builder is
Python and not a DuckDB query.

Job layout (`/nasa/data/d12/yaak/{job-id}/`):

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
    relative_ego_pos      pozyx pose + pallet tag         WORLD ego, pallet, target
    relative_dropoff_pos  pozyx pose + dropoff.parquet    -> ego frame IN THE CONFIG

The two `relative_*` tokens are the intention (see the constants block). This
module does NOT compute them: it emits the WORLD-frame ego pose, live pallet
position, and the precomputed dropoff target, and the dataset config's DuckDB
stage rotates them into the vehicle frame and scales them - the same split the car
model uses for waypoints. The row builder and the episode windower are two
pipefunc stages with that transform in between.

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

import math
from pathlib import Path
from typing import Final, NamedTuple, final

import polars as pl
from pydantic import InstanceOf, validate_call
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

# Where the traction TARGET comes from. The names matter: the previous ones inverted the
# semantics and cost us a model.
#
#   operator  0x193 on the Linde bus -- the "traction/drive command frame ... raw joystick
#             mirror" (coppernic `linde-d12-decode`), i.e. the operator's own demand, which
#             is what an imitation target should be. +/-500 counts = +/-100%, so the quantum
#             is 0.2 pp. MEASURED on 2026-08-31--16-47-19: 83 Hz, 799 distinct values in the
#             drive, changing 33 times a second.
#   applied   what the Lindelot controller applied (`applied_traction`). 10 Hz, 155 distinct,
#             quantum 1.0 pp. The plant's output, not the operator's intent.
#   uds_diag  UDS DID 0x3039, folded into the running `pb::VehicleState` snapshot. NOT a bus
#             signal: `lindelot/vehicle_state` is republished on every UDS reading, so its
#             11 Hz publish rate is not the refresh rate of any individual field. MEASURED:
#             `traction_command_pct` changes 0.50 times a second, median gap between changes
#             1.63 s and never below 1.51 s, giving 44 distinct values across a whole drive.
#             Nearest-asof onto the 10 Hz grid replicates each value ~16x, so a 5 s chunk
#             holds about 3 independent values.
#
# `uds_diag` was the default until 2026-09-01 because it was named `command` and the joystick
# frame was named `reported` -- and "command is the imitation target" is the correct
# principle, applied to the wrong topic. Both signals are legitimately called a traction
# command (one is the CAN command frame, the other a DID literally named
# DID_TRACTION_COMMAND), so the old names are kept as aliases rather than deleted: runs
# 5d3a592p / jabwefrp / 3cov0k09 / 9lf54of4 / o2ei7gh0 were all trained through them and
# must stay reproducible.
TRACTION_SOURCES: Final = {
    "operator": ("linde/traction", "traction_pct"),
    "applied": ("lindelot/applied", "traction_pct"),
    "uds_diag": ("lindelot/vehicle_state", "traction_command_pct"),
    # deprecated aliases -- misleading names, retained for reproducibility
    "reported": ("linde/traction", "traction_pct"),
    "command": ("lindelot/vehicle_state", "traction_command_pct"),
}

# --- intention: two ego-frame position tokens -----------------------------------
#
# The pozyx indoor system carries both the vehicle pose and the pallet position in
# ONE world frame (gnss is all-zero indoors and unused). `relative_ego_pos` points
# the model at the pallet to pick up; `relative_dropoff_pos` at where to leave it -
# the dropoff, which is NOT a sensor reading and comes from `dropoff.parquet`
# (written by `scripts.prepare_dropoff`, joined by time so it can vary within a
# job). This module emits both targets and the ego pose in WORLD metres; the
# dataset config translates by -ego, rotates by -heading and scales to ~[-1, 1],
# so the frame transform lives in one place, exactly as the car model's waypoints.
#
# These are inputs, never targets.
POSE_TOPIC: Final = "pozyx/pose"
TAG_TOPIC: Final = "pozyx/tag"
PALLET_LABEL: Final = "pallet"
DROPOFF_FILE: Final = "dropoff.parquet"

# the world-frame columns the row builder emits for the config to transform
WORLD_POSITION_COLUMNS: Final = (
    "ego_x",
    "ego_y",
    "ego_heading",
    "pallet_x",
    "pallet_y",
    "target_x",
    "target_y",
)

POSITION_DIM: Final = 2  # (forward, lateral); z/height is `fork_above_300`'s job

# the ego-frame token columns the config's DuckDB stage produces from the above
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


def row_table(
    job_dir: Path,
    *,
    cameras: tuple[str, ...],
    reference: str,
    traction_source: str,
    max_skew_s: float,
) -> pl.DataFrame:
    """One row per reference-camera frame: frame indices, signals, WORLD positions.

    The ego-frame rotation and metre scaling are NOT done here - they live in the
    dataset config's DuckDB stage (see `WORLD_POSITION_COLUMNS`). This decodes,
    aligns the camera clocks and signals, joins the world-frame ego pose, live
    pallet position and precomputed dropoff target, and drops not-yet-valid rows.

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

    table = _world_positions(table, topics, job_dir)

    return _drop_invalid(table, validity).rename({
        "frame_idx": f"{reference}/frame_idx"
    })


def _read_dropoff(job_dir: Path) -> pl.DataFrame:
    """The precomputed world-frame dropoff target(s), sorted by time.

    Raises:
        ValueError: if the file is missing - it is a separate preprocessing step.
    """
    path = job_dir / DROPOFF_FILE
    if not path.is_file():
        msg = (
            f"{path} missing: run `python -m rmind.scripts.prepare_dropoff "
            f"--job-dir {job_dir}` first"
        )
        raise ValueError(msg)

    return (
        pl
        .read_parquet(path)
        .sort("log_time")
        .select(
            "log_time",
            pl.col("x_m").cast(pl.Float32).alias("target_x"),
            pl.col("y_m").cast(pl.Float32).alias("target_y"),
        )
    )


def _world_positions(
    table: pl.DataFrame, topics: dict[str, pl.DataFrame], job_dir: Path
) -> pl.DataFrame:
    """Join world-frame ego pose, live pallet, and dropoff target onto the table.

    No rotation or scaling: those are the config's job. The dropoff uses a BACKWARD
    asof (the target in effect at or before each frame); any frames earlier than
    the first target row inherit it by back-filling, so a single-row file is simply
    a constant goal.

    Raises:
        ValueError: if the job carries no pallet tag fix.
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
            (pl.col("x_mm") / 1000).cast(pl.Float32).alias("pallet_x"),
            (pl.col("y_mm") / 1000).cast(pl.Float32).alias("pallet_y"),
        )
    )
    if pallet.is_empty():
        msg = f"no {PALLET_LABEL!r} tag fixes in the job"
        raise ValueError(msg)

    target = _read_dropoff(job_dir)

    return (
        table
        .join_asof(ego, on="log_time", strategy="nearest")
        .join_asof(pallet, on="log_time", strategy="nearest")
        .join_asof(target, on="log_time", strategy="backward")
        # frames before the first target row inherit it
        .with_columns(pl.col("target_x", "target_y").fill_null(strategy="backward"))
    )


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


def episodes(
    table: pl.DataFrame,
    *,
    cameras: tuple[str, ...],
    episode_length: int,
    episode_step: int,
    episode_stride: int,
    episode_offset: int = 0,
) -> pl.DataFrame:
    """Sliding-window episodes over a positioned row table, on the RAW grid.

    The clip grid is laid on the RAW (undecimated) timeline with a step of
    `episode_stride` raw frames, and each clip then gathers `episode_length`
    frames `episode_step` raw frames apart. This is the D12 form of the retile in
    `config/_templates/dataset/yaak/train.yaml` (`episode_stride: 31`), and it
    exists for the same reason.

    The previous implementation decimated FIRST (`table.gather_every(every_nth)`)
    and strode over the decimated rows, which means every clip in every epoch
    started on raw-frame phase 0 of `episode_step`. At `episode_step: 3` that made
    2 of every 3 extracted frames unreachable by any clip, forever — the grid is
    deterministic, so this was not sampling variance but a fixed 33% of the data
    never being read. Striding on the raw grid with a stride COPRIME to the step
    rotates successive clip starts through every phase: 31 is coprime to 3, so the
    starts cycle 0, 1, 2, 0, ... and coverage goes to ~100%.

    `episode_offset` drops the first N raw rows before the grid is laid, shifting
    every clip start by N. It is baked at instantiation (rbyte builds the sample
    table once and locks it into shared memory), so rotate it across restart
    segments rather than expecting per-epoch variation.

    Expects the two `POSITIONS` columns already present (the config's DuckDB stage
    adds them as 2-lists); `input_id` is carried through from the row builder.

    Raises:
        ValueError: if `episode_step`/`episode_stride` are not positive, or the job
            is too short for one clip.
    """
    if episode_step < 1:
        msg = f"episode_step must be >= 1, got {episode_step}"
        raise ValueError(msg)
    if episode_stride < 1:
        msg = f"episode_stride must be >= 1, got {episode_stride}"
        raise ValueError(msg)
    if episode_offset < 0:
        msg = f"episode_offset must be >= 0, got {episode_offset}"
        raise ValueError(msg)

    rows = table.slice(episode_offset) if episode_offset else table

    # raw frames a clip spans: L readouts, (L-1) gaps of `episode_step`
    span = (episode_length - 1) * episode_step + 1
    if len(rows) < span:
        msg = (
            f"{len(rows)} usable frames < clip span {span} "
            f"(episode_length {episode_length} x episode_step {episode_step})"
        )
        raise ValueError(msg)

    if math.gcd(episode_stride, episode_step) != 1:
        # not fatal - a non-coprime stride still trains, it just cannot reach
        # every phase - but it is almost always a mistake, so say so loudly
        logger.warning(
            "episode_stride is not coprime to episode_step: the clip grid can "
            "only reach some raw-frame phases",
            episode_stride=episode_stride,
            episode_step=episode_step,
            reachable_phases=episode_step // math.gcd(episode_stride, episode_step),
        )

    input_id = rows["input_id"][0]  # constant within a job

    # scalar columns window to Array(dtype, L); the position columns are 2-lists
    # per row, so they window to Array(_, (L, 2))
    scalar_columns = [f"{c}/frame_idx" for c in cameras] + list(SIGNALS)
    starts = range(0, len(rows) - span + 1, episode_stride)
    # absolute row indices of one clip, relative to its start
    picks = [i * episode_step for i in range(episode_length)]

    columns = {
        column: rows[column].to_list() for column in [*scalar_columns, *POSITIONS]
    }
    windows = {
        column: [[values[start + p] for p in picks] for start in starts]
        for column, values in columns.items()
    }

    schema: dict[str, pl.DataType] = {"input_id": pl.String()}
    for column in scalar_columns:
        dtype = pl.Int32() if column.endswith("frame_idx") else pl.Float32()
        schema[column] = pl.Array(dtype, episode_length)
    for column in POSITIONS:
        schema[column] = pl.Array(pl.Float32(), (episode_length, POSITION_DIM))

    logger.info(
        "windowed d12 episodes",
        clips=len(starts),
        span=span,
        phases_reached=min(episode_step, len(starts)),
    )

    return pl.DataFrame({"input_id": [input_id] * len(starts)} | windows, schema=schema)


@final
class D12RowTableBuilder:
    """Pipefunc stage: one D12 job directory -> its per-frame row table.

    Emits the WORLD-frame ego/pallet/target columns; the dataset config's DuckDB
    stage turns them into the ego-frame tokens before `D12EpisodeWindower` windows
    the result. Data-layout parameters (cameras, reference timeline, pairing guard)
    are fixed per dataset and set here.
    """

    @validate_call
    def __init__(
        self,
        *,
        cameras: tuple[str, ...],
        reference: str = "cam_fork",
        traction_source: str = "command",
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
        self._max_skew_s = max_skew_s

    @validate_call
    def __call__(self, *, job_dir: Path) -> pl.DataFrame:
        table = row_table(
            job_dir,
            cameras=self._cameras,
            reference=self._reference,
            traction_source=self._traction_source,
            max_skew_s=self._max_skew_s,
        ).with_columns(pl.lit(job_dir.name).alias("input_id"))

        logger.info(
            "built d12 row table",
            job=job_dir.name,
            frames=len(table),
            signal_ranges=table.select(list(SIGNALS)).describe().to_dicts(),
            # world metres, to sanity-check the scale applied downstream
            world_ranges=table.select(WORLD_POSITION_COLUMNS).describe().to_dicts(),
        )

        return table


@final
class DuckDBStage:
    """A pipefunc stage that runs one DuckDB query over a `row_table` input.

    The query itself lives in the dataset config (that is the point - the
    intention transform is config, like the car waypoints); this only gives it a
    named signature pipefunc can bind, so it needs no `makefun`. The query must
    read `FROM row_table`.
    """

    @validate_call
    def __init__(self, *, query: str) -> None:
        self._query = query

    def __call__(self, *, row_table: InstanceOf[pl.DataFrame]) -> pl.DataFrame:
        from rbyte.samples.duckdb import DuckDBQuery  # noqa: PLC0415

        return DuckDBQuery(query=self._query)(row_table=row_table)


@final
class D12EpisodeWindower:
    """Pipefunc stage: a positioned row table -> its sliding-window episodes.

    The training parameters that shape the episodes are constructor arguments so
    the dataset config can wire them to the experiment's values.
    """

    @validate_call
    def __init__(
        self,
        *,
        cameras: tuple[str, ...],
        episode_length: int = 6,
        episode_step: int = 3,
        episode_stride: int = 31,
        episode_offset: int = 0,
    ) -> None:
        self._cameras = cameras
        self._episode_length = episode_length
        self._episode_step = episode_step
        self._episode_stride = episode_stride
        self._episode_offset = episode_offset

    @validate_call
    def __call__(self, *, frames: InstanceOf[pl.DataFrame]) -> pl.DataFrame:
        samples = episodes(
            frames,
            cameras=self._cameras,
            episode_length=self._episode_length,
            episode_step=self._episode_step,
            episode_stride=self._episode_stride,
            episode_offset=self._episode_offset,
        )

        return samples
