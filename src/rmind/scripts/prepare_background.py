"""Turn real palletjack recordings into a "do nothing" negative class for the card rig.

The command-card model has only ever seen cards, so an ordinary warehouse scene is
out of distribution and its output is whatever the network happens to extrapolate.
That is the one thing a rig driving real actuators must not do. These frames are
labelled (0, 0, 0) so that anything which is *not* a card commands a full stop.

Input is chunked `cam_left_backward--N.mp4` files copied off the kit (`/data/<recording>/`).
No CAN or mcap decoding is needed: the label is a constant, not a recorded action.

    just generate-cards                 # the positive class
    uv run python -m rmind.scripts.prepare_background --raw-dir data/palletjack/background/raw

Emits, under `--output-dir`:

    frames/{:06d}.jpg   decoded frames, resized to the card frame size
    samples.parquet     rbyte samples: sliding windows of consecutive frames, all-zero actions

Episodes are windows of *consecutive* frames, unlike the cards (which repeat one
frame), so the model also sees a moving real scene and still commands zero.
"""

import argparse
import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import]
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Final

import polars as pl
from structlog import get_logger

logger = get_logger(__name__)

# must match `generate_cards.FRAME_SIZE`; the kit camera is 1920x1080, same 16:9
FRAME_WIDTH: Final = 576
FRAME_HEIGHT: Final = 324


def extract(chunk: Path, out_dir: Path, *, start_index: int, every_nth: int) -> int:
    """Decode every `every_nth` frame of `chunk` into `out_dir`; return the count written.

    Decodes into a scratch directory and moves the results into place, so the
    count reflects what this chunk actually produced rather than whatever the
    output directory already held - re-running over a populated `frames/` would
    otherwise inflate every count after the first.

    Raises:
        ValueError: if `ffmpeg` is not on PATH.
    """
    if (ffmpeg := shutil.which("ffmpeg")) is None:
        msg = "ffmpeg not found on PATH"
        raise ValueError(msg)

    with TemporaryDirectory(dir=out_dir.parent) as scratch:
        scratch_dir = Path(scratch)
        # recordings on the kit contain zero-byte and truncated chunks; skip
        # rather than abort a run over a whole directory
        result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
            [
                ffmpeg,
                "-v",
                "error",
                "-i",
                chunk.as_posix(),
                "-vf",
                f"select=not(mod(n\\,{every_nth})),scale={FRAME_WIDTH}:{FRAME_HEIGHT}",
                "-vsync",
                "0",
                "-q:v",
                "2",
                (scratch_dir / "%06d.jpg").as_posix(),
            ],
            check=False,
            capture_output=True,
        )
        if result.returncode != 0:
            logger.warning(
                "skipping undecodable chunk",
                chunk=chunk.name,
                returncode=result.returncode,
                stderr=result.stderr.decode(errors="replace").strip()[:200],
            )

        frames = sorted(scratch_dir.glob("*.jpg"))
        for offset, frame in enumerate(frames):
            frame.replace(out_dir / f"{start_index + offset:06d}.jpg")

    return len(frames)


def build_samples(
    groups: list[list[int]], *, episode_length: int, stride: int
) -> pl.DataFrame:
    """Sliding windows within each chunk - never spanning a chunk boundary.

    Raises:
        ValueError: if no chunk holds enough frames for one episode.
    """
    windows = [
        frames[start : start + episode_length]
        for frames in groups
        for start in range(0, len(frames) - episode_length + 1, stride)
    ]
    if not windows:
        msg = (
            "no episodes: not enough frames per chunk for the requested episode_length"
        )
        raise ValueError(msg)

    zeros = [[0.0] * episode_length for _ in windows]

    return pl.DataFrame(
        {
            "input_id": ["background"] * len(windows),
            # no CAN decode: zero, like the cards. See generate_cards.
            "speed": zeros,
            "frame_idx": windows,
            "traction": zeros,
            "steering": zeros,
            "fork1": zeros,
        },
        schema={
            "input_id": pl.String,
            "speed": pl.Array(pl.Float32, episode_length),
            "frame_idx": pl.Array(pl.Int32, episode_length),
            "traction": pl.Array(pl.Float32, episode_length),
            "steering": pl.Array(pl.Float32, episode_length),
            "fork1": pl.Array(pl.Float32, episode_length),
        },
    )


def main(  # ruff: ignore[too-many-arguments]
    raw_dir: Path,
    output_dir: Path,
    *,
    glob: str,
    episode_length: int,
    every_nth: int,
    stride: int,
) -> None:
    """Decode the chunks and write the samples table.

    Raises:
        ValueError: if no chunk matches `glob`.
    """
    chunks = sorted(raw_dir.glob(glob))
    if not chunks:
        msg = f"no chunks matching {glob!r} in {raw_dir}"
        raise ValueError(msg)

    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    groups: list[list[int]] = []
    next_index = 0
    for chunk in chunks:
        written = extract(
            chunk, frames_dir, start_index=next_index, every_nth=every_nth
        )
        groups.append(list(range(next_index, next_index + written)))
        logger.debug("extracted", chunk=chunk.name, frames=written)
        next_index += written

    samples = build_samples(groups, episode_length=episode_length, stride=stride)
    samples.write_parquet(output_dir / "samples.parquet")

    logger.info(
        "prepared background",
        output_dir=output_dir.resolve().as_posix(),
        chunks=len(chunks),
        frames=next_index,
        episodes=len(samples),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/palletjack/background/raw"),
        help="directory of cam_left_backward--N.mp4 chunks copied off the kit",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/palletjack/background")
    )
    parser.add_argument(
        "--glob",
        default="*_lb-*.mp4",
        help="which chunks to use - keep it to a single camera",
    )
    parser.add_argument("--episode-length", type=int, default=6)
    parser.add_argument(
        "--every-nth", type=int, default=6, help="frame decimation (60 fps source)"
    )
    parser.add_argument(
        "--stride", type=int, default=1, help="sliding-window stride, in decoded frames"
    )
    args = parser.parse_args()

    main(
        args.raw_dir,
        args.output_dir,
        glob=args.glob,
        episode_length=args.episode_length,
        every_nth=args.every_nth,
        stride=args.stride,
    )
