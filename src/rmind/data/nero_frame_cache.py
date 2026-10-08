"""Preprocessed nero camera frames on local disk (training-only, opt-in).

The robot-native patch runs are bound by video decoding, not by the GPU: every
32-frame window decodes 3 native 1080p/800p mp4 streams, and stride-7 windows
overlap ~13x, so the same frames are decoded over and over (the documented cost
is ~3.3 s per 2-sample batch against a ~0.24 s GPU step). The model only ever
sees `rmind.data.nero_image.preprocess`'s output -- a 140x224 uint8 letterboxed
frame -- so caching THAT output once per mp4 frame removes the decode from the
loop without changing a single input byte.

Layout, per take and camera (`<cache>/<take>/<camera>.npy` + `.json`):

* `<camera>.npy`: uint8 `(N, 3, H_in, W_in)`, row `i` = `preprocess(frame_i)` for
  EVERY frame `i` of the mp4 in decode order, `N = decoder.metadata.num_frames`.
  Rows are indexed by the same mp4 ordinal (`frame_index.<camera>`) that
  `rbyte.streams.video.TorchCodecVideoSource` takes, so the dataset config only
  swaps the stream source.
* `<camera>.json`: the manifest (`nero_frame_cache/1`): the preprocessing id and
  the SHA256 of `nero_image.py` (any edit of the function invalidates the cache),
  `input_hw`, the frame count, and the source mp4's size plus a SHA256 of its
  first and last MiB. `NeroFrameCacheSource` refuses a cache whose manifest does
  not match the current preprocessing, the requested grid or the mp4 it is
  configured against. The manifest is written LAST, so a half-built cache is
  never accepted.

Same bytes as the decode path: frames come from the same torchcodec decoder
settings rbyte uses (NCHW, exact seek mode, uint8) and go through the same
`preprocess`. `tests/test_nero_bimanual.py` pins cached == decoded on sampled
indices of a real take, and `nero_frame_cache --verify` repeats that check on
the built cache.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from torch import Tensor

from rmind.data.nero_image import PREPROCESSING_ID, preprocess, preprocessing_sha256

if TYPE_CHECKING:
    import os

__all__ = [
    "MANIFEST_SCHEMA",
    "NeroFrameCacheSource",
    "build_camera_cache",
    "cache_paths",
    "manifest_problems",
    "video_fingerprint",
]

MANIFEST_SCHEMA = "nero_frame_cache/1"
_FINGERPRINT_BYTES = 1 << 20
_DECODE_CHUNK = 64


def cache_paths(
    cache_root: str | os.PathLike[str], take: str, camera: str
) -> tuple[Path, Path]:
    """`(<cache>/<take>/<camera>.npy, <cache>/<take>/<camera>.json)`."""
    base = Path(cache_root) / take
    return base / f"{camera}.npy", base / f"{camera}.json"


def video_fingerprint(video: str | os.PathLike[str]) -> dict[str, Any]:
    """Size + SHA256 of the first and last MiB: cheap, and survives a copy without -a."""
    path = Path(video)
    size = path.stat().st_size
    digest = hashlib.sha256()
    with path.open("rb") as f:
        digest.update(f.read(_FINGERPRINT_BYTES))
        f.seek(max(0, size - _FINGERPRINT_BYTES))
        digest.update(f.read(_FINGERPRINT_BYTES))
    return {"size": size, "head_tail_sha256": digest.hexdigest()}


def _decoder(video: str | os.PathLike[str], num_threads: int) -> Any:
    from torchcodec.decoders import VideoDecoder  # noqa: PLC0415

    # the settings rbyte's TorchCodecVideoSource uses on the decode path
    return VideoDecoder(
        Path(video).resolve().as_posix(),
        dimension_order="NCHW",
        num_ffmpeg_threads=num_threads,
        seek_mode="exact",
    )


def build_camera_cache(
    video: str | os.PathLike[str],
    npy: str | os.PathLike[str],
    *,
    input_hw: tuple[int, int],
    num_threads: int = 4,
) -> dict[str, Any]:
    """Decode every frame of `video`, preprocess it, write `npy` + its manifest.

    Returns the manifest.

    Raises:
        RuntimeError: if the decoder yields a different frame count than its
            metadata announces (the rbyte frame index would not line up).
    """
    npy = Path(npy)
    manifest_path = npy.with_suffix(".json")
    manifest_path.unlink(missing_ok=True)  # a stale manifest must not bless a new npy
    npy.parent.mkdir(parents=True, exist_ok=True)
    decoder = _decoder(video, num_threads)
    n = decoder.metadata.num_frames
    if n is None:
        msg = f"{video}: decoder metadata has no frame count"
        raise RuntimeError(msg)
    h, w = int(input_hw[0]), int(input_hw[1])
    tmp = npy.with_suffix(".npy.partial")
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(n, 3, h, w))
    for start in range(0, n, _DECODE_CHUNK):
        stop = min(n, start + _DECODE_CHUNK)
        frames = decoder.get_frames_in_range(start=start, stop=stop).data
        if frames.shape[0] != stop - start:
            msg = f"{video}: decoded {frames.shape[0]} frames for [{start}, {stop})"
            raise RuntimeError(msg)
        out[start:stop] = preprocess(frames, (h, w)).numpy()
    out.flush()
    del out
    tmp.replace(npy)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "preprocessing_id": PREPROCESSING_ID,
        "preprocessing_sha256": preprocessing_sha256(),
        "input_hw": [h, w],
        "num_frames": int(n),
        "video": Path(video).name,
        "video_fingerprint": video_fingerprint(video),
    }
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def manifest_problems(
    npy: str | os.PathLike[str],
    *,
    input_hw: Sequence[int],
    video: str | os.PathLike[str] | None = None,
) -> list[str]:
    """Why the cache at `npy` must not be used (empty = usable)."""
    npy = Path(npy)
    manifest_path = npy.with_suffix(".json")
    if not npy.is_file() or not manifest_path.is_file():
        return [f"{npy}: no cache (or no manifest: an unfinished build)"]
    m = json.loads(manifest_path.read_text())
    problems = []
    if m.get("schema") != MANIFEST_SCHEMA:
        problems.append(f"schema {m.get('schema')!r} != {MANIFEST_SCHEMA!r}")
    if m.get("preprocessing_sha256") != preprocessing_sha256():
        problems.append(
            "built with a different rmind.data.nero_image (preprocessing_sha256 "
            f"{m.get('preprocessing_sha256')} != {preprocessing_sha256()})"
        )
    if list(m.get("input_hw", [])) != [int(v) for v in input_hw]:
        problems.append(f"input_hw {m.get('input_hw')} != {list(input_hw)}")
    shape = np.load(npy, mmap_mode="r").shape
    if shape[0] != m.get("num_frames") or tuple(shape[2:]) != tuple(
        int(v) for v in input_hw
    ):
        problems.append(f"array shape {shape} disagrees with the manifest")
    if video is not None and m.get("video_fingerprint") != video_fingerprint(video):
        problems.append(f"built from a different {Path(video).name}")
    return [f"{npy}: {p}" for p in problems]


class NeroFrameCacheSource:
    """rbyte `StreamSource` over a cached camera: `[i]` == preprocess(decode(i)).

    A drop-in for `TransformedSource(TorchCodecVideoSource(video),
    NeroImagePreprocess(input_hw))`: an int index gives `(3, H, W)`, a sequence
    `(n, 3, H, W)`, both uint8. The manifest is checked on construction (see
    `manifest_problems`); the array is memory-mapped lazily (and re-opened after
    unpickling), so many sources cost no RAM beyond the page cache.

    Raises:
        ValueError: if the cache is missing, unfinished or stale.
    """

    def __init__(
        self,
        *,
        path: str | os.PathLike[str],
        input_hw: Sequence[int],
        video: str | os.PathLike[str] | None = None,
    ) -> None:
        self._path = Path(path)
        problems = manifest_problems(self._path, input_hw=input_hw, video=video)
        if problems:
            msg = (
                "frame cache refused (rebuild it with "
                "`python -m rmind.scripts.nero_frame_cache`):\n  "
                + "\n  ".join(problems)
            )
            raise ValueError(msg)
        self._array: np.ndarray | None = None

    def _frames(self) -> np.ndarray:
        if self._array is None:
            self._array = np.load(self._path, mmap_mode="r")
        return self._array

    def __len__(self) -> int:
        return int(self._frames().shape[0])

    def __getitem__(self, indexes: int | Sequence[int]) -> Tensor:
        frames = self._frames()
        if isinstance(indexes, Sequence):
            rows = list(cast("Sequence[int]", indexes))
            return torch.from_numpy(frames[rows])  # fancy index: a copy
        return torch.from_numpy(np.array(frames[int(indexes)]))  # copy off the map

    def __getstate__(self) -> dict[str, Any]:
        return {"_path": self._path, "_array": None}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
