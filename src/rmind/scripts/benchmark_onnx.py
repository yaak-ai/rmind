"""Benchmark ControlTransformer on raw ride data — ONNX and/or wandb model.

Both backends can be active at once so their predictions are compared side by side.
Fixed run parameters (data_dir, start_frame, num_episodes, ...) live in
config/benchmark_onnx.yaml — override just `onnx=` and/or `wandb_model=` per run.

`wandb_model=` always requires `export=...` (e.g.
export=yaak/control_transformer/finetuned, or export=yaak/patch_policy/finetuned):
the PyTorch model is then loaded the same way that export strips its in-graph
image preprocessing (hparams_jq for ControlTransformer,
PatchPolicy.load_for_export for PatchPolicy), so both backends run
architecturally identical, export-stripped models on the same externally
preprocessed input — see tests/test_benchmark_onnx_preprocessing.py for why
that external preprocessing is equivalent to the original, unstripped one.
PatchPolicy checkpoints are only benchmarked on cam_front_left — see the
NOTE in _WandbBackend.run.

`frame_sources=` selects where cam_front_left frames come from: "video"
(default) decodes cam_front_left.pii.mp4; "jpg" reads dvc.yaml's
pre-extracted frames/cam_front_left.pii.mp4/<W>x<H>/*.jpg instead — the
same fixed-size JPEGs training itself was built from, skipping
_preprocess_image's approximate video-decode downscale. "video_jpeg"
decodes video like "video" but additionally round-trips the downscaled
frame through ffmpeg's mjpeg quantization tables at `jpeg_qv=` (default
16, matching the offline extraction pipeline's `-q:v 16` — byte-identical
DQT against the real training jpgs, asserted in
tests/test_benchmark_onnx_preprocessing.py) before continuing preprocessing —
this emulates the offline pipeline's JPEG-compression step without
needing `scale_npp` (that resampling kernel still has no local
equivalent; only the *encode* half is reproduced here). "ffmpeg" is the
BASELINE: it re-runs extract_frames' own ffmpeg command (NVDEC hevc
decode, GPU resize in NV12, real mjpeg encoder), substituting only
scale_cuda for the unavailable scale_npp, so it shows how close any
inference-side preprocessing can get — 1.59 uint8 MAE against the real
training jpgs, vs 3.80 for "video_jpeg" and 4.42 for "video" (two
adjacent real frames differ by 1.34). Pass multiple
(e.g. frame_sources=[video,jpg,video_jpeg]) to run every backend against
all of them — every table then carries one row/column per (backend,
source) pair, and _print_validation's pairwise diff covers
source-vs-source too.

Usage:
    # ONNX only (compare with driver's benchmark_all_models.py)
    just benchmark-onnx onnx=~/rmind/outputs/.../model.onnx

    # wandb PyTorch only
    just benchmark-onnx \\
        wandb_model=yaak/rmind/model-XXXXXXXX:vN \\
        export=yaak/control_transformer/finetuned

    # Both side by side (ONNX vs torch, same batches)
    just benchmark-onnx \\
        onnx=~/rmind/outputs/.../model.onnx \\
        wandb_model=yaak/rmind/model-XXXXXXXX:vN \\
        export=yaak/control_transformer/finetuned

    # video-decoded frames vs pre-extracted jpg frames, same backend(s)
    just benchmark-onnx \\
        onnx=~/rmind/outputs/.../model.onnx \\
        frame_sources=[video,jpg]

    # ...plus a video source that also simulates the offline JPEG re-encode
    just benchmark-onnx \\
        onnx=~/rmind/outputs/.../model.onnx \\
        frame_sources=[video,jpg,video_jpeg]
"""

from __future__ import annotations

import bisect
import csv
import json
import mmap
import operator
import time
from collections.abc import Sequence  # noqa: TC003 — needed at runtime by pydantic
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, ClassVar, cast

import hydra
import numpy as np
import structlog
import torch
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, field_validator

logger = structlog.get_logger(__name__)

EMBED_DIM = 384
NUM_TIMESTEPS = 6
NUM_WAYPOINTS = 10
DEFAULT_IMAGE_SIZE = (324, 576)  # (H, W)
_ONNX_IMAGE_INPUT_NDIM = 5  # [B, T, C, H, W]
_TARGET_LATENCY_MS = 100
_MIN_BACKENDS_FOR_COMPARISON = 2
_VALIDATION_TOLERANCE = 1e-3

# Matches raw.yaml's Normalize step (applied via _normalize_cam in
# _preprocess_image), which both the exported ONNX graph and the
# hparams_jq-stripped PyTorch model no longer apply internally.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

# Lowercase batch_data_* keys — match driver's naming for case-insensitive ONNX input matching
_K_CAM = "batch_data_cam_front_left"
_K_SPEED = "batch_data_meta_vehiclemotion_speed"
_K_GAS = "batch_data_meta_vehiclemotion_gas_pedal_normalized"
_K_BRAKE = "batch_data_meta_vehiclemotion_brake_pedal_normalized"
_K_STEER = "batch_data_meta_vehiclemotion_steering_angle_normalized"
_K_TURN = "batch_data_meta_vehiclestate_turn_signal"
_K_WP = "batch_data_waypoints_xy_normalized"

# Nested dict keys for the PyTorch model (under the "data" key)
_PT_CAM = "cam_front_left"
_PT_SPEED = "meta/VehicleMotion/speed"
_PT_GAS = "meta/VehicleMotion/gas_pedal_normalized"
_PT_BRAKE = "meta/VehicleMotion/brake_pedal_normalized"
_PT_STEER = "meta/VehicleMotion/steering_angle_normalized"
_PT_TURN = "meta/VehicleState/turn_signal"
_PT_WP = "waypoints/xy_normalized"


# ── Lightweight data containers ───────────────────────────────────────────────


@dataclass
class Predictions:
    gas: float
    brake: float
    steer: float
    turn: int
    time_ms: float = 0.0


@dataclass
class GroundTruth:
    gas: float
    brake: float
    steer: float
    turn: int


@dataclass
class _VehicleState:
    timestamp: float
    speed: float = 0.0
    gas_pedal: float = 0.0
    brake_pedal: float = 0.0
    steering_angle: float = 0.0
    turn_signal: int = 0


@dataclass
class _GnssPosition:
    timestamp: float
    latitude: float = 0.0
    longitude: float = 0.0
    heading: float = 0.0


# ── Metadata reader (mirrors driver's RbyteMetadataReader) ────────────────────


@dataclass
class _MetadataReader:
    metadata_path: Path
    camera_name: str = "cam_front_left"
    _motion: list[_VehicleState] = field(init=False, default_factory=list)
    _gnss: list[_GnssPosition] = field(init=False, default_factory=list)
    _frame_ts: dict[int, float] = field(init=False, default_factory=dict)
    _turn_entries: list[tuple[float, int]] = field(init=False, default_factory=list)

    def load(self) -> None:
        from rbyte.io.yaak.metadata.message_iterator import (  # noqa: PLC0415
            YaakMetadataMessageIterator,
        )

        with Path(self.metadata_path).open("rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            for msg_cls, msg_bytes in YaakMetadataMessageIterator(mm):
                msg = msg_cls()
                msg.ParseFromString(msg_bytes)
                ts = msg.time_stamp.ToMicroseconds() / 1_000_000.0  # ty:ignore[unresolved-attribute]
                name = msg_cls.__name__
                if name == "VehicleMotion":
                    self._motion.append(
                        _VehicleState(
                            timestamp=ts,
                            speed=getattr(msg, "speed", 0.0),
                            gas_pedal=getattr(msg, "gas_pedal_normalized", 0.0),
                            brake_pedal=getattr(msg, "brake_pedal_normalized", 0.0),
                            steering_angle=getattr(
                                msg, "steering_angle_normalized", 0.0
                            ),
                        )
                    )
                elif (
                    name == "ImageMetadata"
                    and getattr(msg, "camera_name", None) == self.camera_name
                ):
                    self._frame_ts[getattr(msg, "frame_idx", 0)] = ts
                elif name == "VehicleState":
                    self._turn_entries.append((ts, int(getattr(msg, "turn_signal", 0))))
                elif name == "Gnss":
                    self._gnss.append(
                        _GnssPosition(
                            timestamp=ts,
                            latitude=getattr(msg, "latitude", 0.0),
                            longitude=getattr(msg, "longitude", 0.0),
                            heading=getattr(msg, "heading", 0.0),
                        )
                    )
            mm.close()

        self._gnss.sort(key=lambda x: x.timestamp)
        self._turn_entries.sort(key=operator.itemgetter(0))
        logger.info(
            "Metadata loaded",
            motion=len(self._motion),
            gnss=len(self._gnss),
            frames=len(self._frame_ts),
        )

    @staticmethod
    def _nearest_ts(entries: Sequence[_VehicleState | _GnssPosition], ts: float) -> int:
        times = [e.timestamp for e in entries]
        idx = bisect.bisect_left(times, ts)
        if idx >= len(times):
            return len(times) - 1
        if idx > 0 and abs(times[idx - 1] - ts) < abs(times[idx] - ts):
            return idx - 1
        return idx

    def _turn_at(self, ts: float) -> int:
        if not self._turn_entries:
            return 0
        times = [t for t, _ in self._turn_entries]
        idx = bisect.bisect_left(times, ts)
        if idx >= len(times):
            idx = len(times) - 1
        elif idx > 0 and abs(times[idx - 1] - ts) < abs(times[idx] - ts):
            idx -= 1
        return self._turn_entries[idx][1]

    def _state_at(self, ts: float) -> _VehicleState:
        if not self._motion:
            return _VehicleState(timestamp=0.0)
        idx = self._nearest_ts(self._motion, ts)
        e = self._motion[idx]
        return _VehicleState(
            timestamp=e.timestamp,
            speed=e.speed,
            gas_pedal=e.gas_pedal,
            brake_pedal=e.brake_pedal,
            steering_angle=e.steering_angle,
            turn_signal=self._turn_at(e.timestamp),
        )

    def _gnss_at(self, ts: float) -> _GnssPosition:
        if not self._gnss:
            return _GnssPosition(timestamp=0.0)
        return self._gnss[self._nearest_ts(self._gnss, ts)]

    def _frame_lookup(self, frame_idx: int) -> float:
        if frame_idx in self._frame_ts:
            return self._frame_ts[frame_idx]
        if self._frame_ts:
            closest = min(self._frame_ts, key=lambda f: abs(f - frame_idx))
            return self._frame_ts[closest]
        return 0.0

    def get_state_for_frame(self, frame_idx: int) -> _VehicleState:
        return self._state_at(self._frame_lookup(frame_idx))

    def get_gnss_for_frame(self, frame_idx: int) -> _GnssPosition:
        return self._gnss_at(self._frame_lookup(frame_idx))


# ── Waypoint loader (mirrors driver's WaypointLoader) ─────────────────────────


@dataclass
class _WaypointLoader:
    waypoints_path: Path
    _wps: list[dict] = field(init=False, default_factory=list)
    _times: list[float] = field(init=False, default_factory=list)

    def load(self) -> None:
        from pyproj import Transformer  # noqa: PLC0415

        t = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)
        with Path(self.waypoints_path).open(encoding="utf-8") as f:
            data = json.load(f)
        for feat in data.get("features", []):
            geom = feat.get("geometry", {})
            props = feat.get("properties", {})
            if geom.get("type") == "Point":
                lon, lat = geom["coordinates"][:2]
                x, y = t.transform(lon, lat)
                self._wps.append({
                    "timestamp": props.get("timestamp", 0.0),
                    "heading": props.get("heading", 0.0),
                    "lon": lon,
                    "lat": lat,
                    "x": x,
                    "y": y,
                })
        self._wps.sort(key=operator.itemgetter("timestamp"))
        self._times = [w["timestamp"] for w in self._wps]
        logger.info("Waypoints loaded", count=len(self._wps))

    def get_for_gnss(self, gnss: _GnssPosition, n: int = NUM_WAYPOINTS) -> np.ndarray:
        from pyproj import Transformer  # noqa: PLC0415

        if not self._wps:
            return np.zeros((n, 2), dtype=np.float32)

        idx = bisect.bisect_left(self._times, gnss.timestamp)
        wps = self._wps[idx : idx + n]
        if not wps:
            wps = self._wps[-n:]

        coords = np.array([[w["x"], w["y"]] for w in wps], dtype=np.float64)
        if len(coords) < n:
            last = coords[-1] if len(coords) else np.zeros(2)
            coords = np.vstack([coords, np.tile(last, (n - len(coords), 1))])

        t = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)
        ego_x, ego_y = t.transform(gnss.longitude, gnss.latitude)
        coords[:, 0] -= ego_x
        coords[:, 1] -= ego_y

        heading_rad = np.radians(gnss.heading)
        cos_h, sin_h = np.cos(heading_rad), np.sin(heading_rad)
        x_rot = coords[:, 0] * cos_h - coords[:, 1] * sin_h
        y_rot = coords[:, 0] * sin_h + coords[:, 1] * cos_h
        return np.stack([x_rot, y_rot], axis=1).astype(np.float32)


# ── Batch loading ─────────────────────────────────────────────────────────────


# Matches raw.yaml's CenterCrop([320, 576]) step, applied after downscaling to
# DEFAULT_IMAGE_SIZE (training's native JPEG frame resolution) and before the
# final resize — see input_transform's image branch.
_CENTER_CROP_SIZE = (320, 576)  # (H, W)


# The offline dvc.yaml extract_frames stage's ffmpeg command ends in
# `-f image2 -q:v 16 ...`. That -q:v is ffmpeg's native mjpeg quantizer scale
# (2-31, lower=better) -- NOT a 0-100 libjpeg "quality", and no libjpeg
# quality reproduces its table: the closest (quality 50) is still ~20 mean
# absolute DQT coefficients away, and passing 16 straight through as a libjpeg
# quality -- which this module used to do -- is the single worst match in the
# whole 0-100 sweep (~96 away), i.e. drastically over-compressed. That bug
# made the "video_jpeg" source land *further* from the real training jpgs than
# plain "video" (5.59 vs 4.31 uint8 MAE) when its whole purpose is to land
# closer (3.83).
#
# ffmpeg builds the table itself, scaling ff_mpeg1_default_intra_matrix by the
# requested -q:v: the DC coefficient stays pinned at 8, and every AC
# coefficient becomes base times qv floor-divided by 8, clamped to 1..255.
# Verified byte-identical to `ffmpeg -q:v N -pix_fmt yuvj420p` for every N in
# 2..31, and the qv=16 table reproduces the embedded DQT of the real training
# jpgs under <data_dir>/frames/cam_front_left.pii.mp4/576x324/ exactly (all 64
# coefficients; extract_frames emits a single table shared by luma and chroma)
# -- see tests/test_benchmark_onnx_preprocessing.py, which asserts this
# against real drive data.
#
# This is a pure encoder-side setting, independent of the decode/scale half of
# that pipeline (which uses scale_npp and has no local equivalent, see
# _preprocess_image's docstring).
# The actual implementation lives in rmind.utils.jpeg so non-training
# consumers (drivr's live camera pipeline) can reuse this exact, tested
# quantization logic without reaching into script internals. Re-exported here
# under their original private names for tests/test_benchmark_onnx_preprocessing.py.
from rmind.utils.jpeg import (  # noqa: E402
    DEFAULT_JPEG_QV as _DEFAULT_JPEG_QV,
    FFMPEG_MJPEG_BASE_QTABLE as _FFMPEG_MJPEG_BASE_QTABLE,
    JPEG_QV_RANGE as _JPEG_QV_RANGE,
    JPEG_SUBSAMPLING_420 as _JPEG_SUBSAMPLING_420,
    ffmpeg_mjpeg_qtable as _ffmpeg_mjpeg_qtable,
    simulate_offline_jpeg as _simulate_offline_jpeg,
)


def _preprocess_image(
    image: np.ndarray, image_size: tuple[int, int], *, jpeg_qv: int | None = None
) -> np.ndarray:
    """HWC uint8 → resize → center crop → resize → scale → normalize → CHW float32.

    Mirrors raw.yaml's image input_transform end to end (Rearrange ->
    CenterCrop([320, 576]) -> Resize -> ToDtype -> Normalize): downscale to
    DEFAULT_IMAGE_SIZE approximates the offline extraction pipeline
    (torchcodec.transforms.Resize, per rbyte's dataset config) that produced
    training's fixed-size JPEG frames — not reproduced bit-exactly here — then
    crop, resize, scale and normalize exactly as training does. Every backend
    this benchmark runs is built from the hparams_jq-stripped config, so none
    of them do this internally anymore — see
    tests/test_benchmark_onnx_preprocessing.py for the equivalence proof.

    `jpeg_qv`, when set (frame_sources=[video_jpeg]), additionally round-trips
    the downscaled frame through ffmpeg's real mjpeg encoder — see
    _simulate_offline_jpeg. Only meaningful for video-decoded frames; the
    "jpg" source is already real compressed data and never passes this.
    """
    import cv2  # noqa: PLC0415
    from torchvision.transforms import v2 as T  # noqa: PLC0415

    # The v2 *classes* (not torchvision.transforms.functional's functional API)
    # — raw.yaml instantiates these same classes, and the two aren't numerically
    # identical (functional.resize leaves a ~1/255-per-pixel rounding residual
    # even with matching interpolation/antialias args, per
    # tests/test_benchmark_onnx_preprocessing.py's investigation history).
    crop = T.CenterCrop(list(_CENTER_CROP_SIZE))
    resize_final = T.Resize(list(image_size))

    # cv2.INTER_CUBIC (not torchvision's Resize) for this first downscale —
    # empirically closer to the real extraction pipeline's scale_npp output
    # than torchvision's antialiased Resize, per this benchmark's own
    # video-vs-jpg investigation.
    h, w = DEFAULT_IMAGE_SIZE
    native_hwc = cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC)

    if jpeg_qv is not None:
        native_hwc = _simulate_offline_jpeg(native_hwc, qv=jpeg_qv)

    # uint8 throughout crop/resize (matching ToDtype running *after* Resize in
    # raw.yaml) to minimize rounding drift relative to the real pipeline.
    tensor = torch.from_numpy(native_hwc).permute(2, 0, 1).unsqueeze(0)  # uint8
    cropped = crop(tensor)
    resized = resize_final(cropped)
    scaled = (resized.float() / 255.0)[0].numpy()  # CHW float32 [0, 1]
    return _normalize_cam(scaled)


def _normalize_cam(cam: np.ndarray) -> np.ndarray:
    """ImageNet mean/std normalize a CHW float32 image in [0, 1]."""
    return (cam - _IMAGENET_MEAN) / _IMAGENET_STD


class _VideoFrameSource:
    """Reads frames straight out of cam_front_left.pii.mp4 (original behavior)."""

    def __init__(self, video_path: Path) -> None:
        import cv2  # noqa: PLC0415

        self._cap = cv2.VideoCapture(str(video_path))
        if not self._cap.isOpened():
            msg = f"Cannot open video: {video_path}"
            raise RuntimeError(msg)

    def read(self, frame_idx: int) -> np.ndarray:
        import cv2  # noqa: PLC0415

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame_bgr = self._cap.read()
        if not ret:
            msg = f"Cannot read frame {frame_idx}"
            raise ValueError(msg)
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        self._cap.release()


# /home/alex/data/dvc.yaml's extract_frames stage runs ffmpeg with -vsync 0
# (every decoded frame kept, none dropped/duplicated) through the image2
# muxer, which numbers output files 1-based in source order — so cv2's
# 0-based frame_idx N is jpg file N+1.
_JPG_INDEX_OFFSET = 1
_JPG_FILENAME_PATTERN = "{idx:09d}.jpg"


class _JpgFrameSource:
    """Reads pre-extracted frames/cam_front_left.pii.mp4/<W>x<H>/*.jpg frames —
    the same fixed-size JPEGs training itself was built from (see
    _preprocess_image's docstring), so this source skips the video decode
    (and its approximate downscale-to-native-size step) entirely.
    """

    def __init__(self, frames_dir: Path) -> None:
        if not frames_dir.is_dir():
            msg = f"Frames dir not found: {frames_dir}"
            raise RuntimeError(msg)
        self._frames_dir = frames_dir

    def read(self, frame_idx: int) -> np.ndarray:
        import cv2  # noqa: PLC0415

        path = self._frames_dir / _JPG_FILENAME_PATTERN.format(
            idx=frame_idx + _JPG_INDEX_OFFSET
        )
        frame_bgr = cv2.imread(str(path))
        if frame_bgr is None:
            msg = f"Cannot read frame {path}"
            raise ValueError(msg)
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        pass


# /home/alex/data/dvc.yaml's extract_frames stage, verbatim:
#
#   ffmpeg -y -vsync 0 -threads 0 -hwaccel cuda -hwaccel_output_format cuda
#          -c:v hevc_cuvid -i <video>
#          -filter_complex "scale_npp=576:324,hwdownload,format=nv12"
#          -f image2 -q:v 16 <out>/%09d.jpg
#
# The "ffmpeg" frame source reproduces this, so it is the best case any
# inference-side preprocessing can reach: real NVDEC hevc decode, the resize
# done in NV12 (chroma at half resolution, never round-tripping through RGB --
# which is what the cv2 path cannot reproduce), and the real mjpeg encoder.
#
# One substitution: scale_npp is unavailable outside NVIDIA's proprietary NPP
# build, so scale_cuda stands in. That single difference is the whole residual:
# measured against the real training jpgs (uint8 MAE, 576x324),
#
#   ffmpeg + scale_cuda bicubic (this source) 1.59   <- the floor
#   ffmpeg + scale_cuda bilinear             2.40
#   ffmpeg software scale, still in NV12      2.84
#   cv2 INTER_CUBIC + mjpeg q16 ("video_jpeg") 3.80
#   cv2 INTER_CUBIC alone ("video")           4.42
#   two ADJACENT real training frames         1.34
#
# so the ideal case lands just above one frame of scene motion, and ~2.4x
# closer than the best pure-Python path.
_FFMPEG_INTERP_ALGO = "bicubic"  # scale_cuda's default; also its closest to NPP
# -vsync 0 is a deprecated global spelling of this output option; byte-identical
# output was verified against extract_frames' own `-vsync 0` on real drive data.
_FFMPEG_FPS_MODE = "passthrough"
# Frames are extracted in blocks so consecutive episodes reuse one ffmpeg run.
# `select` counts decoded frames, so a block costs a decode from frame 0 up to
# its end (~3s for 512 frames at start_frame~2900 on an RTX 5090); this keeps
# indexing exact, which -ss seeking would not.
_FFMPEG_BLOCK_FRAMES = 512


class _FfmpegFrameSource:
    """Re-runs extract_frames' own ffmpeg command to produce frames.

    The baseline source: every other source approximates this pipeline, so the
    gap between "ffmpeg" and "jpg" is the irreducible floor (scale_npp vs
    scale_cuda) and any source further from "jpg" than this one is losing
    something recoverable. Frames come out already at DEFAULT_IMAGE_SIZE and
    already mjpeg-compressed, exactly like the "jpg" source, so _preprocess_image
    must NOT additionally apply jpeg_qv to them.
    """

    # _load_batch builds a fresh source per episode while consecutive episodes
    # overlap by 5 of their 6 frames, so extraction is cached for the life of
    # the process rather than per instance — otherwise every episode re-runs
    # ffmpeg over the same block.
    _tmpdir: ClassVar[Any] = None
    _blocks: ClassVar[set[tuple[str, bool, int]]] = set()

    def __init__(self, video_path: Path, *, hwaccel: bool = True) -> None:
        import shutil  # noqa: PLC0415
        import tempfile  # noqa: PLC0415

        if shutil.which("ffmpeg") is None:
            msg = "frame_sources=[ffmpeg] needs `ffmpeg` on PATH"
            raise RuntimeError(msg)
        self._video_path = video_path
        self._hwaccel = hwaccel
        if type(self)._tmpdir is None:  # noqa: SLF001
            type(self)._tmpdir = tempfile.TemporaryDirectory(  # noqa: SLF001
                prefix="rmind-ffmpeg-frames-"
            )
        self._root = Path(type(self)._tmpdir.name)  # noqa: SLF001

    def _extract_block(self, block: int) -> Path:
        import subprocess  # noqa: PLC0415, S404 — fixed argv, no shell

        key = (str(self._video_path), self._hwaccel, block)
        out_dir = self._root / f"{'hw' if self._hwaccel else 'sw'}-{block}"
        if key in type(self)._blocks:  # noqa: SLF001
            return out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        first = block * _FFMPEG_BLOCK_FRAMES
        last = first + _FFMPEG_BLOCK_FRAMES - 1
        height, width = DEFAULT_IMAGE_SIZE
        decode: list[str] = []
        if self._hwaccel:
            decode = [
                "-hwaccel", "cuda",
                "-hwaccel_output_format", "cuda",
                "-c:v", "hevc_cuvid",
            ]  # fmt: skip
            scale = f"scale_cuda={width}:{height}:interp_algo={_FFMPEG_INTERP_ALGO}"
            chain = f"{scale},hwdownload,format=nv12"
        else:
            # keep the resize in NV12 like the real pipeline, minus the GPU
            chain = f"format=nv12,scale={width}:{height},format=nv12"

        command = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-threads", "0",
            *decode,
            "-i", str(self._video_path),
            "-filter_complex", rf"select='between(n\,{first}\,{last})',{chain}",
            "-fps_mode", _FFMPEG_FPS_MODE,
            "-frames:v", str(_FFMPEG_BLOCK_FRAMES),
            "-f", "image2",
            "-q:v", str(_DEFAULT_JPEG_QV),
            str(out_dir / _JPG_FILENAME_PATTERN.replace("{idx:09d}", "%09d")),
        ]  # fmt: skip
        result = subprocess.run(command, capture_output=True, text=True, check=False)  # noqa: S603
        if result.returncode != 0:
            msg = (
                f"ffmpeg failed extracting frames {first}..{last} "
                f"(hwaccel={self._hwaccel}): {result.stderr.strip()[:500]}"
            )
            raise RuntimeError(msg)

        type(self)._blocks.add(key)  # noqa: SLF001
        logger.debug(
            "ffmpeg extracted frame block",
            frames=f"{first}..{last}",
            hwaccel=self._hwaccel,
        )
        return out_dir

    def read(self, frame_idx: int) -> np.ndarray:
        import cv2  # noqa: PLC0415

        block, offset = divmod(frame_idx, _FFMPEG_BLOCK_FRAMES)
        # ffmpeg's image2 muxer numbers the SELECTED frames 1-based, so the
        # block's first frame is file 1 — same convention as extract_frames'
        # whole-video run, just rebased (see _JPG_INDEX_OFFSET).
        path = self._extract_block(block) / _JPG_FILENAME_PATTERN.format(
            idx=offset + _JPG_INDEX_OFFSET
        )
        frame_bgr = cv2.imread(str(path))
        if frame_bgr is None:
            msg = f"Cannot read ffmpeg-extracted frame {frame_idx} at {path}"
            raise ValueError(msg)
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        # extracted blocks are process-wide (see _blocks); the TemporaryDirectory
        # cleans itself up at interpreter exit
        pass


_FrameSource = _VideoFrameSource | _JpgFrameSource | _FfmpegFrameSource


def _jpg_frames_dir(data_dir: Path) -> Path:
    # dvc.yaml's extract_frames stage names the output dir <width>x<height>
    # (params.yaml's frames.scale), which is DEFAULT_IMAGE_SIZE's (H, W) —
    # training's native JPEG frame resolution — spelled WxH.
    h, w = DEFAULT_IMAGE_SIZE
    return data_dir / "frames" / "cam_front_left.pii.mp4" / f"{w}x{h}"


@dataclass
class _BatchRequest:
    video_path: Path
    frames_dir: Path
    metadata: _MetadataReader
    waypoints: _WaypointLoader
    start_frame: int
    frame_step: int
    image_size: tuple[int, int]
    source: str = "video"  # "video" | "jpg" | "video_jpeg" | "ffmpeg"
    jpeg_qv: int = _DEFAULT_JPEG_QV  # only applied when source == "video_jpeg"
    ffmpeg_hwaccel: bool = True  # only applied when source == "ffmpeg"

    def make_frame_source(self) -> _FrameSource:
        if self.source == "jpg":
            return _JpgFrameSource(self.frames_dir)
        if self.source == "ffmpeg":
            return _FfmpegFrameSource(self.video_path, hwaccel=self.ffmpeg_hwaccel)
        # "video" and "video_jpeg" both decode video
        return _VideoFrameSource(self.video_path)


def _read_timestep(
    source: _FrameSource, frame_idx: int, request: _BatchRequest
) -> tuple[np.ndarray, _VehicleState, np.ndarray]:
    frame_rgb = source.read(frame_idx)
    state = request.metadata.get_state_for_frame(frame_idx)
    gnss = request.metadata.get_gnss_for_frame(frame_idx)
    jpeg_qv = request.jpeg_qv if request.source == "video_jpeg" else None
    image = _preprocess_image(frame_rgb, request.image_size, jpeg_qv=jpeg_qv)
    wp = request.waypoints.get_for_gnss(gnss)
    return image, state, wp


def _read_ground_truth(
    source: _FrameSource, gt_idx: int, metadata: _MetadataReader
) -> GroundTruth:
    source.read(gt_idx)  # validates the GT frame exists; pixels unused
    gt_state = metadata.get_state_for_frame(gt_idx)
    return GroundTruth(
        gas=gt_state.gas_pedal,
        brake=gt_state.brake_pedal,
        steer=gt_state.steering_angle,
        turn=gt_state.turn_signal,
    )


def _read_episode(
    source: _FrameSource, request: _BatchRequest
) -> tuple[list[np.ndarray], list[_VehicleState], list[np.ndarray], GroundTruth]:
    images, states, wps_list = [], [], []

    for t in range(NUM_TIMESTEPS):
        frame_idx = request.start_frame + t * request.frame_step
        image, state, wp = _read_timestep(source, frame_idx, request)
        images.append(image)
        states.append(state)
        wps_list.append(wp)

    # Ground truth: frame AFTER the episode (matches driver)
    gt_idx = request.start_frame + NUM_TIMESTEPS * request.frame_step
    gt = _read_ground_truth(source, gt_idx, request.metadata)
    return images, states, wps_list, gt


def _load_batch(request: _BatchRequest) -> tuple[dict[str, np.ndarray], GroundTruth]:
    source = request.make_frame_source()
    try:
        images, states, wps_list, gt = _read_episode(source, request)
    finally:
        source.close()

    wp_array = np.clip(np.stack(wps_list) / 100.0, -1.0, 1.0)  # [T, N, 2], 100m horizon

    batch = {
        _K_CAM: np.stack(images)[np.newaxis].astype(np.float32),  # [1, T, 3, H, W]
        _K_SPEED: np.array([s.speed for s in states], dtype=np.float32).reshape(
            1, -1, 1
        ),
        _K_GAS: np.array([s.gas_pedal for s in states], dtype=np.float32).reshape(
            1, -1, 1
        ),
        _K_BRAKE: np.array([s.brake_pedal for s in states], dtype=np.float32).reshape(
            1, -1, 1
        ),
        _K_STEER: np.array(
            [s.steering_angle for s in states], dtype=np.float32
        ).reshape(1, -1, 1),
        _K_TURN: np.array([s.turn_signal for s in states], dtype=np.int32).reshape(
            1, -1, 1
        ),
        _K_WP: wp_array[np.newaxis].astype(np.float32),  # [1, T, N, 2]
    }
    return batch, gt


# ── Inference backends ────────────────────────────────────────────────────────


class _ONNXBackend:
    def __init__(self, model_path: Path) -> None:
        import onnxruntime as ort  # noqa: PLC0415

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if torch.cuda.is_available()
            else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_names = {inp.name for inp in self.session.get_inputs()}
        self.output_map: dict[str, int] = {
            o.name: i for i, o in enumerate(self.session.get_outputs())
        }
        self.image_size = self._read_image_size()
        logger.info(
            "ONNX model loaded",
            path=str(model_path),
            image_size=self.image_size,
            providers=providers,
        )

    def _read_image_size(self) -> tuple[int, int]:
        for inp in self.session.get_inputs():
            if (
                "cam_front_left" in inp.name.lower()
                and len(inp.shape or []) >= _ONNX_IMAGE_INPUT_NDIM
            ):
                h, w = inp.shape[3], inp.shape[4]
                if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
                    return (h, w)
        return DEFAULT_IMAGE_SIZE

    def run(
        self, batch: dict[str, np.ndarray], cache: np.ndarray | None
    ) -> tuple[Predictions, np.ndarray | None]:
        inputs = dict(batch)
        inputs["cached_projected_embeddings"] = (
            cache
            if cache is not None
            else np.zeros((1, 0, EMBED_DIM), dtype=np.float32)
        )
        # ONNX input names are the short canonical names (e.g. "cam_front_left"),
        # while batch keys carry a "batch_data_..." prefix — match by suffix,
        # case-insensitively (identical to driver's ONNXModel.run).
        matched = {
            n: v
            for n in self.input_names
            for k, v in inputs.items()
            if k.lower().endswith(n.lower())
        }

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = self.session.run(None, matched)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        def _get(key: str, fallback_idx: int) -> np.ndarray:
            return cast("np.ndarray", outputs[self.output_map.get(key, fallback_idx)])

        preds = Predictions(
            gas=float(_get("policy.continuous.gas_pedal", 0).squeeze()),
            brake=float(_get("policy.continuous.brake_pedal", 1).squeeze()),
            steer=float(_get("policy.continuous.steering_angle", 2).squeeze()),
            turn=int(_get("policy.discrete.turn_signal", 3).squeeze()),
            time_ms=elapsed_ms,
        )
        new_cache = (
            cast("np.ndarray", outputs[self.output_map["cached_projected_embeddings"]])
            if "cached_projected_embeddings" in self.output_map
            else None
        )
        return preds, new_cache


_PATCH_POLICY_TARGET_MARKER = "rmind.models.patch_policy.PatchPolicy"


class _WandbBackend:
    def __init__(
        self,
        artifact: str,
        *,
        target: str | None = None,
        hparams_jq: str | None = None,
        strict: bool | None = None,
    ) -> None:
        if target is not None and _PATCH_POLICY_TARGET_MARKER in target:
            # PatchPolicy.load_for_export already applies the same "strip the
            # in-graph image preprocessing" trick load_from_wandb_artifact's
            # hparams_jq does for ControlTransformer (see its docstring): it sets
            # input_transform[2]["image"] = nn.Identity() and sample_codes=False.
            # There is no hparams_jq/strict knob to thread through here.
            from rmind.models.patch_policy import PatchPolicy  # noqa: PLC0415

            self.model = PatchPolicy.load_for_export(artifact).eval()
            self.is_patch_policy = True
        else:
            from rmind.models.control_transformer import ControlTransformer

            if hparams_jq is None:
                # _validate_config should have already rejected this — defense in depth.
                msg = "wandb_model= for a ControlTransformer export requires hparams_jq"
                raise ValueError(msg)

            # map_location="cpu": we .to(self.device) right below anyway, and it avoids
            # torch.load trying to restore the checkpoint's original CUDA tensors when
            # hparams_jq is set (that path doesn't default map_location like Lightning's
            # own load_from_checkpoint does), which errors out on a CPU-only machine.
            kwargs: dict[str, Any] = {"map_location": "cpu", "hparams_jq": hparams_jq}
            if strict is not None:
                kwargs["strict"] = strict
            self.model = ControlTransformer.load_from_wandb_artifact(
                artifact, **kwargs
            ).eval()
            self.is_patch_policy = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        logger.info("Wandb model loaded", artifact=artifact, device=self.device)

    def run(self, onnx_batch: dict[str, np.ndarray]) -> Predictions:
        dev = self.device

        def _t(arr: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(arr).to(dev)

        # hparams_jq (ControlTransformer) / load_for_export (PatchPolicy) —
        # see __init__ — both strip the model's own image-encoder Rearrange
        # along with the rest of input_transform, so it now expects CHW
        # straight through — same layout _K_CAM already has.
        #
        # NOTE: only cam_front_left is fed. That matches config/model/yaak/
        # patch_policy/raw.yaml's Remapper (single-camera image path) for
        # every current PatchPolicy experiment, but config/export/yaak/
        # patch_policy/finetuned.yaml's ONNX graph additionally declares
        # cam_left_forward/cam_right_forward inputs. If a checkpoint's own
        # saved hparams actually reads those (this benchmark can't introspect
        # a wandb artifact's hparams ahead of loading it), its predictions
        # here would be wrong rather than erroring — this local benchmark
        # harness (whether sourcing frames from cam_front_left.pii.mp4 or its
        # pre-extracted jpg frames, see frame_sources=) only ever has
        # front-camera footage to feed. Extending _load_batch/_MetadataReader
        # to source left/right camera frames is out of scope of this fix.
        cam = _t(onnx_batch[_K_CAM])

        # Reconstruct the nested {"data": {...}} batch that ControlTransformer/
        # PatchPolicy.forward expect (both use the same Remapper path names)
        data: dict = {
            _PT_CAM: cam,
            _PT_SPEED: _t(onnx_batch[_K_SPEED]),
            _PT_WP: _t(onnx_batch[_K_WP]),
        }
        # PatchPolicy.forward never reads the action chunk (require_chunk=False) —
        # see config/export/yaak/patch_policy/finetuned.yaml's docstring — but its
        # ChunkFields step unconditionally unfolds gas/brake/steer/turn into rolling
        # (episode_length, action_horizon) windows, which needs
        # episode_length + action_horizon - 1 raw timesteps of history; we only have
        # NUM_TIMESTEPS. Omitting the keys entirely (rather than supplying
        # NUM_TIMESTEPS-long real values) makes Remapper/ChunkFields propagate None
        # instead, which ChunkFields short-circuits on — matching how the real ONNX
        # export (which never had these fields to begin with) behaves.
        if not self.is_patch_policy:
            data.update({
                _PT_GAS: _t(onnx_batch[_K_GAS]),
                _PT_BRAKE: _t(onnx_batch[_K_BRAKE]),
                _PT_STEER: _t(onnx_batch[_K_STEER]),
                _PT_TURN: _t(onnx_batch[_K_TURN]),
            })
        batch: dict = {"data": data}

        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = self.model(batch)
        if dev == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        policy = out["policy"]
        return Predictions(
            gas=float(policy["continuous"]["gas_pedal"].squeeze()),
            brake=float(policy["continuous"]["brake_pedal"].squeeze()),
            steer=float(policy["continuous"]["steering_angle"].squeeze()),
            turn=int(policy["discrete"]["turn_signal"].squeeze()),
            time_ms=elapsed_ms,
        )


# ── Output helpers ────────────────────────────────────────────────────────────


def _print_timing_table(
    label_backend: dict[str, Any],
    all_preds: dict[str, list[Predictions]],
    n_episodes: int,
) -> None:
    from tabulate import tabulate  # noqa: PLC0415

    rows = []
    has_cpu_star = False
    for label, backend in label_backend.items():
        times = np.array([p.time_ms for p in all_preds[label]])
        if isinstance(backend, _ONNXBackend):
            providers = backend.session.get_providers()
            on_gpu = "CUDAExecutionProvider" in providers
            device = "CUDA" if on_gpu else "CPU*"
            if not on_gpu:
                has_cpu_star = True
        else:
            device = backend.device.upper()
        rows.append([
            label,
            device,
            "full",
            n_episodes,
            f"{times.mean():.1f}",
            f"{times.min():.1f}",
            f"{times.max():.1f}",
            f"{times.std():.1f}",
            "✓" if times.mean() < _TARGET_LATENCY_MS else "✗",
        ])

    sep = "=" * 100
    print(f"\n{sep}")  # noqa: T201
    print(f"TIMING RESULTS (GPU) - {n_episodes} episodes")  # noqa: T201
    print(sep)  # noqa: T201
    print(  # noqa: T201
        tabulate(
            rows,
            headers=[
                "Model",
                "Device",
                "Type",
                "Episodes",
                "Mean (ms)",
                "Min (ms)",
                "Max (ms)",
                "Std (ms)",
                "10 Hz OK?",
            ],
            tablefmt="grid",
            colalign=(
                "left",
                "left",
                "left",
                "right",
                "right",
                "right",
                "right",
                "right",
                "right",
            ),
        )
    )
    if has_cpu_star:
        print("  * ONNX runs on CPU (no CUDAExecutionProvider). GPU not available.")  # noqa: T201


def _print_error_table(
    labels: Sequence[str],
    all_preds: dict[str, list[Predictions]],
    all_gt: list[GroundTruth],
) -> None:
    from tabulate import tabulate  # noqa: PLC0415

    rows = []
    for label in labels:
        preds = all_preds[label]
        gas_e = [abs(p.gas - g.gas) for p, g in zip(preds, all_gt, strict=False)]
        brake_e = [abs(p.brake - g.brake) for p, g in zip(preds, all_gt, strict=False)]
        steer_e = [abs(p.steer - g.steer) for p, g in zip(preds, all_gt, strict=False)]
        turn_match = (
            np.mean([p.turn == g.turn for p, g in zip(preds, all_gt, strict=False)])
            * 100
        )
        rows.append([
            label,
            f"{np.mean(gas_e):.6f}",
            f"{np.mean(brake_e):.6f}",
            f"{np.mean(steer_e):.6f}",
            f"{np.max(gas_e):.6f}",
            f"{np.max(brake_e):.6f}",
            f"{np.max(steer_e):.6f}",
            f"{turn_match:.1f}%",
        ])

    print("\nERROR VS GROUND TRUTH:")  # noqa: T201
    print(  # noqa: T201
        tabulate(
            rows,
            headers=[
                "Model",
                "Gas MAE",
                "Brake MAE",
                "Steer MAE",
                "Gas Max",
                "Brake Max",
                "Steer Max",
                "Turn Match %",
            ],
            tablefmt="grid",
        )
    )


def _print_validation(
    labels: Sequence[str], all_preds: dict[str, list[Predictions]]
) -> None:
    from tabulate import tabulate  # noqa: PLC0415

    labels = list(labels)
    sep = "=" * 100
    print("\nVALIDATION CHECKS")  # noqa: T201
    print(sep)  # noqa: T201
    if len(labels) < _MIN_BACKENDS_FOR_COMPARISON:
        print("  Only one backend — no cross-model comparison.")  # noqa: T201
        return

    # Max is a single-frame statistic: one VQ-head bin flip on a near-tie frame
    # dominates it (see frame_source_parity_report.md), so it alone makes a
    # source pair look wildly discordant even when the rest of the run agrees
    # tightly. p95/p99 sit between that and the median's blindness to tails.
    rows = []
    any_violation = False
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            la, lb = labels[i], labels[j]
            pa, pb = all_preds[la], all_preds[lb]
            for chan in ("gas", "brake", "steer"):
                diffs = np.abs([
                    getattr(a, chan) - getattr(b, chan)
                    for a, b in zip(pa, pb, strict=False)
                ])
                chan_max = float(np.max(diffs))
                ok = chan_max <= _VALIDATION_TOLERANCE
                any_violation |= not ok
                rows.append([
                    f"{la} vs {lb}",
                    chan,
                    f"{np.median(diffs):.8f}",
                    f"{np.percentile(diffs, 95):.8f}",
                    f"{np.percentile(diffs, 99):.8f}",
                    f"{chan_max:.8f}",
                    "✓" if ok else "⚠",
                ])

    print(  # noqa: T201
        tabulate(
            rows,
            headers=["Pair", "Chan", "Median", "P95", "P99", "Max", "OK?"],
            tablefmt="grid",
        )
    )
    if any_violation:
        print(  # noqa: T201
            "  ⚠ rows above tolerance may be a single bin-flip frame, not a"
            " systemic diff — compare Max against P99 to tell them apart."
        )


def _print_per_episode(
    labels: Sequence[str], rows: list[dict], all_gt: list[GroundTruth]
) -> None:
    from tabulate import tabulate  # noqa: PLC0415

    labels = list(labels)
    headers = (
        ["Ep", "Frame", "Chan"]
        # + [f"T={i}" for i in range(NUM_TIMESTEPS)]
        + ["GT"]
        + labels
    )
    print("\nPER-EPISODE PREDICTIONS")  # noqa: T201

    table_rows = []
    for row, gt in zip(rows, all_gt, strict=False):
        ep = row["episode"] + 1
        frame = row["frame"]
        for chan_key, chan_label, gt_val in [
            ("brake", "brake", gt.brake),
            ("gas", "gas", gt.gas),
            ("steer", "steer", gt.steer),
        ]:
            hist = row[f"history_{chan_key}"]
            [f"{v:.6f}" for v in hist]
            pred_vals = [f"{row[f'{lbl}_{chan_key}']:.6f}" for lbl in labels]
            table_rows.append([
                ep,
                frame,
                chan_label,
                # *t_vals,
                f"{gt_val:.6f}",
                *pred_vals,
            ])
    print(  # noqa: T201
        tabulate(
            table_rows,
            headers=headers,
            tablefmt="grid",
            colalign=("right",) * len(headers),
        )
    )


def _print_summary_footer(
    labels: Sequence[str],
    all_preds: dict[str, list[Predictions]],
    all_gt: list[GroundTruth],
) -> None:
    labels = list(labels)
    sep = "=" * 100
    print(f"\n{sep}")  # noqa: T201
    print("SUMMARY")  # noqa: T201
    print(sep)  # noqa: T201
    for label in labels:
        times = np.array([p.time_ms for p in all_preds[label]])
        hz = 1000.0 / times.mean() if times.mean() > 0 else 0.0
        print(f"{label}:  {times.mean():.1f} ms ({hz:.1f} Hz)")  # noqa: T201

    all_gas_max = max(
        max(abs(p.gas - g.gas) for p, g in zip(all_preds[lbl], all_gt, strict=False))
        for lbl in labels
    )
    all_steer_max = max(
        max(
            abs(p.steer - g.steer) for p, g in zip(all_preds[lbl], all_gt, strict=False)
        )
        for lbl in labels
    )
    print(f"Max error vs GT:    Gas={all_gas_max:.6f}, Steer={all_steer_max:.6f}")  # noqa: T201

    if len(labels) >= _MIN_BACKENDS_FOR_COMPARISON:
        la, lb = labels[0], labels[1]
        pa, pb = all_preds[la], all_preds[lb]
        max(abs(a.gas - b.gas) for a, b in zip(pa, pb, strict=False))
        max(abs(a.brake - b.brake) for a, b in zip(pa, pb, strict=False))
        max(abs(a.steer - b.steer) for a, b in zip(pa, pb, strict=False))


# ── Main ──────────────────────────────────────────────────────────────────────


_VALID_FRAME_SOURCES = frozenset({"video", "jpg", "video_jpeg", "ffmpeg"})


class Config(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    data_dir: Path
    start_frame: int = 0
    num_episodes: int = 10
    frame_step: int = 10
    onnx: Sequence[Path] | None = None
    # not `model` — that name collides with Hydra's config/model/ group, which makes
    # `model=...` a defaults-list override attempt instead of a plain value override.
    wandb_model: Sequence[str] | None = None
    image_size: tuple[int, int] | None = None
    output: Path | None = None
    warmup: int = 1
    # "video" decodes cam_front_left.pii.mp4 (original behavior); "jpg" reads
    # dvc.yaml's pre-extracted frames/cam_front_left.pii.mp4/<W>x<H>/*.jpg
    # instead — the same fixed-size JPEGs training itself was built from, so
    # it skips _preprocess_image's approximate downscale-to-native-size step.
    # "video_jpeg" decodes video like "video" but also round-trips the
    # downscaled frame through ffmpeg's real mjpeg encoder at jpeg_qv= (see
    # _simulate_offline_jpeg), emulating the offline pipeline's JPEG-encode
    # step without needing scale_npp. "ffmpeg" re-runs extract_frames' own
    # command (see _FfmpegFrameSource) and is the baseline every other source
    # approximates. Pass multiple (e.g.
    # frame_sources=[video,jpg,video_jpeg,ffmpeg]) to run every backend
    # against all of them and compare predictions side by side.
    frame_sources: Sequence[str] = ("video",)
    # ffmpeg mjpeg -q:v (2..31, lower=better) used by the "video_jpeg" source
    # — see _simulate_offline_jpeg for why 16 is the confirmed default. Sweep
    # it on a single source to probe the model's sensitivity to compression;
    # that is a cleaner test than comparing decode paths, whose pixel
    # differences sit near the quantization floor either way.
    jpeg_qv: int = _DEFAULT_JPEG_QV
    # "ffmpeg" source: run extract_frames' real NVDEC + scale_cuda + mjpeg
    # chain (the baseline). Set false to swap the GPU resize for a software
    # one, still in NV12 — measured 2.84 vs 1.59 uint8 MAE against the real
    # training jpgs, so only use it where CUDA is unavailable.
    ffmpeg_hwaccel: bool = True

    @field_validator("onnx", "wandb_model", mode="before")
    @classmethod
    def _coerce_to_list(cls, v: Any) -> Any:
        # lets a single `onnx=path`/`wandb_model=artifact` override stand in for a list
        return v if v is None or isinstance(v, list | tuple) else [v]

    @field_validator("frame_sources", mode="before")
    @classmethod
    def _coerce_frame_sources(cls, v: Any) -> Any:
        return v if isinstance(v, list | tuple) else [v]

    @field_validator("frame_sources")
    @classmethod
    def _check_frame_sources(cls, v: Sequence[str]) -> Sequence[str]:
        if not v:
            msg = "frame_sources must be non-empty"
            raise ValueError(msg)
        bad = sorted(set(v) - _VALID_FRAME_SOURCES)
        if bad:
            msg = (
                f"Unknown frame_sources {bad} (expected {sorted(_VALID_FRAME_SOURCES)})"
            )
            raise ValueError(msg)
        return v

    @field_validator("jpeg_qv")
    @classmethod
    def _check_jpeg_qv(cls, v: int) -> int:
        # fail here rather than mid-run on the first video_jpeg frame
        _ffmpeg_mjpeg_qtable(v)
        return v


def _validate_config(
    config: Config, *, target: str | None, hparams_jq: str | None
) -> None:
    if not config.onnx and not config.wandb_model:
        msg = "At least one of onnx= or wandb_model= is required"
        raise ValueError(msg)

    is_patch_policy = target is not None and _PATCH_POLICY_TARGET_MARKER in target
    if config.wandb_model and target is None:
        msg = (
            "wandb_model= requires export=... (e.g. "
            "export=yaak/control_transformer/finetuned or "
            "export=yaak/patch_policy/finetuned) so the PyTorch model is loaded "
            "with the same export-time stripping as the ONNX export, for the "
            "comparison to be apples-to-apples. See module docstring."
        )
        raise ValueError(msg)
    if config.wandb_model and not is_patch_policy and hparams_jq is None:
        msg = (
            "wandb_model= with a ControlTransformer export= requires that "
            "export's model.hparams_jq — the PyTorch model must be loaded with "
            "the same hparams_jq as the ONNX export for the comparison to be "
            "apples-to-apples. See module docstring."
        )
        raise ValueError(msg)

    for fname in ("metadata.log", "waypoints.json"):
        if not (config.data_dir / fname).exists():
            msg = f"Missing: {config.data_dir / fname}"
            raise FileNotFoundError(msg)

    if {"video", "video_jpeg", "ffmpeg"} & set(config.frame_sources):
        video_path = config.data_dir / "cam_front_left.pii.mp4"
        if not video_path.exists():
            msg = f"Missing: {video_path}"
            raise FileNotFoundError(msg)

    if "jpg" in config.frame_sources:
        frames_dir = _jpg_frames_dir(config.data_dir)
        if not frames_dir.is_dir() or next(frames_dir.glob("*.jpg"), None) is None:
            msg = f"Missing (or empty) jpg frames dir: {frames_dir}"
            raise FileNotFoundError(msg)


def _build_backends(
    config: Config,
    *,
    target: str | None = None,
    hparams_jq: str | None = None,
    strict: bool | None = None,
) -> tuple[dict[str, _ONNXBackend | _WandbBackend], tuple[int, int]]:
    # ordered dict: label → backend
    backends: dict[str, _ONNXBackend | _WandbBackend] = {}
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE

    onnx_paths = config.onnx or []
    for idx, path in enumerate(onnx_paths):
        label = "ONNX" if len(onnx_paths) == 1 else f"ONNX {idx + 1}"
        backend = _ONNXBackend(path)
        backends[label] = backend
        image_size = backend.image_size  # last ONNX wins if multiple

    wandb_artifacts = config.wandb_model or []
    if wandb_artifacts:
        if target is None:
            # _validate_config should have already rejected this — defense in depth.
            msg = "wandb_model= requires export=..."
            raise ValueError(msg)
        for idx, artifact in enumerate(wandb_artifacts):
            label = "PyTorch" if len(wandb_artifacts) == 1 else f"PyTorch {idx + 1}"
            backends[label] = _WandbBackend(
                artifact, target=target, hparams_jq=hparams_jq, strict=strict
            )

    if config.image_size:
        image_size = config.image_size

    logger.info("Backends", labels=list(backends), image_size=image_size)
    return backends, image_size


def _warmup_backends(
    onnx_backends: dict[str, _ONNXBackend], request: _BatchRequest, num_warmup: int
) -> None:
    if not onnx_backends or num_warmup <= 0:
        return

    logger.info("Warming up", runs=num_warmup)
    try:
        wb, _ = _load_batch(request)
        for _ in range(num_warmup):
            for backend in onnx_backends.values():
                backend.run(wb, None)
    except Exception as e:  # noqa: BLE001 — best-effort warmup, must not abort the run
        logger.warning("Warmup failed", error=str(e))


def _run_benchmark(
    backends: dict[str, _ONNXBackend | _WandbBackend],
    request: _BatchRequest,
    num_episodes: int,
) -> tuple[dict[str, list[Predictions]], list[GroundTruth], list[dict]]:
    all_preds: dict[str, list[Predictions]] = {label: [] for label in backends}
    all_gt: list[GroundTruth] = []
    rows: list[dict] = []

    for i in range(num_episodes):
        ep_request = replace(
            request, start_frame=request.start_frame + i * request.frame_step
        )
        try:
            batch, gt = _load_batch(ep_request)
        except (ValueError, RuntimeError) as e:
            logger.warning("Skipping episode", episode=i, error=str(e))
            break

        row: dict = {
            "episode": i,
            "frame": ep_request.start_frame,
            "history_gas": batch[_K_GAS][0, :, 0].tolist(),
            "history_brake": batch[_K_BRAKE][0, :, 0].tolist(),
            "history_steer": batch[_K_STEER][0, :, 0].tolist(),
        }

        for label, backend in backends.items():
            if isinstance(backend, _ONNXBackend):
                # Always pass None (zeros) — matches driver's full-forward behavior
                preds, _ = backend.run(batch, None)
            else:
                preds = backend.run(batch)  # type: ignore[assignment]

            all_preds[label].append(preds)
            row[f"{label}_gas"] = round(preds.gas, 6)
            row[f"{label}_brake"] = round(preds.brake, 6)
            row[f"{label}_steer"] = round(preds.steer, 6)
            row[f"{label}_turn"] = preds.turn
            row[f"{label}_time_ms"] = round(preds.time_ms, 3)

        all_gt.append(gt)
        rows.append(row)

    return all_preds, all_gt, rows


def _resolve_export_model_cfg(
    cfg: DictConfig,
) -> tuple[str | None, str | None, bool | None]:
    """Pull `(_target_, hparams_jq, strict)` out of an `export=...` group's `model:` block.

    `export=yaak/control_transformer/finetuned` (same group export_onnx.py uses)
    injects a top-level `model:` block (it's `@package _global_`) with the exact
    hparams_jq/strict that export applies — select it to load PyTorch with an
    identically-stripped input_transform, for an apples-to-apples ONNX comparison.
    `export=yaak/patch_policy/finetuned` doesn't define hparams_jq/strict at all —
    `PatchPolicy.load_for_export` strips its own in-graph image preprocessing
    unconditionally, no jq patch needed — so those two come back `None` for it.
    """
    export_model_cfg = OmegaConf.select(cfg, "model", default=None)
    if export_model_cfg is None:
        return None, None, None
    return (
        OmegaConf.select(export_model_cfg, "_target_", default=None),
        OmegaConf.select(export_model_cfg, "hparams_jq", default=None),
        OmegaConf.select(export_model_cfg, "strict", default=None),
    )


def _run_all_sources(
    config: Config,
    backends: dict[str, _ONNXBackend | _WandbBackend],
    base_request: _BatchRequest,
) -> tuple[
    dict[str, _ONNXBackend | _WandbBackend],
    dict[str, list[Predictions]],
    list[GroundTruth],
    list[dict],
]:
    """Run every backend against every configured frame source.

    When comparing more than one source, each backend's label gets suffixed
    with its source so every (backend, source) pair shows up as its own
    row/column — _print_validation's generic pairwise diff then automatically
    includes source-vs-source comparisons too.
    """
    multi_source = len(config.frame_sources) > 1
    all_preds: dict[str, list[Predictions]] = {}
    label_backend: dict[str, _ONNXBackend | _WandbBackend] = {}
    all_gt_by_source: dict[str, list[GroundTruth]] = {}
    rows_by_source: dict[str, list[dict]] = {}

    for source in config.frame_sources:
        request = replace(base_request, source=source)
        labeled = (
            {f"{label} ({source})": b for label, b in backends.items()}
            if multi_source
            else backends
        )
        label_backend.update(labeled)
        preds, gt, rows = _run_benchmark(labeled, request, config.num_episodes)
        all_preds.update(preds)
        all_gt_by_source[source] = gt
        rows_by_source[source] = rows

    counts = {source: len(gt) for source, gt in all_gt_by_source.items()}
    if len(set(counts.values())) > 1:
        logger.warning("Episode count differs across frame sources", counts=counts)

    # Ground truth comes from metadata alone (pixels never factor in), so
    # it's identical across sources modulo how many episodes each source's
    # frame reader let complete — use whichever ran fewest.
    canonical_source = min(all_gt_by_source, key=lambda s: counts[s])
    all_gt = all_gt_by_source[canonical_source]

    # Merge every source's per-episode backend columns onto the canonical
    # source's rows (episode/frame/history_* are source-independent already).
    rows = rows_by_source[canonical_source]
    for source, source_rows in rows_by_source.items():
        if source == canonical_source:
            continue
        for row, source_row in zip(rows, source_rows, strict=False):
            row.update({
                k: v
                for k, v in source_row.items()
                if k not in {"episode", "frame"} and not k.startswith("history_")
            })

    return label_backend, all_preds, all_gt, rows


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    target, hparams_jq, strict = _resolve_export_model_cfg(cfg)

    config = Config(**OmegaConf.to_container(cfg, resolve=True))  # ty:ignore[invalid-argument-type]
    _validate_config(config, target=target, hparams_jq=hparams_jq)

    backends, image_size = _build_backends(
        config, target=target, hparams_jq=hparams_jq, strict=strict
    )

    meta = _MetadataReader(config.data_dir / "metadata.log")
    meta.load()
    wps = _WaypointLoader(config.data_dir / "waypoints.json")
    wps.load()

    base_request = _BatchRequest(
        video_path=config.data_dir / "cam_front_left.pii.mp4",
        frames_dir=_jpg_frames_dir(config.data_dir),
        metadata=meta,
        waypoints=wps,
        start_frame=config.start_frame,
        frame_step=config.frame_step,
        image_size=image_size,
        source=config.frame_sources[0],
        jpeg_qv=config.jpeg_qv,
        ffmpeg_hwaccel=config.ffmpeg_hwaccel,
    )

    _warmup_backends(
        {label: b for label, b in backends.items() if isinstance(b, _ONNXBackend)},
        base_request,
        config.warmup,
    )

    logger.info("=" * 70)
    logger.info(
        "Benchmark: %d episodes, start_frame=%d, frame_step=%d, frame_sources=%s",
        config.num_episodes,
        config.start_frame,
        config.frame_step,
        list(config.frame_sources),
    )
    logger.info("=" * 70)

    label_backend, all_preds, all_gt, rows = _run_all_sources(
        config, backends, base_request
    )

    n = len(all_gt)
    if n == 0:
        logger.error("No episodes completed")
        return

    labels = list(all_preds)
    _print_timing_table(label_backend, all_preds, n)
    _print_error_table(labels, all_preds, all_gt)
    _print_validation(labels, all_preds)
    _print_per_episode(labels, rows, all_gt)
    _print_summary_footer(labels, all_preds, all_gt)

    if config.output and rows:
        config.output.parent.mkdir(parents=True, exist_ok=True)
        with Path(config.output).open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        logger.info("Predictions saved", path=str(config.output))
