"""Benchmark ControlTransformer on raw ride data — ONNX and/or wandb model.

Both backends can be active at once so their predictions are compared side by side.
Fixed run parameters (data_dir, start_frame, num_episodes, ...) live in
config/benchmark_onnx.yaml — override just `onnx=` and/or `wandb_model=` per run.

`wandb_model=` always requires `export=...` (e.g.
export=yaak/control_transformer/finetuned): the PyTorch model is then loaded
with the same hparams_jq used to produce the ONNX export, so both backends run
architecturally identical, hparams_jq-stripped models on the same externally
preprocessed input — see tests/test_benchmark_onnx_preprocessing.py for why
that external preprocessing is equivalent to the original, unstripped one.

Usage:
    # ONNX only (compare with drivr's benchmark_all_models.py)
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

# Lowercase batch_data_* keys — match drivr's naming for case-insensitive ONNX input matching
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


# ── Metadata reader (mirrors drivr's RbyteMetadataReader) ────────────────────


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


# ── Waypoint loader (mirrors drivr's WaypointLoader) ─────────────────────────


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


def _preprocess_image(image: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
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
    """
    from torchvision.transforms import v2 as T  # noqa: PLC0415

    # The v2 *classes* (not torchvision.transforms.functional's functional API)
    # — raw.yaml instantiates these same classes, and the two aren't numerically
    # identical (functional.resize leaves a ~1/255-per-pixel rounding residual
    # even with matching interpolation/antialias args, per
    # tests/test_benchmark_onnx_preprocessing.py's investigation history).
    resize_native = T.Resize(list(DEFAULT_IMAGE_SIZE))
    crop = T.CenterCrop(list(_CENTER_CROP_SIZE))
    resize_final = T.Resize(list(image_size))

    # uint8 throughout crop/resize (matching ToDtype running *after* Resize in
    # raw.yaml) to minimize rounding drift relative to the real pipeline.
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)  # uint8
    native = resize_native(tensor)
    cropped = crop(native)
    resized = resize_final(cropped)
    scaled = (resized.float() / 255.0)[0].numpy()  # CHW float32 [0, 1]
    return _normalize_cam(scaled)


def _normalize_cam(cam: np.ndarray) -> np.ndarray:
    """ImageNet mean/std normalize a CHW float32 image in [0, 1]."""
    return (cam - _IMAGENET_MEAN) / _IMAGENET_STD


@dataclass
class _BatchRequest:
    video_path: Path
    metadata: _MetadataReader
    waypoints: _WaypointLoader
    start_frame: int
    frame_step: int
    image_size: tuple[int, int]


def _read_timestep(
    cap: Any, frame_idx: int, request: _BatchRequest
) -> tuple[np.ndarray, _VehicleState, np.ndarray]:
    import cv2  # noqa: PLC0415

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame_bgr = cap.read()
    if not ret:
        msg = f"Cannot read frame {frame_idx}"
        raise ValueError(msg)

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    state = request.metadata.get_state_for_frame(frame_idx)
    gnss = request.metadata.get_gnss_for_frame(frame_idx)
    image = _preprocess_image(frame_rgb, request.image_size)
    wp = request.waypoints.get_for_gnss(gnss)
    return image, state, wp


def _read_ground_truth(cap: Any, gt_idx: int, metadata: _MetadataReader) -> GroundTruth:
    import cv2  # noqa: PLC0415

    cap.set(cv2.CAP_PROP_POS_FRAMES, gt_idx)
    ret, _ = cap.read()
    if not ret:
        msg = f"Cannot read GT frame {gt_idx}"
        raise ValueError(msg)
    gt_state = metadata.get_state_for_frame(gt_idx)
    return GroundTruth(
        gas=gt_state.gas_pedal,
        brake=gt_state.brake_pedal,
        steer=gt_state.steering_angle,
        turn=gt_state.turn_signal,
    )


def _read_episode(
    cap: Any, request: _BatchRequest
) -> tuple[list[np.ndarray], list[_VehicleState], list[np.ndarray], GroundTruth]:
    images, states, wps_list = [], [], []

    for t in range(NUM_TIMESTEPS):
        frame_idx = request.start_frame + t * request.frame_step
        image, state, wp = _read_timestep(cap, frame_idx, request)
        images.append(image)
        states.append(state)
        wps_list.append(wp)

    # Ground truth: frame AFTER the episode (matches drivr)
    gt_idx = request.start_frame + NUM_TIMESTEPS * request.frame_step
    gt = _read_ground_truth(cap, gt_idx, request.metadata)
    return images, states, wps_list, gt


def _load_batch(request: _BatchRequest) -> tuple[dict[str, np.ndarray], GroundTruth]:
    import cv2  # noqa: PLC0415

    cap = cv2.VideoCapture(str(request.video_path))
    if not cap.isOpened():
        msg = f"Cannot open video: {request.video_path}"
        raise RuntimeError(msg)

    try:
        images, states, wps_list, gt = _read_episode(cap, request)
    finally:
        cap.release()

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
        # case-insensitively (identical to drivr's ONNXModel.run).
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


class _WandbBackend:
    def __init__(
        self,
        artifact: str,
        *,
        hparams_jq: str,
        strict: bool | None = None,
    ) -> None:
        from rmind.models.control_transformer import ControlTransformer  # noqa: PLC0415

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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        logger.info("Wandb model loaded", artifact=artifact, device=self.device)

    def run(self, onnx_batch: dict[str, np.ndarray]) -> Predictions:
        dev = self.device

        def _t(arr: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(arr).to(dev)

        # hparams_jq (required, see __init__) strips the model's own image
        # encoder Rearrange along with the rest of input_transform, so it now
        # expects CHW straight through — same layout _K_CAM already has.
        cam = _t(onnx_batch[_K_CAM])

        # Reconstruct the nested {"data": {...}} batch that ControlTransformer.forward expects
        batch: dict = {
            "data": {
                _PT_CAM: cam,
                _PT_SPEED: _t(onnx_batch[_K_SPEED]),
                _PT_GAS: _t(onnx_batch[_K_GAS]),
                _PT_BRAKE: _t(onnx_batch[_K_BRAKE]),
                _PT_STEER: _t(onnx_batch[_K_STEER]),
                _PT_TURN: _t(onnx_batch[_K_TURN]),
                _PT_WP: _t(onnx_batch[_K_WP]),
            }
        }

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
    backends: dict, all_preds: dict[str, list[Predictions]], n_episodes: int
) -> None:
    from tabulate import tabulate  # noqa: PLC0415

    rows = []
    has_cpu_star = False
    for label, backend in backends.items():
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
    backends: dict, all_preds: dict[str, list[Predictions]], all_gt: list[GroundTruth]
) -> None:
    from tabulate import tabulate  # noqa: PLC0415

    rows = []
    for label in backends:
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


def _print_validation(backends: dict, all_preds: dict[str, list[Predictions]]) -> None:
    labels = list(backends)
    sep = "=" * 100
    print("\nVALIDATION CHECKS")  # noqa: T201
    print(sep)  # noqa: T201
    if len(labels) < _MIN_BACKENDS_FOR_COMPARISON:
        print("  Only one backend — no cross-model comparison.")  # noqa: T201
        return

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            la, lb = labels[i], labels[j]
            pa, pb = all_preds[la], all_preds[lb]
            gas_max = max(abs(a.gas - b.gas) for a, b in zip(pa, pb, strict=False))
            brake_max = max(
                abs(a.brake - b.brake) for a, b in zip(pa, pb, strict=False)
            )
            steer_max = max(
                abs(a.steer - b.steer) for a, b in zip(pa, pb, strict=False)
            )
            ok = (
                gas_max <= _VALIDATION_TOLERANCE
                and brake_max <= _VALIDATION_TOLERANCE
                and steer_max <= _VALIDATION_TOLERANCE
            )
            icon = "✓" if ok else "⚠"
            verb = "agree within" if ok else "diffs exceed"
            print(  # noqa: T201
                f"{icon} {la} vs {lb} {verb} tolerance "
                f"(max: gas={gas_max:.8f}, brake={brake_max:.8f}, steer={steer_max:.8f})"
            )


def _print_per_episode(
    backends: dict, rows: list[dict], all_gt: list[GroundTruth]
) -> None:
    from tabulate import tabulate  # noqa: PLC0415

    labels = list(backends)
    headers = (
        ["Ep", "Field"] + [f"T={i}" for i in range(NUM_TIMESTEPS)] + ["GT"] + labels
    )
    sep = "=" * 150
    print("\nPER-EPISODE PREDICTIONS")  # noqa: T201
    print(sep)  # noqa: T201

    for row, gt in zip(rows, all_gt, strict=False):
        ep = row["episode"] + 1
        table_rows = []
        for field_key, field_label, gt_val in [
            ("brake", "brake", gt.brake),
            ("gas", "gas", gt.gas),
            ("steer", "steer", gt.steer),
        ]:
            hist = row[f"history_{field_key}"]
            t_vals = [f"{v:.6f}" for v in hist]
            pred_vals = [f"{row[f'{lbl}_{field_key}']:.6f}" for lbl in labels]
            table_rows.append([ep, field_label, *t_vals, f"{gt_val:.6f}", *pred_vals])
        print(tabulate(table_rows, headers=headers, tablefmt="grid"))  # noqa: T201


def _print_summary_footer(
    backends: dict, all_preds: dict[str, list[Predictions]], all_gt: list[GroundTruth]
) -> None:
    sep = "=" * 100
    print(f"\n{sep}")  # noqa: T201
    print("SUMMARY")  # noqa: T201
    print(sep)  # noqa: T201
    for label in backends:
        times = np.array([p.time_ms for p in all_preds[label]])
        hz = 1000.0 / times.mean() if times.mean() > 0 else 0.0
        print(f"{label}:  {times.mean():.1f} ms ({hz:.1f} Hz)")  # noqa: T201

    all_gas_max = max(
        max(abs(p.gas - g.gas) for p, g in zip(all_preds[lbl], all_gt, strict=False))
        for lbl in backends
    )
    all_steer_max = max(
        max(
            abs(p.steer - g.steer) for p, g in zip(all_preds[lbl], all_gt, strict=False)
        )
        for lbl in backends
    )
    print(f"Max error vs GT:    Gas={all_gas_max:.6f}, Steer={all_steer_max:.6f}")  # noqa: T201

    labels = list(backends)
    if len(labels) >= _MIN_BACKENDS_FOR_COMPARISON:
        la, lb = labels[0], labels[1]
        pa, pb = all_preds[la], all_preds[lb]
        max(abs(a.gas - b.gas) for a, b in zip(pa, pb, strict=False))
        max(abs(a.brake - b.brake) for a, b in zip(pa, pb, strict=False))
        max(abs(a.steer - b.steer) for a, b in zip(pa, pb, strict=False))


# ── Main ──────────────────────────────────────────────────────────────────────


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

    @field_validator("onnx", "wandb_model", mode="before")
    @classmethod
    def _coerce_to_list(cls, v: Any) -> Any:
        # lets a single `onnx=path`/`wandb_model=artifact` override stand in for a list
        return v if v is None or isinstance(v, list | tuple) else [v]


def _validate_config(config: Config, *, hparams_jq: str | None) -> None:
    if not config.onnx and not config.wandb_model:
        msg = "At least one of onnx= or wandb_model= is required"
        raise ValueError(msg)

    if config.wandb_model and hparams_jq is None:
        msg = (
            "wandb_model= requires export=... (e.g. "
            "export=yaak/control_transformer/finetuned) — the PyTorch model must "
            "be loaded with the same hparams_jq as the ONNX export for the "
            "comparison to be apples-to-apples. See module docstring."
        )
        raise ValueError(msg)

    for fname in ("cam_front_left.pii.mp4", "metadata.log", "waypoints.json"):
        if not (config.data_dir / fname).exists():
            msg = f"Missing: {config.data_dir / fname}"
            raise FileNotFoundError(msg)


def _build_backends(
    config: Config, *, hparams_jq: str | None = None, strict: bool | None = None
) -> tuple[dict[str, _ONNXBackend | _WandbBackend], tuple[int, int]]:
    # ordered dict: label → backend
    backends: dict[str, _ONNXBackend | _WandbBackend] = {}
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE

    onnx_paths = config.onnx or []
    for idx, path in enumerate(onnx_paths):
        label = "ONNX Full" if len(onnx_paths) == 1 else f"ONNX Full {idx + 1}"
        backend = _ONNXBackend(path)
        backends[label] = backend
        image_size = backend.image_size  # last ONNX wins if multiple

    wandb_artifacts = config.wandb_model or []
    if wandb_artifacts:
        if hparams_jq is None:
            # _validate_config should have already rejected this — defense in depth.
            msg = "wandb_model= requires export=..."
            raise ValueError(msg)
        for idx, artifact in enumerate(wandb_artifacts):
            label = (
                "PyTorch Native"
                if len(wandb_artifacts) == 1
                else f"PyTorch Native {idx + 1}"
            )
            backends[label] = _WandbBackend(
                artifact, hparams_jq=hparams_jq, strict=strict
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
                # Always pass None (zeros) — matches drivr's full-forward behavior
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


def _resolve_export_hparams(cfg: DictConfig) -> tuple[str | None, bool | None]:
    # `export=yaak/control_transformer/finetuned` (same group export_onnx.py uses)
    # injects a top-level `model:` block (it's `@package _global_`) with the exact
    # hparams_jq/strict that export applies — select it to load PyTorch with an
    # identically-stripped input_transform, for an apples-to-apples ONNX comparison.
    export_model_cfg = OmegaConf.select(cfg, "model", default=None)
    if export_model_cfg is None:
        return None, None
    return export_model_cfg.hparams_jq, export_model_cfg.strict


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    hparams_jq, strict = _resolve_export_hparams(cfg)

    config = Config(**OmegaConf.to_container(cfg, resolve=True))  # ty:ignore[invalid-argument-type]
    _validate_config(config, hparams_jq=hparams_jq)

    backends, image_size = _build_backends(config, hparams_jq=hparams_jq, strict=strict)

    meta = _MetadataReader(config.data_dir / "metadata.log")
    meta.load()
    wps = _WaypointLoader(config.data_dir / "waypoints.json")
    wps.load()

    request = _BatchRequest(
        video_path=config.data_dir / "cam_front_left.pii.mp4",
        metadata=meta,
        waypoints=wps,
        start_frame=config.start_frame,
        frame_step=config.frame_step,
        image_size=image_size,
    )

    onnx_backends = {
        label: b for label, b in backends.items() if isinstance(b, _ONNXBackend)
    }
    _warmup_backends(onnx_backends, request, config.warmup)

    logger.info("=" * 70)
    logger.info(
        "Benchmark: %d episodes, start_frame=%d, frame_step=%d",
        config.num_episodes,
        config.start_frame,
        config.frame_step,
    )
    logger.info("=" * 70)

    all_preds, all_gt, rows = _run_benchmark(backends, request, config.num_episodes)

    n = len(all_gt)
    if n == 0:
        logger.error("No episodes completed")
        return

    _print_timing_table(backends, all_preds, n)
    _print_error_table(backends, all_preds, all_gt)
    _print_validation(backends, all_preds)
    _print_per_episode(backends, rows, all_gt)
    _print_summary_footer(backends, all_preds, all_gt)

    if config.output and rows:
        config.output.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0].keys())
        with Path(config.output).open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        logger.info("Predictions saved", path=str(config.output))
