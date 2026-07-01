"""Benchmark ControlTransformer on raw ride data — ONNX and/or wandb model.

Both backends can be active at once so their predictions are compared side by side.

Usage:
    # ONNX only (compare with drivr's benchmark_all_models.py)
    just benchmark-onnx \\
        --onnx ~/rmind/outputs/.../model.onnx \\
        --data-dir ~/data/Niro122-HQ/2023-05-25--09-34-14 \\
        --start-frame 2910 --num-episodes 50

    # wandb PyTorch only
    just benchmark-onnx \\
        --model yaak/rmind/model-XXXXXXXX:vN \\
        --data-dir ~/data/... --start-frame 2910 --num-episodes 50

    # Both side by side (ONNX vs torch, same batches)
    just benchmark-onnx \\
        --onnx ~/rmind/outputs/.../model.onnx \\
        --model yaak/rmind/model-XXXXXXXX:vN \\
        --data-dir ~/data/... --start-frame 2910 --num-episodes 50 \\
        --output /tmp/rmind.csv
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import mmap
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import structlog
import torch

logger = structlog.get_logger(__name__)

EMBED_DIM = 384
NUM_TIMESTEPS = 6
NUM_WAYPOINTS = 10
DEFAULT_IMAGE_SIZE = (324, 576)  # (H, W)

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
        from rbyte.io.yaak.metadata.message_iterator import YaakMetadataMessageIterator  # noqa: PLC0415

        with open(self.metadata_path, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            for msg_cls, msg_bytes in YaakMetadataMessageIterator(mm):
                msg = msg_cls()
                msg.ParseFromString(msg_bytes)
                ts = msg.time_stamp.ToMicroseconds() / 1_000_000.0
                name = msg_cls.__name__
                if name == "VehicleMotion":
                    self._motion.append(_VehicleState(
                        timestamp=ts,
                        speed=getattr(msg, "speed", 0.0),
                        gas_pedal=getattr(msg, "gas_pedal_normalized", 0.0),
                        brake_pedal=getattr(msg, "brake_pedal_normalized", 0.0),
                        steering_angle=getattr(msg, "steering_angle_normalized", 0.0),
                    ))
                elif name == "ImageMetadata" and msg.camera_name == self.camera_name:
                    self._frame_ts[msg.frame_idx] = ts
                elif name == "VehicleState":
                    self._turn_entries.append((ts, int(getattr(msg, "turn_signal", 0))))
                elif name == "Gnss":
                    self._gnss.append(_GnssPosition(
                        timestamp=ts,
                        latitude=msg.latitude,
                        longitude=msg.longitude,
                        heading=getattr(msg, "heading", 0.0),
                    ))
            mm.close()

        self._gnss.sort(key=lambda x: x.timestamp)
        self._turn_entries.sort(key=lambda x: x[0])
        logger.info(
            "Metadata loaded",
            motion=len(self._motion),
            gnss=len(self._gnss),
            frames=len(self._frame_ts),
        )

    def _nearest_ts(self, entries: list, ts: float, key=lambda x: x.timestamp) -> int:
        times = [key(e) for e in entries]
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
        with open(self.waypoints_path) as f:
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
                    "lon": lon, "lat": lat, "x": x, "y": y,
                })
        self._wps.sort(key=lambda w: w["timestamp"])
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

def _preprocess_image(image: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    """HWC uint8 → resize → CHW float32 [0, 1]. Mirrors drivr's preprocess_image."""
    from torchvision.transforms import functional as TF  # noqa: PLC0415

    tensor = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0)
    resized = TF.resize(
        tensor,
        list(image_size),
        interpolation=TF.InterpolationMode.BILINEAR,
        antialias=True,
    )
    return (resized / 255.0)[0].numpy()


def _load_batch(
    video_path: Path,
    metadata: _MetadataReader,
    waypoints: _WaypointLoader,
    start_frame: int,
    frame_step: int,
    image_size: tuple[int, int],
) -> tuple[dict[str, np.ndarray], GroundTruth]:
    import cv2  # noqa: PLC0415

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    try:
        images, speeds, gases, brakes, steers, turns, wps_list = [], [], [], [], [], [], []

        for t in range(NUM_TIMESTEPS):
            frame_idx = start_frame + t * frame_step
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame_bgr = cap.read()
            if not ret:
                raise ValueError(f"Cannot read frame {frame_idx}")

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            state = metadata.get_state_for_frame(frame_idx)
            gnss = metadata.get_gnss_for_frame(frame_idx)

            images.append(_preprocess_image(frame_rgb, image_size))
            speeds.append(state.speed)
            gases.append(state.gas_pedal)
            brakes.append(state.brake_pedal)
            steers.append(state.steering_angle)
            turns.append(state.turn_signal)
            wps_list.append(waypoints.get_for_gnss(gnss))

        # Ground truth: frame AFTER the episode (matches drivr)
        gt_idx = start_frame + NUM_TIMESTEPS * frame_step
        cap.set(cv2.CAP_PROP_POS_FRAMES, gt_idx)
        ret, _ = cap.read()
        if not ret:
            raise ValueError(f"Cannot read GT frame {gt_idx}")
        gt_state = metadata.get_state_for_frame(gt_idx)
        gt = GroundTruth(
            gas=gt_state.gas_pedal,
            brake=gt_state.brake_pedal,
            steer=gt_state.steering_angle,
            turn=gt_state.turn_signal,
        )
    finally:
        cap.release()

    wp_array = np.clip(np.stack(wps_list) / 100.0, -1.0, 1.0)  # [T, N, 2], 100m horizon

    batch = {
        _K_CAM: np.stack(images)[np.newaxis].astype(np.float32),          # [1, T, 3, H, W]
        _K_SPEED: np.array(speeds, dtype=np.float32).reshape(1, -1, 1),
        _K_GAS: np.array(gases, dtype=np.float32).reshape(1, -1, 1),
        _K_BRAKE: np.array(brakes, dtype=np.float32).reshape(1, -1, 1),
        _K_STEER: np.array(steers, dtype=np.float32).reshape(1, -1, 1),
        _K_TURN: np.array(turns, dtype=np.int32).reshape(1, -1, 1),
        _K_WP: wp_array[np.newaxis].astype(np.float32),                   # [1, T, N, 2]
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
        logger.info("ONNX model loaded", path=str(model_path), image_size=self.image_size, providers=providers)

    def _read_image_size(self) -> tuple[int, int]:
        for inp in self.session.get_inputs():
            if "cam_front_left" in inp.name.lower() and len(inp.shape or []) >= 5:
                h, w = inp.shape[3], inp.shape[4]
                if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
                    return (h, w)
        return DEFAULT_IMAGE_SIZE

    def run(
        self,
        batch: dict[str, np.ndarray],
        cache: np.ndarray | None,
    ) -> tuple[Predictions, np.ndarray | None]:
        inputs = dict(batch)
        inputs["cached_projected_embeddings"] = (
            cache if cache is not None else np.zeros((1, 0, EMBED_DIM), dtype=np.float32)
        )
        # Case-insensitive name matching (identical to drivr's ONNXModel.run)
        matched = {n: inputs[n.lower()] for n in self.input_names if n.lower() in inputs}

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = self.session.run(None, matched)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        def _get(key: str, fallback_idx: int) -> np.ndarray:
            return outputs[self.output_map.get(key, fallback_idx)]

        preds = Predictions(
            gas=float(_get("policy.continuous.gas_pedal", 0).squeeze()),
            brake=float(_get("policy.continuous.brake_pedal", 1).squeeze()),
            steer=float(_get("policy.continuous.steering_angle", 2).squeeze()),
            turn=int(_get("policy.discrete.turn_signal", 3).squeeze()),
            time_ms=elapsed_ms,
        )
        new_cache = (
            outputs[self.output_map["cached_projected_embeddings"]]
            if "cached_projected_embeddings" in self.output_map
            else None
        )
        return preds, new_cache


class _WandbBackend:
    def __init__(self, artifact: str) -> None:
        from rmind.models.control_transformer import ControlTransformer  # noqa: PLC0415

        self.model = ControlTransformer.load_from_wandb_artifact(artifact).eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.image_size = DEFAULT_IMAGE_SIZE
        logger.info("Wandb model loaded", artifact=artifact, device=self.device)

    def run(self, onnx_batch: dict[str, np.ndarray]) -> Predictions:
        dev = self.device

        def _t(arr: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(arr).to(dev)

        # The model's image encoder starts with Rearrange('... h w c -> ... c h w'),
        # so it expects HWC input [B, T, H, W, C]. Our batch has CHW [B, T, C, H, W].
        cam_hwc = _t(onnx_batch[_K_CAM]).permute(0, 1, 3, 4, 2)  # [B, T, H, W, C]

        # Reconstruct the nested {"data": {...}} batch that ControlTransformer.forward expects
        batch: dict = {"data": {
            _PT_CAM: cam_hwc,
            _PT_SPEED: _t(onnx_batch[_K_SPEED]),
            _PT_GAS: _t(onnx_batch[_K_GAS]),
            _PT_BRAKE: _t(onnx_batch[_K_BRAKE]),
            _PT_STEER: _t(onnx_batch[_K_STEER]),
            _PT_TURN: _t(onnx_batch[_K_TURN]),
            _PT_WP: _t(onnx_batch[_K_WP]),
        }}

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
    backends: dict,
    all_preds: dict[str, list[Predictions]],
    n_episodes: int,
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
            label, device, "full", n_episodes,
            f"{times.mean():.1f}", f"{times.min():.1f}",
            f"{times.max():.1f}", f"{times.std():.1f}",
            "✓" if times.mean() < 100 else "✗",
        ])

    sep = "=" * 100
    print(f"\n{sep}")
    print(f"TIMING RESULTS (GPU) - {n_episodes} episodes")
    print(sep)
    print(tabulate(
        rows,
        headers=["Model", "Device", "Type", "Episodes",
                 "Mean (ms)", "Min (ms)", "Max (ms)", "Std (ms)", "10 Hz OK?"],
        tablefmt="grid",
        colalign=("left", "left", "left", "right",
                  "right", "right", "right", "right", "right"),
    ))
    if has_cpu_star:
        print("  * ONNX runs on CPU (no CUDAExecutionProvider). GPU not available.")


def _print_error_table(
    backends: dict,
    all_preds: dict[str, list[Predictions]],
    all_gt: list[GroundTruth],
) -> None:
    from tabulate import tabulate  # noqa: PLC0415

    rows = []
    for label in backends:
        preds = all_preds[label]
        gas_e = [abs(p.gas - g.gas) for p, g in zip(preds, all_gt)]
        brake_e = [abs(p.brake - g.brake) for p, g in zip(preds, all_gt)]
        steer_e = [abs(p.steer - g.steer) for p, g in zip(preds, all_gt)]
        turn_match = np.mean([p.turn == g.turn for p, g in zip(preds, all_gt)]) * 100
        rows.append([
            label,
            f"{np.mean(gas_e):.6f}", f"{np.mean(brake_e):.6f}", f"{np.mean(steer_e):.6f}",
            f"{np.max(gas_e):.6f}", f"{np.max(brake_e):.6f}", f"{np.max(steer_e):.6f}",
            f"{turn_match:.1f}%",
        ])

    print("\nERROR VS GROUND TRUTH:")
    print(tabulate(
        rows,
        headers=["Model", "Gas MAE", "Brake MAE", "Steer MAE",
                 "Gas Max", "Brake Max", "Steer Max", "Turn Match %"],
        tablefmt="grid",
    ))


def _print_validation(
    backends: dict,
    all_preds: dict[str, list[Predictions]],
) -> None:
    labels = list(backends)
    sep = "=" * 100
    print("\nVALIDATION CHECKS")
    print(sep)
    if len(labels) < 2:
        print("  Only one backend — no cross-model comparison.")
        return

    TOLERANCE = 1e-3
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            la, lb = labels[i], labels[j]
            pa, pb = all_preds[la], all_preds[lb]
            gas_max = max(abs(a.gas - b.gas) for a, b in zip(pa, pb))
            brake_max = max(abs(a.brake - b.brake) for a, b in zip(pa, pb))
            steer_max = max(abs(a.steer - b.steer) for a, b in zip(pa, pb))
            ok = gas_max <= TOLERANCE and brake_max <= TOLERANCE and steer_max <= TOLERANCE
            icon = "✓" if ok else "⚠"
            verb = "agree within" if ok else "diffs exceed"
            print(
                f"{icon} {la} vs {lb} {verb} tolerance "
                f"(max: gas={gas_max:.8f}, brake={brake_max:.8f}, steer={steer_max:.8f})"
            )


def _print_per_episode(
    backends: dict,
    rows: list[dict],
    all_gt: list[GroundTruth],
) -> None:
    from tabulate import tabulate  # noqa: PLC0415

    labels = list(backends)
    headers = (
        ["Ep", "Field"]
        + [f"T={i}" for i in range(NUM_TIMESTEPS)]
        + ["GT"]
        + labels
    )
    sep = "=" * 150
    print(f"\nPER-EPISODE PREDICTIONS")
    print(sep)

    for row, gt in zip(rows, all_gt):
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
            table_rows.append([ep, field_label] + t_vals + [f"{gt_val:.6f}"] + pred_vals)
        print(tabulate(table_rows, headers=headers, tablefmt="grid"))


def _print_summary_footer(
    backends: dict,
    all_preds: dict[str, list[Predictions]],
    all_gt: list[GroundTruth],
) -> None:
    sep = "=" * 100
    print(f"\n{sep}")
    print("SUMMARY")
    print(sep)
    for label in backends:
        times = np.array([p.time_ms for p in all_preds[label]])
        hz = 1000.0 / times.mean() if times.mean() > 0 else 0.0
        print(f"{label}:  {times.mean():.1f} ms ({hz:.1f} Hz)")

    all_gas_max = max(
        max(abs(p.gas - g.gas) for p, g in zip(all_preds[lbl], all_gt))
        for lbl in backends
    )
    all_steer_max = max(
        max(abs(p.steer - g.steer) for p, g in zip(all_preds[lbl], all_gt))
        for lbl in backends
    )
    print(f"Max error vs GT:    Gas={all_gas_max:.6f}, Steer={all_steer_max:.6f}")

    labels = list(backends)
    if len(labels) >= 2:
        la, lb = labels[0], labels[1]
        pa, pb = all_preds[la], all_preds[lb]
        gas_d = max(abs(a.gas - b.gas) for a, b in zip(pa, pb))
        brake_d = max(abs(a.brake - b.brake) for a, b in zip(pa, pb))
        steer_d = max(abs(a.steer - b.steer) for a, b in zip(pa, pb))
        print(f"Max diff {la}→{lb}:  Gas={gas_d:.6f}, Brake={brake_d:.6f}, Steer={steer_d:.6f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark ControlTransformer on raw ride data (ONNX and/or wandb)"
    )
    parser.add_argument(
        "--data-dir", type=Path, required=True,
        help="Ride directory containing cam_front_left.pii.mp4, metadata.log, waypoints.json",
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument(
        "--frame-step", type=int, default=10,
        help="Frames between timesteps within an episode, and between episode start frames",
    )
    parser.add_argument(
        "--onnx", type=Path, nargs="+", metavar="PATH",
        help="Path(s) to exported .onnx model(s)",
    )
    parser.add_argument(
        "--model", type=str, nargs="+", metavar="ARTIFACT",
        help="Wandb artifact string(s) for PyTorch inference (e.g. yaak/rmind/model-XXXXXXXX:vN)",
    )
    parser.add_argument(
        "--image-size", type=int, nargs=2, metavar=("H", "W"),
        help="Override image dimensions (default: read from ONNX model or 324 576)",
    )
    parser.add_argument("--output", type=Path, help="Write per-episode CSV predictions here")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs before timing")
    cli = parser.parse_args()

    if not cli.onnx and not cli.model:
        parser.error("At least one of --onnx or --model is required")

    # ── Build backends ────────────────────────────────────────────────────────
    # ordered dict: label → backend
    backends: dict[str, _ONNXBackend | _WandbBackend] = {}
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE

    onnx_paths = cli.onnx or []
    for idx, path in enumerate(onnx_paths):
        label = "ONNX Full" if len(onnx_paths) == 1 else f"ONNX Full {idx + 1}"
        b = _ONNXBackend(path)
        backends[label] = b
        image_size = b.image_size  # last ONNX wins if multiple

    wandb_artifacts = cli.model or []
    for idx, artifact in enumerate(wandb_artifacts):
        label = "PyTorch Native" if len(wandb_artifacts) == 1 else f"PyTorch Native {idx + 1}"
        backends[label] = _WandbBackend(artifact)

    if cli.image_size:
        image_size = (cli.image_size[0], cli.image_size[1])

    logger.info("Backends", labels=list(backends), image_size=image_size)

    # ── Data ──────────────────────────────────────────────────────────────────
    data_dir = cli.data_dir
    for fname in ("cam_front_left.pii.mp4", "metadata.log", "waypoints.json"):
        if not (data_dir / fname).exists():
            parser.error(f"Missing: {data_dir / fname}")

    meta = _MetadataReader(data_dir / "metadata.log")
    meta.load()
    wps = _WaypointLoader(data_dir / "waypoints.json")
    wps.load()
    video_path = data_dir / "cam_front_left.pii.mp4"

    # ── Warmup (ONNX backends only, matching drivr) ───────────────────────────
    onnx_backends = {l: b for l, b in backends.items() if isinstance(b, _ONNXBackend)}
    if onnx_backends and cli.warmup > 0:
        logger.info("Warming up", runs=cli.warmup)
        try:
            wb, _ = _load_batch(video_path, meta, wps, cli.start_frame, cli.frame_step, image_size)
            for _ in range(cli.warmup):
                for b in onnx_backends.values():
                    b.run(wb, None)
        except Exception as e:
            logger.warning("Warmup failed", error=str(e))

    # ── Benchmark ─────────────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("Benchmark: %d episodes, start_frame=%d, frame_step=%d",
                cli.num_episodes, cli.start_frame, cli.frame_step)
    logger.info("=" * 70)

    # per-backend accumulators
    all_preds: dict[str, list[Predictions]] = {l: [] for l in backends}
    all_gt: list[GroundTruth] = []
    rows: list[dict] = []

    for i in range(cli.num_episodes):
        ep_start = cli.start_frame + i * cli.frame_step
        try:
            batch, gt = _load_batch(video_path, meta, wps, ep_start, cli.frame_step, image_size)
        except (ValueError, RuntimeError) as e:
            logger.warning("Skipping episode", episode=i, error=str(e))
            break

        row: dict = {
            "episode": i,
            "frame": ep_start,
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

    n = len(all_gt)
    if n == 0:
        logger.error("No episodes completed")
        return

    # ── Formatted output ──────────────────────────────────────────────────────
    _print_timing_table(backends, all_preds, n)
    _print_error_table(backends, all_preds, all_gt)
    _print_validation(backends, all_preds)
    _print_per_episode(backends, rows, all_gt)
    _print_summary_footer(backends, all_preds, all_gt)

    # ── CSV output ────────────────────────────────────────────────────────────
    if cli.output and rows:
        cli.output.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0].keys())
        with open(cli.output, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        logger.info("Predictions saved", path=str(cli.output))
