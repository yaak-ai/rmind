"""Benchmark 4 models on real driving data from disk.

Compares:
1. PyTorch Native — baseline model, 6 timesteps, no cache
2. PyTorch Cache-enabled — CacheEnabledControlTransformer wrapper, 6 timesteps
3. ONNX Full Forward — full forward ONNX model, 6 timesteps with empty cache
4. ONNX Incremental — 5 cached timesteps + 1 new timestep (incremental)

Data is read from a driving session directory containing:
- cam_front_left.pii.mp4 — video frames
- metadata.log — vehicle state metadata
- waypoints.json — route waypoints

Usage:
    uv run python -m rmind.scripts.benchmark_real_data \
        --data-dir /path/to/session \
        --full-model outputs/ControlTransformer_cache.onnx \
        --incremental-model outputs/ControlTransformer_cache_incremental.onnx \
        --artifact yaak/cargpt/model-y93ejvgg:v9
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mmap

import cv2
import numpy as np
import torch
from prettytable import PrettyTable
from structlog import get_logger
from torch import Tensor

from rbyte.io.yaak.metadata.message_iterator import YaakMetadataMessageIterator
from rbyte.io.yaak.proto import can_pb2, sensor_pb2

logger = get_logger(__name__)

NUM_WAYPOINTS = 10


# ==================== Data Loading Helpers ====================

def load_metadata(metadata_path: Path) -> tuple[list[can_pb2.VehicleMotion], list[can_pb2.VehicleState], list[sensor_pb2.Gnss], list[sensor_pb2.ImageMetadata]]:
    """Parse metadata.log and return lists of protobuf messages sorted by session_timestamp."""
    motions: list[can_pb2.VehicleMotion] = []
    states: list[can_pb2.VehicleState] = []
    gnss_msgs: list[sensor_pb2.Gnss] = []
    image_metas: list[sensor_pb2.ImageMetadata] = []

    with open(metadata_path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        for msg_type, msg_buf in YaakMetadataMessageIterator(mm):
            msg = msg_type()
            msg.ParseFromString(msg_buf)
            if msg_type is can_pb2.VehicleMotion:
                motions.append(msg)
            elif msg_type is can_pb2.VehicleState:
                states.append(msg)
            elif msg_type is sensor_pb2.Gnss:
                gnss_msgs.append(msg)
            elif msg_type is sensor_pb2.ImageMetadata:
                image_metas.append(msg)

    motions.sort(key=lambda m: m.session_timestamp)
    states.sort(key=lambda m: m.session_timestamp)
    gnss_msgs.sort(key=lambda m: m.session_timestamp)
    image_metas.sort(key=lambda m: m.session_timestamp)

    return motions, states, gnss_msgs, image_metas


def _find_nearest(messages: list, timestamp: int) -> Any:
    """Binary search for the message with session_timestamp closest to timestamp."""
    if not messages:
        return None
    lo, hi = 0, len(messages) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if messages[mid].session_timestamp < timestamp:
            lo = mid + 1
        else:
            hi = mid
    # Check lo and lo-1
    if lo > 0 and abs(messages[lo - 1].session_timestamp - timestamp) < abs(messages[lo].session_timestamp - timestamp):
        return messages[lo - 1]
    return messages[lo]


def build_frame_timestamp_map(
    image_metas: list[sensor_pb2.ImageMetadata],
    camera_name: str = "cam_front_left",
) -> dict[int, int]:
    """Build mapping from frame_idx -> session_timestamp for a specific camera."""
    frame_to_ts: dict[int, int] = {}
    for meta in image_metas:
        if meta.camera_name == camera_name:
            frame_to_ts[meta.frame_idx] = meta.session_timestamp
    return frame_to_ts


def load_waypoints(waypoints_path: Path) -> list[dict]:
    """Load GeoJSON waypoints."""
    with open(waypoints_path) as f:
        data = json.load(f)
    return data["features"]


def get_nearest_waypoints(
    waypoints: list[dict],
    lat: float,
    lon: float,
    num_waypoints: int,
) -> torch.Tensor:
    """Get the nearest waypoints to a lat/lon position, returned as normalized xy offsets.

    Returns tensor of shape [num_waypoints, 2] with (dx, dy) in a local frame.
    """
    # Find nearest waypoint by geodesic approximation
    def _dist(wp: dict) -> float:
        coords = wp["geometry"]["coordinates"]
        return (coords[0] - lon) ** 2 + (coords[1] - lat) ** 2

    # Sort by distance, take nearest as anchor, then next num_waypoints along the route
    # Actually, waypoints are ordered along the route. Find the closest one and take the next N.
    min_idx = 0
    min_dist = float("inf")
    for i, wp in enumerate(waypoints):
        d = _dist(wp)
        if d < min_dist:
            min_dist = d
            min_idx = i

    # Take num_waypoints starting from the nearest
    selected = []
    for i in range(num_waypoints):
        idx = min(min_idx + i, len(waypoints) - 1)
        coords = waypoints[idx]["geometry"]["coordinates"]
        selected.append(coords)

    # Convert to local frame offsets relative to current position
    # Use equirectangular approximation
    cos_lat = math.cos(math.radians(lat))
    result = []
    for wlon, wlat in selected:
        dx = (wlon - lon) * cos_lat * 111320.0  # meters
        dy = (wlat - lat) * 111320.0  # meters
        result.append([dx / 1000.0, dy / 1000.0])  # normalize to km

    return torch.tensor(result, dtype=torch.float32)


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    name: str
    times_ms: list[float] = field(default_factory=list)

    @property
    def mean_ms(self) -> float:
        return float(np.mean(self.times_ms)) if self.times_ms else 0.0

    @property
    def std_ms(self) -> float:
        return float(np.std(self.times_ms)) if self.times_ms else 0.0


@dataclass
class PredictionResult:
    """Predictions from a model run."""

    brake_pedal: float
    gas_pedal: float
    steering_angle: float
    turn_signal: int


def flatten_batch_to_onnx(batch: dict[str, Any], prefix: str = "batch") -> dict[str, np.ndarray]:
    """Flatten nested batch dict to ONNX input format."""
    result = {}

    def _recurse(obj: Any, current_prefix: str) -> None:
        if isinstance(obj, Tensor):
            name = current_prefix.lower().replace("/", "_")
            arr = obj.numpy()
            # ONNX models expect int32, not int64
            if arr.dtype == np.int64:
                arr = arr.astype(np.int32)
            result[name] = arr
        elif isinstance(obj, dict):
            for key, value in obj.items():
                new_prefix = f"{current_prefix}_{key}"
                _recurse(value, new_prefix)

    _recurse(batch, prefix)
    return result


def match_onnx_inputs(onnx_input_names: set[str], inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Match input dict keys to ONNX input names (case-insensitive)."""
    matched = {}
    for onnx_name in onnx_input_names:
        for our_name, value in inputs.items():
            if our_name.lower() == onnx_name.lower():
                matched[onnx_name] = value
                break
    return matched


def slice_batch(batch: dict[str, Any], timestep_slice: slice) -> dict[str, Any]:
    """Slice batch tensors along the timestep dimension (dim 1)."""
    result = {}
    for key, value in batch.items():
        if isinstance(value, Tensor):
            result[key] = value[:, timestep_slice]
        elif isinstance(value, dict):
            result[key] = slice_batch(value, timestep_slice)
        else:
            result[key] = value
    return result


def to_device(obj: Any, device: torch.device) -> Any:
    """Recursively move tensors to device."""
    if isinstance(obj, Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    return obj


def to_cpu(obj: Any) -> Any:
    """Recursively move tensors to CPU."""
    if isinstance(obj, Tensor):
        return obj.cpu()
    elif isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    return obj


def load_episode_data(
    video_cap: cv2.VideoCapture,
    motions: list[can_pb2.VehicleMotion],
    states: list[can_pb2.VehicleState],
    gnss_msgs: list[sensor_pb2.Gnss],
    frame_to_ts: dict[int, int],
    waypoints: list[dict],
    ep_start: int,
    episode_length: int,
    frame_step: int,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, float | int]] | None:
    """Load one episode of real data from disk.

    Returns:
        Tuple of (batch_dict, ground_truth_dict) or None if not enough frames.
        ground_truth_dict contains the last timestep's actual values.
    """
    images = []
    speeds = []
    gas_pedals = []
    brake_pedals = []
    steering_angles = []
    turn_signals = []
    waypoints_per_timestep = []

    for t in range(episode_length):
        frame_idx = ep_start + t * frame_step

        if frame_idx not in frame_to_ts:
            return None

        # Read video frame via cv2
        video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame_bgr = video_cap.read()
        if not ret or frame_bgr is None:
            return None
        # BGR -> RGB, resize to 256x256, uint8 -> float [0,1], HWC -> CHW
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, (256, 256))
        frame_tensor = torch.from_numpy(frame_rgb).float().permute(2, 0, 1) / 255.0  # [C, H, W]
        images.append(frame_tensor)

        # Look up metadata by timestamp
        ts = frame_to_ts[frame_idx]
        motion = _find_nearest(motions, ts)
        state = _find_nearest(states, ts)
        gnss = _find_nearest(gnss_msgs, ts)

        speeds.append(motion.speed)
        gas_pedals.append(motion.gas_pedal_normalized)
        brake_pedals.append(motion.brake_pedal_normalized)
        steering_angles.append(motion.steering_angle_normalized)
        turn_signals.append(int(state.turn_signal))

        # Waypoints
        wp = get_nearest_waypoints(waypoints, gnss.latitude, gnss.longitude, NUM_WAYPOINTS)
        waypoints_per_timestep.append(wp)

    if len(images) < episode_length:
        return None

    # Ground truth from last timestep
    ground_truth = {
        "brake_pedal": brake_pedals[-1],
        "gas_pedal": gas_pedals[-1],
        "steering_angle": steering_angles[-1],
        "turn_signal": turn_signals[-1],
    }

    # Build batch: [B=1, T, ...]
    waypoints_tensor = torch.stack(waypoints_per_timestep).unsqueeze(0).to(device)

    batch = {
        "data": {
            "cam_front_left": torch.stack(images).unsqueeze(0).to(device),  # [1, T, H, W, C]
            "meta/VehicleMotion/speed": torch.tensor(speeds, dtype=torch.float32)
            .view(1, -1, 1)
            .to(device),
            "meta/VehicleMotion/gas_pedal_normalized": torch.tensor(gas_pedals, dtype=torch.float32)
            .view(1, -1, 1)
            .to(device),
            "meta/VehicleMotion/brake_pedal_normalized": torch.tensor(brake_pedals, dtype=torch.float32)
            .view(1, -1, 1)
            .to(device),
            "meta/VehicleMotion/steering_angle_normalized": torch.tensor(steering_angles, dtype=torch.float32)
            .view(1, -1, 1)
            .to(device),
            "meta/VehicleState/turn_signal": torch.tensor(turn_signals, dtype=torch.long)
            .view(1, -1, 1)
            .to(device),
            "waypoints/xy_normalized": waypoints_tensor,  # [1, T, num_waypoints, 2]
        },
    }

    return batch, ground_truth


def run_pytorch_native(
    model: Any,
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[PredictionResult, float]:
    """Run PyTorch native model and return predictions + time_ms."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    outputs = model(batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000

    policy = outputs["policy"]
    preds = PredictionResult(
        brake_pedal=float(policy["continuous", "brake_pedal"].squeeze().cpu()),
        gas_pedal=float(policy["continuous", "gas_pedal"].squeeze().cpu()),
        steering_angle=float(policy["continuous", "steering_angle"].squeeze().cpu()),
        turn_signal=int(policy["discrete", "turn_signal"].squeeze().cpu()),
    )
    return preds, elapsed_ms


def run_pytorch_cache(
    cache_model: Any,
    batch: dict[str, Any],
    empty_proj_emb: Tensor,
    empty_kv: Tensor,
    device: torch.device,
) -> tuple[PredictionResult, float]:
    """Run cache-enabled PyTorch model and return predictions + time_ms."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    outputs = cache_model(batch, empty_proj_emb, empty_kv)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000

    preds = PredictionResult(
        brake_pedal=float(outputs[0].squeeze().cpu()),
        gas_pedal=float(outputs[1].squeeze().cpu()),
        steering_angle=float(outputs[2].squeeze().cpu()),
        turn_signal=int(outputs[3].squeeze().cpu()),
    )
    return preds, elapsed_ms


def run_onnx_full(
    session: Any,
    input_names: set[str],
    batch_cpu: dict[str, Any],
    embed_dim: int,
    num_layers: int,
) -> tuple[PredictionResult, float, np.ndarray, np.ndarray]:
    """Run ONNX full forward model. Returns predictions, time_ms, proj_emb, kv."""
    onnx_inputs = flatten_batch_to_onnx(batch_cpu)
    onnx_inputs["cached_projected_embeddings"] = np.empty((1, 0, embed_dim), dtype=np.float32)
    onnx_inputs["cached_kv"] = np.empty((num_layers, 2, 1, 0, embed_dim), dtype=np.float32)
    matched = match_onnx_inputs(input_names, onnx_inputs)

    start = time.perf_counter()
    outputs = session.run(None, matched)
    elapsed_ms = (time.perf_counter() - start) * 1000

    preds = PredictionResult(
        brake_pedal=float(outputs[0].flatten()[0]),
        gas_pedal=float(outputs[1].flatten()[0]),
        steering_angle=float(outputs[2].flatten()[0]),
        turn_signal=int(outputs[3].flatten()[0]),
    )
    return preds, elapsed_ms, outputs[-2], outputs[-1]


def run_onnx_incremental(
    incr_session: Any,
    incr_input_names: set[str],
    batch_1_cpu: dict[str, Any],
    proj_emb_5: np.ndarray,
    kv_cache_5: np.ndarray,
) -> tuple[PredictionResult, float]:
    """Run ONNX incremental model with cached context. Returns predictions, time_ms."""
    onnx_inputs = flatten_batch_to_onnx(batch_1_cpu)
    onnx_inputs["cached_projected_embeddings"] = proj_emb_5
    onnx_inputs["cached_kv"] = kv_cache_5
    matched = match_onnx_inputs(incr_input_names, onnx_inputs)

    start = time.perf_counter()
    outputs = incr_session.run(None, matched)
    elapsed_ms = (time.perf_counter() - start) * 1000

    preds = PredictionResult(
        brake_pedal=float(outputs[0].flatten()[0]),
        gas_pedal=float(outputs[1].flatten()[0]),
        steering_angle=float(outputs[2].flatten()[0]),
        turn_signal=int(outputs[3].flatten()[0]),
    )
    return preds, elapsed_ms


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark 4 models on real driving data")
    parser.add_argument("--data-dir", type=Path, required=True, help="Path to driving session directory")
    parser.add_argument("--full-model", type=Path, required=True, help="Path to full forward ONNX model")
    parser.add_argument("--incremental-model", type=Path, required=True, help="Path to incremental ONNX model")
    parser.add_argument("--artifact", type=str, default=None, help="W&B artifact for trained weights")
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of episodes to run")
    parser.add_argument("--episode-length", type=int, default=6, help="Timesteps per episode")
    parser.add_argument("--frame-step", type=int, default=10, help="Frames between timesteps")
    parser.add_argument("--episode-spacing", type=int, default=300, help="Frames between episodes")
    parser.add_argument("--num-warmup", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device (cpu or cuda:N)")
    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info(
        "starting benchmark",
        data_dir=str(args.data_dir),
        device=str(device),
        num_episodes=args.num_episodes,
        episode_length=args.episode_length,
    )

    # Validate paths — try both video naming conventions
    video_path = args.data_dir / "cam_front_left.pii.mp4"
    if not video_path.exists():
        video_path = args.data_dir / "cam_front_left.defish.mp4"
    metadata_path = args.data_dir / "metadata.log"
    waypoints_path = args.data_dir / "waypoints.json"
    for p in [video_path, metadata_path, waypoints_path]:
        if not p.exists():
            logger.error("file not found", path=str(p))
            return
    if not args.full_model.exists():
        logger.error("full model not found", path=str(args.full_model))
        return
    if not args.incremental_model.exists():
        logger.error("incremental model not found", path=str(args.incremental_model))
        return

    # ==================== Load Data ====================
    logger.info("loading metadata", path=str(metadata_path))
    motions, vehicle_states, gnss_msgs, image_metas = load_metadata(metadata_path)
    logger.info("metadata loaded", motions=len(motions), states=len(vehicle_states), gnss=len(gnss_msgs), images=len(image_metas))

    frame_to_ts = build_frame_timestamp_map(image_metas)
    frame_indices = sorted(frame_to_ts.keys())
    start_frame = frame_indices[0]
    logger.info("frame map built", num_frames=len(frame_indices), start_frame=start_frame)

    logger.info("loading waypoints", path=str(waypoints_path))
    waypoints = load_waypoints(waypoints_path)
    logger.info("waypoints loaded", num_waypoints=len(waypoints))

    logger.info("opening video", path=str(video_path))
    video_cap = cv2.VideoCapture(str(video_path))
    if not video_cap.isOpened():
        logger.error("failed to open video", path=str(video_path))
        return
    num_video_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info("video opened", num_frames=num_video_frames)

    # ==================== Load PyTorch Model ====================
    logger.info("loading PyTorch model via Hydra")
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate as hydra_instantiate
    from omegaconf import OmegaConf

    config_path = Path(__file__).parents[3] / "config"
    with initialize_config_dir(config_dir=str(config_path), version_base=None):
        cfg = compose(config_name="export/onnx_cache")

    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    pytorch_model = hydra_instantiate(model_cfg).to(device).eval()

    if args.artifact:
        from rmind.scripts.export_onnx_cache import _load_weights_from_artifact
        _load_weights_from_artifact(pytorch_model, args.artifact)

    logger.info("PyTorch model loaded")

    # ==================== Load ONNX Models ====================
    import onnxruntime as ort

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "cuda" in args.device
        else ["CPUExecutionProvider"]
    )

    logger.info("loading ONNX models")
    full_session = ort.InferenceSession(str(args.full_model), providers=providers)
    incr_session = ort.InferenceSession(str(args.incremental_model), providers=providers)

    full_input_names = {inp.name for inp in full_session.get_inputs()}
    incr_input_names = {inp.name for inp in incr_session.get_inputs()}
    logger.info("ONNX models loaded", full_inputs=len(full_input_names), incr_inputs=len(incr_input_names))

    # ==================== Extract ONNX Dimensions ====================
    # Get tokens_per_timestep from ONNX model for incremental inference slicing
    import onnx
    from onnx.numpy_helper import to_array

    onnx_model_proto = onnx.load(str(args.full_model))
    onnx_pe_shape = None
    for initializer in onnx_model_proto.graph.initializer:
        if '_position_embeddings_packed' in initializer.name:
            onnx_pe_shape = to_array(initializer).shape
            logger.info("ONNX position_embeddings shape", name=initializer.name, shape=onnx_pe_shape)
            break
    del onnx_model_proto

    embed_dim = int(pytorch_model.encoder.layers[0].embedding_dim)
    num_layers = len(list(pytorch_model.encoder.layers))
    onnx_tokens_per_timestep = onnx_pe_shape[1] // args.episode_length if onnx_pe_shape else None
    logger.info("model dimensions", embed_dim=embed_dim, num_layers=num_layers, onnx_tokens_per_ts=onnx_tokens_per_timestep)

    # ==================== Create Cache-Enabled Model (with foresight) ====================
    from rmind.scripts.export_onnx_cache import CacheEnabledControlTransformer
    from rmind.components.mask import TorchAttentionMaskLegend
    from rmind.components.objectives.policy import PolicyObjective

    # Build mask and PE from the model's own episode builder
    with torch.inference_mode():
        pe_data = load_episode_data(
            video_cap, motions, vehicle_states, gnss_msgs, frame_to_ts, waypoints,
            start_frame, args.episode_length, args.frame_step, device,
        )
        if pe_data is None:
            logger.error("cannot load episode for mask/PE computation")
            return
        pe_batch, _ = pe_data

        # Build episode (eval mode + timestep_offset=0 gives deterministic PE)
        pe_episode = pytorch_model.episode_builder(pe_batch, timestep_offset=0)

        # Compute mask from the model's own PolicyObjective (matches native forward exactly)
        cache_mask = PolicyObjective.build_attention_mask(
            pe_episode.index, pe_episode.timestep, legend=TorchAttentionMaskLegend
        ).mask.to(device)

        # Compute position embeddings
        cache_pe = pe_episode.embeddings_packed - pe_episode.projected_embeddings_packed

    logger.info("cache model setup",
                mask_shape=tuple(cache_mask.shape),
                pe_shape=tuple(cache_pe.shape))

    cache_model = CacheEnabledControlTransformer(
        pytorch_model,
        mask=cache_mask,
        position_embeddings_packed=cache_pe,
    ).to(device).eval()

    # Empty cache tensors for full forward
    empty_proj_emb = torch.zeros(1, 0, embed_dim, device=device)
    empty_kv = torch.zeros(num_layers, 2, 1, 0, embed_dim, device=device)

    # ==================== Warmup ====================
    logger.info("running warmup", num_warmup=args.num_warmup)
    warmup_start = start_frame
    warmup_ep_end = warmup_start + (args.episode_length - 1) * args.frame_step
    if warmup_ep_end <= frame_indices[-1]:
        warmup_data = load_episode_data(
            video_cap, motions, vehicle_states, gnss_msgs, frame_to_ts, waypoints,
            warmup_start, args.episode_length, args.frame_step, device,
        )
        if warmup_data is not None:
            warmup_batch, _ = warmup_data
            warmup_batch_cpu = to_cpu(warmup_batch)
            warmup_batch_1_cpu = to_cpu(slice_batch(warmup_batch, slice(5, 6)))

            with torch.inference_mode():
                for _ in range(args.num_warmup):
                    pytorch_model(warmup_batch)
                    cache_model(warmup_batch, empty_proj_emb, empty_kv)
                    if device.type == "cuda":
                        torch.cuda.synchronize()

                # ONNX warmup
                warmup_onnx_inputs = flatten_batch_to_onnx(warmup_batch_cpu)
                warmup_onnx_inputs["cached_projected_embeddings"] = np.empty((1, 0, embed_dim), dtype=np.float32)
                warmup_onnx_inputs["cached_kv"] = np.empty((num_layers, 2, 1, 0, embed_dim), dtype=np.float32)
                warmup_matched = match_onnx_inputs(full_input_names, warmup_onnx_inputs)
                for _ in range(args.num_warmup):
                    full_session.run(None, warmup_matched)

                # Incremental warmup
                warmup_full_out = full_session.run(None, warmup_matched)
                warmup_proj = warmup_full_out[-2][:, :5 * onnx_tokens_per_timestep, :]
                warmup_kv = warmup_full_out[-1][:, :, :, :5 * onnx_tokens_per_timestep, :]
                warmup_incr_inputs = flatten_batch_to_onnx(warmup_batch_1_cpu)
                warmup_incr_inputs["cached_projected_embeddings"] = warmup_proj
                warmup_incr_inputs["cached_kv"] = warmup_kv
                warmup_incr_matched = match_onnx_inputs(incr_input_names, warmup_incr_inputs)
                for _ in range(args.num_warmup):
                    incr_session.run(None, warmup_incr_matched)

            logger.info("warmup complete")
    else:
        logger.warning("not enough frames for warmup")

    # ==================== Run Episodes ====================
    logger.info("running episodes", num_episodes=args.num_episodes)

    timing_native = BenchmarkResult(name="PyTorch Native")
    timing_cache = BenchmarkResult(name="PyTorch Cache-enabled")
    timing_onnx_full = BenchmarkResult(name="ONNX Full Forward")
    timing_onnx_incr = BenchmarkResult(name="ONNX Incremental")

    # Accumulators for MAE and per-episode predictions
    mae_native = {"brake": [], "gas": [], "steering": []}
    mae_cache = {"brake": [], "gas": [], "steering": []}
    mae_onnx_full = {"brake": [], "gas": [], "steering": []}
    mae_onnx_incr = {"brake": [], "gas": [], "steering": []}

    all_preds: dict[str, list[PredictionResult]] = {
        "native": [], "cache": [], "onnx_full": [], "onnx_incr": [],
    }

    # Per-episode results table
    episode_table = PrettyTable()
    episode_table.field_names = [
        "Ep", "Field",
        "GT", "Native", "Cache", "ONNX Full", "ONNX Incr",
    ]
    episode_table.float_format = ".6"
    episode_table.align = "r"
    episode_table.align["Field"] = "l"

    episodes_run = 0

    with torch.inference_mode():
        for ep_idx in range(args.num_episodes):
            ep_start = start_frame + ep_idx * args.episode_spacing

            ep_end = ep_start + (args.episode_length - 1) * args.frame_step
            if ep_end > frame_indices[-1]:
                logger.warning("not enough frames remaining", episode=ep_idx + 1)
                break

            episode_data = load_episode_data(
                video_cap, motions, vehicle_states, gnss_msgs, frame_to_ts, waypoints,
                ep_start, args.episode_length, args.frame_step, device,
            )
            if episode_data is None:
                logger.warning("failed to load episode", episode=ep_idx + 1)
                continue

            batch, ground_truth = episode_data
            batch_cpu = to_cpu(batch)
            batch_1 = slice_batch(batch, slice(args.episode_length - 1, args.episode_length))
            batch_1_cpu = to_cpu(batch_1)

            # 1. PyTorch Native
            native_preds, native_ms = run_pytorch_native(pytorch_model, batch, device)
            timing_native.times_ms.append(native_ms)

            # 2. PyTorch Cache-enabled
            cache_preds, cache_ms = run_pytorch_cache(cache_model, batch, empty_proj_emb, empty_kv, device)
            timing_cache.times_ms.append(cache_ms)

            # 3. ONNX Full Forward
            onnx_full_preds, onnx_full_ms, proj_emb_out, kv_out = run_onnx_full(
                full_session, full_input_names, batch_cpu, embed_dim, num_layers,
            )
            timing_onnx_full.times_ms.append(onnx_full_ms)

            # 4. ONNX Incremental (5 cached + 1 new)
            proj_emb_5 = proj_emb_out[:, :5 * onnx_tokens_per_timestep, :]
            kv_cache_5 = kv_out[:, :, :, :5 * onnx_tokens_per_timestep, :]
            onnx_incr_preds, onnx_incr_ms = run_onnx_incremental(
                incr_session, incr_input_names, batch_1_cpu, proj_emb_5, kv_cache_5,
            )
            timing_onnx_incr.times_ms.append(onnx_incr_ms)

            # Store predictions for consistency check
            all_preds["native"].append(native_preds)
            all_preds["cache"].append(cache_preds)
            all_preds["onnx_full"].append(onnx_full_preds)
            all_preds["onnx_incr"].append(onnx_incr_preds)

            # Accumulate MAE (ground truth vs predictions)
            gt = ground_truth
            for mae_dict, preds in [
                (mae_native, native_preds),
                (mae_cache, cache_preds),
                (mae_onnx_full, onnx_full_preds),
                (mae_onnx_incr, onnx_incr_preds),
            ]:
                mae_dict["brake"].append(abs(gt["brake_pedal"] - preds.brake_pedal))
                mae_dict["gas"].append(abs(gt["gas_pedal"] - preds.gas_pedal))
                mae_dict["steering"].append(abs(gt["steering_angle"] - preds.steering_angle))

            # Per-episode table rows
            ep_label = str(ep_idx + 1)
            for field, gt_key, pred_attr in [
                ("brake", "brake_pedal", "brake_pedal"),
                ("gas", "gas_pedal", "gas_pedal"),
                ("steering", "steering_angle", "steering_angle"),
                ("turn_sig", "turn_signal", "turn_signal"),
            ]:
                episode_table.add_row([
                    ep_label if field == "brake" else "",
                    field,
                    f"{gt[gt_key]:.6f}" if isinstance(gt[gt_key], float) else str(gt[gt_key]),
                    f"{getattr(native_preds, pred_attr):.6f}" if isinstance(getattr(native_preds, pred_attr), float) else str(getattr(native_preds, pred_attr)),
                    f"{getattr(cache_preds, pred_attr):.6f}" if isinstance(getattr(cache_preds, pred_attr), float) else str(getattr(cache_preds, pred_attr)),
                    f"{getattr(onnx_full_preds, pred_attr):.6f}" if isinstance(getattr(onnx_full_preds, pred_attr), float) else str(getattr(onnx_full_preds, pred_attr)),
                    f"{getattr(onnx_incr_preds, pred_attr):.6f}" if isinstance(getattr(onnx_incr_preds, pred_attr), float) else str(getattr(onnx_incr_preds, pred_attr)),
                ])

            episodes_run += 1
            logger.info(
                "episode complete",
                episode=ep_idx + 1,
                native_ms=f"{native_ms:.1f}",
                cache_ms=f"{cache_ms:.1f}",
                onnx_full_ms=f"{onnx_full_ms:.1f}",
                onnx_incr_ms=f"{onnx_incr_ms:.1f}",
            )

    if episodes_run == 0:
        logger.error("no episodes completed")
        return

    # ==================== Results ====================
    print("\n" + "=" * 120)
    print("PER-EPISODE PREDICTIONS")
    print("=" * 120)
    print(episode_table)

    # Summary: Timing
    print("\n" + "=" * 120)
    print("TIMING SUMMARY")
    print("=" * 120)

    timing_table = PrettyTable()
    timing_table.field_names = ["Model", "Mean (ms)", "Std (ms)", "Speedup"]
    timing_table.float_format = ".2"
    timing_table.align = "r"
    timing_table.align["Model"] = "l"

    baseline = timing_native.mean_ms
    for result in [timing_native, timing_cache, timing_onnx_full, timing_onnx_incr]:
        speedup = baseline / result.mean_ms if result.mean_ms > 0 else 0.0
        timing_table.add_row([result.name, f"{result.mean_ms:.2f}", f"{result.std_ms:.2f}", f"{speedup:.2f}x"])

    print(timing_table)

    # Summary: MAE vs Ground Truth
    print("\n" + "=" * 120)
    print("MAE vs GROUND TRUTH")
    print("=" * 120)

    mae_table = PrettyTable()
    mae_table.field_names = ["Model", "Brake MAE", "Gas MAE", "Steering MAE"]
    mae_table.float_format = ".6"
    mae_table.align = "r"
    mae_table.align["Model"] = "l"

    mae_entries = [
        ("PyTorch Native", mae_native),
        ("PyTorch Cache-enabled", mae_cache),
        ("ONNX Full Forward", mae_onnx_full),
        ("ONNX Incremental", mae_onnx_incr),
    ]
    for name, mae_dict in mae_entries:
        mae_table.add_row([
            name,
            f"{np.mean(mae_dict['brake']):.6f}",
            f"{np.mean(mae_dict['gas']):.6f}",
            f"{np.mean(mae_dict['steering']):.6f}",
        ])

    print(mae_table)

    # Summary: Cross-model consistency (direct pairwise prediction diffs)
    print("\n" + "=" * 120)
    print("CROSS-MODEL CONSISTENCY (max abs diff across all episodes)")
    print("=" * 120)

    def compute_max_diffs(preds_a: list[PredictionResult], preds_b: list[PredictionResult]) -> dict[str, float]:
        max_brake = max(abs(a.brake_pedal - b.brake_pedal) for a, b in zip(preds_a, preds_b))
        max_gas = max(abs(a.gas_pedal - b.gas_pedal) for a, b in zip(preds_a, preds_b))
        max_steer = max(abs(a.steering_angle - b.steering_angle) for a, b in zip(preds_a, preds_b))
        return {"brake": max_brake, "gas": max_gas, "steering": max_steer}

    consistency_table = PrettyTable()
    consistency_table.field_names = ["Comparison", "Brake", "Gas", "Steering"]
    consistency_table.align = "r"
    consistency_table.align["Comparison"] = "l"

    pairwise_diffs = {}
    pairwise_diffs["Native vs Cache"] = compute_max_diffs(all_preds["native"], all_preds["cache"])
    pairwise_diffs["Cache vs ONNX Full"] = compute_max_diffs(all_preds["cache"], all_preds["onnx_full"])
    pairwise_diffs["ONNX Full vs ONNX Incr"] = compute_max_diffs(all_preds["onnx_full"], all_preds["onnx_incr"])

    for label, diffs in pairwise_diffs.items():
        consistency_table.add_row([
            label,
            f"{diffs['brake']:.2e}",
            f"{diffs['gas']:.2e}",
            f"{diffs['steering']:.2e}",
        ])

    print(consistency_table)

    # Verification verdicts
    TOLERANCE = 1e-4
    print("\n--- Verification ---")
    for label, diffs in pairwise_diffs.items():
        max_diff = max(diffs.values())
        status = "PASS" if max_diff < TOLERANCE else "WARN"
        print(f"  [{status}] {label}: max diff = {max_diff:.2e}")

    print(f"\nEpisodes completed: {episodes_run}/{args.num_episodes}")
    print("=" * 120)


if __name__ == "__main__":
    main()
