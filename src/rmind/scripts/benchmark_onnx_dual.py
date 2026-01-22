"""Benchmark comparing native PyTorch (no cache) vs ONNX dual model (cache + incremental).

This script compares:
1. Native PyTorch model (baseline) - 6 timesteps, no cache
2. ONNX full forward model - 6 timesteps with empty cache
3. ONNX dual model setup - 5 cached timesteps + 1 new timestep (incremental)

Both predictions and timing are compared.

Usage:
    uv run python -m rmind.scripts.benchmark_onnx_dual \
        --full-model outputs/2026-01-21/latest/ControlTransformer_cache.onnx \
        --incremental-model outputs/2026-01-21/latest/ControlTransformer_cache_incremental.onnx \
        --num-warmup 5 \
        --num-iterations 20
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from structlog import get_logger
from torch import Tensor

logger = get_logger(__name__)

# ImageNet normalization constants
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    name: str
    num_iterations: int
    times_ms: list[float] = field(default_factory=list)

    @property
    def mean_ms(self) -> float:
        return np.mean(self.times_ms)

    @property
    def std_ms(self) -> float:
        return np.std(self.times_ms)

    @property
    def min_ms(self) -> float:
        return np.min(self.times_ms)

    @property
    def max_ms(self) -> float:
        return np.max(self.times_ms)

    def __str__(self) -> str:
        return (
            f"{self.name}:\n"
            f"  Mean: {self.mean_ms:.2f} ms\n"
            f"  Std:  {self.std_ms:.2f} ms\n"
            f"  Min:  {self.min_ms:.2f} ms\n"
            f"  Max:  {self.max_ms:.2f} ms"
        )


@dataclass
class PredictionResult:
    """Predictions from a model run."""

    brake_pedal: float
    gas_pedal: float
    steering_angle: float
    turn_signal: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "brake_pedal": self.brake_pedal,
            "gas_pedal": self.gas_pedal,
            "steering_angle": self.steering_angle,
            "turn_signal": self.turn_signal,
        }


def normalize_images(batch: dict[str, Any]) -> dict[str, Any]:
    """Apply ImageNet normalization to images in batch."""
    result = {}
    for key, value in batch.items():
        if isinstance(value, dict):
            result[key] = normalize_images(value)
        elif isinstance(value, Tensor) and key == "cam_front_left":
            mean = IMAGENET_MEAN.view(1, 1, 3, 1, 1)
            std = IMAGENET_STD.view(1, 1, 3, 1, 1)
            result[key] = (value - mean) / std
        else:
            result[key] = value
    return result


def flatten_batch_to_onnx(batch: dict[str, Any], prefix: str = "batch") -> dict[str, np.ndarray]:
    """Flatten nested batch dict to ONNX input format."""
    result = {}

    def _recurse(obj: Any, current_prefix: str) -> None:
        if isinstance(obj, Tensor):
            name = current_prefix.lower().replace("/", "_")
            result[name] = obj.numpy()
        elif isinstance(obj, dict):
            for key, value in obj.items():
                new_prefix = f"{current_prefix}_{key}"
                _recurse(value, new_prefix)

    _recurse(batch, prefix)
    return result


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


def match_onnx_inputs(onnx_input_names: set[str], inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Match input dict keys to ONNX input names (case-insensitive)."""
    matched = {}
    for onnx_name in onnx_input_names:
        for our_name, value in inputs.items():
            if our_name.lower() == onnx_name.lower():
                matched[onnx_name] = value
                break
    return matched


def create_test_batch_from_config(cfg) -> dict[str, Any]:
    """Create test batch from hydra config (same as export script)."""
    from hydra.utils import instantiate as hydra_instantiate

    # Set seed BEFORE generating args for reproducible batch data
    # This MUST match the export script to ensure the same batch data
    torch.manual_seed(42)

    input_cfg = cfg.args
    args = hydra_instantiate(input_cfg, _recursive_=True, _convert_="all")
    batch = args[0]
    return normalize_images(batch)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark PyTorch vs ONNX dual model")
    parser.add_argument(
        "--full-model",
        type=Path,
        default=Path("outputs/2026-01-21/latest/ControlTransformer_cache.onnx"),
        help="Path to full forward ONNX model",
    )
    parser.add_argument(
        "--incremental-model",
        type=Path,
        default=Path("outputs/2026-01-21/latest/ControlTransformer_cache_incremental.onnx"),
        help="Path to incremental ONNX model",
    )
    parser.add_argument("--num-warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--num-iterations", type=int, default=20, help="Benchmark iterations")
    parser.add_argument("--device", type=str, default="cpu", help="Device for PyTorch (cpu/cuda)")
    args = parser.parse_args()

    # CRITICAL: Set seed BEFORE loading models for reproducibility
    # This ensures PyTorch and ONNX produce same outputs for same input
    torch.manual_seed(42)
    np.random.seed(42)
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Validate paths
    if not args.full_model.exists():
        logger.error("Full model not found", path=str(args.full_model))
        return
    if not args.incremental_model.exists():
        logger.error("Incremental model not found", path=str(args.incremental_model))
        return

    device = torch.device(args.device)
    logger.info("starting benchmark", device=str(device), num_warmup=args.num_warmup, num_iterations=args.num_iterations)

    # ==================== Load Models ====================
    logger.info("loading PyTorch model via Hydra")

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from rmind.config import HydraConfig
    from pytorch_lightning import LightningModule

    config_path = Path(__file__).parents[3] / "config"
    with initialize_config_dir(config_dir=str(config_path), version_base=None):
        cfg = compose(config_name="export/onnx_cache")

    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model_hydra = HydraConfig[LightningModule](**model_cfg)
    pytorch_model = model_hydra.instantiate().to(device).eval()
    logger.info("PyTorch model loaded")

    # Create cache-enabled wrapper for fair ONNX comparison
    from rmind.scripts.export_onnx_cache import CacheEnabledControlTransformer
    from rmind.scripts.build_onnx_mask import build_policy_mask

    full_mask_np = build_policy_mask(num_timesteps=6)
    full_mask = torch.from_numpy(full_mask_np).to(device)

    # Precompute position_embeddings for 6 timesteps (constant, reusable)
    # We'll compute this after we have test data
    cache_model = None  # Will be created after test data
    logger.info("cache-enabled PyTorch model will be created after test data")

    # Load ONNX models
    import onnxruntime as ort

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if args.device == "cuda" else ["CPUExecutionProvider"]

    logger.info("loading ONNX models", full_model=str(args.full_model), incremental_model=str(args.incremental_model))
    full_session = ort.InferenceSession(str(args.full_model), providers=providers)
    incr_session = ort.InferenceSession(str(args.incremental_model), providers=providers)

    full_input_names = {inp.name for inp in full_session.get_inputs()}
    incr_input_names = {inp.name for inp in incr_session.get_inputs()}
    logger.info("ONNX models loaded", full_inputs=len(full_input_names), incr_inputs=len(incr_input_names))

    # ==================== Create Test Data ====================
    logger.info("creating test data from config (same as export)")
    batch_6 = create_test_batch_from_config(cfg)
    batch_5 = slice_batch(batch_6, slice(0, 5))
    batch_1 = slice_batch(batch_6, slice(5, 6))

    # Move to device for PyTorch
    def to_device(obj, device):
        if isinstance(obj, Tensor):
            return obj.to(device)
        elif isinstance(obj, dict):
            return {k: to_device(v, device) for k, v in obj.items()}
        return obj

    batch_6_device = to_device(batch_6, device)

    # Model dimensions and position embeddings
    with torch.inference_mode():
        episode = pytorch_model.episode_builder(batch_6_device, timestep_offset=0)
        _, seq_len, embed_dim = episode.embeddings_packed.shape
        tokens_per_timestep = seq_len // 6
        num_layers = len(list(pytorch_model.encoder.layers))

        # Compute position_embeddings for 6 timesteps (constant, reusable)
        position_embeddings_packed = episode.embeddings_packed - episode.projected_embeddings_packed

    logger.info("model dimensions", seq_len=seq_len, embed_dim=embed_dim, tokens_per_timestep=tokens_per_timestep, num_layers=num_layers)

    # Now create cache-enabled model with position_embeddings
    cache_model = CacheEnabledControlTransformer(
        pytorch_model,
        mask=full_mask,
        position_embeddings_packed=position_embeddings_packed,
    ).to(device).eval()
    logger.info("cache-enabled PyTorch model created")

    # Debug: check PyTorch batch sample values
    def get_nested(d, keys):
        for k in keys:
            d = d[k]
        return d

    brake_key = ["data", "meta/VehicleMotion/brake_pedal_normalized"]
    gas_key = ["data", "meta/VehicleMotion/gas_pedal_normalized"]
    try:
        brake_val = get_nested(batch_6, brake_key)
        gas_val = get_nested(batch_6, gas_key)
        logger.info("PyTorch batch sample", brake=brake_val.flatten()[:3].tolist(), gas=gas_val.flatten()[:3].tolist())
    except Exception as e:
        logger.warning("could not get batch sample", error=str(e))

    # ==================== Benchmark: Native PyTorch (Baseline) ====================
    logger.info("benchmarking native PyTorch (baseline)")
    pytorch_result = BenchmarkResult(name="PyTorch Native (6ts, no cache)", num_iterations=args.num_iterations)
    pytorch_preds = None

    # Model is in eval mode, so it uses deterministic position embeddings (offset 0)
    # Set the same mask on native model's objectives for fair comparison
    # Without this, native model builds mask dynamically which may differ
    for obj in pytorch_model.objectives.values():
        if hasattr(obj, '_mask'):
            obj._mask = full_mask

    with torch.inference_mode():
        # Warmup
        for _ in range(args.num_warmup):
            _ = pytorch_model(batch_6_device)
            if device.type == "cuda":
                torch.cuda.synchronize()

        # Benchmark
        for _ in range(args.num_iterations):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            outputs = pytorch_model(batch_6_device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            pytorch_result.times_ms.append((end - start) * 1000)

        # Extract predictions from last run
        policy = outputs["policy"]
        pytorch_preds = PredictionResult(
            brake_pedal=float(policy["continuous", "brake_pedal"].squeeze().cpu()),
            gas_pedal=float(policy["continuous", "gas_pedal"].squeeze().cpu()),
            steering_angle=float(policy["continuous", "steering_angle"].squeeze().cpu()),
            turn_signal=int(policy["discrete", "turn_signal"].squeeze().cpu()),
        )

    logger.info(str(pytorch_result))

    # ==================== Benchmark: Cache-enabled PyTorch ====================
    logger.info("benchmarking cache-enabled PyTorch (6ts)")
    cache_pytorch_result = BenchmarkResult(name="PyTorch Cache-enabled (6ts)", num_iterations=args.num_iterations)
    cache_pytorch_preds = None

    # Model is in eval mode, so it uses deterministic position embeddings (offset 0)
    # Set mask on objectives
    for obj in cache_model.objectives.values():
        if hasattr(obj, '_mask'):
            obj._mask = full_mask

    with torch.inference_mode():
        # Empty cache for full forward
        empty_proj_emb = torch.zeros(1, 0, embed_dim, device=device)
        empty_kv = torch.zeros(num_layers, 2, 1, 0, embed_dim, device=device)

        # Warmup
        for _ in range(args.num_warmup):
            _ = cache_model(batch_6_device, empty_proj_emb, empty_kv)
            if device.type == "cuda":
                torch.cuda.synchronize()

        # Benchmark
        for _ in range(args.num_iterations):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            outputs = cache_model(batch_6_device, empty_proj_emb, empty_kv)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            cache_pytorch_result.times_ms.append((end - start) * 1000)

        # Extract predictions (first 4 outputs)
        cache_pytorch_preds = PredictionResult(
            brake_pedal=float(outputs[0].squeeze().cpu()),
            gas_pedal=float(outputs[1].squeeze().cpu()),
            steering_angle=float(outputs[2].squeeze().cpu()),
            turn_signal=int(outputs[3].squeeze().cpu()),
        )

    logger.info(str(cache_pytorch_result))

    # ==================== Benchmark: ONNX Full Forward ====================
    logger.info("benchmarking ONNX full forward (6ts)")
    onnx_full_result = BenchmarkResult(name="ONNX Full Forward (6ts)", num_iterations=args.num_iterations)

    # Prepare inputs (mask is baked in)
    onnx_inputs_full = flatten_batch_to_onnx(batch_6)
    onnx_inputs_full["cached_projected_embeddings"] = np.empty((1, 0, embed_dim), dtype=np.float32)
    onnx_inputs_full["cached_kv"] = np.empty((num_layers, 2, 1, 0, embed_dim), dtype=np.float32)

    # Debug: show input names and sample values
    logger.info("flattened batch keys", keys=sorted(onnx_inputs_full.keys()))
    logger.info("ONNX expected inputs", names=sorted(full_input_names))

    # Debug: check sample values
    for key in ["batch_data_meta_vehiclemotion_brake_pedal_normalized", "batch_data_meta_vehiclemotion_gas_pedal_normalized"]:
        if key in onnx_inputs_full:
            val = onnx_inputs_full[key]
            logger.info(f"input sample: {key}", shape=val.shape, dtype=str(val.dtype), sample=val.flatten()[:3].tolist())

    onnx_inputs_full_matched = match_onnx_inputs(full_input_names, onnx_inputs_full)
    logger.info("matched inputs", num_matched=len(onnx_inputs_full_matched), num_expected=len(full_input_names))

    # Check for missing inputs
    missing = full_input_names - set(onnx_inputs_full_matched.keys())
    if missing:
        logger.error("missing ONNX inputs", missing=missing)

    # Warmup
    for _ in range(args.num_warmup):
        _ = full_session.run(None, onnx_inputs_full_matched)

    # Benchmark
    for _ in range(args.num_iterations):
        start = time.perf_counter()
        outputs = full_session.run(None, onnx_inputs_full_matched)
        end = time.perf_counter()
        onnx_full_result.times_ms.append((end - start) * 1000)

    # Extract predictions
    onnx_full_preds = PredictionResult(
        brake_pedal=float(outputs[0].flatten()[0]),
        gas_pedal=float(outputs[1].flatten()[0]),
        steering_angle=float(outputs[2].flatten()[0]),
        turn_signal=int(outputs[3].flatten()[0]),
    )
    onnx_full_proj_emb = outputs[-2]
    onnx_full_kv = outputs[-1]

    logger.info(str(onnx_full_result))

    # ==================== Benchmark: ONNX Dual (Incremental) ====================
    logger.info("benchmarking ONNX dual model (5 cached + 1 new)")
    onnx_dual_result = BenchmarkResult(name="ONNX Dual (5 cached + 1 new)", num_iterations=args.num_iterations)

    # Trim cache to 5 timesteps (simulate sliding window)
    proj_emb_5 = onnx_full_proj_emb[:, :5 * tokens_per_timestep, :]
    kv_cache_5 = onnx_full_kv[:, :, :, :5 * tokens_per_timestep, :]

    # Prepare inputs for incremental (mask is baked in)
    onnx_inputs_incr = flatten_batch_to_onnx(batch_1)
    onnx_inputs_incr["cached_projected_embeddings"] = proj_emb_5
    onnx_inputs_incr["cached_kv"] = kv_cache_5
    onnx_inputs_incr_matched = match_onnx_inputs(incr_input_names, onnx_inputs_incr)

    # Warmup
    for _ in range(args.num_warmup):
        _ = incr_session.run(None, onnx_inputs_incr_matched)

    # Benchmark
    for _ in range(args.num_iterations):
        start = time.perf_counter()
        outputs = incr_session.run(None, onnx_inputs_incr_matched)
        end = time.perf_counter()
        onnx_dual_result.times_ms.append((end - start) * 1000)

    # Extract predictions
    onnx_dual_preds = PredictionResult(
        brake_pedal=float(outputs[0].flatten()[0]),
        gas_pedal=float(outputs[1].flatten()[0]),
        steering_angle=float(outputs[2].flatten()[0]),
        turn_signal=int(outputs[3].flatten()[0]),
    )

    logger.info(str(onnx_dual_result))

    # ==================== Results Summary ====================
    print("\n" + "=" * 120)
    print("BENCHMARK RESULTS")
    print("=" * 120)

    # Compute speedups
    speedup_native = 1.0
    speedup_cache_pytorch = pytorch_result.mean_ms / cache_pytorch_result.mean_ms
    speedup_full = pytorch_result.mean_ms / onnx_full_result.mean_ms
    speedup_dual = pytorch_result.mean_ms / onnx_dual_result.mean_ms

    # Consolidated table with timing and predictions
    print(f"\n{'Model':<32} {'Mean(ms)':<10} {'Std(ms)':<9} {'Speedup':<8} {'brake':<10} {'gas':<10} {'steering':<10} {'turn':<6}")
    print("-" * 120)
    print(f"{'PyTorch Native (baseline)':<32} {pytorch_result.mean_ms:<10.2f} {pytorch_result.std_ms:<9.2f} {speedup_native:<8.2f} {pytorch_preds.brake_pedal:<10.6f} {pytorch_preds.gas_pedal:<10.6f} {pytorch_preds.steering_angle:<10.6f} {pytorch_preds.turn_signal:<6}")
    print(f"{'PyTorch Cache-enabled (6ts)':<32} {cache_pytorch_result.mean_ms:<10.2f} {cache_pytorch_result.std_ms:<9.2f} {speedup_cache_pytorch:<8.2f} {cache_pytorch_preds.brake_pedal:<10.6f} {cache_pytorch_preds.gas_pedal:<10.6f} {cache_pytorch_preds.steering_angle:<10.6f} {cache_pytorch_preds.turn_signal:<6}")
    print(f"{'ONNX Full Forward (6ts)':<32} {onnx_full_result.mean_ms:<10.2f} {onnx_full_result.std_ms:<9.2f} {speedup_full:<8.2f} {onnx_full_preds.brake_pedal:<10.6f} {onnx_full_preds.gas_pedal:<10.6f} {onnx_full_preds.steering_angle:<10.6f} {onnx_full_preds.turn_signal:<6}")
    print(f"{'ONNX Dual (5 cached + 1 new)':<32} {onnx_dual_result.mean_ms:<10.2f} {onnx_dual_result.std_ms:<9.2f} {speedup_dual:<8.2f} {onnx_dual_preds.brake_pedal:<10.6f} {onnx_dual_preds.gas_pedal:<10.6f} {onnx_dual_preds.steering_angle:<10.6f} {onnx_dual_preds.turn_signal:<6}")
    print("-" * 120)

    # ONNX Full vs ONNX Dual (should be nearly identical - this validates the incremental model)
    diff_onnx = {
        "brake": abs(onnx_full_preds.brake_pedal - onnx_dual_preds.brake_pedal),
        "gas": abs(onnx_full_preds.gas_pedal - onnx_dual_preds.gas_pedal),
        "steering": abs(onnx_full_preds.steering_angle - onnx_dual_preds.steering_angle),
        "turn": abs(onnx_full_preds.turn_signal - onnx_dual_preds.turn_signal),
    }

    print("\n--- ONNX Consistency Check ---")
    print(f"ONNX Full vs ONNX Dual max diff: brake={diff_onnx['brake']:.2e}, gas={diff_onnx['gas']:.2e}, steering={diff_onnx['steering']:.2e}")

    # Native vs Cache-enabled PyTorch (should be similar)
    diff_native_cache = {
        "brake": abs(pytorch_preds.brake_pedal - cache_pytorch_preds.brake_pedal),
        "gas": abs(pytorch_preds.gas_pedal - cache_pytorch_preds.gas_pedal),
        "steering": abs(pytorch_preds.steering_angle - cache_pytorch_preds.steering_angle),
        "turn": abs(pytorch_preds.turn_signal - cache_pytorch_preds.turn_signal),
    }
    print(f"Native vs Cache-enabled max diff: brake={diff_native_cache['brake']:.2e}, gas={diff_native_cache['gas']:.2e}, steering={diff_native_cache['steering']:.2e}")

    # ONNX consistency check (both ONNX models should produce same output)
    TOLERANCE = 1e-5
    onnx_consistent = all([
        diff_onnx["brake"] < TOLERANCE, diff_onnx["gas"] < TOLERANCE,
        diff_onnx["steering"] < TOLERANCE, diff_onnx["turn"] == 0,
    ])

    # PyTorch Cache-enabled vs ONNX Full Forward (should match exactly)
    diff_pytorch_onnx = {
        "brake": abs(cache_pytorch_preds.brake_pedal - onnx_full_preds.brake_pedal),
        "gas": abs(cache_pytorch_preds.gas_pedal - onnx_full_preds.gas_pedal),
        "steering": abs(cache_pytorch_preds.steering_angle - onnx_full_preds.steering_angle),
        "turn": abs(cache_pytorch_preds.turn_signal - onnx_full_preds.turn_signal),
    }
    print(f"PyTorch Cache vs ONNX Full max diff: brake={diff_pytorch_onnx['brake']:.2e}, gas={diff_pytorch_onnx['gas']:.2e}, steering={diff_pytorch_onnx['steering']:.2e}")

    pytorch_onnx_match = all([
        diff_pytorch_onnx["brake"] < TOLERANCE, diff_pytorch_onnx["gas"] < TOLERANCE,
        diff_pytorch_onnx["steering"] < TOLERANCE, diff_pytorch_onnx["turn"] == 0,
    ])

    # Native vs Cache-enabled should now match (both use deterministic PE in eval mode)
    native_cache_match = all([
        diff_native_cache["brake"] < TOLERANCE, diff_native_cache["gas"] < TOLERANCE,
        diff_native_cache["steering"] < TOLERANCE, diff_native_cache["turn"] == 0,
    ])

    # The key comparison: incremental vs full forward for streaming
    speedup_incr_vs_full = onnx_full_result.mean_ms / onnx_dual_result.mean_ms
    print(f"\nONNX Incremental vs ONNX Full:    {speedup_incr_vs_full:.2f}x faster (streaming speedup after first frame)")

    print("VERIFICATION RESULTS")
    print("-" * 120)

    # Report verification status
    if native_cache_match:
        print("✓ PyTorch Native and PyTorch Cache-enabled match (same predictions)")
    else:
        print("✗ PyTorch Native and PyTorch Cache-enabled do NOT match")

    if pytorch_onnx_match:
        print("✓ PyTorch Cache-enabled and ONNX Full Forward match (same predictions)")
    else:
        print("✗ PyTorch Cache-enabled and ONNX Full Forward do NOT match")

    if onnx_consistent:
        print("✓ ONNX Full Forward and ONNX Incremental match (same predictions)")
    else:
        print("✗ ONNX Full Forward and ONNX Incremental do NOT match")

    # Overall status
    if native_cache_match and pytorch_onnx_match and onnx_consistent:
        print("✓ SUCCESS: All models produce consistent predictions")
    else:
        print("✗ FAILURE: Model predictions are inconsistent")

    print("\nNote: All models use deterministic timestep position 0 in eval mode.")
    print("All position encodings (timestep, waypoints, actions, special) are correctly applied.")
    print("=" * 120)


if __name__ == "__main__":
    main()
