"""Benchmark inference speed for ControlTransformer models.

Compares:
1. PyTorch model - full forward pass
2. ONNX model (standard) - full forward pass
3. ONNX model (cache-enabled) - full forward pass (first call)
4. ONNX model (cache-enabled) - incremental inference (subsequent calls)

Usage:
    uv run python -m rmind.scripts.benchmark_inference \
        --pytorch-model path/to/checkpoint.ckpt \
        --onnx-model path/to/model.onnx \
        --onnx-cache-model path/to/model_cache.onnx \
        --num-warmup 5 \
        --num-iterations 50
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from structlog import get_logger

logger = get_logger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    name: str
    num_iterations: int
    total_time_ms: float
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float

    def __str__(self) -> str:
        return (
            f"{self.name}:\n"
            f"  Iterations: {self.num_iterations}\n"
            f"  Mean: {self.mean_time_ms:.2f} ms\n"
            f"  Std:  {self.std_time_ms:.2f} ms\n"
            f"  Min:  {self.min_time_ms:.2f} ms\n"
            f"  Max:  {self.max_time_ms:.2f} ms\n"
        )


def create_dummy_batch(
    batch_size: int = 1,
    num_timesteps: int = 6,
    device: torch.device | str = "cuda",
) -> dict[str, Any]:
    """Create a dummy batch for benchmarking."""
    # This creates a minimal batch structure
    # Adjust based on your actual model input requirements
    return {
        "data": {
            "cam_front_left": torch.randint(
                0, 255, (batch_size, num_timesteps, 3, 128, 128),
                dtype=torch.uint8, device=device
            ),
            "meta": {
                "vehicle_motion": {
                    "speed": torch.rand(
                        batch_size, num_timesteps, 1,
                        dtype=torch.float32, device=device
                    ) * 30,
                    "steering_angle": torch.rand(
                        batch_size, num_timesteps, 1,
                        dtype=torch.float32, device=device
                    ) * 2 - 1,
                    "gas_pedal_normalized": torch.rand(
                        batch_size, num_timesteps, 1,
                        dtype=torch.float32, device=device
                    ),
                    "brake_pedal_normalized": torch.rand(
                        batch_size, num_timesteps, 1,
                        dtype=torch.float32, device=device
                    ),
                },
            },
        },
    }


def benchmark_pytorch(
    model: torch.nn.Module,
    batch: dict[str, Any],
    mask: torch.Tensor,
    num_warmup: int,
    num_iterations: int,
    device: torch.device,
) -> tuple[BenchmarkResult, BenchmarkResult]:
    """Benchmark PyTorch model inference.

    Returns two results:
    1. Full pipeline (episode_builder + encoder)
    2. Encoder only (pre-computed embeddings)
    """
    model = model.to(device).eval()

    # Get encoder
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        for obj in model.objectives.values():
            if hasattr(obj, "encoder") and obj.encoder is not None:
                encoder = obj.encoder
                break

    with torch.inference_mode():
        # Pre-compute embeddings for encoder-only benchmark
        episode = model.episode_builder(batch)
        embeddings = episode.embeddings_packed.clone()

        # Warmup
        for _ in range(num_warmup):
            _ = encoder(src=embeddings, mask=mask)
            torch.cuda.synchronize()

        # Benchmark encoder only
        encoder_times = []
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = encoder(src=embeddings, mask=mask)
            torch.cuda.synchronize()
            end = time.perf_counter()
            encoder_times.append((end - start) * 1000)

        # Warmup full pipeline
        for _ in range(num_warmup):
            episode = model.episode_builder(batch)
            _ = encoder(src=episode.embeddings_packed, mask=mask)
            torch.cuda.synchronize()

        # Benchmark full pipeline
        full_times = []
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            episode = model.episode_builder(batch)
            _ = encoder(src=episode.embeddings_packed, mask=mask)
            torch.cuda.synchronize()
            end = time.perf_counter()
            full_times.append((end - start) * 1000)

    encoder_times = np.array(encoder_times)
    full_times = np.array(full_times)

    return (
        BenchmarkResult(
            name="PyTorch encoder only (full sequence)",
            num_iterations=num_iterations,
            total_time_ms=encoder_times.sum(),
            mean_time_ms=encoder_times.mean(),
            std_time_ms=encoder_times.std(),
            min_time_ms=encoder_times.min(),
            max_time_ms=encoder_times.max(),
        ),
        BenchmarkResult(
            name="PyTorch full pipeline (episode_builder + encoder)",
            num_iterations=num_iterations,
            total_time_ms=full_times.sum(),
            mean_time_ms=full_times.mean(),
            std_time_ms=full_times.std(),
            min_time_ms=full_times.min(),
            max_time_ms=full_times.max(),
        ),
    )


def benchmark_pytorch_with_cache(
    model: torch.nn.Module,
    batch_full: dict[str, Any],
    batch_single: dict[str, Any],
    mask_full: torch.Tensor,
    mask_incremental: torch.Tensor,
    num_warmup: int,
    num_iterations: int,
    device: torch.device,
    tokens_per_timestep: int,
) -> tuple[BenchmarkResult, BenchmarkResult, BenchmarkResult]:
    """Benchmark PyTorch model with KV cache (incremental inference).

    Returns three results:
    1. Encoder only with KV cache (pre-computed embeddings)
    2. Full incremental pipeline (episode_builder + encoder with cache)
    3. Episode builder only (single timestep)
    """
    model = model.to(device).eval()

    # Get encoder
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        for obj in model.objectives.values():
            if hasattr(obj, "encoder") and obj.encoder is not None:
                encoder = obj.encoder
                break

    num_layers = len(list(encoder.layers))
    embed_dim = encoder.layers[0].embedding_dim

    with torch.inference_mode():
        # First, do a full forward to get initial cache
        episode_full = model.episode_builder(batch_full)
        _, kv_cache = encoder.forward_with_kv_tensor(
            episode_full.embeddings_packed,
            mask_full,
            torch.empty(num_layers, 2, 1, 0, embed_dim, device=device),
        )

        # Trim cache to simulate sliding window (remove first timestep)
        cached_seq_len = kv_cache.shape[3] - tokens_per_timestep
        kv_cache_trimmed = kv_cache[:, :, :, -cached_seq_len:].contiguous()

        # Pre-compute embeddings for encoder-only benchmark
        episode_single = model.episode_builder(batch_single)
        new_embeddings = model.episode_builder.apply_timestep_position_embeddings(
            episode_single.projected_embeddings_packed,
            num_timesteps=1,
            timestep_offset=5,
        ).clone()

        # Warmup encoder only
        for _ in range(num_warmup):
            _ = encoder.forward_with_kv_tensor(
                new_embeddings, mask_incremental, kv_cache_trimmed
            )
            torch.cuda.synchronize()

        # Benchmark encoder only with KV cache
        encoder_times = []
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = encoder.forward_with_kv_tensor(
                new_embeddings, mask_incremental, kv_cache_trimmed
            )
            torch.cuda.synchronize()
            end = time.perf_counter()
            encoder_times.append((end - start) * 1000)

        # Warmup episode builder only
        for _ in range(num_warmup):
            _ = model.episode_builder(batch_single)
            torch.cuda.synchronize()

        # Benchmark episode builder only (single timestep)
        episode_builder_times = []
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model.episode_builder(batch_single)
            torch.cuda.synchronize()
            end = time.perf_counter()
            episode_builder_times.append((end - start) * 1000)

        # Warmup full incremental pipeline
        for _ in range(num_warmup):
            episode_single = model.episode_builder(batch_single)
            emb = model.episode_builder.apply_timestep_position_embeddings(
                episode_single.projected_embeddings_packed,
                num_timesteps=1,
                timestep_offset=5,
            )
            _ = encoder.forward_with_kv_tensor(emb, mask_incremental, kv_cache_trimmed)
            torch.cuda.synchronize()

        # Benchmark full incremental pipeline
        full_times = []
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            episode_single = model.episode_builder(batch_single)
            emb = model.episode_builder.apply_timestep_position_embeddings(
                episode_single.projected_embeddings_packed,
                num_timesteps=1,
                timestep_offset=5,
            )
            _ = encoder.forward_with_kv_tensor(emb, mask_incremental, kv_cache_trimmed)
            torch.cuda.synchronize()
            end = time.perf_counter()
            full_times.append((end - start) * 1000)

    encoder_times = np.array(encoder_times)
    episode_builder_times = np.array(episode_builder_times)
    full_times = np.array(full_times)

    return (
        BenchmarkResult(
            name="PyTorch encoder only (incremental + KV cache)",
            num_iterations=num_iterations,
            total_time_ms=encoder_times.sum(),
            mean_time_ms=encoder_times.mean(),
            std_time_ms=encoder_times.std(),
            min_time_ms=encoder_times.min(),
            max_time_ms=encoder_times.max(),
        ),
        BenchmarkResult(
            name="PyTorch full pipeline (incremental + KV cache)",
            num_iterations=num_iterations,
            total_time_ms=full_times.sum(),
            mean_time_ms=full_times.mean(),
            std_time_ms=full_times.std(),
            min_time_ms=full_times.min(),
            max_time_ms=full_times.max(),
        ),
        BenchmarkResult(
            name="PyTorch episode_builder only (single timestep)",
            num_iterations=num_iterations,
            total_time_ms=episode_builder_times.sum(),
            mean_time_ms=episode_builder_times.mean(),
            std_time_ms=episode_builder_times.std(),
            min_time_ms=episode_builder_times.min(),
            max_time_ms=episode_builder_times.max(),
        ),
    )


def benchmark_onnx(
    session,
    inputs: dict[str, np.ndarray],
    output_names: list[str],
    num_warmup: int,
    num_iterations: int,
    name: str,
) -> BenchmarkResult:
    """Benchmark ONNX Runtime inference."""
    # Warmup
    for _ in range(num_warmup):
        _ = session.run(output_names, inputs)

    # Benchmark
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        _ = session.run(output_names, inputs)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    times = np.array(times)
    return BenchmarkResult(
        name=name,
        num_iterations=num_iterations,
        total_time_ms=times.sum(),
        mean_time_ms=times.mean(),
        std_time_ms=times.std(),
        min_time_ms=times.min(),
        max_time_ms=times.max(),
    )


def flatten_batch_to_numpy(
    batch: dict[str, Any],
    prefix: str = "batch",
    onnx_input_names: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Flatten nested batch dict and convert to numpy for ONNX.

    Args:
        batch: Nested batch dict
        prefix: Prefix for keys (default "batch")
        onnx_input_names: If provided, only include keys that match ONNX input names

    Returns:
        Flattened dict with numpy arrays
    """
    result = {}

    def _flatten(obj: Any, current_prefix: str) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_prefix = f"{current_prefix}_{key}".lower()
                _flatten(value, new_prefix)
        elif isinstance(obj, torch.Tensor):
            result[current_prefix] = obj.cpu().numpy()

    _flatten(batch, prefix)

    # Filter to only include expected ONNX inputs
    if onnx_input_names:
        filtered = {}
        for name in onnx_input_names:
            if name in result:
                filtered[name] = result[name]
            else:
                # Try to find a matching key (handle naming variations)
                for key in result:
                    if key.replace("/", "_").lower() == name.lower():
                        filtered[name] = result[key]
                        break
        return filtered

    return result


def build_causal_mask(seq_new: int, seq_total: int) -> np.ndarray:
    """Build causal attention mask."""
    start_idx = seq_total - seq_new
    mask = np.ones((seq_new, seq_total), dtype=bool)
    for i in range(seq_new):
        mask[i, : start_idx + i + 1] = False
    return mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark inference speed")
    parser.add_argument(
        "--config",
        type=str,
        default="yaak/control_transformer/raw_export",
        help="Model config path for Hydra",
    )
    parser.add_argument(
        "--onnx-cache-model",
        type=Path,
        default=None,
        help="Path to cache-enabled ONNX model",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--num-timesteps", type=int, default=6, help="Number of timesteps")
    parser.add_argument("--num-warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--num-iterations", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    args = parser.parse_args()

    device = torch.device(args.device)
    results: list[BenchmarkResult] = []

    # Load PyTorch model via Hydra
    logger.info("Loading PyTorch model...")
    import hydra
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    # Use Hydra to load model config
    config_path = Path(__file__).parents[3] / "config"
    with initialize_config_dir(config_dir=str(config_path), version_base=None):
        cfg = compose(
            config_name="export/onnx_cache",
            overrides=[f"model={args.config}"],
        )

    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    from rmind.config import HydraConfig
    from pytorch_lightning import LightningModule

    model_hydra = HydraConfig[LightningModule](**model_cfg)
    model = model_hydra.instantiate().to(device).eval()

    # Create dummy batches
    logger.info("Creating dummy batches...")

    # Use the model's input config to create proper batch
    input_cfg = OmegaConf.to_container(cfg.args, resolve=True)
    batch_full = instantiate(input_cfg, _recursive_=True, _convert_="all")[0]

    # Move batch to device
    def move_to_device(obj, device):
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        elif isinstance(obj, dict):
            return {k: move_to_device(v, device) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [move_to_device(v, device) for v in obj]
        return obj

    batch_full = move_to_device(batch_full, device)

    # Create single timestep batch (last timestep)
    def slice_last_timestep(obj):
        if isinstance(obj, torch.Tensor) and obj.dim() >= 2:
            return obj[:, -1:].contiguous()
        elif isinstance(obj, dict):
            return {k: slice_last_timestep(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [slice_last_timestep(v) for v in obj]
        return obj

    batch_single = slice_last_timestep(batch_full)

    # Build episode to get dimensions
    with torch.inference_mode():
        episode = model.episode_builder(batch_full)
        embeddings = episode.embeddings_packed
        batch_size, seq_len, embed_dim = embeddings.shape
        num_timesteps = args.num_timesteps
        tokens_per_timestep = seq_len // num_timesteps

    logger.info(
        "Model dimensions",
        batch_size=batch_size,
        seq_len=seq_len,
        embed_dim=embed_dim,
        tokens_per_timestep=tokens_per_timestep,
    )

    # Get encoder info
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        for obj in model.objectives.values():
            if hasattr(obj, "encoder") and obj.encoder is not None:
                encoder = obj.encoder
                break
    num_layers = len(list(encoder.layers))

    # Build masks
    from rmind.components.mask import TorchAttentionMaskLegend
    from rmind.components.objectives.policy import PolicyObjective

    mask_full = PolicyObjective.build_attention_mask(
        episode.index, episode.timestep, legend=TorchAttentionMaskLegend
    ).mask.to(device)

    # Incremental mask for single timestep
    seq_cached = seq_len - tokens_per_timestep
    seq_new = tokens_per_timestep
    seq_total = seq_len
    mask_incremental = torch.from_numpy(
        build_causal_mask(seq_new, seq_total)
    ).to(device)

    # ==================== PyTorch Benchmarks ====================
    logger.info("Running PyTorch full forward benchmark...")
    pytorch_encoder_full, pytorch_pipeline_full = benchmark_pytorch(
        model=model,
        batch=batch_full,
        mask=mask_full,
        num_warmup=args.num_warmup,
        num_iterations=args.num_iterations,
        device=device,
    )
    results.append(pytorch_encoder_full)
    results.append(pytorch_pipeline_full)
    logger.info(str(pytorch_encoder_full))
    logger.info(str(pytorch_pipeline_full))

    logger.info("Running PyTorch incremental (with KV cache) benchmark...")
    pytorch_encoder_incr, pytorch_pipeline_incr, pytorch_episode_single = benchmark_pytorch_with_cache(
        model=model,
        batch_full=batch_full,
        batch_single=batch_single,
        mask_full=mask_full,
        mask_incremental=mask_incremental,
        num_warmup=args.num_warmup,
        num_iterations=args.num_iterations,
        device=device,
        tokens_per_timestep=tokens_per_timestep,
    )
    results.append(pytorch_encoder_incr)
    results.append(pytorch_pipeline_incr)
    results.append(pytorch_episode_single)
    logger.info(str(pytorch_encoder_incr))
    logger.info(str(pytorch_pipeline_incr))
    logger.info(str(pytorch_episode_single))

    # ==================== ONNX Benchmarks ====================
    if args.onnx_cache_model is None:
        logger.info("No ONNX cache model specified. Skipping ONNX benchmarks.")
        logger.info("To export: just export-onnx-cache")
    elif not args.onnx_cache_model.exists():
        logger.warning("ONNX cache model not found", path=str(args.onnx_cache_model))
        logger.info("To export: just export-onnx-cache")

    if args.onnx_cache_model and args.onnx_cache_model.exists():
        import onnxruntime as ort

        logger.info("Loading ONNX cache model...", path=str(args.onnx_cache_model))

        # Configure ONNX Runtime
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        session = ort.InferenceSession(str(args.onnx_cache_model), providers=providers)

        input_names = [inp.name for inp in session.get_inputs()]
        output_names = [out.name for out in session.get_outputs()]

        logger.info("ONNX inputs", names=input_names)
        logger.info("ONNX outputs", names=output_names)

        # Get batch-related input names (exclude cache inputs)
        batch_input_names = [
            name for name in input_names
            if name not in ("cached_projected_embeddings", "cached_kv", "mask")
        ]

        # Prepare ONNX inputs for full forward
        batch_np = flatten_batch_to_numpy(batch_full, onnx_input_names=batch_input_names)
        empty_proj_emb = np.empty((batch_size, 0, embed_dim), dtype=np.float32)
        empty_kv = np.empty((num_layers, 2, batch_size, 0, embed_dim), dtype=np.float32)
        mask_full_np = mask_full.cpu().numpy()

        onnx_inputs_full = {
            **batch_np,
            "cached_projected_embeddings": empty_proj_emb,
            "cached_kv": empty_kv,
            "mask": mask_full_np,
        }

        logger.info("Running ONNX full forward benchmark...")
        result_onnx_full = benchmark_onnx(
            session=session,
            inputs=onnx_inputs_full,
            output_names=output_names,
            num_warmup=args.num_warmup,
            num_iterations=args.num_iterations,
            name="ONNX (full forward)",
        )
        results.append(result_onnx_full)
        logger.info(str(result_onnx_full))

        # Run one full forward to get cache for incremental benchmark
        outputs = session.run(output_names, onnx_inputs_full)

        # Find the projected_embeddings and kv_cache outputs by shape
        # projected_embeddings: [B, S, D] = [1, 1644, 384]
        # kv_cache: [L, 2, B, S, D] = [8, 2, 1, 1644, 384]
        cached_proj_emb = None
        cached_kv = None
        for i, (name, out) in enumerate(zip(output_names, outputs)):
            logger.debug(f"Output {i} ({name}): shape={out.shape}")
            if out.ndim == 3 and out.shape[2] == embed_dim:
                cached_proj_emb = out
            elif out.ndim == 5 and out.shape[0] == num_layers:
                cached_kv = out

        if cached_proj_emb is None or cached_kv is None:
            logger.error("Could not find projected_embeddings or kv_cache in outputs")
            logger.error(f"Output shapes: {[(name, out.shape) for name, out in zip(output_names, outputs)]}")
        else:
            logger.info("Found cache outputs", proj_emb_shape=cached_proj_emb.shape, kv_shape=cached_kv.shape)

            # Trim cache (remove first timestep for sliding window)
            cached_proj_emb_trimmed = cached_proj_emb[:, tokens_per_timestep:]
            cached_kv_trimmed = cached_kv[:, :, :, tokens_per_timestep:]

            # Prepare inputs for incremental inference
            batch_single_np = flatten_batch_to_numpy(batch_single, onnx_input_names=batch_input_names)
            mask_incremental_np = mask_incremental.cpu().numpy()

            onnx_inputs_incremental = {
                **batch_single_np,
                "cached_projected_embeddings": cached_proj_emb_trimmed,
                "cached_kv": cached_kv_trimmed,
                "mask": mask_incremental_np,
            }

            logger.info("Running ONNX incremental (with cache) benchmark...")
            try:
                result_onnx_cache = benchmark_onnx(
                    session=session,
                    inputs=onnx_inputs_incremental,
                    output_names=output_names,
                    num_warmup=args.num_warmup,
                    num_iterations=args.num_iterations,
                    name="ONNX (incremental with cache)",
                )
                results.append(result_onnx_cache)
                logger.info(str(result_onnx_cache))
            except Exception as e:
                logger.warning(
                    "ONNX incremental benchmark failed - model may have static shapes",
                    error=str(e),
                )
                logger.info(
                    "Note: For ONNX incremental inference, export with dynamic shapes "
                    "or use TensorRT with optimization profiles"
                )

    # ==================== Summary ====================
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    for result in results:
        print(f"\n{result}")

    # Calculate speedups
    print("\n" + "=" * 60)
    print("SPEEDUP ANALYSIS")
    print("=" * 60)

    # Encoder-only speedup (full sequence vs incremental with KV cache)
    print("\n--- Encoder-only speedup (KV cache effect) ---")
    print(f"  Full sequence:     {pytorch_encoder_full.mean_time_ms:.2f} ms")
    print(f"  Incremental+cache: {pytorch_encoder_incr.mean_time_ms:.2f} ms")
    encoder_speedup = pytorch_encoder_full.mean_time_ms / pytorch_encoder_incr.mean_time_ms
    print(f"  Speedup: {encoder_speedup:.2f}x")

    # Full pipeline speedup
    print("\n--- Full pipeline speedup (episode_builder + encoder) ---")
    print(f"  Full sequence:     {pytorch_pipeline_full.mean_time_ms:.2f} ms")
    print(f"  Incremental+cache: {pytorch_pipeline_incr.mean_time_ms:.2f} ms")
    pipeline_speedup = pytorch_pipeline_full.mean_time_ms / pytorch_pipeline_incr.mean_time_ms
    print(f"  Speedup: {pipeline_speedup:.2f}x")

    # Episode builder overhead
    episode_builder_time_full = pytorch_pipeline_full.mean_time_ms - pytorch_encoder_full.mean_time_ms
    print("\n--- Episode builder timing ---")
    print(f"  Full (6 timesteps):  {episode_builder_time_full:.2f} ms (computed)")
    print(f"  Single timestep:     {pytorch_episode_single.mean_time_ms:.2f} ms (measured)")
    episode_speedup = episode_builder_time_full / pytorch_episode_single.mean_time_ms
    print(f"  Speedup (6 vs 1):    {episode_speedup:.2f}x")

    # Breakdown analysis
    print("\n--- Time breakdown (single timestep inference) ---")
    print(f"  Episode builder:     {pytorch_episode_single.mean_time_ms:.2f} ms")
    print(f"  Encoder + KV cache:  {pytorch_encoder_incr.mean_time_ms:.2f} ms")
    print(f"  Total:               {pytorch_pipeline_incr.mean_time_ms:.2f} ms")

    # ONNX comparison if available
    onnx_results = [r for r in results if "ONNX" in r.name]
    if onnx_results:
        print("\n--- ONNX vs PyTorch comparison ---")
        for onnx_result in onnx_results:
            if "full forward" in onnx_result.name.lower():
                print(f"  ONNX full forward: {onnx_result.mean_time_ms:.2f} ms")
                print(f"    vs PyTorch encoder: {pytorch_encoder_full.mean_time_ms / onnx_result.mean_time_ms:.2f}x")
            elif "incremental" in onnx_result.name.lower():
                print(f"  ONNX incremental:  {onnx_result.mean_time_ms:.2f} ms")
                print(f"    vs PyTorch encoder incr: {pytorch_encoder_incr.mean_time_ms / onnx_result.mean_time_ms:.2f}x")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
