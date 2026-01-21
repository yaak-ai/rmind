"""Verify dual ONNX model setup matches native PyTorch predictions.

This script:
1. Loads both full forward and incremental ONNX models
2. Creates test data with 7 timesteps
3. Runs full forward model with first 6 timesteps → get cache
4. Runs incremental model with 7th timestep using cache
5. Runs native PyTorch CacheEnabledControlTransformer with same setup
6. Compares predictions

Usage:
    python -m rmind.scripts.verify_dual_onnx \
        full_model=<path_to_full_onnx> \
        incremental_model=<path_to_incremental_onnx> \
        model=<model_config>
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any, ClassVar, Iterator

import hydra
import numpy as np
import onnxruntime as ort
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pydantic import AfterValidator, BaseModel, ConfigDict
from pytorch_lightning import LightningModule
from structlog import get_logger
from torch import Tensor

from rmind.config import HydraConfig

logger = get_logger(__name__)

# ImageNet normalization constants (used by DINOv2/DINOv3 models)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)


def normalize_images(batch: dict[str, Any]) -> dict[str, Any]:
    """Apply ImageNet normalization to images in batch.

    Normalizes images using ImageNet mean and std:
        normalized = (image - mean) / std

    Args:
        batch: Batch dict containing 'data/cam_front_left' with shape [B, T, C, H, W]

    Returns:
        Batch with normalized images
    """
    result = {}
    for key, value in batch.items():
        if isinstance(value, dict):
            result[key] = normalize_images(value)
        elif isinstance(value, Tensor) and key == "cam_front_left":
            # Image tensor: [B, T, C, H, W]
            # Normalize along channel dimension
            mean = IMAGENET_MEAN.view(1, 1, 3, 1, 1)
            std = IMAGENET_STD.view(1, 1, 3, 1, 1)
            result[key] = (value - mean) / std
            logger.info(
                "applied ImageNet normalization",
                key=key,
                shape=value.shape,
                input_range=(float(value.min()), float(value.max())),
                output_range=(float(result[key].min()), float(result[key].max())),
            )
        else:
            result[key] = value
    return result


def _get_tensor_leaves(tree: Any) -> Iterator[Tensor]:
    """Recursively yield all tensor leaves from a nested dict/TensorDict structure."""
    if isinstance(tree, Tensor):
        yield tree
    elif isinstance(tree, dict):
        for v in tree.values():
            yield from _get_tensor_leaves(v)
    elif hasattr(tree, "values"):  # TensorDict-like
        for v in tree.values():
            yield from _get_tensor_leaves(v)


def flatten_batch_to_onnx(batch: dict[str, Any], prefix: str = "batch") -> dict[str, np.ndarray]:
    """Flatten a nested batch dict to ONNX input format."""
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
            # Slice along dim 1 (timestep dimension)
            result[key] = value[:, timestep_slice]
        elif isinstance(value, dict):
            result[key] = slice_batch(value, timestep_slice)
        else:
            result[key] = value
    return result


class Config(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")

    model: HydraConfig[LightningModule]
    full_model: Path
    incremental_model: Path


@hydra.main(version_base=None)
@torch.inference_mode()
def main(cfg: DictConfig) -> None:
    config = Config(**OmegaConf.to_container(cfg, resolve=True))

    # Set seed for reproducible batch data
    torch.manual_seed(42)

    logger.info("loading PyTorch model")
    base_model = config.model.instantiate().eval()

    # Import here to avoid circular imports
    from rmind.components.mask import TorchAttentionMaskLegend
    from rmind.components.objectives.policy import PolicyObjective
    from rmind.scripts.export_onnx_cache import CacheEnabledControlTransformer

    # Create cache-enabled wrapper
    cache_model = CacheEnabledControlTransformer(base_model).eval()
    num_layers = cache_model.num_layers
    embed_dim = cache_model.embedding_dim
    logger.info("model config", num_layers=num_layers, embedding_dim=embed_dim)

    # Create test data with 6 timesteps
    # The incremental ONNX model expects 5 cached timesteps + 1 new timestep
    # So we compare: PyTorch with 6 timesteps vs ONNX dual (5 + 1)
    batch_6 = {
        "data": {
            "cam_front_left": torch.testing.make_tensor(
                (1, 6, 3, 256, 256), dtype=torch.float, device="cpu", low=0, high=1
            ),
            "meta/VehicleMotion/brake_pedal_normalized": torch.testing.make_tensor(
                (1, 6, 1), dtype=torch.float32, device="cpu", low=0, high=1
            ),
            "meta/VehicleMotion/gas_pedal_normalized": torch.testing.make_tensor(
                (1, 6, 1), dtype=torch.float32, device="cpu", low=0, high=1
            ),
            "meta/VehicleMotion/steering_angle_normalized": torch.testing.make_tensor(
                (1, 6, 1), dtype=torch.float32, device="cpu", low=-1, high=1
            ),
            "meta/VehicleMotion/speed": torch.testing.make_tensor(
                (1, 6, 1), dtype=torch.float32, device="cpu", low=0, high=130
            ),
            "meta/VehicleState/turn_signal": torch.testing.make_tensor(
                (1, 6, 1), dtype=torch.int, device="cpu", low=0, high=3
            ),
            "waypoints/xy_normalized": torch.testing.make_tensor(
                (1, 6, 10, 2), dtype=torch.float32, device="cpu", low=0, high=20
            ),
        }
    }

    # Apply ImageNet normalization to images
    # This matches the preprocessing expected by DINOv2/DINOv3 backbone
    batch_6 = normalize_images(batch_6)

    # Split into 5-timestep batch and 1-timestep batch
    # (incremental ONNX model expects 5 cached timesteps)
    batch_5 = slice_batch(batch_6, slice(0, 5))
    batch_1 = slice_batch(batch_6, slice(5, 6))

    logger.info("created test batches", batch_6_shape="6 timesteps", batch_5_shape="5 timesteps", batch_1_shape="1 timestep")

    # Build episodes and masks BEFORE patching _is_exporting()
    # (Episode type has .index with .max() method, EpisodeExport doesn't)

    # === Build 6-timestep episode and mask (full PyTorch comparison) ===
    logger.info("building 6-timestep episode and mask")
    episode_6 = base_model.episode_builder(batch_6)
    proj_embeddings_6 = episode_6.projected_embeddings_packed
    batch_size, seq_len_6, _ = proj_embeddings_6.shape
    logger.info("6-timestep sequence", seq_len=seq_len_6)

    # Get tokens per timestep
    proj_emb_struct = episode_6.projected_embeddings
    first_leaf = next(
        leaf for leaf in _get_tensor_leaves(proj_emb_struct)
        if leaf is not None and leaf.dim() >= 2
    )
    tokens_per_timestep = seq_len_6 // 6
    logger.info("tokens per timestep", tokens_per_timestep=tokens_per_timestep)

    # Build mask for 6 timesteps
    mask_6 = PolicyObjective.build_attention_mask(
        episode_6.index, episode_6.timestep, legend=TorchAttentionMaskLegend
    ).mask

    # === Build 5-timestep episode and mask ===
    logger.info("building 5-timestep episode and mask")
    episode_5 = base_model.episode_builder(batch_5)
    proj_embeddings_5 = episode_5.projected_embeddings_packed
    seq_len_5 = proj_embeddings_5.shape[1]

    mask_5 = PolicyObjective.build_attention_mask(
        episode_5.index, episode_5.timestep, legend=TorchAttentionMaskLegend
    ).mask

    # Build incremental mask for timestep 6 (attending to all 5 cached + 1 new)
    seq_len_1 = tokens_per_timestep
    total_seq_len = seq_len_5 + seq_len_1

    # Incremental mask: [S_new, S_total] - new positions attend to all previous + self
    incr_mask = torch.ones(seq_len_1, total_seq_len, dtype=torch.bool)
    for i in range(seq_len_1):
        # Position i (in new tokens) can attend to all cached + positions 0..i in new
        incr_mask[i, : seq_len_5 + i + 1] = False

    logger.info("masks built", mask_6_shape=mask_6.shape, mask_5_shape=mask_5.shape, incr_mask_shape=incr_mask.shape)

    # Patch _is_exporting() to return True for consistent behavior
    import rmind.components.episode as episode_module
    original_is_exporting = episode_module._is_exporting
    episode_module._is_exporting = lambda: True

    try:
        # === Step 1: Run PyTorch with full 6 timesteps ===
        logger.info("running PyTorch with 6 timesteps")

        # Set mask on objectives
        for obj in cache_model.objectives.values():
            if hasattr(obj, '_mask'):
                obj._mask = mask_6

        # Empty cache for full forward
        cached_proj_emb_empty = torch.zeros(batch_size, 0, embed_dim)
        cached_kv_empty = torch.zeros(num_layers, 2, batch_size, 0, embed_dim)

        # Run full forward with 6 timesteps
        pytorch_6_outputs = cache_model(batch_6, cached_proj_emb_empty, cached_kv_empty, mask_6)
        pytorch_6_preds = pytorch_6_outputs[:4]  # First 4 outputs are predictions
        logger.info("PyTorch 6-timestep predictions computed")

        # === Step 2: Run PyTorch with dual model setup (5 + 1) ===
        logger.info("running PyTorch dual model setup (5 + 1 timesteps)")

        # Set mask on objectives for 5-timestep forward
        for obj in cache_model.objectives.values():
            if hasattr(obj, '_mask'):
                obj._mask = mask_5

        # Run full forward with 5 timesteps
        pytorch_5_outputs = cache_model(batch_5, cached_proj_emb_empty, cached_kv_empty, mask_5)
        proj_emb_5 = pytorch_5_outputs[-2]  # Second to last is projected embeddings
        kv_cache_5 = pytorch_5_outputs[-1]  # Last is KV cache
        logger.info("PyTorch 5-timestep forward done", proj_emb_shape=proj_emb_5.shape, kv_shape=kv_cache_5.shape)

        # Set incremental mask on objectives
        for obj in cache_model.objectives.values():
            if hasattr(obj, '_mask'):
                obj._mask = incr_mask

        # Run incremental forward with 1 timestep
        pytorch_incr_outputs = cache_model(batch_1, proj_emb_5, kv_cache_5, incr_mask)
        pytorch_incr_preds = pytorch_incr_outputs[:4]  # First 4 outputs are predictions
        logger.info("PyTorch incremental predictions computed")

        # === Step 3: Load and run ONNX models ===
        logger.info("loading ONNX models")

        full_session = ort.InferenceSession(str(config.full_model), providers=["CPUExecutionProvider"])
        incr_session = ort.InferenceSession(str(config.incremental_model), providers=["CPUExecutionProvider"])

        # Get ONNX input names
        full_input_names = {inp.name for inp in full_session.get_inputs()}
        incr_input_names = {inp.name for inp in incr_session.get_inputs()}
        logger.info("ONNX inputs", full_model=len(full_input_names), incremental=len(incr_input_names))

        # The full ONNX model was exported with 6 timesteps (fixed shapes with dynamo)
        # The incremental ONNX model was exported with 5 cached + 1 new = 1644 total
        # These are incompatible, so we verify them separately:
        # 1. ONNX full (6) vs PyTorch full (6)
        # 2. ONNX incremental (using PyTorch-generated cache) vs PyTorch incremental

        # === ONNX Full Model (6 timesteps) ===
        onnx_inputs_full = flatten_batch_to_onnx(batch_6)
        onnx_inputs_full["cached_projected_embeddings"] = cached_proj_emb_empty.numpy()
        onnx_inputs_full["cached_kv"] = cached_kv_empty.numpy()
        onnx_inputs_full["mask"] = mask_6.numpy()

        # Match input names for full model
        onnx_inputs_full_matched = {}
        for onnx_name in full_input_names:
            for our_name, value in onnx_inputs_full.items():
                if our_name.lower() == onnx_name.lower():
                    onnx_inputs_full_matched[onnx_name] = value
                    break

        # Run full forward ONNX with 6 timesteps
        logger.info("running ONNX full forward (6 timesteps)")
        onnx_full_outputs = full_session.run(None, onnx_inputs_full_matched)
        onnx_full_preds = onnx_full_outputs[:4]  # First 4 are predictions
        onnx_proj_emb_full = onnx_full_outputs[-2]
        onnx_kv_cache_full = onnx_full_outputs[-1]
        logger.info("ONNX 6-timestep forward done", proj_emb_shape=onnx_proj_emb_full.shape, kv_shape=onnx_kv_cache_full.shape)

        # === ONNX Incremental Model (1 timestep with PyTorch-generated 5-timestep cache) ===
        # Use PyTorch-generated cache (from pytorch_5_outputs) for ONNX incremental
        # This tests that ONNX incremental model produces correct results given a valid cache
        onnx_inputs_1 = flatten_batch_to_onnx(batch_1)
        onnx_inputs_1["cached_projected_embeddings"] = proj_emb_5.numpy()  # Use PyTorch cache
        onnx_inputs_1["cached_kv"] = kv_cache_5.numpy()  # Use PyTorch cache
        onnx_inputs_1["mask"] = incr_mask.numpy()

        # Match input names
        onnx_inputs_1_matched = {}
        for onnx_name in incr_input_names:
            for our_name, value in onnx_inputs_1.items():
                if our_name.lower() == onnx_name.lower():
                    onnx_inputs_1_matched[onnx_name] = value
                    break

        # Run incremental ONNX
        logger.info("running ONNX incremental (1 timestep)")
        onnx_incr_outputs = incr_session.run(None, onnx_inputs_1_matched)
        onnx_incr_preds = onnx_incr_outputs[:4]
        logger.info("ONNX incremental predictions computed")

        # === Step 4: Compare all predictions ===
        logger.info("comparing predictions")

        prediction_names = [
            "policy_continuous_brake_pedal",
            "policy_continuous_gas_pedal",
            "policy_continuous_steering_angle",
            "policy_discrete_turn_signal",
        ]

        # Tolerances
        PREDICTION_TOL = 0.01  # 1% tolerance for predictions

        all_match = True

        # Compare PyTorch 6-timestep vs dual model (5+1)
        logger.info("=== PyTorch 6-timestep vs PyTorch dual (5+1) ===")
        for i, (pt6, pt_incr) in enumerate(zip(pytorch_6_preds, pytorch_incr_preds)):
            pt6_np = pt6.numpy() if isinstance(pt6, Tensor) else pt6
            pt_incr_np = pt_incr.numpy() if isinstance(pt_incr, Tensor) else pt_incr
            diff = np.abs(pt6_np - pt_incr_np)
            max_diff = diff.max()
            match_status = "MATCH" if max_diff < PREDICTION_TOL else "MISMATCH"
            if max_diff >= PREDICTION_TOL:
                all_match = False
            logger.info(
                f"{prediction_names[i]}",
                pytorch_6=float(pt6_np.flatten()[0]),
                pytorch_dual=float(pt_incr_np.flatten()[0]),
                max_diff=float(max_diff),
                status=match_status,
            )

        # Compare ONNX full 6-timestep vs PyTorch 6-timestep
        logger.info("=== ONNX full (6) vs PyTorch full (6) ===")
        for i, (onnx_pred, pt6) in enumerate(zip(onnx_full_preds, pytorch_6_preds)):
            pt6_np = pt6.numpy() if isinstance(pt6, Tensor) else pt6
            diff = np.abs(onnx_pred - pt6_np)
            max_diff = diff.max()
            match_status = "MATCH" if max_diff < PREDICTION_TOL else "MISMATCH"
            if max_diff >= PREDICTION_TOL:
                all_match = False
            logger.info(
                f"{prediction_names[i]}",
                onnx_full=float(onnx_pred.flatten()[0]),
                pytorch_full=float(pt6_np.flatten()[0]),
                max_diff=float(max_diff),
                status=match_status,
            )

        # Compare ONNX dual model vs PyTorch dual model
        logger.info("=== ONNX dual (5+1) vs PyTorch dual (5+1) ===")
        for i, (onnx_pred, pt_incr) in enumerate(zip(onnx_incr_preds, pytorch_incr_preds)):
            pt_incr_np = pt_incr.numpy() if isinstance(pt_incr, Tensor) else pt_incr
            diff = np.abs(onnx_pred - pt_incr_np)
            max_diff = diff.max()
            match_status = "MATCH" if max_diff < PREDICTION_TOL else "MISMATCH"
            if max_diff >= PREDICTION_TOL:
                all_match = False
            logger.info(
                f"{prediction_names[i]}",
                onnx_dual=float(onnx_pred.flatten()[0]),
                pytorch_dual=float(pt_incr_np.flatten()[0]),
                max_diff=float(max_diff),
                status=match_status,
            )

        # Compare ONNX dual model vs PyTorch 6-timestep (the ultimate test)
        logger.info("=== ONNX dual (5+1) vs PyTorch full (6) ===")
        for i, (onnx_pred, pt6) in enumerate(zip(onnx_incr_preds, pytorch_6_preds)):
            pt6_np = pt6.numpy() if isinstance(pt6, Tensor) else pt6
            diff = np.abs(onnx_pred - pt6_np)
            max_diff = diff.max()
            match_status = "MATCH" if max_diff < PREDICTION_TOL else "MISMATCH"
            if max_diff >= PREDICTION_TOL:
                all_match = False
            logger.info(
                f"{prediction_names[i]}",
                onnx_dual=float(onnx_pred.flatten()[0]),
                pytorch_full=float(pt6_np.flatten()[0]),
                max_diff=float(max_diff),
                status=match_status,
            )

        if all_match:
            logger.info("SUCCESS: All predictions match within tolerance")
        else:
            logger.error("FAILURE: Some predictions differ beyond tolerance")

    finally:
        # Restore original function
        episode_module._is_exporting = original_is_exporting


if __name__ == "__main__":
    main()
