"""Verify that PyTorch and ONNX cache-enabled models produce the same outputs.

This script:
1. Loads the PyTorch model and ONNX model
2. Creates a test batch with deterministic inputs
3. Builds and sets the same mask on both models
4. Runs both models and compares outputs

Usage:
    python -m rmind.scripts.verify_onnx_cache \
        model=<model_config> \
        args=<args_config> \
        onnx_path=<path_to_onnx_model>
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
    """Flatten a nested batch dict to ONNX input format.

    ONNX input names are lowercase with underscores, e.g.:
    batch['data']['cam_front_left'] -> 'batch_data_cam_front_left'
    """
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


class Config(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")

    model: HydraConfig[LightningModule]
    args: Annotated[Sequence[Any], AfterValidator(instantiate)]
    onnx_path: Path


@hydra.main(version_base=None)
@torch.inference_mode()
def main(cfg: DictConfig) -> None:
    config = Config(**OmegaConf.to_container(cfg, resolve=True))

    # Set seed BEFORE generating args for reproducible batch data
    # This MUST match the export script to ensure the same batch data
    torch.manual_seed(42)

    logger.info("loading PyTorch model")
    args = instantiate(config.args, _recursive_=True, _convert_="all")
    base_model = config.model.instantiate().eval()

    # Set deterministic mode for reproducible forward passes
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Import here to avoid circular imports
    from rmind.components.mask import TorchAttentionMaskLegend
    from rmind.components.objectives.policy import PolicyObjective
    from rmind.scripts.export_onnx_cache import CacheEnabledControlTransformer

    # Create cache-enabled wrapper
    cache_model = CacheEnabledControlTransformer(base_model).eval()

    # Get model dimensions
    num_layers = cache_model.num_layers
    embed_dim = cache_model.embedding_dim
    logger.info("model config", num_layers=num_layers, embedding_dim=embed_dim)

    # Build episode BEFORE patching _is_exporting() to get the mask
    # (Episode type has .index with .max() method, EpisodeExport doesn't)
    batch = args[0]
    episode = base_model.episode_builder(batch)
    proj_embeddings = episode.projected_embeddings_packed
    batch_size, seq_len, _ = proj_embeddings.shape
    logger.info("projected_embeddings shape", shape=proj_embeddings.shape)

    # Compute tokens per timestep
    proj_emb_struct = episode.projected_embeddings
    first_leaf = next(
        leaf for leaf in _get_tensor_leaves(proj_emb_struct)
        if leaf is not None and leaf.dim() >= 2
    )
    num_timesteps = first_leaf.shape[1]
    tokens_per_timestep = seq_len // num_timesteps
    logger.info(
        "sequence structure",
        num_timesteps=num_timesteps,
        tokens_per_timestep=tokens_per_timestep,
        total_seq_len=seq_len,
    )

    # Build the mask using PolicyObjective.build_attention_mask
    # This is the canonical mask for this episode
    mask = PolicyObjective.build_attention_mask(
        episode.index, episode.timestep, legend=TorchAttentionMaskLegend
    ).mask.to(proj_embeddings.device)
    logger.info("mask shape", shape=mask.shape)

    # CRITICAL: Set the mask on PolicyObjective before PyTorch forward
    # This ensures the objective's encoder call uses the same mask
    for obj_name, obj in cache_model.objectives.items():
        if hasattr(obj, '_mask'):
            logger.info("setting mask on objective", objective=obj_name)
            obj._mask = mask

    # Empty cache for full forward
    cached_proj_emb = torch.zeros(batch_size, 0, embed_dim, device=proj_embeddings.device)
    cached_kv = torch.zeros(
        num_layers, 2, batch_size, 0, embed_dim, device=proj_embeddings.device
    )

    # Patch _is_exporting() to return True during PyTorch forward
    # This ensures position embeddings use offset=0 (same as during ONNX export)
    import rmind.components.episode as episode_module
    original_is_exporting = episode_module._is_exporting

    def patched_is_exporting() -> bool:
        return True

    episode_module._is_exporting = patched_is_exporting
    logger.info("patched _is_exporting() to return True for deterministic position embeddings")

    # Run PyTorch model
    logger.info("running PyTorch forward pass")
    try:
        pytorch_outputs = cache_model(batch, cached_proj_emb, cached_kv, mask)
    finally:
        # Restore original function
        episode_module._is_exporting = original_is_exporting
    logger.info("PyTorch outputs", num_outputs=len(pytorch_outputs))

    # Load ONNX model
    logger.info("loading ONNX model", path=str(config.onnx_path))
    session = ort.InferenceSession(
        str(config.onnx_path),
        providers=["CPUExecutionProvider"],
    )

    # Get ONNX input names
    onnx_input_names = {inp.name for inp in session.get_inputs()}
    logger.info("ONNX input names", names=sorted(onnx_input_names))

    # Prepare ONNX inputs
    onnx_inputs = flatten_batch_to_onnx(batch)
    onnx_inputs["cached_projected_embeddings"] = cached_proj_emb.numpy()
    onnx_inputs["cached_kv"] = cached_kv.numpy()
    onnx_inputs["mask"] = mask.numpy()

    # Match input names (case-insensitive)
    onnx_inputs_matched = {}
    for onnx_name in onnx_input_names:
        onnx_name_lower = onnx_name.lower()
        matched = False
        for our_name, value in onnx_inputs.items():
            if our_name.lower() == onnx_name_lower:
                onnx_inputs_matched[onnx_name] = value
                matched = True
                break
        if not matched:
            logger.warning("unmatched ONNX input", name=onnx_name)

    # Check for missing inputs
    missing = onnx_input_names - set(onnx_inputs_matched.keys())
    if missing:
        logger.error("missing ONNX inputs", missing=missing)
        raise ValueError(f"Missing ONNX inputs: {missing}")

    # Run ONNX model
    logger.info("running ONNX forward pass")
    onnx_outputs = session.run(None, onnx_inputs_matched)
    logger.info("ONNX outputs", num_outputs=len(onnx_outputs))

    # Get ONNX output names
    onnx_output_names = [out.name for out in session.get_outputs()]
    logger.info("ONNX output names", names=onnx_output_names)

    # Compare outputs
    logger.info("comparing outputs")

    # PyTorch outputs: (*predictions, proj_emb, kv_cache)
    # The predictions are sorted alphabetically by key
    prediction_names = [
        "policy_continuous_brake_pedal",
        "policy_continuous_gas_pedal",
        "policy_continuous_steering_angle",
        "policy_discrete_turn_signal",
    ]

    # Tolerances matching export script
    # Predictions: 1% tolerance (0.01) - scalar outputs with many ops
    # Embeddings: 0.2% tolerance (0.002) - larger outputs, more sensitive to drift
    PREDICTION_TOL = 0.01  # 1% tolerance for scalar predictions
    EMBEDDING_TOL = 0.002  # 0.2% tolerance for embeddings and KV cache

    all_match = True
    for i, (pytorch_out, onnx_out) in enumerate(zip(pytorch_outputs, onnx_outputs)):
        if i < len(prediction_names):
            name = prediction_names[i]
            tolerance = PREDICTION_TOL
        elif i == len(prediction_names):
            name = "projected_embeddings"
            tolerance = EMBEDDING_TOL
        else:
            name = "kv_cache"
            tolerance = EMBEDDING_TOL

        pytorch_np = pytorch_out.numpy() if isinstance(pytorch_out, Tensor) else pytorch_out
        diff = np.abs(pytorch_np - onnx_out)
        max_diff = diff.max()
        mean_diff = diff.mean()

        match_status = "MATCH" if max_diff < tolerance else "MISMATCH"
        if max_diff >= tolerance:
            all_match = False

        logger.info(
            f"output comparison: {name}",
            shape=pytorch_np.shape,
            max_diff=float(max_diff),
            mean_diff=float(mean_diff),
            status=match_status,
            tolerance=tolerance,
        )

    if all_match:
        logger.info("SUCCESS: All outputs match within tolerance")
    else:
        logger.error("FAILURE: Some outputs differ beyond tolerance")


if __name__ == "__main__":
    main()
