"""Export cache-enabled ControlTransformer model to ONNX.

This script exports a model that supports efficient sequential inference:

Inputs:
    batch: Input batch dict (full batch for initial call, single timestep for subsequent)
    cached_projected_embeddings: [B, S_cached, D] - cached projected embeddings WITHOUT position
                                 embeddings (zeros with S_cached=0 for first call)
    cached_kv: [L, 2, B, S_cached, D] - cached KV (zeros with S_cached=0 for first call)
    mask: Attention mask [S_new, S_total] where S_total = S_cached + S_new

Outputs:
    predictions: Model predictions
    all_projected_embeddings: [B, S_total, D] - all projected embeddings WITHOUT position
                              embeddings (cache this for next call)
    kv_cache: [L, 2, B, S_total, D] - updated KV cache (cache this for next call)

IMPORTANT: This model caches projected_embeddings (before position embedding), not
embeddings (after position embedding). Position embeddings are applied after
concatenating cached and new projected embeddings. This ensures correct position
encoding for the full sequence.

Usage:
    # First call (process all timesteps)
    preds, proj_emb, kv = model(full_batch, empty_emb, empty_kv, full_mask)

    # Subsequent calls (process single timestep)
    preds, proj_emb, kv = model(single_timestep_batch, cached_proj_emb, cached_kv, incr_mask)
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from collections.abc import Iterator
from typing import Annotated, Any, ClassVar, Literal

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pydantic import AfterValidator, BaseModel, ConfigDict
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.model_summary.model_summary import ModelSummary
from structlog import get_logger
from torch import Tensor, nn

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


def _get_batch_input_names(
    batch: dict[str, Any], prefix: str = "batch"
) -> Iterator[tuple[str, int]]:
    """Recursively yield (input_name, timestep_dim) for batch tensors.

    ONNX flattens nested dicts into names like 'batch_data_cam_front_left'.
    For tensors with shape [B, T, ...], the timestep dimension is 1.
    """
    def _recurse(obj: Any, current_prefix: str) -> Iterator[tuple[str, int]]:
        if isinstance(obj, Tensor):
            # Tensors with dim >= 2 have timestep at dim 1 (after batch)
            if obj.dim() >= 2:
                # ONNX uses lowercase and underscores for input names
                name = current_prefix.lower().replace("/", "_")
                yield (name, 1)
        elif isinstance(obj, dict):
            for key, value in obj.items():
                # ONNX uses underscore to join nested keys
                new_prefix = f"{current_prefix}_{key}"
                yield from _recurse(value, new_prefix)

    yield from _recurse(batch, prefix)





class CacheEnabledControlTransformer(nn.Module):
    """Cache-enabled ControlTransformer wrapper for ONNX export.

    Supports efficient sequential inference by:
    1. Computing projected embeddings (WITHOUT position embeddings) for the input batch
    2. Concatenating with cached projected embeddings from previous calls
    3. Applying position embeddings to the full concatenated sequence
    4. Running encoder with KV cache for efficient attention

    IMPORTANT: This wrapper caches projected_embeddings (before position embedding),
    not embeddings_packed (after position embedding). This ensures correct position
    encoding when concatenating cached and new embeddings:
    - Cached projected embeddings for timesteps 0-4 get PE positions 0-4
    - New projected embeddings for timestep 5 get PE position 5
    - Total sequence gets PE positions 0-5 (not 0-4, 5, 5 which would happen
      if we cached embeddings with PE already applied)

    The model always takes cached_projected_embeddings and cached_kv as inputs.
    For the first call, pass zero-sized tensors (S_cached=0).
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.episode_builder = model.episode_builder
        self.objectives = model.objectives

        # Get encoder from model or first objective
        self.encoder = getattr(model, "encoder", None)
        if self.encoder is None:
            for objective in self.objectives.values():
                if hasattr(objective, "encoder") and objective.encoder is not None:
                    self.encoder = objective.encoder
                    break

        if self.encoder is None:
            msg = "No encoder found in model or objectives"
            raise ValueError(msg)

        # Model dimensions
        self.num_layers = len(list(self.encoder.layers))
        self.embedding_dim = self.encoder.layers[0].embedding_dim

        # Compute tokens per timestep from episode builder
        self._tokens_per_timestep: int | None = None

    @property
    def tokens_per_timestep(self) -> int:
        """Number of tokens per timestep (computed lazily)."""
        if self._tokens_per_timestep is None:
            msg = "tokens_per_timestep not set. Call forward() first or set manually."
            raise RuntimeError(msg)
        return self._tokens_per_timestep

    def _compute_policy_predictions(
        self,
        encoder_output: Tensor,
        episode: Any,
    ) -> dict[str, Tensor]:
        """Compute policy predictions directly from encoder output.

        This uses the encoder output from the KV-cached forward path, ensuring
        a single encoder pass is used for both cache updates and predictions.

        Args:
            encoder_output: [B, S_new, D] - encoder output for new tokens only
            episode: Episode or EpisodeExport with token indices

        Returns:
            Dict mapping prediction names to tensors
        """
        from einops import rearrange
        from rmind.components.episode import EpisodeExport, Modality, SpecialToken

        # Get the policy objective
        policy_obj = self.objectives.get("policy")
        if policy_obj is None:
            return {}

        # Extract token embeddings using episode indices (same logic as PolicyObjective._compute_logits)
        if isinstance(episode, EpisodeExport):
            # EpisodeExport uses plain dict indices
            observation_summary = encoder_output[
                :,
                episode.index[Modality.SPECIAL.value][
                    SpecialToken.OBSERVATION_SUMMARY.value
                ][-1],
            ]

            observation_history = encoder_output[
                :,
                episode.index[Modality.SPECIAL.value][
                    SpecialToken.OBSERVATION_HISTORY.value
                ][-1],
            ]

            waypoints = encoder_output[
                :, episode.index[Modality.CONTEXT.value]["waypoints"][-1]
            ].mean(dim=1, keepdim=True)
        else:
            # Episode uses TensorDict indices
            embeddings = (
                episode
                .index[-1]
                .select(
                    (Modality.SPECIAL, SpecialToken.OBSERVATION_HISTORY),
                    (Modality.SPECIAL, SpecialToken.OBSERVATION_SUMMARY),
                    (Modality.CONTEXT, "waypoints"),
                )
                .parse(encoder_output)
            )

            observation_history = embeddings.get((
                Modality.SPECIAL,
                SpecialToken.OBSERVATION_HISTORY,
            ))

            observation_summary = embeddings.get((
                Modality.SPECIAL,
                SpecialToken.OBSERVATION_SUMMARY,
            ))

            waypoints = embeddings.get((Modality.CONTEXT, "waypoints")).mean(
                dim=1, keepdim=True
            )

        # Combine features (same as PolicyObjective._compute_logits)
        features = rearrange(
            [observation_summary, observation_history.detach(), waypoints],
            "i b 1 d -> b 1 (i d)",
        )

        # Get predictions from policy heads
        logits = policy_obj.heads(features)

        # Convert logits to predictions (same as PolicyObjective.forward)
        from rmind.components.episode import Modality
        from rmind.utils.functional import non_zero_signal_with_threshold

        def to_prediction(path: tuple, logit: Tensor) -> Tensor:
            # path elements are MappingKey objects, extract the key string
            modality_str = path[0].key if hasattr(path[0], "key") else str(path[0])
            action_name = path[1].key if hasattr(path[1], "key") else str(path[1])
            action_type = (Modality(modality_str), action_name)
            match action_type:
                case (Modality.CONTINUOUS, _):
                    return logit[..., 0]
                case (Modality.DISCRETE, "turn_signal"):
                    return non_zero_signal_with_threshold(logit).class_idx
                case _:
                    msg = f"Invalid action type: {action_type}"
                    raise NotImplementedError(msg)

        from torch.utils._pytree import tree_map_with_path  # noqa: PLC2701
        return tree_map_with_path(to_prediction, logits)

    def forward(
        self,
        batch: dict,
        cached_projected_embeddings: Tensor,
        cached_kv: Tensor,
        mask: Tensor,
    ) -> tuple[dict, Tensor, Tensor]:
        """Forward pass with cache support.

        Args:
            batch: Input batch dict (full batch or single timestep)
            cached_projected_embeddings: [B, S_cached, D] - cached projected embeddings
                                         WITHOUT position embeddings (S_cached=0 for first call)
            cached_kv: [L, 2, B, S_cached, D] - cached KV (S_cached=0 for first call)
            mask: Attention mask [S_new, S_total]
                  - First call: [S_full, S_full] (S_cached=0, so S_new=S_total)
                  - Subsequent: [S_new, S_total] where S_total = S_cached + S_new

        Returns:
            predictions: Dict of prediction tensors
            all_projected_embeddings: [B, S_total, D] - all projected embeddings for caching
                                      (WITHOUT position embeddings)
            kv_cache: [L, 2, B, S_total, D] - updated KV cache
        """
        # Build episode and get projected embeddings (WITHOUT position embeddings)
        episode = self.episode_builder(batch)
        new_projected_embeddings = episode.projected_embeddings_packed

        # Compute and cache tokens_per_timestep
        batch_size, new_seq_len, embed_dim = new_projected_embeddings.shape
        # Get number of timesteps from episode's projected_embeddings structure
        # The projected_embeddings has shape [B, T, ...] where T is num_timesteps
        proj_emb = episode.projected_embeddings
        # Get first leaf tensor to find num_timesteps
        first_leaf = next(
            leaf for leaf in _get_tensor_leaves(proj_emb)
            if leaf is not None and leaf.dim() >= 2
        )
        num_new_timesteps = first_leaf.shape[1]
        self._tokens_per_timestep = new_seq_len // num_new_timesteps

        # Concatenate with cached projected embeddings (without PE)
        all_projected_embeddings = torch.cat(
            [cached_projected_embeddings, new_projected_embeddings], dim=1
        )

        # Compute total number of timesteps
        total_seq_len = all_projected_embeddings.shape[1]
        total_timesteps = total_seq_len // self._tokens_per_timestep

        # Apply position embeddings to the full concatenated sequence
        all_embeddings = self.episode_builder.apply_timestep_position_embeddings(
            all_projected_embeddings, total_timesteps, timestep_offset=0
        )

        # Get new embeddings (with PE) for encoder - only the new portion
        new_embeddings = all_embeddings[:, -new_seq_len:]

        # Run encoder with KV cache (uses tensor interface for export compatibility)
        encoder_output, kv_cache = self.encoder.forward_with_kv_tensor(
            new_embeddings, mask, cached_kv
        )

        # Compute predictions directly from encoder output (single encoder path)
        # This avoids running a second encoder call in PolicyObjective
        predictions = self._compute_policy_predictions(encoder_output, episode)

        # Flatten TensorDict/dict outputs to plain tensors for ONNX compatibility
        # Expected structure: policy -> continuous/discrete -> individual predictions
        flat_predictions = {}
        for cat_name, cat_output in predictions.items():
            if hasattr(cat_output, 'items'):  # Nested dict/TensorDict
                for pred_name, pred_tensor in cat_output.items():
                    flat_predictions[f"policy_{cat_name}_{pred_name}"] = pred_tensor
            elif isinstance(cat_output, Tensor):
                flat_predictions[f"policy_{cat_name}"] = cat_output

        # Return as tuple: (brake_pedal, gas_pedal, steering_angle, turn_signal, proj_emb, kv)
        # Sorted by key for consistent ordering
        sorted_preds = [flat_predictions[k] for k in sorted(flat_predictions.keys())]

        # Return projected embeddings (without PE) for caching
        return (*sorted_preds, all_projected_embeddings, kv_cache)


class Config(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")

    model: HydraConfig[LightningModule]
    args: Annotated[Sequence[Any], AfterValidator(instantiate)]
    f: Path
    opset_version: int | None = None
    dynamo: bool = True  # Use dynamo-based export (True) or legacy export (False)
    external_data: bool = False
    optimize: bool = True
    verify: bool = False
    report: bool = True
    artifacts_dir: Path = Path.cwd()
    dynamic_shapes: bool = True  # Enable dynamic shapes for cache inputs
    dynamic_batch: bool = False  # Enable dynamic shapes for batch inputs and mask
    verify_outputs: bool = True  # Verify PyTorch vs ONNX outputs after export


@hydra.main(version_base=None)
@torch.inference_mode()
def main(cfg: DictConfig) -> None:
    config = Config(**OmegaConf.to_container(cfg, resolve=True))  # ty:ignore[invalid-argument-type]

    # Set seed BEFORE generating args for reproducible batch data
    # This ensures standalone verification can use the same batch data
    torch.manual_seed(42)

    logger.debug("instantiating", target=config.model.target)
    args = instantiate(config.args, _recursive_=True, _convert_="all")
    base_model = config.model.instantiate().eval()
    logger.debug(f"model summary:\n{ModelSummary(base_model)}")  # noqa: G004

    # Create cache-enabled wrapper
    logger.debug("creating cache-enabled model wrapper")
    cache_model = CacheEnabledControlTransformer(base_model).eval()

    num_layers = cache_model.num_layers
    embed_dim = cache_model.embedding_dim
    logger.debug("model config", num_layers=num_layers, embedding_dim=embed_dim)

    # Build episode to get shapes
    logger.debug("building episode for shape inference")
    batch = args[0]
    episode = base_model.episode_builder(batch)
    proj_embeddings = episode.projected_embeddings_packed
    batch_size, seq_len, _ = proj_embeddings.shape
    logger.debug("projected_embeddings shape", shape=proj_embeddings.shape)

    # Compute tokens per timestep
    proj_emb_struct = episode.projected_embeddings
    first_leaf = next(
        leaf for leaf in _get_tensor_leaves(proj_emb_struct)
        if leaf is not None and leaf.dim() >= 2
    )
    num_timesteps = first_leaf.shape[1]
    tokens_per_timestep = seq_len // num_timesteps
    logger.debug(
        "sequence structure",
        num_timesteps=num_timesteps,
        tokens_per_timestep=tokens_per_timestep,
        total_seq_len=seq_len,
    )

    # Determine if this is an incremental export (single timestep batch)
    is_incremental = num_timesteps == 1
    if is_incremental:
        logger.info("incremental export mode detected (1 timestep batch)")
        # For incremental export, use non-empty cache (5 timesteps worth)
        # This creates an ONNX model with fixed 1-timestep batch shape
        num_cached_timesteps = 5
        cached_seq_len = num_cached_timesteps * tokens_per_timestep
        total_seq_len = cached_seq_len + seq_len

        # Create dummy cache tensors
        cached_proj_emb = torch.randn(
            batch_size, cached_seq_len, embed_dim, device=proj_embeddings.device
        )
        cached_kv = torch.randn(
            num_layers, 2, batch_size, cached_seq_len, embed_dim,
            device=proj_embeddings.device
        )

        # Build incremental mask: [S_new, S_total] - new positions attend to all cached + self
        mask = torch.ones(seq_len, total_seq_len, dtype=torch.bool, device=proj_embeddings.device)
        for i in range(seq_len):
            # Position i (in new tokens) can attend to all cached + positions 0..i in new
            mask[i, : cached_seq_len + i + 1] = False

        logger.debug(
            "incremental export config",
            cached_seq_len=cached_seq_len,
            new_seq_len=seq_len,
            total_seq_len=total_seq_len,
            mask_shape=mask.shape,
        )
    else:
        # Full forward export (standard case)
        from rmind.components.mask import TorchAttentionMaskLegend
        from rmind.components.objectives.policy import PolicyObjective

        mask = PolicyObjective.build_attention_mask(
            episode.index, episode.timestep, legend=TorchAttentionMaskLegend
        ).mask.to(proj_embeddings.device)

        # Empty cache for full forward
        cached_proj_emb = torch.zeros(batch_size, 0, embed_dim, device=proj_embeddings.device)
        cached_kv = torch.zeros(
            num_layers, 2, batch_size, 0, embed_dim, device=proj_embeddings.device
        )

    # IMPORTANT: Set the mask on PolicyObjective before export
    # This ensures the objective's encoder call uses the same mask as the
    # CacheEnabledControlTransformer's encoder call. Without this, the objective
    # would use mask=None during export (since episode is EpisodeExport, not Episode).
    for obj_name, obj in cache_model.objectives.items():
        if hasattr(obj, '_mask'):
            logger.debug("setting mask on objective", objective=obj_name, mask_shape=mask.shape)
            # Use setattr to update the buffer value (register_buffer fails if already exists)
            obj._mask = mask

    # Set deterministic mode for reproducible results
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Test forward pass - patch _is_exporting() to ensure deterministic position embeddings
    # This makes PyTorch use the same PE positions (0-5) as will be traced into ONNX
    import rmind.components.episode as episode_module
    original_is_exporting = episode_module._is_exporting
    episode_module._is_exporting = lambda: True

    logger.debug("testing forward pass", is_incremental=is_incremental)
    outputs = cache_model(batch, cached_proj_emb, cached_kv, mask)

    # Restore original function before export (torch.export needs real is_exporting behavior)
    episode_module._is_exporting = original_is_exporting
    # Outputs: (*predictions, proj_emb, kv_cache) - last two are cache outputs
    proj_emb_out, kv_out = outputs[-2], outputs[-1]
    logger.debug(
        "forward output",
        num_outputs=len(outputs),
        proj_emb_shape=proj_emb_out.shape,
        kv_shape=kv_out.shape,
    )

    # Export args
    logger.debug("exporting cache-enabled model", dynamic_shapes=config.dynamic_shapes)
    export_args = (batch, cached_proj_emb, cached_kv, mask)

    try:
        # Build dynamic axes for legacy ONNX export
        dynamic_axes = None
        if config.dynamic_shapes:
            dynamic_axes = {
                # Cache inputs: sequence dimension is dynamic
                "cached_projected_embeddings": {1: "s_cached"},
                "cached_kv": {3: "s_cached"},
            }
            logger.debug("using dynamic axes for cache inputs")

        # Add dynamic axes for batch inputs and mask (enables incremental inference)
        if config.dynamic_batch:
            if dynamic_axes is None:
                dynamic_axes = {}

            # Add dynamic timestep dimension for all batch inputs
            for input_name, timestep_dim in _get_batch_input_names(batch):
                dynamic_axes[input_name] = {timestep_dim: "timesteps"}
                logger.debug("adding dynamic axis", input=input_name, dim=timestep_dim)

            # Add dynamic dimensions for mask [S_new, S_total]
            dynamic_axes["mask"] = {0: "s_new", 1: "s_total"}
            logger.debug("adding dynamic axes for mask")

            logger.info(
                "dynamic_batch enabled - batch inputs and mask have dynamic shapes",
                num_batch_inputs=len([k for k in dynamic_axes if k.startswith("batch")]),
            )

        if config.dynamo:
            # Use dynamo-based export (torch.export -> torch.onnx.export)
            # For dynamo export, we need to use torch.export.Dim() for dynamic shapes
            # The dynamic_axes parameter is only used by torch.onnx.export for naming
            logger.debug("using dynamo-based export")

            # Build dynamic_shapes for torch.export.export()
            export_dynamic_shapes: dict[str, Any] | None = None

            # NOTE: torch.export.Dim() dynamic shapes don't work with this model because:
            # 1. Model code has operations that specialize dimensions (integer division, slicing)
            # 2. Model code accesses .shape on dicts during non-strict tracing
            # 3. The episode builder uses mit.one() which doesn't work with symbolic tracing
            #
            # The workaround is to use the dual-model approach:
            # - export-onnx-cache: Full forward model (6 timesteps)
            # - export-onnx-incremental: Incremental model (1 timestep)
            #
            # For TensorRT, use optimization profiles on the incremental model to handle
            # different cache lengths by exporting with different num_cached_timesteps.
            logger.debug(
                "dynamo export does not support dynamic shapes for this model - "
                "using static shapes. See docs/INCREMENTAL_INFERENCE_REQUEST.md for workarounds."
            )

            exported = torch.export.export(
                mod=cache_model,
                args=export_args,
                strict=True,
            )

            torch.onnx.export(
                model=exported,
                f=config.f,
                opset_version=config.opset_version,
                dynamo=True,
                external_data=config.external_data,
                optimize=config.optimize,
                verify=config.verify,
                report=config.report,
                artifacts_dir=config.artifacts_dir,
                dynamic_axes=dynamic_axes,
            )
        else:
            # Use legacy torch.onnx.export (better dynamic_axes support)
            logger.debug("using legacy ONNX export")

            # Build input names for legacy export
            input_names = list(_get_batch_input_names(batch))
            input_names = [name for name, _ in input_names]
            input_names.extend(["cached_projected_embeddings", "cached_kv", "mask"])

            # Output names match the flattened predictions + cache outputs
            # Predictions are sorted alphabetically, then proj_emb and kv_cache
            output_names = [
                "policy_continuous_brake_pedal",
                "policy_continuous_gas_pedal",
                "policy_continuous_steering_angle",
                "policy_discrete_turn_signal",
                "projected_embeddings",
                "kv_cache",
            ]

            # NOTE: Legacy export with dynamic_axes doesn't work due to unsupported
            # operators (aten::_native_multi_head_attention from nn.MultiheadAttention).
            # Use dynamo=True instead. This code path is kept for reference.
            torch.onnx.export(
                model=cache_model,
                args=export_args,
                f=str(config.f),
                opset_version=config.opset_version or 17,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                do_constant_folding=True,
            )

        logger.debug(
            "exported",
            model=config.f.resolve().as_posix(),
            artifacts=config.artifacts_dir.resolve().as_posix(),
        )

        # Verify PyTorch vs ONNX outputs
        if config.verify_outputs:
            import numpy as np
            import onnxruntime as ort

            logger.info("verifying PyTorch vs ONNX outputs")

            # Flatten batch to ONNX input format
            def flatten_batch_to_onnx(batch_dict: dict[str, Any], prefix: str = "batch") -> dict[str, Any]:
                result = {}
                def _recurse(obj: Any, current_prefix: str) -> None:
                    if isinstance(obj, Tensor):
                        name = current_prefix.lower().replace("/", "_")
                        result[name] = obj.numpy()
                    elif isinstance(obj, dict):
                        for key, value in obj.items():
                            new_prefix = f"{current_prefix}_{key}"
                            _recurse(value, new_prefix)
                _recurse(batch_dict, prefix)
                return result

            # Load ONNX model
            session = ort.InferenceSession(str(config.f), providers=["CPUExecutionProvider"])
            onnx_input_names = {inp.name for inp in session.get_inputs()}

            # Prepare ONNX inputs using the SAME inputs as export
            onnx_inputs = flatten_batch_to_onnx(batch)
            onnx_inputs["cached_projected_embeddings"] = cached_proj_emb.numpy()
            onnx_inputs["cached_kv"] = cached_kv.numpy()
            onnx_inputs["mask"] = mask.numpy()

            # Match input names (case-insensitive)
            onnx_inputs_matched = {}
            for onnx_name in onnx_input_names:
                onnx_name_lower = onnx_name.lower()
                for our_name, value in onnx_inputs.items():
                    if our_name.lower() == onnx_name_lower:
                        onnx_inputs_matched[onnx_name] = value
                        break

            # Run ONNX model
            onnx_outputs = session.run(None, onnx_inputs_matched)

            # Run PyTorch model with the SAME inputs used during export tracing
            # Note: We need to run in export context to match the traced behavior
            pytorch_outputs = outputs  # Use the outputs from the test forward pass

            # Compare outputs
            prediction_names = [
                "policy_continuous_brake_pedal",
                "policy_continuous_gas_pedal",
                "policy_continuous_steering_angle",
                "policy_discrete_turn_signal",
                "projected_embeddings",
                "kv_cache",
            ]

            # With unified encoder path, differences should be minimal (FP epsilon level)
            PREDICTION_TOL = 1e-5  # ~0.001% tolerance for scalar predictions
            EMBEDDING_TOL = 0.002  # 0.2% tolerance for embeddings (larger due to more ops)

            all_match = True
            for i, (pytorch_out, onnx_out) in enumerate(zip(pytorch_outputs, onnx_outputs)):
                name = prediction_names[i] if i < len(prediction_names) else f"output_{i}"
                pytorch_np = pytorch_out.numpy() if isinstance(pytorch_out, Tensor) else pytorch_out
                diff = np.abs(pytorch_np - onnx_out)
                max_diff = float(diff.max())
                mean_diff = float(diff.mean())

                # Use appropriate tolerance based on output type
                is_prediction = i < 4  # First 4 outputs are predictions
                tol = PREDICTION_TOL if is_prediction else EMBEDDING_TOL
                match_status = "MATCH" if max_diff < tol else "MISMATCH"
                if max_diff >= tol:
                    all_match = False

                logger.info(
                    f"output comparison: {name}",
                    shape=pytorch_np.shape,
                    max_diff=max_diff,
                    mean_diff=mean_diff,
                    tolerance=tol,
                    status=match_status,
                )

            if all_match:
                logger.info("SUCCESS: All outputs match within tolerance")
            else:
                logger.warning("WARNING: Some outputs differ beyond tolerance - this may indicate issues with the export")

    except Exception as e:
        logger.error("export failed", error=str(e))
        raise


if __name__ == "__main__":
    main()
