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
from pytorch_lightning.utilities.model_summary.model_summary import ModelSummary
from structlog import get_logger
from torch import Tensor, nn


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
            logger.debug(
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


def _repeat_batch_timesteps(batch: dict[str, Any], target_timesteps: int) -> dict[str, Any]:
    """Repeat a batch to have target_timesteps by duplicating data along timestep dimension.

    Args:
        batch: Batch dict with tensors of shape [B, T, ...] where T is current timesteps
        target_timesteps: Target number of timesteps

    Returns:
        Batch with tensors expanded to [B, target_timesteps, ...]
    """
    result = {}
    for key, value in batch.items():
        if isinstance(value, dict):
            result[key] = _repeat_batch_timesteps(value, target_timesteps)
        elif isinstance(value, Tensor) and value.dim() >= 2:
            # Tensor with shape [B, T, ...] - repeat along timestep dimension
            current_t = value.shape[1]
            if current_t < target_timesteps:
                repeats = (target_timesteps + current_t - 1) // current_t
                # Repeat and slice to exact target
                repeated = value.repeat(1, repeats, *([1] * (value.dim() - 2)))
                result[key] = repeated[:, :target_timesteps]
            else:
                result[key] = value[:, :target_timesteps]
        else:
            result[key] = value
    return result



class CacheEnabledControlTransformer(nn.Module):
    """Cache-enabled ControlTransformer wrapper for ONNX export.

    Supports efficient sequential inference by:
    1. Computing projected_embeddings (no PE) for the input batch
    2. Concatenating with cached projected_embeddings from previous calls
    3. Adding precomputed position_embeddings to the full sequence
    4. Running encoder with KV cache for efficient attention

    Position embedding strategy:
    - Cache `projected_embeddings` (no position embeddings at all)
    - `position_embeddings_packed` for 6 timesteps is CONSTANT - precompute once and reuse
    - After concatenation (always 6 timesteps = 1644 tokens), add precomputed position_embeddings

    The model takes cached_projected_embeddings and cached_kv as inputs.
    For the first call, pass zero-sized tensors (S_cached=0).

    The attention mask can optionally be baked into the model at construction time.
    Pass the mask to __init__ to bake it in, or omit it to pass mask as a forward() argument.
    """

    def __init__(
        self,
        model: nn.Module,
        mask: Tensor | None = None,
        position_embeddings_packed: Tensor | None = None,
    ) -> None:
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

        # Optional baked-in mask (if provided at construction time)
        self._baked_mask = mask
        if mask is not None:
            logger.debug("baked-in mask set", mask_shape=tuple(mask.shape))

        # Precomputed position embeddings for full sequence (6 timesteps)
        # This is constant and can be reused for all forward calls
        if position_embeddings_packed is not None:
            self.register_buffer("_position_embeddings_packed", position_embeddings_packed)
            logger.debug("position_embeddings_packed set", shape=tuple(position_embeddings_packed.shape))
        else:
            self._position_embeddings_packed: Tensor | None = None

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
        from rmind.components.episode import EpisodeExport, Modality, SummaryToken

        # Get the policy objective
        policy_obj = self.objectives.get("policy")
        if policy_obj is None:
            return {}

        # Extract token embeddings using episode indices (same logic as PolicyObjective._compute_logits)
        if isinstance(episode, EpisodeExport):
            # EpisodeExport uses plain dict indices
            observation_summary = encoder_output[
                :,
                episode.index[Modality.SUMMARY.value][
                    SummaryToken.OBSERVATION_SUMMARY.value
                ][-1],
            ]

            observation_history = encoder_output[
                :,
                episode.index[Modality.SUMMARY.value][
                    SummaryToken.OBSERVATION_HISTORY.value
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
                    (Modality.SUMMARY, SummaryToken.OBSERVATION_HISTORY),
                    (Modality.SUMMARY, SummaryToken.OBSERVATION_SUMMARY),
                    (Modality.CONTEXT, "waypoints"),
                )
                .parse(encoder_output)
            )

            observation_history = embeddings.get((
                Modality.SUMMARY,
                SummaryToken.OBSERVATION_HISTORY,
            ))

            observation_summary = embeddings.get((
                Modality.SUMMARY,
                SummaryToken.OBSERVATION_SUMMARY,
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

    def _forward_impl(
        self,
        batch: dict,
        cached_projected_embeddings: Tensor,
        cached_kv: Tensor,
        mask: Tensor,
    ) -> tuple[dict, Tensor, Tensor]:
        """Internal forward implementation.

        Args:
            batch: Input batch dict (full batch or single timestep)
            cached_projected_embeddings: [B, S_cached, D] - cached projected embeddings
                                         WITHOUT any position embeddings (S_cached=0 for first call)
            cached_kv: [L, 2, B, S_cached, D] - cached KV (S_cached=0 for first call)
            mask: Attention mask [S_new, S_total]

        Returns:
            predictions: Dict of prediction tensors
            all_projected_embeddings: [B, S_total, D] - projected embeddings for caching (no PE)
            kv_cache: [L, 2, B, S_total, D] - updated KV cache
        """
        # Build episode with timestep_offset=0 for consistent position embeddings
        episode = self.episode_builder(batch, timestep_offset=0)

        # Get projected embeddings (no PE)
        new_projected_embeddings = episode.projected_embeddings_packed
        batch_size, new_seq_len, embed_dim = new_projected_embeddings.shape

        # Compute tokens_per_timestep
        proj_emb = episode.projected_embeddings
        first_leaf = next(
            leaf for leaf in _get_tensor_leaves(proj_emb)
            if leaf is not None and leaf.dim() >= 2
        )
        num_new_timesteps = first_leaf.shape[1]
        self._tokens_per_timestep = new_seq_len // num_new_timesteps

        # Get position embeddings for the new tokens
        # position_embeddings = embeddings - projected_embeddings
        new_position_embeddings = episode.embeddings_packed - new_projected_embeddings

        # Concatenate projected embeddings (no PE)
        all_projected_embeddings = torch.cat(
            [cached_projected_embeddings, new_projected_embeddings], dim=1
        )
        total_seq_len = all_projected_embeddings.shape[1]

        # Get position embeddings for full sequence
        if self._position_embeddings_packed is not None:
            # Use precomputed position embeddings (for incremental model)
            # For incremental: cached has 5ts, new has 1ts -> total 6ts
            # Position embeddings are for full 6 timesteps
            position_embeddings = self._position_embeddings_packed
        else:
            # Compute from episode (for full model with 6 timesteps)
            # Store for potential reuse
            position_embeddings = new_position_embeddings

        # Add position embeddings to get final embeddings
        all_embeddings = all_projected_embeddings + position_embeddings

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

        # Return projected embeddings (no PE) for caching
        return (*sorted_preds, all_projected_embeddings, kv_cache)

    def forward(
        self,
        batch: dict,
        cached_projected_embeddings: Tensor,
        cached_kv: Tensor,
    ) -> tuple[dict, Tensor, Tensor]:
        """Forward pass using baked-in mask.

        This method uses the mask that was provided at construction time.
        Use this for ONNX export when you want the mask baked into the model.

        Args:
            batch: Input batch dict (full batch or single timestep)
            cached_projected_embeddings: [B, S_cached, D] - cached embeddings with non-timestep PEs
                                         (waypoints, actions, special) but WITHOUT timestep PE.
                                         Note: Parameter name kept for ONNX compatibility.
            cached_kv: [L, 2, B, S_cached, D] - cached KV

        Returns:
            Tuple of (predictions..., all_embeddings_with_non_ts_pe, kv_cache)
        """
        if self._baked_mask is None:
            msg = "No baked-in mask set. Use forward_with_mask() or pass mask to __init__."
            raise RuntimeError(msg)
        return self._forward_impl(batch, cached_projected_embeddings, cached_kv, self._baked_mask)

    def forward_with_mask(
        self,
        batch: dict,
        cached_projected_embeddings: Tensor,
        cached_kv: Tensor,
        mask: Tensor,
    ) -> tuple[dict, Tensor, Tensor]:
        """Forward pass with explicit mask argument.

        Use this when you want to pass the mask at runtime (e.g., for testing
        or when not using baked-in mask).

        Args:
            batch: Input batch dict (full batch or single timestep)
            cached_projected_embeddings: [B, S_cached, D] - cached projected embeddings
            cached_kv: [L, 2, B, S_cached, D] - cached KV
            mask: Attention mask [S_new, S_total]

        Returns:
            predictions + cache outputs
        """
        return self._forward_impl(batch, cached_projected_embeddings, cached_kv, mask)


class Config(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")

    model: dict[str, Any]  # Raw Hydra config dict (supports method _target_ like load_from_wandb_artifact)
    args: Annotated[Sequence[Any], AfterValidator(instantiate)]
    f: Path
    artifact: str | None = None  # W&B artifact to load weights from (e.g., "yaak/cargpt/model-xxx:v1")
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


def _load_weights_from_artifact(model: nn.Module, artifact: str) -> None:
    """Load model weights from a W&B artifact checkpoint."""
    import wandb

    logger.info("loading weights from artifact", artifact=artifact)

    # Download artifact
    api = wandb.Api()
    artifact_obj = api.artifact(artifact, type="model")
    artifact_dir = artifact_obj.download()
    ckpt_path = Path(artifact_dir) / "model.ckpt"

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)

    # Load weights into model
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning("missing keys when loading weights", missing=missing)
    if unexpected:
        logger.debug("unexpected keys when loading weights", unexpected=unexpected)

    logger.info("loaded weights from artifact", artifact=artifact)


@hydra.main(version_base=None)
@torch.inference_mode()
def main(cfg: DictConfig) -> None:
    config = Config(**OmegaConf.to_container(cfg, resolve=True))  # ty:ignore[invalid-argument-type]

    # Set seed BEFORE generating args for reproducible batch data
    # This ensures standalone verification can use the same batch data
    torch.manual_seed(42)

    logger.debug("instantiating", target=config.model.get("_target_"))
    args = instantiate(config.args, _recursive_=True, _convert_="all")
    base_model = instantiate(config.model).eval()

    # Load weights from artifact if provided
    if config.artifact:
        _load_weights_from_artifact(base_model, config.artifact)

    logger.debug(f"model summary:\n{ModelSummary(base_model)}")  # noqa: G004

    # Create cache-enabled wrapper
    # Build episode to get shapes BEFORE creating the model wrapper
    logger.debug("building episode for shape inference")
    batch = args[0]

    episode = base_model.episode_builder(batch)
    proj_embeddings = episode.projected_embeddings_packed
    batch_size, seq_len, embed_dim = proj_embeddings.shape
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

    # Precompute position_embeddings for 6 timesteps (constant, can be reused)
    # position_embeddings = embeddings - projected_embeddings
    if is_incremental:
        # For incremental, we need 6-timestep position_embeddings
        # Build a 6-timestep batch by repeating the 1-timestep data
        batch_6ts = _repeat_batch_timesteps(batch, target_timesteps=6)
        episode_6ts = base_model.episode_builder(batch_6ts, timestep_offset=0)
        position_embeddings_packed = episode_6ts.embeddings_packed - episode_6ts.projected_embeddings_packed
    else:
        # For full forward, extract from the current episode
        position_embeddings_packed = episode.embeddings_packed - proj_embeddings

    logger.debug("position_embeddings_packed shape", shape=position_embeddings_packed.shape)

    # Build the mask BEFORE creating the model (so it can be baked in)
    # Use the model's own PolicyObjective.build_attention_mask() to ensure
    # the mask matches the native forward path exactly.
    from rmind.components.mask import TorchAttentionMaskLegend
    from rmind.components.objectives.policy import PolicyObjective

    if is_incremental:
        logger.info("incremental export mode detected (1 timestep batch)")
        # For incremental export, use non-empty cache (5 timesteps worth)
        num_cached_timesteps = 5
        cached_seq_len = num_cached_timesteps * tokens_per_timestep
        total_seq_len_incr = cached_seq_len + seq_len

        # Build full 6-timestep mask from the model, then crop to incremental shape
        full_mask = PolicyObjective.build_attention_mask(
            episode_6ts.index, episode_6ts.timestep, legend=TorchAttentionMaskLegend
        ).mask.to(proj_embeddings.device)
        mask = full_mask[-tokens_per_timestep:, :]  # Crop to incremental shape

        logger.debug(
            "incremental export config",
            cached_seq_len=cached_seq_len,
            new_seq_len=seq_len,
            total_seq_len=total_seq_len_incr,
            mask_shape=tuple(mask.shape),
        )
    else:
        # Full forward export - use model's own mask building
        mask = PolicyObjective.build_attention_mask(
            episode.index, episode.timestep, legend=TorchAttentionMaskLegend
        ).mask.to(proj_embeddings.device)

        logger.debug("full forward mask", mask_shape=tuple(mask.shape))

    # Create cache-enabled wrapper with baked-in mask and position embeddings
    logger.debug("creating cache-enabled model wrapper with baked-in mask and position_embeddings")
    cache_model = CacheEnabledControlTransformer(
        base_model,
        mask=mask,
        position_embeddings_packed=position_embeddings_packed,
    ).eval()

    num_layers = cache_model.num_layers
    logger.debug("model config", num_layers=num_layers, embedding_dim=embed_dim)

    # Create cache tensors
    if is_incremental:
        cached_proj_emb = torch.randn(
            batch_size, cached_seq_len, embed_dim, device=proj_embeddings.device
        )
        cached_kv = torch.randn(
            num_layers, 2, batch_size, cached_seq_len, embed_dim,
            device=proj_embeddings.device
        )
    else:
        # Empty cache for full forward
        cached_proj_emb = torch.zeros(batch_size, 0, embed_dim, device=proj_embeddings.device)
        cached_kv = torch.zeros(
            num_layers, 2, batch_size, 0, embed_dim, device=proj_embeddings.device
        )

    # IMPORTANT: Set the baked-in mask on PolicyObjective before export
    # This ensures the objective's encoder call uses the same mask as the
    # CacheEnabledControlTransformer's encoder call.
    for obj_name, obj in cache_model.objectives.items():
        if hasattr(obj, '_mask'):
            logger.debug("setting mask on objective", objective=obj_name, mask_shape=tuple(mask.shape))
            obj._mask = mask

    # Set deterministic mode for reproducible results
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Test forward pass - patch _is_exporting() to ensure deterministic position embeddings
    # This makes PyTorch use the same PE positions (0-5) as will be traced into ONNX
    import rmind.components.episode as episode_module
    original_is_exporting = episode_module._is_exporting
    episode_module._is_exporting = lambda: True

    logger.debug("testing forward pass (using baked-in mask)", is_incremental=is_incremental)
    outputs = cache_model(batch, cached_proj_emb, cached_kv)  # No mask arg - uses baked-in

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

    # Export args - mask is baked in, so no mask argument needed
    logger.debug("exporting cache-enabled model (mask baked in)", dynamic_shapes=config.dynamic_shapes)
    export_args = (batch, cached_proj_emb, cached_kv)

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

        # Add dynamic axes for batch inputs (enables incremental inference)
        # Note: Mask is baked in, so no dynamic axes needed for mask
        if config.dynamic_batch:
            if dynamic_axes is None:
                dynamic_axes = {}

            # Add dynamic timestep dimension for all batch inputs
            for input_name, timestep_dim in _get_batch_input_names(batch):
                dynamic_axes[input_name] = {timestep_dim: "timesteps"}
                logger.debug("adding dynamic axis", input=input_name, dim=timestep_dim)

            logger.info(
                "dynamic_batch enabled - batch inputs have dynamic shapes",
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

            # Create a wrapper that captures the mask to bake it into the export
            # This avoids issues with torch.export handling of class attributes
            class BakedMaskWrapper(nn.Module):
                def __init__(self, model: CacheEnabledControlTransformer, baked_mask: Tensor):
                    super().__init__()
                    self.model = model
                    # Register mask as a buffer so it gets serialized
                    self.register_buffer("_mask", baked_mask, persistent=False)

                def forward(self, batch: dict, cached_projected_embeddings: Tensor, cached_kv: Tensor):
                    # Use forward_with_mask with the baked-in mask buffer
                    return self.model.forward_with_mask(batch, cached_projected_embeddings, cached_kv, self._mask)

            wrapper = BakedMaskWrapper(cache_model, mask).eval()

            exported = torch.export.export(
                mod=wrapper,
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

            # Build input names for legacy export (mask is baked in)
            input_names = list(_get_batch_input_names(batch))
            input_names = [name for name, _ in input_names]
            input_names.extend(["cached_projected_embeddings", "cached_kv"])

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

            # Prepare ONNX inputs using the SAME inputs as export (mask is baked in)
            onnx_inputs = flatten_batch_to_onnx(batch)
            onnx_inputs["cached_projected_embeddings"] = cached_proj_emb.numpy()
            onnx_inputs["cached_kv"] = cached_kv.numpy()

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
