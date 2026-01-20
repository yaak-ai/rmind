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

        # Get predictions from objectives
        predictions = {name: obj(episode) for name, obj in self.objectives.items()}

        # Return projected embeddings (without PE) for caching
        return predictions, all_projected_embeddings, kv_cache


class Config(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")

    model: HydraConfig[LightningModule]
    args: Annotated[Sequence[Any], AfterValidator(instantiate)]
    f: Path
    opset_version: int | None = None
    dynamo: Literal[True] = True
    external_data: bool = False
    optimize: bool = True
    verify: bool = False
    report: bool = True
    artifacts_dir: Path = Path.cwd()


@hydra.main(version_base=None)
@torch.inference_mode()
def main(cfg: DictConfig) -> None:
    config = Config(**OmegaConf.to_container(cfg, resolve=True))  # ty:ignore[invalid-argument-type]

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

    # Get mask
    from rmind.components.mask import TorchAttentionMaskLegend
    from rmind.components.objectives.policy import PolicyObjective

    mask = PolicyObjective.build_attention_mask(
        episode.index, episode.timestep, legend=TorchAttentionMaskLegend
    ).mask.to(proj_embeddings.device)

    # Test Mode 0: Full forward (no cache)
    logger.debug("testing full forward (no cache)")
    empty_proj_emb = torch.zeros(batch_size, 0, embed_dim, device=proj_embeddings.device)
    empty_kv = torch.zeros(
        num_layers, 2, batch_size, 0, embed_dim, device=proj_embeddings.device
    )

    preds, proj_emb_out, kv_out = cache_model(batch, empty_proj_emb, empty_kv, mask)
    logger.debug(
        "full forward output", proj_emb_shape=proj_emb_out.shape, kv_shape=kv_out.shape
    )

    # Export with full forward args
    logger.debug("exporting cache-enabled model")
    export_args = (batch, empty_proj_emb, empty_kv, mask)

    try:
        exported = torch.export.export(
            mod=cache_model,
            args=export_args,
            strict=True,
        )

        torch.onnx.export(
            model=exported,
            f=config.f,
            opset_version=config.opset_version,
            dynamo=config.dynamo,
            external_data=config.external_data,
            optimize=config.optimize,
            verify=config.verify,
            report=config.report,
            artifacts_dir=config.artifacts_dir,
        )

        logger.debug(
            "exported",
            model=config.f.resolve().as_posix(),
            artifacts=config.artifacts_dir.resolve().as_posix(),
        )

    except Exception as e:
        logger.error("export failed", error=str(e))
        raise


if __name__ == "__main__":
    main()
