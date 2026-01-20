"""Inference utilities for efficient sequential prediction with caching.

This module provides caching mechanisms to avoid redundant computation
during sequential inference where consecutive predictions share overlapping
context windows.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils._pytree import tree_leaves, tree_map  # noqa: PLC2701

from rmind.components.base import TensorTree
from rmind.components.episode import Episode, EpisodeBuilder, TimestepEmbeddings
from rmind.components.llm import KVCache

if TYPE_CHECKING:
    from rmind.components.llm import TransformerEncoder


class EmbeddingCache(Module):
    """Cache for computed embeddings to avoid recomputation during sequential inference.

    When processing sequential data with overlapping context windows (e.g., timesteps
    [0,1,2,3,4] followed by [1,2,3,4,5]), this cache stores embeddings for each
    timestep so that only new timesteps need computation.

    Attributes:
        episode_builder: The underlying EpisodeBuilder to compute embeddings
        max_cached_timesteps: Maximum number of timesteps to cache (LRU eviction)
    """

    def __init__(
        self,
        episode_builder: EpisodeBuilder,
        max_cached_timesteps: int = 32,
    ) -> None:
        super().__init__()
        self.episode_builder = episode_builder
        self.max_cached_timesteps = max_cached_timesteps

        # OrderedDict for LRU eviction - keys are timestep IDs
        # Values are dicts with per-timestep slices of TimestepEmbeddings fields
        self._cache: OrderedDict[int, dict[str, TensorTree]] = OrderedDict()

    def clear(self) -> None:
        """Clear all cached embeddings."""
        self._cache.clear()

    def get_cached_timesteps(self) -> list[int]:
        """Return list of currently cached timestep IDs."""
        return list(self._cache.keys())

    def _slice_batch_at_timestep(
        self, batch: TensorTree, timestep_idx: int
    ) -> TensorTree:
        """Extract a single timestep from a batch along the time dimension."""
        return tree_map(
            lambda x: x[:, timestep_idx : timestep_idx + 1] if x is not None else None,
            batch,
        )

    def _concat_timesteps(
        self, timestep_data: list[dict[str, TensorTree]]
    ) -> dict[str, TensorTree]:
        """Concatenate multiple timestep data dicts along the time dimension."""
        if not timestep_data:
            msg = "Cannot concatenate empty timestep list"
            raise ValueError(msg)

        keys = timestep_data[0].keys()
        result = {}

        for key in keys:
            tensors = [td[key] for td in timestep_data]
            # Concatenate along time dimension (dim=1)
            result[key] = tree_map(
                lambda *xs: (
                    torch.cat(xs, dim=1)
                    if all(x is not None for x in xs)
                    else None
                ),
                *tensors,
            )

        return result

    def compute_embeddings_cached(
        self,
        batch: TensorTree,
        timestep_ids: list[int],
    ) -> TimestepEmbeddings:
        """Compute embeddings using cache for previously seen timesteps.

        Args:
            batch: Input batch with shape [B, T, ...] where T matches len(timestep_ids)
            timestep_ids: Unique identifiers for each timestep in the batch.
                         These should be monotonically increasing frame indices.

        Returns:
            TimestepEmbeddings with embeddings for all requested timesteps
        """
        if len(timestep_ids) == 0:
            msg = "timestep_ids cannot be empty"
            raise ValueError(msg)

        # Identify which timesteps need computation vs are cached
        cached_indices: list[int] = []  # indices into timestep_ids that are cached
        new_indices: list[int] = []  # indices into timestep_ids that need computation

        for idx, tid in enumerate(timestep_ids):
            if tid in self._cache:
                cached_indices.append(idx)
                # Move to end for LRU
                self._cache.move_to_end(tid)
            else:
                new_indices.append(idx)

        # Get batch size and device from input
        batch_size, device = None, None
        for leaf in tree_leaves(batch):
            if leaf is not None:
                batch_size = (leaf.shape[0], 1)  # Single timestep
                device = leaf.device
                break

        if batch_size is None:
            msg = "Could not determine batch size from input"
            raise ValueError(msg)

        # Compute embeddings for new timesteps
        if new_indices:
            for idx in new_indices:
                tid = timestep_ids[idx]
                single_timestep_batch = self._slice_batch_at_timestep(batch, idx)

                # Compute embeddings for this single timestep
                emb = self.episode_builder.compute_embeddings(single_timestep_batch)

                # Store in cache
                self._cache[tid] = {
                    "input": emb.input,
                    "input_tokens": emb.input_tokens,
                    "input_embeddings": emb.input_embeddings,
                    "projected_embeddings": emb.projected_embeddings,
                }

                # Evict old entries if over capacity
                while len(self._cache) > self.max_cached_timesteps:
                    self._cache.popitem(last=False)

        # Assemble all timesteps in order from cache
        ordered_data = [self._cache[tid] for tid in timestep_ids]
        combined = self._concat_timesteps(ordered_data)

        # Determine actual batch size from combined data
        for leaf in tree_leaves(combined["projected_embeddings"]):
            if leaf is not None:
                actual_batch_size = leaf.shape[:2]
                break

        return TimestepEmbeddings(
            input=combined["input"],
            input_tokens=combined["input_tokens"],
            input_embeddings=combined["input_embeddings"],
            projected_embeddings=combined["projected_embeddings"],
            batch_size=actual_batch_size,
            device=device,
        )

    def forward(
        self,
        batch: TensorTree,
        timestep_ids: list[int] | None = None,
    ) -> Episode:
        """Compute episode with embedding caching.

        Args:
            batch: Input batch with shape [B, T, ...]
            timestep_ids: Optional unique identifiers for each timestep.
                         If None, caching is disabled and standard forward is used.

        Returns:
            Episode built from cached + newly computed embeddings
        """
        if timestep_ids is None:
            # Fall back to standard computation without caching
            return self.episode_builder(batch)

        embeddings = self.compute_embeddings_cached(batch, timestep_ids)
        return self.episode_builder.assemble_episode(embeddings)


class InferenceEngine(Module):
    """Orchestrates embedding cache and KV cache for efficient sequential inference.

    This class manages both:
    1. EmbeddingCache: Caches computed embeddings per timestep
    2. KV Cache: Caches key/value projections in the transformer encoder

    For sequential inference where predictions include overlapping context,
    this avoids redundant computation of both embeddings and attention.

    Example usage:
        ```python
        engine = InferenceEngine(model, context_length=5)

        for frame_id, observation in enumerate(observations):
            prediction = engine.step(observation, frame_id)

        # Reset on episode boundary
        engine.reset()
        ```
    """

    def __init__(
        self,
        model: Module,
        context_length: int = 5,
        max_cached_timesteps: int = 32,
    ) -> None:
        """Initialize the inference engine.

        Args:
            model: ControlTransformer model or similar with episode_builder and encoder
            context_length: Number of timesteps in the context window
            max_cached_timesteps: Maximum timesteps to keep in embedding cache
        """
        super().__init__()

        self.model = model
        self.context_length = context_length

        # Embedding cache wraps the episode builder
        self.embedding_cache = EmbeddingCache(
            episode_builder=model.episode_builder,
            max_cached_timesteps=max_cached_timesteps,
        )

        # KV cache state (managed by encoder)
        self._kv_cache: list[KVCache] | None = None
        self._cached_seq_len: int = 0

        # Track the timestep IDs currently in context
        self._current_timestep_ids: list[int] = []

    def reset(self) -> None:
        """Reset all caches. Call this at episode boundaries."""
        self.embedding_cache.clear()
        self._kv_cache = None
        self._cached_seq_len = 0
        self._current_timestep_ids = []

    @property
    def encoder(self) -> TransformerEncoder:
        """Get the encoder from the model."""
        return self.model.encoder

    def _build_context_window(
        self,
        new_observation: TensorTree,
        timestep_id: int,
    ) -> tuple[TensorTree, list[int]]:
        """Build the context window batch for the current timestep.

        Args:
            new_observation: Observation for the new timestep [B, 1, ...]
            timestep_id: Unique ID for this timestep

        Returns:
            Tuple of (batch with full context window, timestep_ids for the window)
        """
        # Determine which timesteps should be in context
        start_id = max(0, timestep_id - self.context_length + 1)
        window_ids = list(range(start_id, timestep_id + 1))

        # For now, we only have the new observation
        # The cached embeddings will be retrieved by EmbeddingCache
        # We need to construct a batch that includes all timesteps

        # This is a simplified version - in practice, you'd need to
        # maintain a buffer of recent observations
        return new_observation, window_ids

    def step(
        self,
        batch: TensorTree,
        timestep_ids: list[int],
        mask: Tensor | None = None,
    ) -> tuple[Tensor, list[KVCache] | None]:
        """Process a batch with caching for efficient sequential inference.

        This method:
        1. Uses embedding cache to avoid recomputing embeddings for cached timesteps
        2. Uses KV cache to avoid recomputing attention for cached positions

        Args:
            batch: Input batch [B, T, ...]
            timestep_ids: Unique IDs for each timestep in the batch
            mask: Optional attention mask

        Returns:
            Tuple of (encoder output, updated KV cache)
        """
        # Get embeddings (cached where possible)
        episode = self.embedding_cache(batch, timestep_ids)

        # Determine which positions are new vs cached in KV cache
        num_new_positions = len(timestep_ids) - len(self._current_timestep_ids)

        if num_new_positions <= 0 or self._kv_cache is None:
            # Full recomputation needed
            embeddings_packed = episode.embeddings_packed

            output, new_kv_cache = self.encoder(
                src=embeddings_packed,
                mask=mask,
                use_cache=True,
                past_key_values=None,
            )

            self._kv_cache = new_kv_cache
            self._current_timestep_ids = timestep_ids.copy()

            return output, new_kv_cache

        # Incremental update: only process new positions
        # Get embeddings for just the new positions
        embeddings_packed = episode.embeddings_packed

        # Calculate how many sequence positions are new
        # (this depends on tokens per timestep)
        tokens_per_timestep = embeddings_packed.shape[1] // len(timestep_ids)
        new_seq_positions = num_new_positions * tokens_per_timestep

        # Extract only new embeddings
        new_embeddings = embeddings_packed[:, -new_seq_positions:]

        # Build incremental mask if provided
        if mask is not None:
            # Extract rows for new positions attending to all positions
            incremental_mask = mask[-new_seq_positions:]
        else:
            incremental_mask = None

        # Run encoder with KV cache
        output, new_kv_cache = self.encoder(
            src=new_embeddings,
            mask=incremental_mask,
            use_cache=True,
            past_key_values=self._kv_cache,
        )

        self._kv_cache = new_kv_cache
        self._current_timestep_ids = timestep_ids.copy()

        return output, new_kv_cache

    def forward(
        self,
        batch: TensorTree,
        timestep_ids: list[int] | None = None,
        mask: Tensor | None = None,
        use_cache: bool = True,
    ) -> Tensor:
        """Forward pass with optional caching.

        Args:
            batch: Input batch
            timestep_ids: Timestep identifiers for caching. If None, no caching.
            mask: Attention mask
            use_cache: Whether to use KV caching

        Returns:
            Encoder output tensor
        """
        if not use_cache or timestep_ids is None:
            # Standard forward without caching
            episode = self.embedding_cache.episode_builder(batch)
            return self.encoder(src=episode.embeddings_packed, mask=mask)

        output, _ = self.step(batch, timestep_ids, mask)
        return output
