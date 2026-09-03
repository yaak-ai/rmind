"""Perceiver-style learned-query attention pooling.

Home for `AttentionPool`, the per-camera side-channel bottleneck behind
`PatchPolicy`'s `compress_cameras`/`camera_pool="attn"` (see
`src/rmind/models/patch_policy.py`): compress a variable-length token set down
to a small, fixed number of latents with one cross-attention layer instead of
feeding every token to the trunk.
"""

from typing import final, override

import torch
from torch import Tensor, nn
from torch.nn import Module


@final
class AttentionPool(Module):
    """A learned `(num_latents, dim)` query cross-attends once over `context`,
    producing `(*batch, num_latents, dim)` regardless of the context length.

    minGPT-style pre-LN block (mirrors `patch_policy.TransformerBlock`): one
    cross-attention (query = the learned latents, key/value = `context`) plus
    one MLP, each with a residual. The query is initialized like
    `PatchPolicy`'s `readout_token`/`register_tokens`
    (`trunc_normal_(std=0.02, a=-0.04, b=0.04)`) -- the same free-parameter,
    born-in-model-space status.
    """

    def __init__(
        self,
        *,
        dim: int,
        num_latents: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        hidden_layer_multiplier: int = 4,
    ) -> None:
        super().__init__()

        # named `camera_latent_queries` (not `query`) so `SelectiveAdamW`
        # (src/rmind/components/optimizers/selective_adamw.py) recognizes it
        # by its last dotted component next to `readout_token`/`register_tokens`
        self.camera_latent_queries = nn.Parameter(torch.empty(num_latents, dim))
        nn.init.trunc_normal_(
            self.camera_latent_queries, mean=0.0, std=0.02, a=-0.04, b=0.04
        )

        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_layer_multiplier * dim),
            nn.GELU(),
            nn.Linear(hidden_layer_multiplier * dim, dim),
        )

    @override
    def forward(self, context: Tensor) -> Tensor:
        """`context`: `(*batch, n, dim)` -> `(*batch, num_latents, dim)`.

        Leading batch dims are flattened for `nn.MultiheadAttention` (which
        accepts only one batch axis) and restored on the way out.
        """
        *batch, n, dim = context.shape
        flat_context = context.reshape(-1, n, dim)
        num_latents = self.camera_latent_queries.shape[0]
        query = self.camera_latent_queries.expand(flat_context.shape[0], -1, -1)

        attn_out, _ = self.attn(
            query=self.attn_norm(query),
            key=flat_context,
            value=flat_context,
            need_weights=False,
        )
        # NOTE: no in-place ops on the residual stream (autograd + checkpointing),
        # mirroring `patch_policy.TransformerBlock.forward`
        h = query + attn_out
        return (h + self.mlp(self.mlp_norm(h))).reshape(*batch, num_latents, dim)
