from typing import override

from pydantic import validate_call
from torch import Tensor, nn

from rmind.components.waypoint_pooler import _Block


class TransformerPolicyHead(nn.Module):
    """Self-attention transformer policy head for the cross-attention path.

    Consumes the full token sequence ``[b, S, dim_model]`` produced by
    ``PolicyObjective`` in its ``_cross_attn`` mode -- i.e. ``cat([waypoints(10),
    obs_summary, obs_history]) = [b, 12, 384]`` -- and runs a ``num_layers``-deep,
    ``num_heads``-head pre-norm self-attention transformer encoder over the ``S``
    tokens, letting every token attend to every other (waypoints <-> summaries).
    The mixed tokens are then mean-pooled to a single token ``[b, 1, dim_model]``
    and projected to ``out_dim`` (2 = mean,logvar for continuous; 3 = classes for
    turn_signal).

    This is a more expressive alternative to ``CrossAttentionPolicyHead``: instead
    of a single learned query cross-attending over a frozen context, all tokens
    participate in self-attention before readout.

    Warm-start: the ``output_projection`` is zero-init, so step-0 outputs are
    exactly ``0`` regardless of the (identity-at-init) encoder blocks -- a mild,
    stable start.
    """

    @validate_call
    def __init__(
        self,
        *,
        dim_model: int = 384,
        out_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()

        self.blocks = nn.ModuleList(
            _Block(dim_model, num_heads, int(mlp_ratio)) for _ in range(num_layers)
        )
        self.output_projection = nn.Linear(dim_model, out_dim)
        # zero-init => step-0 output is exactly 0 (mild, stable warm-start)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    @override
    def forward(self, context: Tensor) -> Tensor:
        x = context
        for block in self.blocks:
            x = block(x)
        pooled = x.mean(dim=1, keepdim=True)
        return self.output_projection(pooled)
