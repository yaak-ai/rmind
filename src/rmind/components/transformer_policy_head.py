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

    Optional architecture knobs (all backward-compatible -- defaults reproduce the
    behavior above exactly):

    - ``dim_inner``: when set, an input ``Linear(dim_model, dim_inner)`` projects
      the tokens to ``dim_inner``, the transformer blocks run at ``dim_inner``, and
      ``output_projection`` reads out from ``dim_inner``. Enables "deeper but less
      wide" heads (more ``num_layers`` at a smaller width). When ``None`` the head
      operates at ``dim_model`` with no input projection (unchanged).
    - ``skip``: input-mean skip connection so the head keeps a direct path to the
      raw token summary. The mean over the INPUT tokens (``[b, 1, dim_model]``,
      projected to the readout width when ``dim_inner`` is set) is added to the
      pooled post-transformer token before ``output_projection``. Output is still
      exactly ``0`` at init (``output_projection`` is zero-init).
    - ``dropout``: passed through to each ``_Block`` (attention + MLP). Default
      ``0.0`` is a no-op.
    """

    @validate_call
    def __init__(  # noqa: PLR0913
        self,
        *,
        dim_model: int = 384,
        out_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dim_inner: int | None = None,
        skip: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        dim_work = dim_inner if dim_inner is not None else dim_model
        # Input projection only when widening/narrowing the working width.
        self.input_projection = (
            nn.Linear(dim_model, dim_inner) if dim_inner is not None else None
        )

        self.blocks = nn.ModuleList(
            _Block(dim_work, num_heads, int(mlp_ratio), dropout)
            for _ in range(num_layers)
        )

        # Input-mean skip: project the raw-token summary to the readout width when
        # the working width differs; identity otherwise.
        self.skip_projection: nn.Linear | None = None
        if skip:
            self.skip_projection = (
                nn.Linear(dim_model, dim_inner) if dim_inner is not None else None
            )
        self._skip = skip

        self.output_projection = nn.Linear(dim_work, out_dim)
        # zero-init => step-0 output is exactly 0 (mild, stable warm-start)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    @override
    def forward(self, context: Tensor) -> Tensor:
        x = context if self.input_projection is None else self.input_projection(context)
        for block in self.blocks:
            x = block(x)
        pooled = x.mean(dim=1, keepdim=True)

        if self._skip:
            summary = context.mean(dim=1, keepdim=True)
            if self.skip_projection is not None:
                summary = self.skip_projection(summary)
            pooled = pooled + summary

        return self.output_projection(pooled)
