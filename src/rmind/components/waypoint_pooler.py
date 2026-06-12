from typing import override

from torch import Tensor, nn

from rmind.components.position_encoding import LearnedSequencePositionEmbedding


class _Block(nn.Module):
    """Pre-norm transformer block with zero-init residual output projections.

    At initialization the attention and MLP contribute exactly zero, so the
    block is the identity. This is what lets the pooler warm-start to ``mean``
    (see ``WaypointTransformerPooler``).
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_out = nn.Linear(dim * mlp_ratio, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.GELU(), mlp_out)
        # zero-init the residual output projections => identity at step 0
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)
        nn.init.zeros_(mlp_out.weight)
        nn.init.zeros_(mlp_out.bias)

    @override
    def forward(self, x: Tensor) -> Tensor:
        normed = self.norm1(x)
        attended = x + self.attn(normed, normed, normed, need_weights=False)[0]
        return attended + self.mlp(self.norm2(attended))


class WaypointTransformerPooler(nn.Module):
    """Self-attention pooler over the per-waypoint tokens: ``[b, n, d] -> [b, 1, d]``.

    A small transformer mixes the ``n`` ordered waypoint tokens (so it can read
    *path shape* / curvature, unlike ``mean`` which only sees the centroid),
    then mean-pools to a single token so the policy head's ``in_channels`` is
    unchanged (drop-in replacement for ``waypoints.mean(dim=1, keepdim=True)``).

    Warm-start: with zero-init residual output projections (``_Block``) and a
    zero-init index embedding, the pooler output equals ``waypoints.mean(dim=1)``
    *exactly* at step 0 -- a strict generalization of mean that diverges only as
    training moves the weights off zero.

    ``use_index_pe`` adds a learned per-waypoint-index embedding so the
    (otherwise permutation-equivariant) attention can use waypoint order. Set it
    to ``False`` when the encoder it consumes already injected waypoint position
    upstream (e.g. a PT checkpoint trained with waypoint PE) -- there it is
    redundant.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        dim: int = 384,
        depth: int = 2,
        num_heads: int = 6,
        num_waypoints: int = 10,
        mlp_ratio: int = 4,
        use_index_pe: bool = True,
    ) -> None:
        super().__init__()
        self.index_pe = (
            LearnedSequencePositionEmbedding(num_waypoints, dim)
            if use_index_pe
            else None
        )
        if self.index_pe is not None:
            # zero so the pooler is exactly mean(inputs) at step 0
            nn.init.zeros_(self.index_pe.embedding.weight)
        self.blocks = nn.ModuleList(
            _Block(dim, num_heads, mlp_ratio) for _ in range(depth)
        )

    @override
    def forward(self, waypoints: Tensor) -> Tensor:
        x = waypoints if self.index_pe is None else self.index_pe(waypoints)
        for block in self.blocks:
            x = block(x)
        return x.mean(dim=1, keepdim=True)
