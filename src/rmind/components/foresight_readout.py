from typing import override

import torch
from pydantic import validate_call
from torch import Tensor, nn


class ForesightAttentionPool(nn.Module):
    """Learned-query multi-head attention pooling over the foresight slots.

    The joint policy head reads observation information only through the single
    learned ``observation_summary`` token (plus ``observation_history``), a
    softmax attention readout that leverage probes showed already extracts most
    of the foresight content but necessarily averages over the 256 foresight
    slots. This gives the head a *wider* learned readout of the same slots:
    ``num_queries`` independent attention queries, concatenated, so the policy
    can keep more than one projection of the foresight representation.

    Spinoff 1.1 ("de-bottleneck the readout"). Additive: its output is
    concatenated onto ``[observation_summary, observation_history]`` in
    ``JointPolicyObjective._features`` — it never replaces them, so a null
    result cleanly means "no incremental readout value".

    Input:  ``(b, n, dim)`` foresight slots (n = 256 for cam_front_left).
    Output: ``(b, num_queries * dim)``.
    """

    @validate_call
    def __init__(
        self,
        *,
        dim: int,
        num_queries: int = 4,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(num_queries, dim) * dim**-0.5)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(dim)
        self.out_features: int = num_queries * dim

    @override
    def forward(self, x: Tensor) -> Tensor:
        # x: (b, n, dim) -> (b, num_queries, dim) -> (b, num_queries * dim)
        query = self.query.expand(x.shape[0], -1, -1)
        pooled, _ = self.attn(query, x, x, need_weights=False)
        return self.norm(pooled).flatten(1)
