from typing import final, override

from einops import rearrange
from pydantic import validate_call
from torch import Tensor, nn


@final
class ConditionBias(nn.Module):
    """Condition-derived additive attention bias over observation tokens.

    The latent queries read the observation summary OS (K = V = OS); the condition
    summary CS (ego speed, route waypoints, current actions) only biases *where* they
    read, never *what* is read:

        logits_qj^h = q_h(query_q) . k_h(OS_j) / sqrt(d_h)  +  b_j^h
        b_j^h       = < W_h OS_j , U_h CS > / sqrt(d_h)      (relevance of OS_j to the condition)

    The bias is key-only (shared by all latent queries of a head), so the output of the
    read is a mixture of OS values whose weights the condition can tilt. Related forms:
    AlphaFold2 pair-biased attention, Graphormer structural bias, context-aware SAN
    (condition only in the logits).
    """

    @validate_call
    def __init__(
        self, *, dim: int, num_heads: int, head_dim: int | None = None
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else dim // num_heads
        self.os_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.cs_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)

    @override
    def forward(
        self, observation: Tensor, condition: Tensor, num_queries: int
    ) -> Tensor:
        """observation (..., n, d), condition (..., m, d) -> bias (..., heads, num_queries, n)."""
        k = rearrange(
            self.os_proj(observation), "... n (h e) -> ... h n e", h=self.num_heads
        )
        c = rearrange(
            self.cs_proj(condition.mean(dim=-2)),
            "... (h e) -> ... h e",
            h=self.num_heads,
        )
        bias = (k * c.unsqueeze(-2)).sum(-1) * self.head_dim**-0.5  # (..., h, n)
        return bias.unsqueeze(-2).expand(*bias.shape[:-1], num_queries, bias.shape[-1])
