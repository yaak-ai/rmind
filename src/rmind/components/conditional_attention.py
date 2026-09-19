from typing import final, override

import torch
from pydantic import validate_call
from torch import Tensor, nn


@final
class ConditionalSelfAttention(nn.Module):
    """Conditional Self-Attention (Xie et al., 2020, arXiv:2002.07338).

    A condition ``c`` gates the input tokens ``x`` by their relevance to ``c``
    (a cross-attention), then a multi-head self-attention mixes the *gated* tokens.

    In the world latent we use ``x = observation summary (OS)`` and
    ``c = action summary (AS)``: the action *updates* the observation summary by
    activating the ``OS`` tokens relevant to it. The values are always ``OS`` --
    the action never enters the token content, only reweights which observation
    tokens are emphasised (``h_i = p_i * x_i``).

    Steps (Eq. 1-7 of the paper):
        p_i  = softmax_i f(x_i, c)      # relevance of each token to the condition
        h_i  = p_i * x_i                # gate tokens by relevance
        OS'  = LN( x + SelfAttn(h) )    # self-attention over the gated tokens (+residual)
    """

    @validate_call
    def __init__(
        self,
        *,
        dim: int,
        num_heads: int = 4,
        hidden_dim: int | None = None,
        attn_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        h = hidden_dim if hidden_dim is not None else dim
        # relevance f(x_i, c): additive (MLP) compatibility -> scalar per token
        self.x_proj = nn.Linear(dim, h)
        self.c_proj = nn.Linear(dim, h)
        self.relevance = nn.Linear(h, 1)
        # self-attention over the relevance-gated tokens
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(dim)

    @override
    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        # x: (..., n, d) tokens (OS);  c: (..., m, d) condition (AS)
        *lead, n, d = x.shape
        cond = c.mean(dim=-2, keepdim=True)  # (..., 1, d) pooled condition
        # Eq. 1-2: per-token relevance to the condition
        rel = self.relevance(torch.tanh(self.x_proj(x) + self.c_proj(cond)))  # (..., n, 1)
        p = rel.squeeze(-1).softmax(dim=-1)  # (..., n) relevance distribution over tokens
        # gate: scale each token by its relevance to the condition
        h = p.unsqueeze(-1) * x  # (..., n, d)
        # Eq. 6-7: self-attention over the gated tokens (batched over the lead dims)
        h_flat = h.reshape(-1, n, d).contiguous()
        attn, _ = self.self_attn(h_flat, h_flat, h_flat, need_weights=False)
        attn = attn.reshape(*lead, n, d)
        # residual keeps the observation content grounded; action only reweights it
        return self.norm(x + attn)
