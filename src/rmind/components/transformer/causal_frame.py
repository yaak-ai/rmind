"""Causal-over-frames transformer trunk with a reusable, bounded KV cache.

This is the decoder-only reformulation of `rmind.models.patch_policy.
BlockCausalTransformer`. Same pre-LN GPT blocks, same widths, same
`nn.MultiheadAttention` parameter layout -- the only architectural change is the
positional encoding, and that change is what makes a KV cache *valid*.

Why the original cannot cache
-----------------------------
`BlockCausalTransformer` adds a learned 1D embedding over the flattened
`num_frames * tokens_per_frame` sequence. That index is **window-absolute**:
slot 0 always means "oldest frame currently in the sliding window". When the
window slides, every frame's positional input changes, so every cached key is
stale. Nothing is reusable even though most frames are unchanged.

The replacement is factorized, and each factor is chosen to be stable under a
sliding window:

* **intra-frame position: a learned `tokens_per_frame`-slot embedding**, tiled
  identically onto every frame. Token *k* of frame *f* always gets
  `intra_position_embedding[k]`, for every *f*. Frame-relative by construction,
  so it never goes stale. This preserves exactly the intra-frame capacity the
  1542-slot embedding had (patch identity, and speed-token vs patch-token
  identity) -- nothing is dropped.
* **inter-frame position: RoPE at *frame* granularity** applied to q and k in
  every layer. All `tokens_per_frame` tokens of a frame share one rotation
  `R_f`, so:

  - intra-frame attention is **exactly unrotated** -- `(R_f q)ᵀ(R_f k) = qᵀk` --
    hence still fully bidirectional and ordered only by the intra-frame
    embedding above;
  - inter-frame attention depends only on `f_q - f_k`, so a key rotated with its
    own absolute frame index stays valid forever, at any future window position.
    That is precisely the cache-safety property.

RoPE is applied with the frame's **episode-absolute** index. Any monotone
counter works (only differences matter); reset it at episode boundaries. The
per-tick `cos`/`sin` are *inputs* to `step`, not computed in-graph, so an
exported engine contains **no `Sin`/`Cos` nodes** (see
`/nasa/max/skills/trt-export/SKILL.md` §2 -- trigonometric ops are the most
fp16-fragile part of the DINOv3 family) and the host keeps the counter in fp64.

Cache contract
--------------
`step` takes `past_k`/`past_v` **read-only** and returns only the *new* frame's
K/V. The ring buffer lives in the host, not the graph: no in-graph scatter, no
`ScatterElements` in TRT, and the cache tensors are plain engine I/O. Layout is
a single stacked tensor per side:

    (num_layers, batch, num_heads, cache_frames * tokens_per_frame, head_dim)

Slot validity is a host-supplied additive `cache_bias` of shape
`(1, 1, 1, cache_frames * tokens_per_frame)` -- `0` for a filled slot, a large
negative value for an empty one. A cold cache is therefore the same graph, and
the same cost, as a warm one.

Streaming/recompute equivalence
-------------------------------
Streaming one frame per tick against a ring of capacity `N` frames is
**exactly** a single full forward over the whole episode under
`frame_block_causal_mask(..., window=N)`: in both, frame *f*'s layer-*l* input
attends to frames `[f-N+1 .. f]`. That equivalence is the correctness gate (see
`tests/test_causal_frame.py`) and it is also the statement "train the way you
infer" -- train with `window=N`, serve with a ring of `N`.

Note what is *not* equal, and cannot be for any bounded window with more than
one layer: re-running frames `[1..6]` as a fresh isolated 6-frame episode. There
frame 5 never saw frame 0, whereas in the stream its layer-2+ K/V were produced
with frame 0 in context. That is a property of bounded attention, not a cache
defect.
"""

from typing import final, override

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from rmind.components.transformer.utils import run_layer_stack

__all__ = [
    "CausalFrameTransformer",
    "CausalFrameTransformerBlock",
    "CausalSelfAttention",
    "frame_block_causal_mask",
    "frame_rope_cos_sin",
]

# additive bias for a masked-out position; finite so an fp16 engine cannot
# produce NaN from -inf * 0 in a fused softmax
MASK_BIAS: float = -1e4


def frame_block_causal_mask(
    num_frames: int,
    tokens_per_frame: int,
    *,
    window: int | None = None,
    device: torch.device | None = None,
) -> Tensor:
    """Bool mask (True = blocked) over the flattened sequence.

    Bidirectional within a frame, causal across frames -- identical to
    `rmind.models.patch_policy.block_causal_mask` when `window is None`.

    `window=N` additionally blocks frames more than `N - 1` older than the
    query's frame, i.e. every frame attends to itself plus the `N - 1` frames
    before it. That is the exact full-sequence equivalent of streaming against a
    KV ring buffer of capacity `N` frames.
    """
    frames = (
        torch.arange(num_frames * tokens_per_frame, device=device) // tokens_per_frame
    )
    delta = frames[:, None] - frames[None, :]  # query frame - key frame
    blocked = delta < 0  # future frames
    if window is not None:
        blocked = blocked | (delta > window - 1)  # evicted frames
    return blocked


def frame_rope_cos_sin(
    frame_index: Tensor,
    *,
    head_dim: int,
    base: float = 1000.0,
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
    """`cos`/`sin` of shape `(*frame_index.shape, head_dim)` for frame RoPE.

    Computed in float64 so a long-episode absolute frame counter stays exact,
    then cast to `dtype` -- float32 by default, which is the serving contract
    (drivr computes these host-side and feeds them as engine inputs). `head_dim`
    must be even; the half-split (GPT-NeoX/Llama) pairing is used, matching every
    mainstream `torch.export`-friendly RoPE implementation.

    Note the intra-frame cancellation `(Rq)ᵀ(Rk) = qᵀk` is algebraically exact but
    numerically only as exact as `cos² + sin² = 1` in `dtype`; at float32 that is
    ~1e-7 relative, well inside the 1e-6 cache gate.
    """
    if head_dim % 2:
        msg = f"head_dim must be even for RoPE, got {head_dim}"
        raise ValueError(msg)
    inv_freq = base ** (
        -torch.arange(0, head_dim, 2, dtype=torch.float64, device=frame_index.device)
        / head_dim
    )
    angle = frame_index.to(torch.float64)[..., None] * inv_freq
    return (
        torch.cat((angle.cos(), angle.cos()), dim=-1).to(dtype),
        torch.cat((angle.sin(), angle.sin()), dim=-1).to(dtype),
    )


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate `x` `(b, h, s, head_dim)` by `cos`/`sin` broadcast over `(s, head_dim)`."""
    x1, x2 = x.chunk(2, dim=-1)
    return x * cos + torch.cat((-x2, x1), dim=-1) * sin


@final
class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with frame RoPE and an optional read-only KV cache.

    Parameter layout is **identical to `nn.MultiheadAttention(batch_first=True)`**
    (`in_proj_weight`, `in_proj_bias`, `out_proj.{weight,bias}`), so a trunk
    trained with `BlockCausalTransformer` loads into this one 1:1 -- only the
    positional embedding differs, and that is the intended change.
    """

    def __init__(
        self, *, dim_model: int, num_heads: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        if dim_model % num_heads:
            msg = f"dim_model {dim_model} not divisible by num_heads {num_heads}"
            raise ValueError(msg)
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.head_dim = dim_model // num_heads
        self.dropout = dropout

        # nn.MultiheadAttention's own initialization, so a randomly-initialized
        # CausalFrameTransformer is distributionally identical to the reference
        self.in_proj_weight = nn.Parameter(torch.empty(3 * dim_model, dim_model))
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * dim_model))
        nn.init.xavier_uniform_(self.in_proj_weight)
        self.out_proj = nn.Linear(dim_model, dim_model)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def _qkv(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        b, s, _ = x.shape
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias)
        return tuple(  # ty:ignore[invalid-return-type]
            t.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
            for t in qkv.chunk(3, dim=-1)
        )

    def _out(self, attn: Tensor) -> Tensor:
        b, _, s, _ = attn.shape
        return self.out_proj(attn.transpose(1, 2).reshape(b, s, self.dim_model))

    @override
    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, mask: Tensor) -> Tensor:
        """Full-sequence forward. `mask` is the bool block-causal mask."""
        q, k, v = self._qkv(x)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=~mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self._out(attn)

    def step(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        past_k: Tensor,
        past_v: Tensor,
        cache_bias: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """One frame's `tokens_per_frame` queries against the cache plus itself.

        No causal mask is needed: every cached key is in the past (causal by
        construction) and every own-frame key is visible (bidirectional within a
        frame). The only masking is `cache_bias`, which zeroes out unfilled ring
        slots.

        Returns `(out, new_k, new_v)` where the K/V are for the new frame only.
        """
        q, k, v = self._qkv(x)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        keys = torch.cat((past_k, k), dim=-2)
        values = torch.cat((past_v, v), dim=-2)
        bias = F.pad(cache_bias, (0, x.shape[1]))  # own-frame keys always visible
        attn = F.scaled_dot_product_attention(q, keys, values, attn_mask=bias)
        return self._out(attn), k, v

    def step_readout(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        past_k: Tensor,
        past_v: Tensor,
        cache_bias: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """`step`, but only the LAST query position's output is computed.

        The head reads a single token per frame (§3.3 of the hand-off), so in the
        final block the attention output and MLP for the other
        `tokens_per_frame - 1` positions are discarded. K/V are still produced
        for all positions -- future frames attend to them.
        """
        q, k, v = self._qkv(x)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        keys = torch.cat((past_k, k), dim=-2)
        values = torch.cat((past_v, v), dim=-2)
        bias = F.pad(cache_bias, (0, x.shape[1]))
        attn = F.scaled_dot_product_attention(
            q[:, :, -1:], keys, values, attn_mask=bias
        )
        return self._out(attn), k, v


@final
class CausalFrameTransformerBlock(nn.Module):
    """Pre-LN GPT block, structurally identical to `patch_policy.TransformerBlock`."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        dim_model: int,
        num_heads: int,
        attn_dropout: float = 0.1,
        resid_dropout: float = 0.1,
        mlp_dropout: float = 0.1,
        hidden_layer_multiplier: int = 4,
    ) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim_model)
        self.attn = CausalSelfAttention(
            dim_model=dim_model, num_heads=num_heads, dropout=attn_dropout
        )
        self.resid_drop = nn.Dropout(resid_dropout)
        self.mlp_norm = nn.LayerNorm(dim_model)
        self.mlp = nn.Sequential(
            nn.Linear(dim_model, hidden_layer_multiplier * dim_model),
            nn.GELU(),
            nn.Linear(hidden_layer_multiplier * dim_model, dim_model),
            nn.Dropout(mlp_dropout),
        )

    @override
    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, mask: Tensor) -> Tensor:
        attn_out = self.attn(self.attn_norm(x), cos, sin, mask)
        # NOTE: no in-place ops on the residual stream (autograd + checkpointing)
        h = x + self.resid_drop(attn_out)
        return h + self.mlp(self.mlp_norm(h))

    def step(  # noqa: PLR0913
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        past_k: Tensor,
        past_v: Tensor,
        cache_bias: Tensor,
        *,
        readout_only: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        fn = self.attn.step_readout if readout_only else self.attn.step
        attn_out, k, v = fn(self.attn_norm(x), cos, sin, past_k, past_v, cache_bias)
        if readout_only:
            x = x[:, -1:]
        h = x + attn_out
        return h + self.mlp(self.mlp_norm(h)), k, v


@final
class CausalFrameTransformer(nn.Module):
    """Decoder over frame blocks: bidirectional intra-frame, causal inter-frame.

    Drop-in for `BlockCausalTransformer` on the training path -- same
    `forward(src, *, num_frames)` signature -- plus a `step` that consumes and
    extends a KV cache. See the module docstring for the positional scheme and
    the cache contract.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        dim_model: int,
        num_layers: int,
        num_heads: int,
        tokens_per_frame: int,
        window: int | None = None,
        rope_base: float = 1000.0,
        attn_dropout: float = 0.1,
        resid_dropout: float = 0.1,
        mlp_dropout: float = 0.1,
        hidden_layer_multiplier: int = 4,
    ) -> None:
        super().__init__()
        self.dim_model = dim_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = dim_model // num_heads
        self.tokens_per_frame = tokens_per_frame
        self.window = window
        self.rope_base = rope_base

        # frame-RELATIVE intra-frame position: tiled onto every frame, so it is
        # invariant to where the frame sits in the window (unlike the 1542-slot
        # window-absolute embedding it replaces)
        self.intra_position_embedding = nn.Embedding(tokens_per_frame, dim_model)
        nn.init.trunc_normal_(
            self.intra_position_embedding.weight, mean=0.0, std=0.02, a=-0.04, b=0.04
        )
        self.layers = nn.ModuleList([
            CausalFrameTransformerBlock(
                dim_model=dim_model,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                resid_dropout=resid_dropout,
                mlp_dropout=mlp_dropout,
                hidden_layer_multiplier=hidden_layer_multiplier,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim_model)

    def _intra(self, num_frames: int, device: torch.device) -> Tensor:
        idx = torch.arange(self.tokens_per_frame, device=device)
        return self.intra_position_embedding(idx).repeat(num_frames, 1)

    @override
    def forward(
        self, src: Tensor, *, num_frames: int, frame_offset: int = 0
    ) -> Tensor:
        """Full-sequence forward over `num_frames * tokens_per_frame` tokens.

        `frame_offset` shifts the episode-absolute frame indices used by RoPE.
        The output must be invariant to it -- that is the property the old
        window-absolute embedding lacked, and it is asserted in the tests.
        """
        b, seq_len, _ = src.shape
        k = self.tokens_per_frame
        if seq_len != num_frames * k:
            msg = f"expected {num_frames * k} tokens, got {seq_len}"
            raise ValueError(msg)

        x = src + self._intra(num_frames, src.device)
        frames = torch.arange(seq_len, device=src.device) // k + frame_offset
        cos, sin = frame_rope_cos_sin(
            frames, head_dim=self.head_dim, base=self.rope_base
        )
        cos, sin = cos.to(src.dtype), sin.to(src.dtype)
        mask = frame_block_causal_mask(
            num_frames, k, window=self.window, device=src.device
        )

        x = run_layer_stack(self.layers, x, cos, sin, mask, training=self.training)
        return self.norm(x)

    def empty_cache(
        self,
        *,
        batch_size: int = 1,
        cache_frames: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """`(past_k, past_v, cache_bias)` for a cold cache.

        `cache_frames` is the number of PAST frames held; it defaults to
        `window - 1`, since a context of `window` frames is the current frame
        plus `window - 1` cached ones. For `window=6` that is 5 * 257 = 1285
        cached keys against 257 queries -- 1542 keys in total, exactly the
        baseline's flattened sequence length.

        K/V are `(num_layers, batch, num_heads, cache_frames * tokens_per_frame,
        head_dim)`; `cache_bias` is `(1, 1, 1, cache_frames * tokens_per_frame)`
        filled with `MASK_BIAS` (nothing valid yet).
        """
        n = cache_frames
        if n is None and self.window is not None:
            n = self.window - 1
        if n is None:
            msg = "cache_frames required when window is None"
            raise ValueError(msg)
        shape = (
            self.num_layers,
            batch_size,
            self.num_heads,
            n * self.tokens_per_frame,
            self.head_dim,
        )
        zeros = torch.zeros(shape, device=device, dtype=dtype)
        bias = torch.full(
            (1, 1, 1, n * self.tokens_per_frame),
            MASK_BIAS,
            device=device,
            dtype=dtype,
        )
        return zeros, zeros.clone(), bias

    def step(  # noqa: PLR0913
        self,
        src: Tensor,
        *,
        past_k: Tensor,
        past_v: Tensor,
        cos: Tensor,
        sin: Tensor,
        cache_bias: Tensor,
        readout_only_final_block: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Decode ONE frame against a read-only cache.

        `src` is `(b, tokens_per_frame, d)`. `cos`/`sin` are `(1, head_dim)` or
        `(head_dim,)` -- the current frame's rotation, computed host-side from
        the episode frame counter (`frame_rope_cos_sin`) so the graph has no
        trigonometric nodes.

        Returns `(out, new_k, new_v)`. `out` is the normed trunk output for the
        frame -- all `tokens_per_frame` positions, or just the readout position
        when `readout_only_final_block`. `new_k`/`new_v` are
        `(num_layers, b, num_heads, tokens_per_frame, head_dim)`: the host writes
        them into its ring buffer.
        """
        x = src + self.intra_position_embedding(
            torch.arange(src.shape[1], device=src.device)
        )
        cos = cos.reshape(1, 1, 1, self.head_dim).to(x.dtype)
        sin = sin.reshape(1, 1, 1, self.head_dim).to(x.dtype)

        new_k: list[Tensor] = []
        new_v: list[Tensor] = []
        last = self.num_layers - 1
        for i, layer in enumerate(self.layers):
            x, k, v = layer.step(
                x,
                cos,
                sin,
                past_k[i],
                past_v[i],
                cache_bias,
                readout_only=readout_only_final_block and i == last,
            )
            new_k.append(k)
            new_v.append(v)
        return self.norm(x), torch.stack(new_k), torch.stack(new_v)
