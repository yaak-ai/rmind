"""Cache-vs-recompute correctness for the decoder-only PatchPolicy trunk.

This is the gate from `/nasa/max/docs/decoder_only_handoff.md` §5.1: warm-cache
streaming must agree with a full recompute to ~1e-6, and if it does not, the
positional encoding is window-absolute (§3.2).

Three things are checked, and the distinction between them matters:

* `test_shift_invariance*` -- the positional scheme does not depend on where a
  frame sits in the window. This is the actual discriminator for the §3.2 trap.
* `test_stream_equals_*_recompute` -- streaming against a ring buffer reproduces
  a single full forward, exactly. With `window=N` the equivalent recompute uses
  `frame_block_causal_mask(..., window=N)`; frame `t`'s window there is
  `[t-N+1 .. t]`, i.e. for N=6 at t=6 exactly the frames `[1..6]` §5.1 names.
* `test_isolated_window_recompute_is_not_identical` -- re-running `[1..6]` as a
  *fresh 6-frame episode* is NOT the same computation and cannot be, for any
  bounded window with more than one layer: there frame 5 never saw frame 0,
  whereas in the stream its layer-2+ K/V were produced with frame 0 in context.
  Recorded as a quantified, expected residual, not a pass.

Every equivalence test is paired with a negative control on a window-absolute
variant of the same trunk, run through the same harness, so a pass is
falsifiable rather than vacuous.
"""

from collections.abc import Callable
from typing import override

import pytest
import torch
from torch import Tensor, nn
from torch.fx.experimental.proxy_tensor import make_fx

from rmind.components.transformer.causal_frame import (
    FLEX_BLOCK_SIZE,
    MASK_BIAS,
    AttentionImpl,
    CausalFrameTransformer,
    apply_rope,
    frame_block_causal_block_mask,
    frame_block_causal_mask,
    frame_block_causal_mask_mod,
    frame_rope_cos_sin,
)
from rmind.models.patch_policy import block_causal_mask

TOKENS_PER_FRAME = 17  # stands in for 257; the property is independent of it
DIM = 32
HEADS = 4
LAYERS = 3
TOL = 1e-6
# a negative control must miss the 1e-6 gate by orders of magnitude, not narrowly
CONTROL_MIN_ERROR = 1e-2

# production geometry, for the flex-vs-sdpa parity gate
PROD_TOKENS_PER_FRAME = 257
# fp32 parity gate, applied SCALE-RELATIVE (max|diff| / max|reference|).
#
# An absolute 1e-5 is the right gate for activations and input gradients (both are
# O(1) here and land at ~1e-6), but it is the wrong gate for a PARAMETER gradient:
# `in_proj_weight.grad` is a sum over all 1542-4112 sequence positions, so its
# entries are O(10) and its fp32 accumulation noise alone is ~1e-5 in absolute
# terms no matter which kernel produced it. Measured on a 5090 with tf32 off, every
# tensor is <= 1.5e-6 scale-relative, and
# `test_flex_is_no_further_from_exact_than_sdpa` shows that residual IS the fp32
# noise: flex sits 1.4x sdpa's own distance from a float64 reference, not 10x.
FLEX_TOL = 1e-5
# 128-block tiling of a 257-token frame: the boundary waste must stay far below the
# 2.23x that padding frames to 384 (3 exact blocks) would cost
MAX_TILE_WASTE = 1.3
# and the block-sparse kernel must compute clearly less than the dense mask does
MAX_DENSE_FRACTION = 0.8


def _trunk(
    *, window: int | None = None, cls: type[CausalFrameTransformer] | None = None
) -> CausalFrameTransformer:
    torch.manual_seed(0)
    trunk = (cls or CausalFrameTransformer)(
        dim_model=DIM,
        num_layers=LAYERS,
        num_heads=HEADS,
        tokens_per_frame=TOKENS_PER_FRAME,
        window=window,
    )
    return trunk.double().eval()


def _tokens(num_frames: int, *, batch: int = 2, seed: int = 1) -> Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(
        batch, num_frames, TOKENS_PER_FRAME, DIM, generator=g, dtype=torch.float64
    )


def _flat(tokens: Tensor) -> Tensor:
    b, t, k, d = tokens.shape
    return tokens.reshape(b, t * k, d)


def _readouts(flat_out: Tensor, num_frames: int) -> Tensor:
    """Last token of every frame -- the readout position (speed token is prepended)."""
    b, s, d = flat_out.shape
    return flat_out.reshape(b, num_frames, s // num_frames, d)[:, :, -1]


class WindowAbsoluteTrunk(CausalFrameTransformer):
    """Negative control: the CURRENT scheme -- a learned embedding over the
    flattened window, indexed window-absolutely, and no RoPE.

    Structurally identical to `CausalFrameTransformer` otherwise, and driven
    through the same streaming harness, so any difference in the results is
    attributable to the positional encoding alone.
    """

    MAX_FRAMES = 16

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # ty:ignore[missing-argument]
        self.absolute_position_embedding = nn.Embedding(
            self.MAX_FRAMES * self.tokens_per_frame, self.dim_model
        )
        nn.init.trunc_normal_(self.absolute_position_embedding.weight, std=0.02)

    def _unit_rope(
        self, device: torch.device, dtype: torch.dtype
    ) -> tuple[Tensor, ...]:
        return (
            torch.ones(1, 1, 1, self.head_dim, device=device, dtype=dtype),
            torch.zeros(1, 1, 1, self.head_dim, device=device, dtype=dtype),
        )

    @override
    def forward(self, src: Tensor, *, num_frames: int, frame_offset: int = 0) -> Tensor:
        _, seq_len, _ = src.shape
        # window-absolute: shifting the frames later in the episode shifts the
        # positional lookup, which is exactly what sliding the window does
        x = src + self.absolute_position_embedding(
            torch.arange(seq_len, device=src.device)
            + frame_offset * self.tokens_per_frame
        )
        cos, sin = self._unit_rope(src.device, src.dtype)
        mask = frame_block_causal_mask(
            num_frames, self.tokens_per_frame, window=self.window, device=src.device
        )
        for layer in self.layers:
            x = layer(x, cos, sin, mask)
        return self.norm(x)

    @override
    def step(
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
        del cos, sin
        # window-absolute: the newest frame always occupies the LAST block of the
        # window, so its positional input changes every time the window slides
        num_cached = cache_bias.shape[-1] // self.tokens_per_frame
        offset = num_cached * self.tokens_per_frame
        idx = torch.arange(src.shape[1], device=src.device) + offset
        x = src + self.absolute_position_embedding(idx)
        cos, sin = self._unit_rope(src.device, src.dtype)

        new_k: list[Tensor] = []
        new_v: list[Tensor] = []
        for i, layer in enumerate(self.layers):
            x, k, v = layer.step(x, cos, sin, past_k[i], past_v[i], cache_bias)
            new_k.append(k)
            new_v.append(v)
        return self.norm(x), torch.stack(new_k), torch.stack(new_v)


def stream(
    trunk: CausalFrameTransformer,
    tokens: Tensor,
    *,
    cache_frames: int,
    ring: bool = False,
) -> Tensor:
    """One-frame-per-tick decode against a host-side ring buffer.

    Emulates exactly what drivr would do: the graph reads `past_k`/`past_v` and
    returns only the new frame's K/V, and the host places them in the ring.
    Returns the readout token per frame, `(b, t, d)`.

    Two placement policies, and they must agree (see
    `test_ring_slot_write_matches_shift_left`):

    * `ring=False` -- shift the whole cache left by one frame block. Simple, but it
      moves the entire cache every tick.
    * `ring=True` -- write into slot `t % cache_frames`, moving nothing. Valid
      because attention is permutation-invariant over keys and each key carries its
      own rotation (RoPE with its absolute frame index) and its own `cache_bias`
      slot. This is what the runtime should do: 257 tokens per layer instead of the
      whole cache, and no overlapping in-place copy.
    """
    b, num_frames, k, _ = tokens.shape
    past_k, past_v, bias = trunk.empty_cache(
        batch_size=b, cache_frames=cache_frames, dtype=tokens.dtype
    )
    outs: list[Tensor] = []
    for t in range(num_frames):
        cos, sin = frame_rope_cos_sin(
            torch.tensor(t), head_dim=trunk.head_dim, base=trunk.rope_base
        )
        out, new_k, new_v = trunk.step(
            tokens[:, t],
            past_k=past_k,
            past_v=past_v,
            cos=cos.to(tokens.dtype),
            sin=sin.to(tokens.dtype),
            cache_bias=bias,
        )
        outs.append(out[:, -1])
        if not cache_frames:
            continue
        if ring:
            slot = slice((t % cache_frames) * k, (t % cache_frames + 1) * k)
            past_k, past_v, bias = past_k.clone(), past_v.clone(), bias.clone()
            past_k[..., slot, :] = new_k
            past_v[..., slot, :] = new_v
            bias[..., slot] = 0.0
        else:
            past_k = torch.cat((past_k[..., k:, :], new_k), dim=-2)
            past_v = torch.cat((past_v[..., k:, :], new_v), dim=-2)
            bias = torch.cat((bias[..., k:], torch.zeros_like(bias[..., :k])), dim=-1)
    return torch.stack(outs, dim=1)


# --------------------------------------------------------------------------- #
# masks
# --------------------------------------------------------------------------- #


def test_unwindowed_mask_matches_reference() -> None:
    """`window=None` is the existing block-causal mask, unchanged."""
    torch.testing.assert_close(
        frame_block_causal_mask(6, TOKENS_PER_FRAME),
        block_causal_mask(6, TOKENS_PER_FRAME),
    )


def test_windowed_mask_geometry() -> None:
    mask = frame_block_causal_mask(8, 1, window=3)
    # frame 5 sees exactly frames 3, 4, 5
    assert (~mask[5]).nonzero().flatten().tolist() == [3, 4, 5]
    # intra-frame is bidirectional: the diagonal block is never blocked
    block = frame_block_causal_mask(4, TOKENS_PER_FRAME, window=2)
    k = TOKENS_PER_FRAME
    assert not block[2 * k : 3 * k, 2 * k : 3 * k].any()


# --------------------------------------------------------------------------- #
# positional scheme
# --------------------------------------------------------------------------- #


def test_intra_frame_attention_is_exactly_unrotated() -> None:
    """RoPE at frame granularity cancels within a frame: `(Rq)ᵀ(Rk) = qᵀk`.

    This is why intra-frame attention stays bidirectional and position-agnostic
    (ordered only by the intra-frame embedding). A pairing/axis bug in
    `apply_rope` breaks this.
    """
    g = torch.Generator().manual_seed(3)
    q = torch.randn(2, HEADS, TOKENS_PER_FRAME, DIM // HEADS, generator=g).double()
    k = torch.randn(2, HEADS, TOKENS_PER_FRAME, DIM // HEADS, generator=g).double()
    cos, sin = frame_rope_cos_sin(
        torch.tensor(37), head_dim=DIM // HEADS, dtype=torch.float64
    )
    cos, sin = cos.reshape(1, 1, 1, -1), sin.reshape(1, 1, 1, -1)
    torch.testing.assert_close(
        apply_rope(q, cos, sin) @ apply_rope(k, cos, sin).transpose(-1, -2),
        q @ k.transpose(-1, -2),
        rtol=0,
        atol=1e-12,
    )


def test_shift_invariance() -> None:
    """The §3.2 discriminator: the SAME frames at a different episode offset must
    produce the SAME output. A window-absolute embedding cannot do this.
    """
    trunk = _trunk()
    x = _flat(_tokens(6))
    torch.testing.assert_close(
        trunk(x, num_frames=6, frame_offset=0),
        trunk(x, num_frames=6, frame_offset=7),
        rtol=0,
        atol=TOL,
    )


def test_shift_invariance_negative_control() -> None:
    """The same test on the window-absolute control -- must FAIL, by a lot.

    Symmetric to `test_shift_invariance`: identical frames, identical blocks,
    only the positional encoding differs. This is the §3.2 trap, reproduced.
    """
    trunk = _trunk(window=6, cls=WindowAbsoluteTrunk)
    x = _flat(_tokens(6))
    err = (
        (
            trunk(x, num_frames=6, frame_offset=0)
            - trunk(x, num_frames=6, frame_offset=7)
        )
        .abs()
        .max()
        .item()
    )
    assert err > CONTROL_MIN_ERROR, (
        f"control should not be shift-invariant, got {err:.3e}"
    )


# --------------------------------------------------------------------------- #
# THE GATE: cache vs recompute
# --------------------------------------------------------------------------- #


def test_stream_equals_full_recompute_unbounded() -> None:
    """Full-history cache: streaming T frames == one causal forward over [0..T-1].

    Checked at EVERY frame's readout, not just the last.
    """
    num_frames = 7
    trunk = _trunk()
    tokens = _tokens(num_frames)
    recompute = _readouts(trunk(_flat(tokens), num_frames=num_frames), num_frames)
    streamed = stream(trunk, tokens, cache_frames=num_frames - 1)
    torch.testing.assert_close(streamed, recompute, rtol=0, atol=TOL)


@pytest.mark.parametrize("window", [2, 3, 6])
def test_stream_equals_sliding_window_recompute(window: int) -> None:
    """§5.1, in the form that is exact.

    Ring of capacity `window - 1` past frames + the current one == a full forward
    under `frame_block_causal_mask(window=window)`. At `window=6`, frame 6's
    context is exactly frames [1..6] -- the case §5.1 names -- and it matches to
    ~1e-6 while sharing no recomputation with the earlier frames.
    """
    num_frames = 7
    trunk = _trunk(window=window)
    tokens = _tokens(num_frames)
    recompute = _readouts(trunk(_flat(tokens), num_frames=num_frames), num_frames)
    streamed = stream(trunk, tokens, cache_frames=window - 1)
    torch.testing.assert_close(streamed, recompute, rtol=0, atol=TOL)


def test_stream_vs_recompute_negative_control() -> None:
    """The same gate on the window-absolute trunk -- must FAIL.

    Demonstrates the gate has teeth: identical harness, identical block
    structure, only the positional encoding differs.
    """
    num_frames = 7
    window = 6
    trunk = _trunk(window=window, cls=WindowAbsoluteTrunk)
    tokens = _tokens(num_frames)
    recompute = _readouts(trunk(_flat(tokens), num_frames=num_frames), num_frames)
    streamed = stream(trunk, tokens, cache_frames=window - 1)
    err = (streamed - recompute).abs().max().item()
    assert err > CONTROL_MIN_ERROR, f"control should diverge, got {err:.3e}"


def test_isolated_window_recompute_is_not_identical() -> None:
    """Documented, expected residual -- NOT a gate.

    Re-running frames [1..6] as a fresh 6-frame episode differs from the stream,
    because there frame 5 never saw frame 0 while in the stream its layer-2+ K/V
    were produced with frame 0 in context. Inherent to any bounded window with
    L > 1; unrelated to the positional encoding.
    """
    trunk = _trunk(window=6)
    tokens = _tokens(7)
    streamed = stream(trunk, tokens, cache_frames=5)[:, -1]
    isolated = _readouts(trunk(_flat(tokens[:, 1:]), num_frames=6, frame_offset=1), 6)[
        :, -1
    ]
    residual = (streamed - isolated).abs().max().item()
    assert residual > TOL, "single-layer coincidence?"
    # sanity: the deviation is a truncation effect, not divergence
    assert residual < 10 * streamed.abs().max().item()


def test_single_layer_isolated_recompute_is_identical() -> None:
    """Corroborates the explanation above: with L=1 there is no layer-2 leakage,
    so the isolated [1..6] recompute IS bit-identical to the stream.
    """
    torch.manual_seed(0)
    trunk = (
        CausalFrameTransformer(
            dim_model=DIM,
            num_layers=1,
            num_heads=HEADS,
            tokens_per_frame=TOKENS_PER_FRAME,
            window=6,
        )
        .double()
        .eval()
    )
    tokens = _tokens(7)
    streamed = stream(trunk, tokens, cache_frames=5)[:, -1]
    isolated = _readouts(trunk(_flat(tokens[:, 1:]), num_frames=6, frame_offset=1), 6)[
        :, -1
    ]
    torch.testing.assert_close(streamed, isolated, rtol=0, atol=TOL)


# --------------------------------------------------------------------------- #
# cache mechanics
# --------------------------------------------------------------------------- #


def test_cold_cache_ignores_unfilled_slots() -> None:
    """A cold cache with garbage in the unfilled slots gives the same answer as
    an empty one -- `cache_bias` is what makes cold-start correct, and it is why
    cold and warm ticks cost the same.
    """
    trunk = _trunk(window=6)
    tokens = _tokens(1)
    past_k, past_v, bias = trunk.empty_cache(
        batch_size=2, cache_frames=5, dtype=tokens.dtype
    )
    cos, sin = frame_rope_cos_sin(torch.tensor(0), head_dim=trunk.head_dim)
    kwargs = {"cos": cos.double(), "sin": sin.double(), "cache_bias": bias}
    clean, *_ = trunk.step(tokens[:, 0], past_k=past_k, past_v=past_v, **kwargs)
    g = torch.Generator().manual_seed(7)
    garbage_k = torch.randn(past_k.shape, generator=g, dtype=torch.float64) * 100
    garbage_v = torch.randn(past_v.shape, generator=g, dtype=torch.float64) * 100
    dirty, *_ = trunk.step(tokens[:, 0], past_k=garbage_k, past_v=garbage_v, **kwargs)
    torch.testing.assert_close(dirty, clean, rtol=0, atol=TOL)
    assert bias.min().item() == MASK_BIAS


def test_readout_only_final_block_matches_full_final_block() -> None:
    """§3.3's free win: skipping the final block's work on the non-readout
    positions changes nothing at the readout, and still emits full K/V.
    """
    trunk = _trunk(window=6)
    tokens = _tokens(1)
    past_k, past_v, bias = trunk.empty_cache(
        batch_size=2, cache_frames=5, dtype=tokens.dtype
    )
    g = torch.Generator().manual_seed(11)
    past_k = torch.randn(past_k.shape, generator=g, dtype=torch.float64)
    past_v = torch.randn(past_v.shape, generator=g, dtype=torch.float64)
    bias = torch.zeros_like(bias)
    cos, sin = frame_rope_cos_sin(torch.tensor(5), head_dim=trunk.head_dim)
    kwargs = {
        "past_k": past_k,
        "past_v": past_v,
        "cos": cos.double(),
        "sin": sin.double(),
        "cache_bias": bias,
    }
    full, full_k, full_v = trunk.step(tokens[:, 0], **kwargs)
    gathered, gath_k, gath_v = trunk.step(
        tokens[:, 0], readout_only_final_block=True, **kwargs
    )
    assert gathered.shape[1] == 1
    torch.testing.assert_close(gathered[:, 0], full[:, -1], rtol=0, atol=TOL)
    torch.testing.assert_close(gath_k, full_k, rtol=0, atol=TOL)
    torch.testing.assert_close(gath_v, full_v, rtol=0, atol=TOL)


def test_multihead_attention_state_dict_is_loadable() -> None:
    """A trunk trained with `BlockCausalTransformer` transfers 1:1, so this is a
    positional-encoding change and not a re-parameterization.
    """
    ref = nn.MultiheadAttention(embed_dim=DIM, num_heads=HEADS, batch_first=True)
    trunk = _trunk(window=6)
    missing, unexpected = trunk.layers[0].attn.load_state_dict(
        ref.state_dict(), strict=False
    )
    assert not missing
    assert not unexpected


# --------------------------------------------------------------------------- #
# FlexAttention path: exact parity with the dense-mask SDPA path
#
# The saving is real only if the kernel is *identical*, not merely similar: the
# block-sparse path is the training path for the SAME weights that serve through
# `step`, so any drift here is a train/infer mismatch on top of a speedup.
# --------------------------------------------------------------------------- #


def _prod_trunk(  # noqa: PLR0913
    *,
    window: int,
    dim: int,
    heads: int,
    impl: AttentionImpl,
    layers: int = 2,
    tokens_per_frame: int = PROD_TOKENS_PER_FRAME,
) -> CausalFrameTransformer:
    """Production-width trunk in fp32 (FlexAttention has no float64 CUDA kernel).

    `attn_dropout=0` is required by the flex path and is set on both arms so the
    two are comparable; `layers=2` is enough to exercise the stack while keeping
    the 4112-token cases cheap.
    """
    torch.manual_seed(0)
    return CausalFrameTransformer(
        dim_model=dim,
        num_layers=layers,
        num_heads=heads,
        tokens_per_frame=tokens_per_frame,
        window=window,
        attn_dropout=0.0,
        resid_dropout=0.0,
        mlp_dropout=0.0,
        attention_impl=impl,
    )


def _pair(
    *, window: int, dim: int, heads: int, device: str, layers: int = 2
) -> tuple[CausalFrameTransformer, CausalFrameTransformer]:
    """Same weights, two attention implementations."""
    sdpa = _prod_trunk(window=window, dim=dim, heads=heads, impl="sdpa", layers=layers)
    flex = _prod_trunk(window=window, dim=dim, heads=heads, impl="flex", layers=layers)
    flex.load_state_dict(sdpa.state_dict())
    return sdpa.to(device), flex.to(device)


def _scale_rel(got: Tensor, ref: Tensor) -> float:
    """`max|got - ref| / max|ref|` -- see the FLEX_TOL comment for why not atol."""
    return ((got.double() - ref.double()).abs().max() / ref.double().abs().max()).item()


def _fwd_bwd(
    trunk: CausalFrameTransformer, x: Tensor, grad_out: Tensor, *, num_frames: int
) -> dict[str, Tensor]:
    """Forward + backward, returning the output and the gradients under comparison.

    An input-gradient-only comparison would miss a parameter whose gradient path
    runs through the mask, so an attention projection at each end of the stack and
    the intra-frame position embedding are included too.
    """
    trunk.zero_grad()
    inp = x.detach().clone().requires_grad_()
    out = trunk(inp, num_frames=num_frames)
    out.backward(grad_out)
    assert inp.grad is not None
    params = dict(trunk.named_parameters())
    last = trunk.num_layers - 1
    wanted = {
        "d_in_proj_weight": "layers.0.attn.in_proj_weight",
        "d_out_proj_weight": f"layers.{last}.attn.out_proj.weight",
        "d_intra_position": "intra_position_embedding.weight",
    }
    grads: dict[str, Tensor] = {"out": out.detach(), "d_input": inp.grad}
    for key, name in wanted.items():
        grad = params[name].grad
        assert grad is not None, name
        grads[key] = grad.clone()
    return grads


def _mutating_ops(mask_mod: Callable[..., Tensor]) -> list[str]:
    """Names of the in-place aten ops a `mask_mod` traces down to."""
    idx = torch.arange(4)
    graph = make_fx(mask_mod)(torch.tensor(0), torch.tensor(0), idx, idx)
    return [
        str(node.target)
        for node in graph.graph.nodes
        if getattr(getattr(node.target, "_schema", None), "is_mutable", False)
    ]


def test_mask_mod_is_free_of_in_place_ops() -> None:
    """Regression guard, and the reason it exists is not hypothetical.

    Inductor lowers a `mask_mod` as a pointwise subgraph, in which no buffer may be
    created -- so a single in-place op makes the flex path fail to compile at EVERY
    shape ("SubgraphLoweringException: Buffers cannot be created while lowering a
    pointwise subgraph"). `ruff check` in this repo runs with `fix = true` and
    `unsafe-fixes = true`, and PLR6104 happily rewrites `keep = keep & x` into
    `keep &= x`, which is exactly that failure. It bit once already.

    This check runs on CPU, so it guards the flex path on machines that cannot run
    the CUDA parity tests at all.
    """
    for window in (None, 6, 16):
        assert not _mutating_ops(
            frame_block_causal_mask_mod(PROD_TOKENS_PER_FRAME, window)
        )


def test_in_place_mask_mod_is_detected() -> None:
    """Negative control for the guard above: the form ruff produces must be caught."""

    def in_place(b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
        del b, h
        delta = q_idx // PROD_TOKENS_PER_FRAME - kv_idx // PROD_TOKENS_PER_FRAME
        keep = delta >= 0
        keep &= delta <= LAYERS  # any bound; the in-place `&=` is the point
        return keep

    assert _mutating_ops(in_place) == ["aten.bitwise_and_.Tensor"]


def test_flex_mask_mod_matches_dense_mask() -> None:
    """The `mask_mod` predicate is the dense mask, inverted. Device-free, exact.

    Checked at the production 257 so the `// tokens_per_frame` arithmetic is
    exercised on a frame width that is not a power of two.
    """
    for window in (None, 2, 6):
        idx = torch.arange(5 * PROD_TOKENS_PER_FRAME)
        mod = frame_block_causal_mask_mod(PROD_TOKENS_PER_FRAME, window)
        keep = mod(torch.tensor(0), torch.tensor(0), idx[:, None], idx[None, :])
        blocked = frame_block_causal_mask(5, PROD_TOKENS_PER_FRAME, window=window)
        torch.testing.assert_close(keep, ~blocked)


@pytest.mark.parametrize(("num_frames", "window"), [(6, 6), (16, 6), (32, 16)])
def test_flex_block_mask_is_block_sparse(num_frames: int, window: int) -> None:
    """The BlockMask really skips blocks, and the 257-vs-128 waste stays small.

    This is the tile-alignment finding as an assertion: a 257-token frame block
    can never tile a 128-element kernel block, so every frame boundary produces a
    partial block that costs a full one. The overhead is a boundary effect and
    amortizes with the number of frames -- but it must never approach the 2.23x
    that padding frames to 384 would cost, which is the alternative this rejects.
    """
    k = PROD_TOKENS_PER_FRAME
    bm = frame_block_causal_block_mask(num_frames, k, window=window)
    # partial blocks (masked elementwise inside the kernel) cost the same as full
    # ones, so both count toward the work actually done
    full_blocks = bm.full_kv_num_blocks
    assert full_blocks is not None, "no fully-unmasked blocks at all?"
    computed = (
        int(bm.kv_num_blocks.sum().item()) + int(full_blocks.sum().item())
    ) * FLEX_BLOCK_SIZE**2
    exact = k * k * sum(min(f + 1, window) for f in range(num_frames))
    dense = (num_frames * k) ** 2
    assert computed < dense, "no blocks skipped -- the mask is not sparse"
    assert 1.0 <= computed / exact < MAX_TILE_WASTE, (
        f"tile waste {computed / exact:.3f}"
    )
    # and the whole point: less work than the dense mask, by the expected margin
    assert computed / dense < MAX_DENSE_FRACTION


def test_flex_forward_matches_sdpa_on_cpu() -> None:
    """Parity without a GPU, so CI covers the mask/plumbing even on a CPU runner.

    Eager FlexAttention (the CPU fallback) has no backward, so this is
    forward-only; the fwd+bwd gate is the CUDA test below.
    """
    sdpa, flex = _pair(window=2, dim=32, heads=4, device="cpu", layers=2)
    g = torch.Generator().manual_seed(5)
    x = torch.randn(1, 4 * PROD_TOKENS_PER_FRAME, 32, generator=g)
    with torch.no_grad():
        torch.testing.assert_close(
            flex(x, num_frames=4), sdpa(x, num_frames=4), rtol=0, atol=FLEX_TOL
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention needs CUDA")
@pytest.mark.parametrize(("dim", "heads"), [(512, 8), (768, 12)])
@pytest.mark.parametrize("window", [6, 16])
def test_flex_matches_sdpa_forward_and_backward(
    dim: int, heads: int, window: int
) -> None:
    """THE GATE: production shapes, fp32, forward and backward.

    `num_frames == window` so the mask is the full block-causal one -- the widest
    case, where nothing is skipped by the window and only causality is sparse. tf32
    is disabled so this measures the kernels, not the tensor cores' mantissa.
    """
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sdpa, flex = _pair(window=window, dim=dim, heads=heads, device="cuda")
    g = torch.Generator(device="cuda").manual_seed(9)
    x = torch.randn(1, window * PROD_TOKENS_PER_FRAME, dim, generator=g, device="cuda")
    grad_out = torch.randn(x.shape, generator=g, device="cuda")

    ref = _fwd_bwd(sdpa, x, grad_out, num_frames=window)
    got = _fwd_bwd(flex, x, grad_out, num_frames=window)
    errors = {key: _scale_rel(got[key], ref[key]) for key in ref}
    assert all(e < FLEX_TOL for e in errors.values()), errors


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention needs CUDA")
@pytest.mark.parametrize("window", [6, 16])
def test_flex_matches_sdpa_under_gradient_checkpointing(window: int) -> None:
    """The same parity in `.train()`, which is a different code path.

    `run_layer_stack` only wraps layers in `checkpoint(use_reentrant=False)` while
    training, so an eval-only test never runs the compiled flex kernel inside a
    checkpoint or its recompute. With every dropout at 0 the two arms are still
    exactly comparable. This is what catches a `BlockMask`-through-`checkpoint`
    regression -- a `BlockMask` is not a tensor, and `checkpoint` has to carry it
    through to the recompute unchanged.
    """
    torch.backends.cuda.matmul.allow_tf32 = False
    sdpa, flex = _pair(window=window, dim=512, heads=8, device="cuda")
    sdpa.train()
    flex.train()
    g = torch.Generator(device="cuda").manual_seed(13)
    x = torch.randn(1, window * PROD_TOKENS_PER_FRAME, 512, generator=g, device="cuda")
    grad_out = torch.randn(x.shape, generator=g, device="cuda")

    ref = _fwd_bwd(sdpa, x, grad_out, num_frames=window)
    got = _fwd_bwd(flex, x, grad_out, num_frames=window)
    errors = {key: _scale_rel(got[key], ref[key]) for key in ref}
    assert all(e < FLEX_TOL for e in errors.values()), errors


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention needs CUDA")
def test_flex_is_no_further_from_exact_than_sdpa() -> None:  # noqa: PLR0914
    """Why FLEX_TOL is not an arbitrary number.

    Both fp32 arms are compared against the SAME trunk run in float64 on the CPU,
    which is the exact answer for this mask. If the flex kernel had a semantic
    difference -- a mis-tiled block, a dropped partial block, a wrong scale -- it
    would sit orders of magnitude further from the fp64 reference than sdpa does.
    Measured on a 5090: sdpa 6.8e-7, flex 9.4e-7 scale-relative on the worst
    tensor, i.e. flex is 1.4x sdpa's own fp32 accumulation noise. The 2x budget
    below is therefore tight, not permissive.

    One layer, `window=6`: enough to be exact, cheap enough to run in float64 on a
    CPU.
    """
    torch.backends.cuda.matmul.allow_tf32 = False
    dim, heads, window = 512, 8, 6
    sdpa, flex = _pair(window=window, dim=dim, heads=heads, device="cuda", layers=1)
    exact = _prod_trunk(
        window=window, dim=dim, heads=heads, impl="sdpa", layers=1
    ).double()
    exact.load_state_dict({k: v.double().cpu() for k, v in sdpa.state_dict().items()})

    g = torch.Generator().manual_seed(9)
    x = torch.randn(1, window * PROD_TOKENS_PER_FRAME, dim, generator=g)
    grad_out = torch.randn(x.shape, generator=g)
    truth = _fwd_bwd(exact, x.double(), grad_out.double(), num_frames=window)
    xc, gc = x.cuda(), grad_out.cuda()
    ref = _fwd_bwd(sdpa, xc, gc, num_frames=window)
    got = _fwd_bwd(flex, xc, gc, num_frames=window)

    for key, exact_value in truth.items():
        sdpa_err = _scale_rel(ref[key].cpu(), exact_value)
        flex_err = _scale_rel(got[key].cpu(), exact_value)
        assert flex_err < FLEX_TOL, (key, flex_err)
        assert flex_err <= 2 * sdpa_err + 1e-9, (key, flex_err, sdpa_err)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention needs CUDA")
def test_flex_is_invariant_to_frame_offset() -> None:
    """RoPE is applied to q/k *before* attention, so it is orthogonal to the
    kernel: the flex path keeps the shift-invariance that makes the cache valid.
    """
    torch.backends.cuda.matmul.allow_tf32 = False
    _, flex = _pair(window=4, dim=512, heads=8, device="cuda")
    flex.eval()
    g = torch.Generator(device="cuda").manual_seed(17)
    x = torch.randn(1, 4 * PROD_TOKENS_PER_FRAME, 512, generator=g, device="cuda")
    with torch.no_grad():
        torch.testing.assert_close(
            flex(x, num_frames=4, frame_offset=11),
            flex(x, num_frames=4, frame_offset=0),
            rtol=0,
            atol=FLEX_TOL,
        )


def test_flex_rejects_attention_dropout() -> None:
    """FlexAttention has no `dropout_p`; silently losing attention dropout when
    switching impl would be an unlogged regularization change.
    """
    with pytest.raises(ValueError, match="dropout"):
        CausalFrameTransformer(
            dim_model=DIM,
            num_layers=1,
            num_heads=HEADS,
            tokens_per_frame=TOKENS_PER_FRAME,
            window=2,
            attention_impl="flex",
        )


def test_unknown_attention_impl_is_rejected() -> None:
    with pytest.raises(ValueError, match="attention_impl"):
        CausalFrameTransformer(
            dim_model=DIM,
            num_layers=1,
            num_heads=HEADS,
            tokens_per_frame=TOKENS_PER_FRAME,
            window=2,
            attn_dropout=0.0,
            attention_impl="flash",  # ty:ignore[invalid-argument-type]
        )


@pytest.mark.parametrize("window", [2, 3, 6])
def test_ring_slot_write_matches_shift_left(window: int) -> None:
    """The host may write the new frame into slot `t % cache_frames` and move
    nothing, instead of shifting the whole cache left.

    Valid because attention is permutation-invariant over keys and every key
    carries its own RoPE rotation (absolute frame index) and its own `cache_bias`
    slot -- so cache ORDER is not information. Worth testing rather than asserting:
    it is the difference between moving 257 tokens per layer per tick and moving the
    entire cache, ~18 MiB versus ~2.3 GiB of device traffic at `_big`/64 frames, and
    the shift-left form is an overlapping in-place copy that is undefined on GPU.
    """
    num_frames = 9
    trunk = _trunk(window=window)
    tokens = _tokens(num_frames)
    shifted = stream(trunk, tokens, cache_frames=window - 1, ring=False)
    slotted = stream(trunk, tokens, cache_frames=window - 1, ring=True)
    torch.testing.assert_close(slotted, shifted, rtol=0, atol=TOL)
