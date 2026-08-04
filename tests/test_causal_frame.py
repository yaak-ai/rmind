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

from typing import override

import pytest
import torch
from torch import Tensor, nn

from rmind.components.transformer.causal_frame import (
    MASK_BIAS,
    CausalFrameTransformer,
    apply_rope,
    frame_block_causal_mask,
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
    trunk: CausalFrameTransformer, tokens: Tensor, *, cache_frames: int
) -> Tensor:
    """One-frame-per-tick decode against a host-side ring buffer.

    Emulates exactly what drivr would do: the graph reads `past_k`/`past_v` and
    returns only the new frame's K/V, and the host shifts them into the ring.
    Returns the readout token per frame, `(b, t, d)`.
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
        if cache_frames:
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
