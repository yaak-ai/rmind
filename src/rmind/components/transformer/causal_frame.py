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

* **intra-frame position: a learned `tokens_per_frame`-slot table**, tiled
  identically onto every frame. Token *k* of frame *f* always gets row *k* of
  that table, for every *f*. Frame-relative by construction, so it never goes
  stale. This preserves exactly the intra-frame capacity the 1542-slot embedding
  had (patch identity, and speed-token vs patch-token identity) -- nothing is
  dropped.

  How that table is scaled and parameterized is configurable along two
  orthogonal axes, `IntraPositionScaling` and `IntraPositionFactorization`,
  whose defaults (`norm_gain`/`flat` -- one free row per slot, LayerNorm'd to a
  common norm and scaled by a learned gain) reproduce the historical behaviour
  bit-for-bit. The factorized modes give a camera its own vector, fed by every
  one of its patch tokens, and share spatial structure across views; the
  panoramic ones lay the three views out in physical left-to-right order, and
  `pano_bearing` makes columns whose fields of view genuinely overlap share a
  learned code. **The composed table is `(tokens_per_frame, dim_model)` in every
  mode**, so the KV-cache layout, the ONNX bindings and the export path are
  identical throughout. `intra_position_applied_table()` is the single
  definition; `forward` and `step` both go through it.
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

Training cost: the mask is block-sparse, so use a block-sparse kernel
------------------------------------------------------------------------
The KV cache makes *serving* cost independent of the window; it does nothing for
training, which is still one dense forward over the whole clip. `attention_impl`
picks how the mask is realized:

* `"sdpa"` (default, unchanged behaviour) -- a materialized bool mask handed to
  `F.scaled_dot_product_attention`. Any `attn_mask` disqualifies the flash
  backend, so every masked position is still computed: cost is `O((F*257)^2)`
  regardless of `window`.
* `"flex"` -- the same mask as a `BlockMask` for `torch.nn.attention.
  flex_attention`, whose Triton kernel skips whole 128x128 blocks. Cost becomes
  proportional to the *unmasked* area, `O(F * window * 257^2)`, i.e. linear in
  context length at fixed window. Measured on a 5090 (bf16, fwd+bwd, one attention
  layer, batch 16, 512-d/8-head, `tests/bench_causal_frame.py`): 6 frames
  4.2 -> 2.1 ms, 32 frames/window 16 95.0 -> 24.4 ms (3.9x), 64 frames/window 16
  377.4 -> 54.7 ms (6.9x). Over the whole trunk at a constant 768 frame-slots per
  step, 32-frame/window-16 training costs 1.08x today's 6-frame dense step
  (dense SDPA would be 2.74x) -- see §11 of docs/decoder_only_kv_cache.md.
  Numerically identical to `"sdpa"` to <=1.5e-6 scale-relative in fp32, forward and
  backward (`tests/test_causal_frame.py`).

  It is NOT a free win at small shapes: at batch 4 / 6 frames flex is slower than
  sdpa (the mask has 13 kv blocks and cannot fill the GPU). The crossover is around
  1000 frame-slots per step, which is why `"sdpa"` remains the default.

Two constraints on `"flex"`, both deliberate rather than incidental:
`attn_dropout` must be 0 (FlexAttention has no `dropout_p`, and the constructor
raises rather than silently dropping it), and the decode `step` is unaffected --
it stays SDPA, because it is the export target and has nothing to skip anyway.

Note what is *not* equal, and cannot be for any bounded window with more than
one layer: re-running frames `[1..6]` as a fresh isolated 6-frame episode. There
frame 5 never saw frame 0, whereas in the stream its layer-2+ K/V were produced
with frame 0 in context. That is a property of bounded attention, not a cache
defect.
"""

import math
from collections.abc import Callable, Iterable, Sequence
from functools import cache, partial
from typing import Literal, cast, final, get_args, override

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention,
)
from torch.utils.checkpoint import checkpoint

from rmind.components.nn import Embedding, default_weight_init_fn
from rmind.components.position_encoding import PatchPositionEmbedding2D

__all__ = [
    "AttentionImpl",
    "CausalFrameTransformer",
    "CausalFrameTransformerBlock",
    "CausalSelfAttention",
    "IntraPositionFactorization",
    "IntraPositionScaling",
    "frame_band_slices",
    "frame_block_causal_block_mask",
    "frame_block_causal_mask",
    "frame_rope_cos_sin",
]

# additive bias for a masked-out position; finite so an fp16 engine cannot
# produce NaN from -inf * 0 in a fused softmax
MASK_BIAS: float = -1e4

AttentionImpl = Literal["sdpa", "flex"]

# FlexAttention's Triton kernels only accept a 128-element block on the shapes
# this trunk uses (`BLOCK_SIZE=64` raises "Q and KV block size must be divisible
# by BLOCK_M and BLOCK_N" from inductor on sm_120/torch 2.12), so the frame block
# of 257 tokens can never tile exactly. See §11 of docs/decoder_only_kv_cache.md
# for the measured cost of that misalignment (6-29% extra computed area, and
# padding the frame to 384 to align it is 2.23x -- strictly worse).
FLEX_BLOCK_SIZE: int = 128

# How the `(tokens_per_frame, dim_model)` intra-frame position table is SCALED
# before it is added to the content tokens. `b846a4f` introduced the LayerNorm +
# learnable-gain balance; it is `norm_gain` here and stays the default.
#
# * `norm_gain`       -- `LayerNorm(T) * gain` over EVERY row (today's behaviour)
# * `patch_norm_gain` -- the same, but over the patch band only; the prefix
#                        (speed) and suffix (register/readout) rows keep their
#                        raw, learned amplitude. The LayerNorm forces every row
#                        to norm `sqrt(dim_model) * gain`, which for a
#                        data-dependent slot like the speed token means its
#                        content is swamped by a constant (measured: the speed
#                        token's content/position ratio fell 1.74x -> 0.19x, i.e.
#                        ~84% of that token became fixed position code).
# * `gain`            -- `T * gain`; one scalar knob, no per-row flattening
# * `none`            -- the raw table; pair it with `intra_position_target_norm`
#                        so "scaling off" does not also mean "back to a 0.45 row
#                        norm 50x below content"
IntraPositionScaling = Literal["norm_gain", "patch_norm_gain", "gain", "none"]

# How that table is PARAMETERIZED. `flat` is one free row per intra-frame slot
# (today's behaviour); the rest factorize the patch band so that a camera's
# identity is a single vector fed by all of its patch tokens, and so that spatial
# structure is shared across views. Every mode composes to the same
# `(tokens_per_frame, dim_model)` shape, so the KV-cache layout, the ONNX
# bindings and the export path are identical in all of them.
#
# * `flat`         -- `flat[slot]`
# * `view`         -- `view[c] + patch[r*cols + j]`
# * `view_2d`      -- `view[c] + row[r] + col[j]`
# * `pano_col`     -- `view[c] + row[r] + gcol[order[c]*cols + j]`, where `order`
#                     is the physical left-to-right camera order derived from the
#                     configured yaws: adjacent columns of adjacent cameras
#                     become adjacent rows of one global column table
# * `pano_bearing` -- `view[c] + row[r] + interp(bearing(c, j))` against one
#                     shared bearing table, so the ~16 deg of FOV two adjacent
#                     cameras genuinely overlap indexes the SAME bins and shares
#                     a code rather than merely sitting next to it
IntraPositionFactorization = Literal[
    "flat", "view", "view_2d", "pano_col", "pano_bearing"
]

# A `trunc_normal_(std=s, a=-2s, b=2s)` draw realizes 0.7737 of the nominal
# variance (the +/-2sigma truncation this repo's `default_weight_init_fn` uses),
# so a d-dimensional row lands at norm ~= sqrt(d) * s * sqrt(0.7737) =
# sqrt(d) * s * 0.8796. `_init_std_for_target_norm` inverts that; the constant is
# pinned empirically by `test_target_norm_is_hit` rather than trusted.
TRUNC_NORMAL_2SIGMA_NORM_FACTOR: float = 0.8796


def frame_band_slices(  # noqa: PLR0913
    *,
    cameras: Sequence[str],
    num_patches: int,
    num_register: int,
    has_readout: bool,
    compress_cameras: Sequence[str] = (),
    num_camera_latents: int = 0,
) -> dict[str, slice]:
    """Intra-frame slot bands, `{"speed", "patch:<camera>"..., "latent:<camera>"...,
    "register"?, "readout"?}`.

    Slot layout `[speed, grid-camera patches (in `cameras` order, skipping
    `compress_cameras`), compressed-camera latents (in `cameras` order),
    register..., readout?]` -- this **must mirror
    `rmind.models.patch_policy.PatchPolicy._frame_tokens`'s `torch.cat` order
    exactly**, which is why it lives here, next to the trunk that consumes
    that layout, rather than being re-derived by each diagnostic that needs it
    (it was duplicated in two of them).

    `compress_cameras`/`num_camera_latents` default to off, reproducing the
    pre-bottleneck layout bit-for-bit.

    Pure index arithmetic, no torch: usable against a bare state dict.
    """
    bands = {"speed": slice(0, 1)}
    i = 1
    for camera in cameras:
        if camera in compress_cameras:
            continue
        bands[f"patch:{camera}"] = slice(i, i + num_patches)
        i += num_patches
    for camera in cameras:
        if camera not in compress_cameras:
            continue
        bands[f"latent:{camera}"] = slice(i, i + num_camera_latents)
        i += num_camera_latents
    if num_register:
        bands["register"] = slice(i, i + num_register)
        i += num_register
    if has_readout:
        bands["readout"] = slice(i, i + 1)
    return bands


def _init_std_for_target_norm(
    target_norm: float, *, dim: int, num_factors: float
) -> float:
    """The per-factor `trunc_normal_` std whose `num_factors`-term additive sum
    lands at row norm `target_norm` in `dim` dimensions.

    The factors are independent, so their variances add: `num_factors` terms of
    per-element variance `v` give `dim * num_factors * v` expected squared norm.
    """
    return target_norm / (
        math.sqrt(dim * num_factors) * TRUNC_NORMAL_2SIGMA_NORM_FACTOR
    )


def _column_bearings_deg(
    *, yaw_deg: Sequence[float], cols: int, hfov_deg: float
) -> Tensor:
    """Bearing of each patch column's centre, `(num_cameras, cols)`, in degrees.

    For a rectilinear camera the mapping from image column to bearing is **not**
    linear -- the sensor is planar, so equal pixel steps subtend smaller angles
    towards the edges:

        x_j     = (2j + 1) / cols - 1                    # column centre in (-1, 1)
        bearing = yaw_c + atan(x_j * tan(hfov / 2))

    Approximations, all deliberate and all correctable from config:

    * `hfov_deg` is assumed HORIZONTAL. The rig documentation gives "FOV 90"
      without saying horizontal or diagonal; if it is diagonal on 16:9 the true
      horizontal FOV is ~82.6 deg and the seam overlap between adjacent views is
      ~12.6 deg rather than ~16.4 deg. Both overlap, which is what matters here,
      and `camera_hfov_deg` is a config value precisely so this can be corrected
      without a code change.
    * Pitch and mount offset are ignored: this is a pure bearing (yaw) model.
    """
    x = (2 * torch.arange(cols, dtype=torch.float64) + 1) / cols - 1
    half = math.tan(math.radians(hfov_deg) / 2)
    offset = torch.rad2deg(torch.atan(x * half))
    return torch.tensor(yaw_deg, dtype=torch.float64).unsqueeze(-1) + offset


def _bearing_interpolation_matrix(bearings: Tensor, *, num_bins: int) -> Tensor:
    """Linear-interpolation weights from patch columns onto a shared bearing table.

    `bearings` is `(num_cameras, cols)` in degrees; the result is
    `(num_cameras * cols, num_bins)`, each row summing to 1 with at most two
    non-zeros. Bins are uniform in bearing over `[min, max]` of the whole rig, so
    the columns two adjacent cameras genuinely share (their FOVs overlap) land in
    the SAME bins with non-zero weight and literally share a learned code.

    Constant -- it depends only on the configured geometry -- so it is registered
    as a non-persistent buffer and the column term becomes one matmul: static
    shape, constant-foldable under `torch.export`.
    """
    flat = bearings.reshape(-1)
    low, high = flat.min(), flat.max()
    span = (high - low).clamp_min(torch.finfo(flat.dtype).eps)
    u = (flat - low) / span * (num_bins - 1)
    lower = u.floor().clamp(0, num_bins - 1).long()
    upper = (lower + 1).clamp(max=num_bins - 1)
    frac = (u - lower.to(u.dtype)).clamp(0.0, 1.0)
    weights = torch.zeros(flat.shape[0], num_bins, dtype=flat.dtype)
    rows = torch.arange(flat.shape[0])
    # index_put_ with accumulate: lower == upper at the very last bin, where both
    # halves of the weight belong to the same column and must SUM to 1
    weights.index_put_((rows, lower), 1.0 - frac, accumulate=True)
    weights.index_put_((rows, upper), frac, accumulate=True)
    return weights


def _validate_factorized_geometry(  # noqa: PLR0913
    *,
    factorization: IntraPositionFactorization,
    tokens_per_frame: int,
    num_cameras: int | None,
    patch_grid: tuple[int, int] | None,
    num_prefix_tokens: int,
    num_suffix_tokens: int,
    camera_yaw_deg: tuple[float, ...] | None,
) -> tuple[int, int]:
    """Check a non-`flat` arm's geometry against the frame layout; returns
    `(rows, cols)`.

    Raises:
        ValueError: if `num_cameras`/`patch_grid` are absent, if the slot
            arithmetic does not reproduce `tokens_per_frame`, or -- for the
            panoramic modes -- if `camera_yaw_deg` is missing, the wrong length,
            or not distinct.
    """
    if num_cameras is None or patch_grid is None:
        msg = (
            f"intra_position_factorization={factorization!r} requires BOTH "
            f"num_cameras (got {num_cameras!r}) and patch_grid (got "
            f"{patch_grid!r}): the trunk cannot infer them -- tokens_per_frame "
            "minus the non-patch slots divides many ways, and the patch grid is "
            "a property of the image encoder, not of a length"
        )
        raise ValueError(msg)

    rows, cols = patch_grid
    expected = num_prefix_tokens + num_cameras * rows * cols + num_suffix_tokens
    if expected != tokens_per_frame:
        msg = (
            "slot arithmetic does not reproduce tokens_per_frame: "
            "PatchPolicy._frame_tokens lays a frame out as [prefix (speed), "
            f"cam_0 patches, ..., cam_{num_cameras - 1} patches, registers, "
            f"readout], i.e. {num_prefix_tokens} + {num_cameras}*{rows}*{cols} "
            f"+ {num_suffix_tokens} = {expected}, but "
            f"tokens_per_frame={tokens_per_frame}"
        )
        raise ValueError(msg)

    if factorization in {"pano_col", "pano_bearing"}:
        if camera_yaw_deg is None or len(camera_yaw_deg) != num_cameras:
            msg = (
                f"intra_position_factorization={factorization!r} requires "
                f"camera_yaw_deg with one yaw per camera (num_cameras="
                f"{num_cameras}), got {camera_yaw_deg!r}"
            )
            raise ValueError(msg)
        if len(set(camera_yaw_deg)) != num_cameras:
            msg = (
                "camera_yaw_deg must be distinct to define a physical "
                f"left-to-right order, got {camera_yaw_deg!r}"
            )
            raise ValueError(msg)

    return rows, cols


def _as_int_pair(value: Iterable[int] | None) -> tuple[int, int] | None:
    """Hydra hands `list`/`ListConfig` where a `tuple` is declared.

    `CausalFrameTransformer.__init__` deliberately carries no `@validate_call`
    (see its docstring), so the normalization pydantic would do happens here.

    Raises:
        ValueError: if the value is not a pair.
    """
    if value is None:
        return None
    pair = tuple(int(v) for v in value)
    if len(pair) != 2:  # noqa: PLR2004
        msg = f"expected two values (rows, cols), got {pair!r}"
        raise ValueError(msg)
    return (pair[0], pair[1])


def _as_float_tuple(value: Iterable[float] | None) -> tuple[float, ...] | None:
    return None if value is None else tuple(float(v) for v in value)


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
        blocked |= delta > window - 1  # evicted frames
    return blocked


def frame_block_causal_mask_mod(
    tokens_per_frame: int, window: int | None
) -> Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]:
    """FlexAttention `mask_mod` for `frame_block_causal_mask`.

    Note the inverted convention: `frame_block_causal_mask` returns True for
    *blocked*, a `mask_mod` returns True for *keep*. The predicate is the same
    frame-delta comparison, evaluated on index tensors instead of materialized.

    ⚠️ Every operation here must be OUT-OF-PLACE. Inductor lowers a `mask_mod` as a
    pointwise subgraph, and an in-place op inside one fails to compile with
    "SubgraphLoweringException: Buffers cannot be created while lowering a pointwise
    subgraph" -- for every shape, so the flex path is simply dead. This is a live
    hazard rather than a hypothetical: `ruff check` (this repo runs it with
    `fix = true, unsafe-fixes = true`) rewrites `keep = keep & x` into `keep &= x`
    under PLR6104 and thereby breaks it, hence the `noqa` below.
    """

    def mask_mod(b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
        del b, h
        delta = q_idx // tokens_per_frame - kv_idx // tokens_per_frame
        keep = delta >= 0
        if window is not None:
            keep = keep & (delta <= window - 1)  # noqa: PLR6104
        return keep

    return mask_mod


@cache
def frame_block_causal_block_mask(
    num_frames: int,
    tokens_per_frame: int,
    *,
    window: int | None = None,
    device: torch.device | None = None,
) -> BlockMask:
    """`BlockMask` equivalent of `frame_block_causal_mask`, for FlexAttention.

    Memoized: building one costs 3-9 ms of launch-bound work (measured on a
    5090), which is per-*layer*-per-*step* if you let it happen naively -- 25 ms
    a step at 8 layers, more than the attention itself. The cache key is the mask
    geometry plus the device, and a `BlockMask` at 64 frames is ~66 KiB, so
    holding them all costs nothing.

    Broadcast over batch and head (`B = H = None`): the mask is the same for every
    sequence in the batch, which is what makes it cheap.

    `device=None` means CPU, matching `frame_block_causal_mask` -- note that
    `create_block_mask`'s own default is `"cuda"`, which would make a CPU-only
    caller fail with "Torch not compiled with CUDA enabled".
    """
    seq_len = num_frames * tokens_per_frame
    return create_block_mask(
        frame_block_causal_mask_mod(tokens_per_frame, window),
        B=None,
        H=None,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=torch.device("cpu") if device is None else device,
        BLOCK_SIZE=FLEX_BLOCK_SIZE,
    )


@cache
def _compiled_flex_attention() -> Callable[..., Tensor]:
    """`torch.compile`d `flex_attention` -- the only form that is block-sparse.

    Eager `flex_attention` is a correctness reference that materializes the score
    matrix; the Triton kernel that actually skips masked blocks only exists after
    `torch.compile`. `dynamic=False` keeps static shapes (one specialization per
    `(batch, seq_len, heads, dtype)`); the trunk sees at most a couple of those,
    but if you vary batch size a lot, raise `torch._dynamo.config.cache_size_limit`
    or dynamo will silently fall back to eager after 8 recompiles.

    Compiled lazily so importing this module (which the ONNX export path does)
    never pays for it.
    """
    return torch.compile(flex_attention, dynamic=False)  # ty:ignore[no-matching-overload]


def flex_frame_attention(
    q: Tensor, k: Tensor, v: Tensor, block_mask: BlockMask
) -> Tensor:
    """Block-sparse attention on CUDA, eager FlexAttention elsewhere.

    FlexAttention has no CPU backward (`NotImplementedError` in
    `_validate_device`), and there is no Triton kernel to gain on CPU anyway, so
    the CPU path is deliberately the eager reference implementation: it exists so
    the parity test can run without a GPU, not to be fast.
    """
    if q.is_cuda:
        return _compiled_flex_attention()(q, k, v, block_mask=block_mask)
    return flex_attention(q, k, v, block_mask=block_mask)


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

    Raises:
        ValueError: if `head_dim` is odd.
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
        self,
        *,
        dim_model: int,
        num_heads: int,
        dropout: float = 0.1,
        attention_impl: AttentionImpl = "sdpa",
    ) -> None:
        super().__init__()
        if dim_model % num_heads:
            msg = f"dim_model {dim_model} not divisible by num_heads {num_heads}"
            raise ValueError(msg)
        if attention_impl not in get_args(AttentionImpl):
            msg = f"attention_impl must be one of {get_args(AttentionImpl)}"
            raise ValueError(msg)
        # FlexAttention has no `dropout_p`. Failing loudly beats silently
        # dropping the trunk's attention dropout when someone flips the impl.
        if attention_impl == "flex" and dropout:
            msg = (
                f"attention_impl='flex' cannot apply attention dropout "
                f"(got attn_dropout={dropout}): FlexAttention has no dropout_p. "
                "Set attn_dropout: 0.0 explicitly (resid_dropout/mlp_dropout are "
                "unaffected) or use attention_impl='sdpa'."
            )
            raise ValueError(msg)
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.head_dim = dim_model // num_heads
        self.dropout = dropout
        self.attention_impl = attention_impl

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
    def forward(
        self, x: Tensor, cos: Tensor, sin: Tensor, mask: Tensor | BlockMask
    ) -> Tensor:
        """Full-sequence forward.

        `mask` is the bool block-causal mask (`attention_impl='sdpa'`) or the
        equivalent `BlockMask` (`attention_impl='flex'`). RoPE is applied to q/k
        *before* attention either way, so the positional scheme is orthogonal to
        the kernel choice -- the frame-granular rotation is already baked into the
        vectors the kernel sees.

        Raises:
            TypeError: if `mask` is not the type `attention_impl` requires.
        """
        q, k, v = self._qkv(x)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if self.attention_impl == "flex":
            if not isinstance(mask, BlockMask):
                msg = f"attention_impl='flex' needs a BlockMask, got {type(mask)}"
                raise TypeError(msg)
            attn = flex_frame_attention(q, k, v, mask)
        else:
            if isinstance(mask, BlockMask):
                msg = "attention_impl='sdpa' needs a bool mask, got a BlockMask"
                raise TypeError(msg)
            attn = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=~mask,
                dropout_p=self.dropout if self.training else 0.0,
            )
        return self._out(attn)

    def step(  # noqa: PLR0913, PLR0917
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
        """One frame's `tokens_per_frame` queries against the cache plus itself.

        No causal mask is needed: every cached key is in the past (causal by
        construction) and every own-frame key is visible (bidirectional within a
        frame). The only masking is `cache_bias`, which zeroes out unfilled ring
        slots.

        `attention_impl` deliberately does NOT apply here: the decode step is
        always plain SDPA. It has 257 queries against a dense, fully-visible
        cache -- there is no sparsity to exploit -- and it is the ONNX/TRT export
        target, which a `torch.compile`d Triton kernel could not be.

        `readout_only` computes the attention output for the LAST query position
        only -- the head reads a single token per frame (§3.3 of the hand-off), so
        in the final block the other `tokens_per_frame - 1` outputs are discarded.
        K/V are still produced for every position; future frames attend to them.

        Returns `(out, new_k, new_v)` where the K/V are for the new frame only.
        """
        q, k, v = self._qkv(x)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        keys = torch.cat((past_k, k), dim=-2)
        values = torch.cat((past_v, v), dim=-2)
        bias = F.pad(cache_bias, (0, x.shape[1]))  # own-frame keys always visible
        attn = F.scaled_dot_product_attention(
            q[:, :, -1:] if readout_only else q, keys, values, attn_mask=bias
        )
        return self._out(attn), k, v


@final
class DropPath(nn.Module):
    """Stochastic depth (https://arxiv.org/abs/1603.09382): drop the whole
    residual branch per sample with probability `drop_prob` during training,
    rescaling survivors by `1/keep`; exact identity in eval, so the streaming
    equivalence gate and the export `step` path are untouched.
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= drop_prob < 1.0:
            msg = f"drop_prob must be in [0, 1), got {drop_prob}"
            raise ValueError(msg)
        self.drop_prob = drop_prob

    @override
    def forward(self, x: Tensor) -> Tensor:
        if not self.drop_prob or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        # fresh tensor, filled in place before entering the autograd graph --
        # no in-place ops on the residual stream (checkpointing)
        gate = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * (gate / keep)


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
        attention_impl: AttentionImpl = "sdpa",
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.drop_path = DropPath(drop_path)
        self.attn_norm = nn.LayerNorm(dim_model)
        self.attn = CausalSelfAttention(
            dim_model=dim_model,
            num_heads=num_heads,
            dropout=attn_dropout,
            attention_impl=attention_impl,
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
    def forward(
        self, x: Tensor, cos: Tensor, sin: Tensor, mask: Tensor | BlockMask
    ) -> Tensor:
        attn_out = self.attn(self.attn_norm(x), cos, sin, mask)
        # NOTE: no in-place ops on the residual stream (autograd + checkpointing)
        h = x + self.drop_path(self.resid_drop(attn_out))
        return h + self.drop_path(self.mlp(self.mlp_norm(h)))

    def step(  # noqa: PLR0913, PLR0917
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
        attn_out, k, v = self.attn.step(
            self.attn_norm(x),
            cos,
            sin,
            past_k,
            past_v,
            cache_bias,
            readout_only=readout_only,
        )
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

    `checkpoint` sets the activation-checkpointing policy used by `forward` while
    training: `True` wraps every block, `False` none, an int `k` every k-th
    block. Wrapping a block trades a full extra forward of it for the memory of
    its activations. `step` (the KV-cached decode path) never checkpoints --
    it is inference-only and the export target.
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
        max_sequence_length: int | None = None,
        attn_dropout: float = 0.1,
        resid_dropout: float = 0.1,
        mlp_dropout: float = 0.1,
        hidden_layer_multiplier: int = 4,
        attention_impl: AttentionImpl = "sdpa",
        drop_path_rate: float = 0.0,
        checkpoint: bool | int = True,
        intra_position_scaling: IntraPositionScaling = "norm_gain",
        intra_position_factorization: IntraPositionFactorization = "flat",
        intra_position_target_norm: float | None = None,
        num_cameras: int | None = None,
        patch_grid: Iterable[int] | None = None,
        num_prefix_tokens: int = 1,
        num_suffix_tokens: int = 0,
        camera_yaw_deg: Iterable[float] | None = None,
        camera_hfov_deg: float = 90.0,
        num_bearing_bins: int | None = None,
    ) -> None:
        super().__init__()
        # normalized to "checkpoint every k-th block", 0 = never
        if isinstance(checkpoint, bool):
            self._checkpoint_every: int = 1 if checkpoint else 0
        elif checkpoint >= 1:
            self._checkpoint_every = checkpoint
        else:
            msg = f"checkpoint must be a bool or an int >= 1, got {checkpoint!r}"
            raise ValueError(msg)
        # There is no learned positional table over the flattened sequence any
        # more, so this trunk has no intrinsic maximum length. The argument
        # exists so the hydra `encoder` node can be overridden in place (the
        # block-causal trunk in config/model/yaak/patch_policy/raw.yaml sets
        # `max_sequence_length: episode_length * (num_patches + 1)`), and it is
        # cross-checked rather than ignored.
        #
        # The check is `>=`, not `==`, deliberately: a clip LONGER than the window is
        # the configuration sliding-window attention exists for (e.g. train on
        # 64-frame clips with a 32-frame window, so most readouts see a full window).
        # A clip SHORTER than the window is the error -- it silently trains a
        # narrower context than will be served, which is the train/infer mismatch
        # the hand-off warns about in §6.
        if max_sequence_length is not None and window is not None:
            minimum = window * tokens_per_frame
            if max_sequence_length < minimum:
                msg = (
                    f"max_sequence_length {max_sequence_length} < window * "
                    f"tokens_per_frame {minimum}: the clip is shorter than the "
                    "attention window, so training would never see a full window "
                    "while serving always will (hand-off §6)"
                )
                raise ValueError(msg)

        self.dim_model = dim_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = dim_model // num_heads
        self.tokens_per_frame = tokens_per_frame
        self.window = window
        self.rope_base = rope_base
        self.attention_impl = attention_impl

        self._init_intra_position(
            scaling=intra_position_scaling,
            factorization=intra_position_factorization,
            target_norm=intra_position_target_norm,
            num_cameras=num_cameras,
            patch_grid=_as_int_pair(patch_grid),
            num_prefix_tokens=num_prefix_tokens,
            num_suffix_tokens=num_suffix_tokens,
            camera_yaw_deg=_as_float_tuple(camera_yaw_deg),
            camera_hfov_deg=camera_hfov_deg,
            num_bearing_bins=num_bearing_bins,
        )
        # stochastic depth: timm-style linear ramp, 0 at the first layer up to
        # drop_path_rate at the last (deeper layers are the more redundant ones)
        self.drop_path_rate = drop_path_rate
        self.layers = nn.ModuleList([
            CausalFrameTransformerBlock(
                dim_model=dim_model,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                resid_dropout=resid_dropout,
                mlp_dropout=mlp_dropout,
                hidden_layer_multiplier=hidden_layer_multiplier,
                attention_impl=attention_impl,
                drop_path=drop_path_rate * i / max(num_layers - 1, 1),
            )
            for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim_model)

    def _init_intra_position(  # noqa: PLR0913
        self,
        *,
        scaling: IntraPositionScaling,
        factorization: IntraPositionFactorization,
        target_norm: float | None,
        num_cameras: int | None,
        patch_grid: tuple[int, int] | None,
        num_prefix_tokens: int,
        num_suffix_tokens: int,
        camera_yaw_deg: tuple[float, ...] | None,
        camera_hfov_deg: float,
        num_bearing_bins: int | None,
    ) -> None:
        """Build the frame-RELATIVE intra-frame position table's parameters.

        The table is tiled onto every frame, so it is invariant to where the
        frame sits in the window (unlike the 1542-slot window-absolute embedding
        it replaces). Two orthogonal, independently-selectable axes govern it:
        `IntraPositionScaling` (how loud it is) and
        `IntraPositionFactorization` (how it is parameterized). **The defaults,
        `norm_gain`/`flat`, reproduce the pre-existing behaviour bit-for-bit** --
        same parameter names, same state-dict keys, same init draw order -- which
        `test_default_intra_position_is_bit_identical` gates.

        On the scaling default (`b846a4f`, kept verbatim): the `trunc_normal_`
        table starts at per-token norm ~`sqrt(dim_model)*0.02`, ~50x smaller than
        the content tokens entering the trunk (`patch_projection` is
        `xavier_uniform_` over a LayerNorm'd input, so its output sits at
        ~`sqrt(dim_model)`). Left alone, the position/camera signal is drowned out
        by content in the residual sum -- the first block's pre-attention
        LayerNorm normalizes that SUM, not each addend, so it cannot rebalance a
        signal that is already 50x too small. Normalizing each position row to
        ~`sqrt(dim_model)` up front and scaling by a learnable gain (init 1.0,
        same as `fusion_patch_gain`/`fusion_goal_gain`) closes that gap
        immediately instead of waiting on gradient descent to grow a tiny table,
        while still letting training dial it back down. `elementwise_affine=
        False` on the LayerNorm: its own per-channel weight would be a redundant
        degree of freedom alongside the scalar gain, so `intra_position_gain`
        stays the single interpretable knob for how much the trunk trusts the
        position signal. `intra_position_target_norm` is the equivalent knob for
        the modes with no LayerNorm -- it sets the init std so that "scaling off"
        does not silently also mean "back to a 0.45 row norm".

        `intra_position_norm`/`intra_position_gain` are created **only in the
        modes that use them**, so the arm is legible in the state dict and
        `SelectiveAdamW`'s literal-name whitelist has nothing to match in the
        others.

        Naming note: the 2D modes hold their patch factors in a
        `PatchPositionEmbedding2D`, so their state-dict keys are
        `patch_position_embedding.{row_embed,col_embed}.weight` in all three of
        `view_2d`/`pano_col`/`pano_bearing`. In `pano_col` the `col_embed` table
        is the *global panoramic* column table (`num_cameras * cols` rows) and in
        `pano_bearing` it is the *shared bearing* table (`num_bearing_bins` rows),
        addressed by a constant interpolation matmul rather than a gather.

        Raises:
            ValueError: on an unknown mode, a negative/oversized prefix or suffix
                count, a factorized mode missing `num_cameras`/`patch_grid`, a
                slot arithmetic that does not reproduce `tokens_per_frame`, or a
                panoramic mode without a usable `camera_yaw_deg`.
        """
        if scaling not in get_args(IntraPositionScaling):
            msg = (
                f"unknown intra_position_scaling {scaling!r}, expected one of "
                f"{get_args(IntraPositionScaling)}"
            )
            raise ValueError(msg)
        if factorization not in get_args(IntraPositionFactorization):
            msg = (
                f"unknown intra_position_factorization {factorization!r}, "
                f"expected one of {get_args(IntraPositionFactorization)}"
            )
            raise ValueError(msg)
        if num_prefix_tokens < 0 or num_suffix_tokens < 0:
            msg = (
                "num_prefix_tokens/num_suffix_tokens must be >= 0, got "
                f"{num_prefix_tokens}/{num_suffix_tokens}"
            )
            raise ValueError(msg)
        if num_prefix_tokens + num_suffix_tokens >= self.tokens_per_frame:
            msg = (
                f"num_prefix_tokens {num_prefix_tokens} + num_suffix_tokens "
                f"{num_suffix_tokens} leaves no patch band in "
                f"tokens_per_frame={self.tokens_per_frame}"
            )
            raise ValueError(msg)

        # public so the diagnostics can read the arm off the trunk instead of
        # re-deriving it from the PatchPolicy that owns it
        self.intra_position_scaling: IntraPositionScaling = scaling
        self.intra_position_factorization: IntraPositionFactorization = factorization
        self.intra_position_target_norm = target_norm
        self.num_cameras = num_cameras
        self.patch_grid = patch_grid
        self.num_prefix_tokens = num_prefix_tokens
        self.num_suffix_tokens = num_suffix_tokens
        self.camera_yaw_deg = camera_yaw_deg
        self.camera_hfov_deg = camera_hfov_deg

        dim = self.dim_model
        if factorization == "flat":
            # verbatim, so the default arm's init draw is unchanged: plain
            # nn.Embedding (whose reset_parameters draws once) then the
            # trunc_normal_ overwrite
            self.intra_position_embedding = nn.Embedding(self.tokens_per_frame, dim)
            std = (
                0.02
                if target_norm is None
                else _init_std_for_target_norm(target_norm, dim=dim, num_factors=1)
            )
            nn.init.trunc_normal_(
                self.intra_position_embedding.weight,
                mean=0.0,
                std=std,
                a=-2 * std,
                b=2 * std,
            )
        else:
            self._init_factorized_intra_position(
                factorization=factorization,
                target_norm=target_norm,
                num_cameras=num_cameras,
                patch_grid=patch_grid,
                camera_yaw_deg=camera_yaw_deg,
                camera_hfov_deg=camera_hfov_deg,
                num_bearing_bins=num_bearing_bins,
            )

        if scaling in {"norm_gain", "patch_norm_gain"}:
            self.intra_position_norm = nn.LayerNorm(dim, elementwise_affine=False)
        if scaling in {"norm_gain", "patch_norm_gain", "gain"}:
            self.intra_position_gain = nn.Parameter(torch.tensor(1.0))

    def _init_factorized_intra_position(  # noqa: PLR0913
        self,
        *,
        factorization: IntraPositionFactorization,
        target_norm: float | None,
        num_cameras: int | None,
        patch_grid: tuple[int, int] | None,
        camera_yaw_deg: tuple[float, ...] | None,
        camera_hfov_deg: float,
        num_bearing_bins: int | None,
    ) -> None:
        """The non-`flat` half of `_init_intra_position`. See it for the contract.

        Geometry validation lives in `_validate_factorized_geometry`, which this
        calls first and which raises `ValueError` on every malformed arm.
        """
        rows, cols = _validate_factorized_geometry(
            factorization=factorization,
            tokens_per_frame=self.tokens_per_frame,
            num_cameras=num_cameras,
            patch_grid=patch_grid,
            num_prefix_tokens=self.num_prefix_tokens,
            num_suffix_tokens=self.num_suffix_tokens,
            camera_yaw_deg=camera_yaw_deg,
        )
        assert num_cameras is not None  # noqa: S101  # validated just above
        dim = self.dim_model
        pre, suf = self.num_prefix_tokens, self.num_suffix_tokens
        panoramic = factorization in {"pano_col", "pano_bearing"}

        # per-patch-row additive factor count, for the target-norm solve:
        # view + patch (`view`), or view + row + column (the rest)
        num_factors = 2.0 if factorization == "view" else 3.0
        num_columns = cols
        if panoramic:
            num_columns, num_factors = self._init_panoramic_columns(
                factorization=factorization,
                num_cameras=num_cameras,
                cols=cols,
                camera_yaw_deg=camera_yaw_deg or (),
                camera_hfov_deg=camera_hfov_deg,
                num_bearing_bins=num_bearing_bins,
            )

        def init_fn(factors: float) -> Callable[[Tensor], None]:
            if target_norm is None:
                return default_weight_init_fn  # ty:ignore[invalid-return-type]
            std = _init_std_for_target_norm(target_norm, dim=dim, num_factors=factors)
            return partial(  # ty:ignore[invalid-return-type]
                nn.init.trunc_normal_, mean=0.0, std=std, a=-2 * std, b=2 * std
            )

        def embedding(num: int, factors: float) -> Embedding:
            return Embedding(num, dim, weight_init_fn=init_fn(factors))

        # Non-patch slots always get their own free rows and NEVER participate in
        # the factorization: a readout token has no camera and no grid position.
        # `M = 1` for them, so they hit the target norm on their own.
        if pre + suf:
            self.special_position_embedding = embedding(pre + suf, 1)
        self.view_position_embedding = embedding(num_cameras, num_factors)
        if factorization == "view":
            self.patch_position_embedding: Embedding | PatchPositionEmbedding2D = (
                embedding(rows * cols, num_factors)
            )
        else:
            # rows are SHARED across cameras (maximum transfer of vertical
            # structure). Known approximation: `cam_front_left` sits at +4 deg
            # pitch and the side cameras ~35 cm lower, so the horizon sits ~1.1
            # rows apart between centre and sides (vfov ~58 deg over 16 rows =
            # 3.6 deg/row). Per-camera rows are a one-line change if it matters.
            self.patch_position_embedding = PatchPositionEmbedding2D(
                (rows, num_columns), dim, weight_init_fn=init_fn(num_factors)
            )

    def _init_panoramic_columns(  # noqa: PLR0913
        self,
        *,
        factorization: IntraPositionFactorization,
        num_cameras: int,
        cols: int,
        camera_yaw_deg: tuple[float, ...],
        camera_hfov_deg: float,
        num_bearing_bins: int | None,
    ) -> tuple[int, float]:
        """Set up the column term of a panoramic factorization.

        Returns `(num_columns, num_factors)`: the width of the shared column
        table, and the effective additive-factor count a patch row sees (which
        the target-norm solve needs).
        """
        if factorization == "pano_col":
            # physical left-to-right order, DERIVED from yaw and never
            # configured directly: for the production rig the permutation is
            # [1, 0, 2], which is its OWN INVERSE, so getting the direction
            # backwards would be completely invisible at runtime and would only
            # show up as a slightly worse arm.
            order = torch.argsort(  # order[k] = camera at physical slot k
                torch.tensor(camera_yaw_deg, dtype=torch.float64)
            )
            slot_of_camera = torch.empty_like(order)
            slot_of_camera[order] = torch.arange(num_cameras)
            index = (slot_of_camera.unsqueeze(-1) * cols + torch.arange(cols)).reshape(
                -1
            )
            self.register_buffer(
                "panorama_column_index", index.long(), persistent=False
            )
            self.panorama_camera_order: tuple[int, ...] = tuple(order.tolist())
            return num_cameras * cols, 3.0

        bins = num_bearing_bins or num_cameras * cols
        bearings = _column_bearings_deg(
            yaw_deg=camera_yaw_deg, cols=cols, hfov_deg=camera_hfov_deg
        )
        weights = _bearing_interpolation_matrix(bearings, num_bins=bins)
        # non-persistent: keeps a derived constant out of the state dict, so
        # `strict=True` loads and warm_start_ckpt's self-check are unaffected
        self.register_buffer("bearing_interpolation", weights.float(), persistent=False)
        self.register_buffer("column_bearing_deg", bearings.float(), persistent=False)
        # the bearing term is a convex combination of <=2 bins, so its variance
        # is `sum_j w_ij^2` of one table row's, not 1x -- fold that in rather
        # than overshooting the target norm
        return bins, 2 + float(weights.pow(2).sum(dim=-1).mean())

    def intra_position_table(self) -> Tensor:
        """The RAW composed intra-frame position table, `(tokens_per_frame, d)`.

        One definition, shared by `forward` and `step` -- see
        `intra_position_applied_table`.
        """
        if self.intra_position_factorization == "flat":
            # an embedding lookup on `arange(n)` is an exact gather of the whole
            # weight, so returning `.weight` is bit-identical and drops an
            # arange+gather pair from the exported decode graph
            return cast("nn.Embedding", self.intra_position_embedding).weight

        num_cameras, (rows, cols) = self._patch_geometry()
        num_patches = rows * cols
        view = self.view_position_embedding.weight

        match self.intra_position_factorization:
            case "view":
                patch = self.patch_position_embedding.weight
            case "view_2d":
                patch = cast(
                    "PatchPositionEmbedding2D", self.patch_position_embedding
                ).table()
            case _:
                patch = None

        if patch is not None:
            # camera-major: camera c owns rows [c*P, (c+1)*P)
            band = view.repeat_interleave(num_patches, dim=0) + patch.repeat(
                num_cameras, 1
            )
        else:
            row = cast(
                "PatchPositionEmbedding2D", self.patch_position_embedding
            ).row_embed.weight
            column = self._panoramic_columns()  # (num_cameras, cols, d)
            band = (
                view.reshape(num_cameras, 1, 1, -1)
                + row.reshape(1, rows, 1, -1)
                + column.reshape(num_cameras, 1, cols, -1)
            ).reshape(num_cameras * num_patches, -1)

        pre, suf = self.num_prefix_tokens, self.num_suffix_tokens
        if not (pre or suf):
            return band
        special = self.special_position_embedding.weight
        return torch.cat([special[:pre], band, special[pre : pre + suf]])

    def _patch_geometry(self) -> tuple[int, tuple[int, int]]:
        """`(num_cameras, (rows, cols))`, non-optional.

        Raises:
            ValueError: if the trunk is not on a factorized arm (where
                `_init_intra_position` has already validated both are set).
        """
        if self.num_cameras is None or self.patch_grid is None:
            msg = "patch geometry is only defined on a factorized position arm"
            raise ValueError(msg)
        return self.num_cameras, self.patch_grid

    def _panoramic_columns(self) -> Tensor:
        """The per-camera column term of a panoramic factorization,
        `(num_cameras, cols, dim_model)`."""
        num_cameras, _ = self._patch_geometry()
        column = cast(
            "PatchPositionEmbedding2D", self.patch_position_embedding
        ).col_embed.weight
        if self.intra_position_factorization == "pano_col":
            gathered = column.index_select(
                0, cast("Tensor", self.panorama_column_index)
            )
        else:
            # one constant matmul against the shared bearing table: overlapping
            # columns of adjacent cameras hit the same bins and share a code
            gathered = (
                cast("Tensor", self.bearing_interpolation).to(column.dtype) @ column
            )
        return gathered.reshape(num_cameras, -1, column.shape[-1])

    def intra_position_applied_table(self) -> Tensor:
        """`intra_position_table()` with `intra_position_scaling` applied -- the
        table the trunk ACTUALLY adds to its content tokens.

        Both `forward` (via `_intra`) and `step` go through here, so the
        expression exists exactly once. `b846a4f` duplicated it across the two
        paths, and a diagnostic that read the raw table while the model applied a
        scaled one already reported a 4.8x amplitude *increase* as a 2.3x
        decrease; the diagnostics now delegate here too.
        """
        table = self.intra_position_table()
        # accessed per branch, not up front: `none`/`gain` do not CREATE the
        # LayerNorm, and `none` does not create the gain either
        match self.intra_position_scaling:
            case "norm_gain":
                return self._position_norm()(table) * self._position_gain()
            case "patch_norm_gain":
                pre = self.num_prefix_tokens
                end = self.tokens_per_frame - self.num_suffix_tokens
                scaled = self._position_norm()(table[pre:end]) * self._position_gain()
                return torch.cat([table[:pre], scaled, table[end:]])
            case "gain":
                return table * self._position_gain()
            case _:
                return table

    def _position_norm(self) -> nn.LayerNorm:
        return self.intra_position_norm

    def _position_gain(self) -> nn.Parameter:
        return self.intra_position_gain

    def intra_position_parameters(self) -> dict[str, nn.Parameter]:
        """The learned position TABLES feeding `intra_position_table()`, by name.

        For a `flat` trunk that is exactly `{"intra_position_embedding.weight":
        ...}`. `intra_position_gain` is deliberately absent: it is a scalar
        calibration, not a table, and the parity gates that consume this compare
        table gradients.

        Enumerated from the known submodule names rather than by matching
        `"position_embedding"` against every parameter, so a subclass carrying an
        unrelated positional table (the tests' window-absolute control does) is
        not silently swept in.
        """
        owners = (
            "intra_position_embedding",
            "special_position_embedding",
            "view_position_embedding",
            "patch_position_embedding",
        )
        return {
            f"{owner}.{name}": param
            for owner in owners
            if (module := getattr(self, owner, None)) is not None
            for name, param in cast("nn.Module", module).named_parameters()
        }

    def _intra(self, num_frames: int) -> Tensor:
        return self.intra_position_applied_table().repeat(num_frames, 1)

    def _should_checkpoint(self, index: int) -> bool:
        return (
            self.training
            and self._checkpoint_every > 0
            and index % self._checkpoint_every == 0
        )

    @override
    def forward(self, src: Tensor, *, num_frames: int, frame_offset: int = 0) -> Tensor:
        """Full-sequence forward over `num_frames * tokens_per_frame` tokens.

        `frame_offset` shifts the episode-absolute frame indices used by RoPE.
        The output must be invariant to it -- that is the property the old
        window-absolute embedding lacked, and it is asserted in the tests.

        Raises:
            ValueError: if `src` is not `num_frames * tokens_per_frame` long.
        """
        _b, seq_len, _ = src.shape
        k = self.tokens_per_frame
        if seq_len != num_frames * k:
            msg = f"expected {num_frames * k} tokens, got {seq_len}"
            raise ValueError(msg)

        x = src + self._intra(num_frames)
        frames = torch.arange(seq_len, device=src.device) // k + frame_offset
        cos, sin = frame_rope_cos_sin(
            frames, head_dim=self.head_dim, base=self.rope_base
        )
        cos, sin = cos.to(src.dtype), sin.to(src.dtype)
        mask: Tensor | BlockMask = (
            frame_block_causal_block_mask(
                num_frames, k, window=self.window, device=src.device
            )
            if self.attention_impl == "flex"
            else frame_block_causal_mask(
                num_frames, k, window=self.window, device=src.device
            )
        )

        # NOTE: deliberately not `run_layer_stack` -- that helper is shared with
        # ControlTransformer's encoder/decoder, so an all-or-nothing checkpointing
        # policy there is not the right one here
        for i, layer in enumerate(self.layers):
            x = (
                checkpoint(layer, x, cos, sin, mask, use_reentrant=False)
                if self._should_checkpoint(i)
                else layer(x, cos, sin, mask)
            )
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

        Raises:
            ValueError: if neither `cache_frames` nor `window` is set.
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
            (1, 1, 1, n * self.tokens_per_frame), MASK_BIAS, device=device, dtype=dtype
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

        Raises:
            ValueError: if `src` is not exactly `tokens_per_frame` long.
        """
        if src.shape[1] != self.tokens_per_frame:
            # `forward` has always validated this; `step` used to index the
            # position table with `arange(src.shape[1])`, so a short `src`
            # silently ran against a PREFIX of the table instead of erroring
            msg = f"step expects {self.tokens_per_frame} tokens, got {src.shape[1]}"
            raise ValueError(msg)

        x = src + self.intra_position_applied_table()
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
