"""Phase 1 §4.4 (a)/(b)/(c) -- camera-identity diagnostics for the 3-camera
causal PatchPolicy, all three in one script (design settled 2026-08-31, see
`here-s-patch-policy-casual-linked-fox.md` §4.4).

(b) is read FIRST -- it is the precondition for interpreting (a), not a
secondary check (§4.4.0): a swap delta of ~0 only means "identity unresolved"
if the side cameras carry signal in the first place.

    (b) per-camera importance   -- is the side camera used AT ALL
    (a) camera-swap sensitivity -- is camera IDENTITY resolved
    (c) probe / attention mass  -- where identity lives, and whether the
                                    trunk USES it

All three share ONE frozen-ViT pass per batch: every manipulation in this
family -- swap, duplicate/copy_from, batch-permute, zero, mean-frame -- is a
gather on the `cam` axis of the `(b, t, cam, p, d)` tensor `model.image_encoder`
returns (`timm_backbone.py:54-77` applies the ViT per `(b, t, cam)`
independently), so `_CachedImageEncoder` below memoizes ONE encoder pass per
batch and every arm just rewrites its output before the trunk sees it. Only
`zero`/`mean-frame` need extra (one-time, cached-for-the-run) encoder work --
see `_zero_patch_embedding`/`_mean_patch_vectors`.

fp32, argmax only throughout (a)/(b) -- sampling noise would swamp the deltas
being hunted, and it makes the swap arm's noise floor exactly zero.

Only supports `CausalFrameTransformer` (the `*_causal*.yaml` arms), same
restriction as `patch_policy_position_audit.py` and for the same reason: no
per-slot `intra_position_embedding` / frame-RoPE to reason about otherwise.

Usage (from a repo checkout with the `val_3cam` rbyte cache built):
    uv run python -m rmind.scripts.patch_policy_camera_probe \
        --artifact yaak/alex-tmp/model-kughoqfi:latest \
        --config-dir /abs/path/to/config \
        [--batches 50] [--attention-batches 8] [--probe-batches 20] \
        [--include-mean-frame] [--skip-attention] [--skip-probe] \
        [--device cuda] [--out results.json]
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from types import MethodType
from typing import TYPE_CHECKING, Any, Self

import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.functional import cosine_similarity
from torch.utils._pytree import MappingKey  # noqa: PLC2701

from rmind.components.transformer.causal_frame import (
    MASK_BIAS,
    CausalFrameTransformer,
    apply_rope,
    frame_block_causal_mask,
    frame_rope_cos_sin,
)
from rmind.models.patch_policy import PatchPolicy
from rmind.scripts.patch_policy_eval import _default_cluster_fn, _to_device
from rmind.utils.pytree import key_get_default

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from rmind.components.transformer.causal_frame import CausalSelfAttention

# §4.5's revised gate: fires if Delta(A,B)/Delta(A,C) < this, on the swap arm's
# readout rel_diff -- see `_report_swap`.
SWAP_GATE_RATIO_THRESHOLD: float = 0.1

# =============================================================================
# Slot layout -- mirrors `patch_policy_position_audit._band_slices`/
# `_slot_layout` exactly (same `[speed, cam0 patches, ..., register, readout?]`
# order `PatchPolicy._frame_tokens` builds), reimplemented locally rather than
# imported so a TypeError here names THIS script, not that one.
# =============================================================================


def _require_causal_frame_trunk(model: PatchPolicy) -> CausalFrameTransformer:
    encoder = model.encoder
    if not isinstance(encoder, CausalFrameTransformer):
        msg = (
            f"patch_policy_camera_probe only supports CausalFrameTransformer "
            f"(the *_causal*.yaml arms), got {type(encoder).__name__}"
        )
        raise TypeError(msg)
    return encoder


def _num_patches_per_camera(model: PatchPolicy, encoder: CausalFrameTransformer) -> int:
    num_register = (
        model.register_tokens.shape[0] if model.register_tokens is not None else 0
    )
    non_patch = 1 + num_register + (1 if model.readout_token is not None else 0)
    num_patches, remainder = divmod(
        encoder.tokens_per_frame - non_patch, len(model.cameras)
    )
    if remainder:
        msg = (
            f"tokens_per_frame {encoder.tokens_per_frame} - {non_patch} non-patch "
            f"slots doesn't divide evenly across {len(model.cameras)} cameras"
        )
        raise ValueError(msg)
    return num_patches


def _slot_layout(model: PatchPolicy) -> dict[str, slice]:
    """`{"speed": slice, "patch:<camera>": slice, "register": slice?, "readout": slice?}`."""
    encoder = _require_causal_frame_trunk(model)
    num_patches = _num_patches_per_camera(model, encoder)
    num_register = (
        model.register_tokens.shape[0] if model.register_tokens is not None else 0
    )
    bands: dict[str, slice] = {"speed": slice(0, 1)}
    i = 1
    for camera in model.cameras:
        bands[f"patch:{camera}"] = slice(i, i + num_patches)
        i += num_patches
    if num_register:
        bands["register"] = slice(i, i + num_register)
        i += num_register
    if model.readout_token is not None:
        bands["readout"] = slice(i, i + 1)
    return bands


def _camera_indices(
    model: PatchPolicy, *, center: str, left: str, right: str
) -> tuple[int, int, int]:
    missing = [c for c in (center, left, right) if c not in model.cameras]
    if missing:
        msg = f"{missing} not in model.cameras={model.cameras!r}"
        raise ValueError(msg)
    idx = {name: i for i, name in enumerate(model.cameras)}
    return idx[center], idx[left], idx[right]


def _load_model(*, artifact: str | None, ckpt: str | None) -> PatchPolicy:
    """`PatchPolicy.load_from_wandb_artifact`/`load_from_checkpoint`, with a
    fallback for checkpoints trained before `b846a4f` (e.g. `03tuy3q9`):
    `CausalFrameTransformer.__init__` unconditionally creates `encoder.
    intra_position_gain`/`intra_position_norm` in the CURRENT code regardless
    of what the checkpoint was trained with, so a strict `load_state_dict`
    fails outright on one that predates them ("Missing key(s): encoder.
    intra_position_gain") -- there is no config flag gating this, it is a
    property of which commit's `causal_frame.py` is installed.

    Retrying with `strict=False` alone would silently run the pre-fix
    checkpoint's NEVER-TRAINED-FOR `intra_position_norm` (a real, non-identity
    `LayerNorm`) over its position table -- exactly the corruption `patch_
    policy_position_audit.py`'s own `_applied_position_table` guards against
    when just READING weights, except here the model would actually FORWARD
    through it. So on that specific failure this monkeypatches `encoder.
    intra_position_norm = nn.Identity()` post-load, restoring "the raw table
    IS the applied table" (§4.3) -- `intra_position_gain` is left at its
    default-init `1.0`, which is then a no-op multiplier, so the net effect is
    bit-identical to the pre-b846a4f `_intra()`.

    Raises:
        RuntimeError: re-raised for any `load_state_dict` failure OTHER than
            the specific pre-`b846a4f` missing-key one this works around.
    """
    kwargs: dict[str, Any] = {"weights_only": False, "map_location": "cpu"}
    loader = (
        (lambda **kw: PatchPolicy.load_from_wandb_artifact(artifact, **kw))
        if artifact
        else (lambda **kw: PatchPolicy.load_from_checkpoint(ckpt, **kw))
    )
    try:
        return loader(**kwargs)
    except RuntimeError as e:
        if "intra_position_gain" not in str(e):
            raise
        print(  # noqa: T201
            f"note: {artifact or ckpt} predates b846a4f (no encoder."
            "intra_position_gain in its state_dict) -- reloading strict=False "
            "and restoring the pre-fix raw-table behavior (intra_position_norm "
            "-> Identity) so this checkpoint runs as it actually trained/served"
        )
        model = loader(**kwargs, strict=False)
        encoder = _require_causal_frame_trunk(model)
        encoder.intra_position_norm = nn.Identity()
        return model


# =============================================================================
# One ViT pass, many arms: `model.image_encoder` gets replaced with this once
# at startup. It memoizes the last `(b, t, cam, p, d)` output keyed by a
# caller-driven `generation` counter -- NOT by tensor identity, because
# `PatchPolicy._features` rebuilds `images` (a fresh `torch.stack`, fresh
# `data_ptr()`) on every call even against the identical logical batch, which
# would make a `data_ptr()`-keyed cache miss every single time. The caller
# sets `generation` once per outer batch and `manipulate` once per arm; same
# generation -> the cached ViT output is reused and only `manipulate` reruns.
# =============================================================================


class _CachedImageEncoder(nn.Module):
    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.generation: object = None
        self.manipulate: Callable[[Tensor], Tensor] | None = None
        self._cached_generation: object = object()  # sentinel, never equals None
        self._patches: Tensor | None = None

    def forward(self, images: Tensor) -> Tensor:
        if self.generation != self._cached_generation:
            self._patches = self.inner(images)
            self._cached_generation = self.generation
        patches = self._patches
        assert patches is not None  # noqa: S101 -- set on the line above
        return patches if self.manipulate is None else self.manipulate(patches)


# =============================================================================
# Camera manipulations -- all operate on `patches: (b, t, cam, p, d)`, the
# frozen ViT's raw output, and return a NEW tensor (cheap: no_grad throughout).
# =============================================================================


def _swap(i: int, j: int) -> Callable[[Tensor], Tensor]:
    def manipulate(patches: Tensor) -> Tensor:
        out = patches.clone()
        out[:, :, i], out[:, :, j] = patches[:, :, j], patches[:, :, i]
        return out

    return manipulate


def _duplicate(source: int, targets: tuple[int, ...]) -> Callable[[Tensor], Tensor]:
    """Copies `patches[:, :, source]` into every index in `targets`: content
    changes, the POSITION/identity assignment of the targets does not."""

    def manipulate(patches: Tensor) -> Tensor:
        out = patches.clone()
        for t in targets:
            out[:, :, t] = patches[:, :, source]
        return out

    return manipulate


def _batch_permute(cam: int, *, seed: int) -> Callable[[Tensor], Tensor]:
    """Shuffles the BATCH axis for camera `cam` only -- preserves the marginal
    per-camera image distribution exactly, destroys only the sample-level
    correlation (the primary permutation-importance ablation, §4.4.1)."""

    def manipulate(patches: Tensor) -> Tensor:
        b = patches.shape[0]
        generator = torch.Generator(device="cpu").manual_seed(seed)
        perm = torch.randperm(b, generator=generator).to(patches.device)
        out = patches.clone()
        out[:, :, cam] = patches[perm][:, :, cam]
        return out

    return manipulate


def _fill_constant(cam: int, vector: Tensor) -> Callable[[Tensor], Tensor]:
    """`vector` is `(p, d)` -- broadcasts over `(b, t)`. Used for both `zero`
    and `mean-frame` (§4.4.1): same mechanism, different constant."""

    def manipulate(patches: Tensor) -> Tensor:
        out = patches.clone()
        out[:, :, cam] = vector.to(dtype=patches.dtype, device=patches.device)
        return out

    return manipulate


def _stacked_images(model: PatchPolicy, batch: Any) -> Tensor:
    inputs = model.input_transform(batch)
    image_by_camera = PatchPolicy._get(inputs, ("image",))  # noqa: SLF001
    return torch.stack([image_by_camera[c] for c in model.cameras], dim=2)


@torch.no_grad()
def _zero_patch_embedding(shim: _CachedImageEncoder, images: Tensor) -> Tensor:
    """The frozen ViT's output on an all-zero POST-`ImageNormalize` frame --
    "zeroing post-transform" in §4.4.1's terms (ImageNet mean-gray), the milder
    of the two zeroing conventions discussed there; raw-uint8 zeroing would
    additionally carry the `-mean/std` shift. The ViT runs per-`(b, t, cam)`
    independently, so a CONSTANT input gives a constant `(p, d)` output --
    computed once here (bypassing the cache, which is keyed by generation, not
    content) and reused for the whole run.
    """
    zeros = torch.zeros_like(images[:1, :1, :1])
    return shim.inner(zeros)[0, 0, 0]


@torch.no_grad()
def _mean_patch_vectors(
    shim: _CachedImageEncoder,
    model: PatchPolicy,
    loader: Iterable[Any],
    *,
    batches: int,
    device: torch.device,
) -> dict[int, Tensor]:
    """Dataset-average per-`(camera, patch-position)` embedding -- the
    "mean-frame" ablation. No trunk, no heads: one ViT pass per batch (cheap),
    amortized ONCE for the whole run (§4.4.4: "the mean frame's patches cache
    for the whole run"). A literal mean RAW FRAME through the ViT would be a
    blurry, semantically meaningless image; averaging the (frozen, linear-ish
    late-layer) PATCH EMBEDDING instead keeps the same "no per-sample
    variation" ablation intent without that confound.
    """
    total: Tensor | None = None
    count = 0
    for i, cpu_batch in enumerate(loader):
        if i >= batches:
            break
        batch = _to_device(cpu_batch, device)
        images = _stacked_images(model, batch)
        patches = shim.inner(images)  # (b, t, cam, p, d) -- cache bypassed on purpose
        summed = patches.sum(dim=(0, 1)).double()  # (cam, p, d)
        total = summed if total is None else total + summed
        count += patches.shape[0] * patches.shape[1]
    if total is None:
        return {}
    mean = (total / count).float()
    return {cam: mean[cam] for cam in range(mean.shape[0])}


# =============================================================================
# Decoding + per-sample metrics shared by (a) and (b)
# =============================================================================


def _decode(model: PatchPolicy, features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """`(code_logits, argmax_codes, argmax_chunk)` -- fp32, argmax only."""
    code_logits, offsets = model._heads(features)  # noqa: SLF001
    codes = code_logits.argmax(dim=-1)
    chunk = model.tokenizer.invert(codes) + model._offset(offsets, codes)  # noqa: SLF001
    return code_logits, codes, chunk


def _readout_metrics(ref: Tensor, arm: Tensor) -> dict[str, Tensor]:
    """`(b, t)` relative-norm difference and cosine between two readout-feature
    tensors -- continuous, near-zero-variance under pairing (§4.4.2)."""
    diff = (arm - ref).norm(dim=-1)
    base = ref.norm(dim=-1).clamp_min(1e-8)
    return {"rel_diff": diff / base, "cos": cosine_similarity(ref, arm, dim=-1)}


def _per_sample_code_ce(code_logits: Tensor, target_codes: Tensor) -> Tensor:
    """Mean per-`(b, t)` cross-entropy across quantizers. NOT `model.losses
    ["code"]` (FocalLoss, batch-reduced to a scalar) -- the paired bootstrap
    below needs a per-CLIP number, so this is a plain, unsmoothed,
    unmodulated substitute; fine for a diagnostic, not the training loss.
    """
    b, t, g, c = code_logits.shape
    ce = F.cross_entropy(
        code_logits.reshape(-1, c), target_codes.reshape(-1), reduction="none"
    ).view(b, t, g)
    return ce.mean(dim=-1)


# =============================================================================
# Accumulation + paired bootstrap CI (§4.4.4: "report paired per-sample delta
# with bootstrap CI over samples, n = batches x b, not two independent
# aggregates")
# =============================================================================


@dataclass
class _Accumulator:
    per_clip: dict[str, list[Tensor]] = field(default_factory=lambda: defaultdict(list))
    per_cluster: dict[str, dict[str, list[float]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(list))
    )

    def add(self, key: str, per_frame: Tensor) -> None:
        """`per_frame` is `(b, t)`. Stores the across-frame-position mean (one
        scalar per CLIP -- the resampling unit below) and the last-frame value
        (the deployment position, `patch_policy_eval.py`'s convention)."""
        self.per_clip[key].append(per_frame.mean(dim=-1).detach().float().cpu())
        self.per_clip[f"{key}@last"].append(per_frame[:, -1].detach().float().cpu())

    def add_cluster(self, key: str, last_frame: Tensor, labels: list[str]) -> None:
        for value, label in zip(last_frame.tolist(), labels, strict=True):
            self.per_cluster[key][label].append(value)

    def cat(self, key: str) -> Tensor:
        return torch.cat(self.per_clip[key])


def _bootstrap_mean_ci(
    values: Tensor, *, n_boot: int, alpha: float, generator: torch.Generator
) -> tuple[float, float, float]:
    n = values.shape[0]
    idx = torch.randint(0, n, (n_boot, n), generator=generator)
    boot_means = values[idx].mean(dim=1)
    lo, hi = torch.quantile(boot_means, torch.tensor([alpha / 2, 1 - alpha / 2]))
    return values.mean().item(), lo.item(), hi.item()


def _bootstrap_ratio_ci(
    numer: Tensor,
    denom: Tensor,
    *,
    n_boot: int,
    alpha: float,
    generator: torch.Generator,
) -> tuple[float, float, float]:
    """Paired bootstrap of `mean(numer) / mean(denom)`: resamples the SAME
    clip indices for both arrays every replicate, since they come from the
    same batches/samples (§4.4.2's headline `Δ(A,B) / Δ(A,C)`)."""
    n = numer.shape[0]
    idx = torch.randint(0, n, (n_boot, n), generator=generator)
    ratios = numer[idx].mean(dim=1) / denom[idx].mean(dim=1).clamp_min(1e-12)
    lo, hi = torch.quantile(ratios, torch.tensor([alpha / 2, 1 - alpha / 2]))
    point = (numer.mean() / denom.mean().clamp_min(1e-12)).item()
    return point, lo.item(), hi.item()


# =============================================================================
# (a) + (b): the shared per-batch matrix
# =============================================================================


@torch.no_grad()
def run_matrix(  # noqa: C901, PLR0913, PLR0914, PLR0915
    model: PatchPolicy,
    shim: _CachedImageEncoder,
    loader: Iterable[Any],
    *,
    device: torch.device,
    max_batches: int,
    center: int,
    left: int,
    right: int,
    seed: int,
    zero_vector: Tensor | None,
    mean_vectors: dict[int, Tensor] | None,
) -> _Accumulator:
    tokenizer = model.tokenizer
    acc = _Accumulator()

    # §4.4.2: three arms, all forwarded on the same batch
    swap_arms: dict[str, Callable[[Tensor], Tensor] | None] = {
        "A_baseline": None,
        "B_swap": _swap(left, right),
        "C_duplicate": _duplicate(center, (left, right)),
    }

    # §4.4.1: per-side-camera ablation family, in the specified priority order
    importance_cams = {"left": left, "right": right}
    importance_kinds: dict[str, Callable[[int, int], Callable[[Tensor], Tensor]]] = {
        "permute": lambda cam, b_idx: _batch_permute(cam, seed=seed * 100_003 + b_idx),
        "copy_from_center": lambda cam, _b: _duplicate(center, (cam,)),
    }
    if mean_vectors:
        importance_kinds["mean_frame"] = lambda cam, _b: _fill_constant(
            cam, mean_vectors[cam]
        )
    if zero_vector is not None:
        importance_kinds["zero"] = lambda cam, _b: _fill_constant(cam, zero_vector)

    for b_idx, cpu_batch in enumerate(loader):
        if b_idx >= max_batches:
            break
        batch = _to_device(cpu_batch, device)
        shim.generation = b_idx

        shim.manipulate = None
        features_ref, chunk = model._features(batch)  # noqa: SLF001
        target_codes = tokenizer(chunk)
        target = tokenizer._normalize(chunk.flatten(-2, -1))  # noqa: SLF001
        _, _, chunk_ref = _decode(model, features_ref)
        recon_ref = (chunk_ref - target).abs().mean(dim=-1)

        try:
            cluster_labels = _default_cluster_fn()(batch, None)
        except (KeyError, TypeError):
            cluster_labels = None

        for arm_name, manipulate in swap_arms.items():
            shim.manipulate = manipulate
            features_arm, _ = model._features(batch)  # noqa: SLF001
            code_logits_arm, _, chunk_arm = _decode(model, features_arm)
            readout = _readout_metrics(features_ref, features_arm)
            recon_arm = (chunk_arm - target).abs().mean(dim=-1)
            acc.add(f"{arm_name}/readout_rel_diff", readout["rel_diff"])
            acc.add(f"{arm_name}/readout_cos", readout["cos"])
            acc.add(
                f"{arm_name}/code_ce",
                _per_sample_code_ce(code_logits_arm, target_codes),
            )
            acc.add(f"{arm_name}/recon_l1", recon_arm)
            acc.add(f"{arm_name}/delta_recon_l1", recon_arm - recon_ref)
            if cluster_labels is not None:
                acc.add_cluster(
                    f"{arm_name}/recon_l1", recon_arm[:, -1], cluster_labels
                )

        for cam_name, cam in importance_cams.items():
            for kind, factory in importance_kinds.items():
                shim.manipulate = factory(cam, b_idx)
                features_arm, _ = model._features(batch)  # noqa: SLF001
                _, _, chunk_arm = _decode(model, features_arm)
                readout = _readout_metrics(features_ref, features_arm)
                recon_arm = (chunk_arm - target).abs().mean(dim=-1)
                tag = f"importance/{cam_name}/{kind}"
                acc.add(f"{tag}/readout_rel_diff", readout["rel_diff"])
                acc.add(f"{tag}/recon_l1", recon_arm)
                acc.add(f"{tag}/delta_recon_l1", recon_arm - recon_ref)
                if cluster_labels is not None:
                    acc.add_cluster(f"{tag}/recon_l1", recon_arm[:, -1], cluster_labels)

        acc.add("A_baseline/recon_l1_self", recon_ref)
        if cluster_labels is not None:
            acc.add_cluster(
                "A_baseline/recon_l1_self", recon_ref[:, -1], cluster_labels
            )

        # each arm above is its own full trunk forward at fp32 (no autocast,
        # §4.4.4) -- release the caching allocator's per-arm activations
        # between batches rather than let 7+ arms' peak usage compound via
        # fragmentation (`torch.cuda.OutOfMemoryError` observed in practice at
        # the val datamodule's default batch_size=32 x 32-frame causal clips).
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return acc


# =============================================================================
# (c), part 1: readout attention mass onto each camera band
# =============================================================================


class _AttentionMassCapture:
    """Temporarily reroutes every `CausalFrameTransformerBlock.attn.forward`
    through a manual, SDPA-equivalent recompute that ALSO buckets the softmax
    weight of each frame's READOUT query onto every camera/slot band.

    Resolves §4.4.3's open question ("does attention-mass capture work at
    window=6 under flex-attention, or does it need an SDPA fallback?"): yes,
    via exactly that fallback. `attention_impl='sdpa'` and `'flex'` are
    numerically identical to <=1.5e-6 relative, forward and backward
    (`tests/test_causal_frame.py`), so this capture always builds the plain
    bool `frame_block_causal_mask` and a manual `softmax(QK^T / sqrt(d))`,
    regardless of the checkpoint's configured `attention_impl` -- there is no
    need to instrument `flex_attention`/`BlockMask` directly. The real
    trunk OUTPUT computed under capture is the SDPA branch too, so it is
    bit-identical to the model's own output up to that same tolerance; (a)/(b)
    never run under capture, only this table does.
    """

    def __init__(
        self, encoder: CausalFrameTransformer, bands: dict[str, slice]
    ) -> None:
        self._encoder = encoder
        self._bands = bands
        self._originals: dict[int, Callable[..., Tensor]] = {}
        # (layer_idx, band_name) -> list of (h, t) tensors, one per batch
        self.mass: dict[tuple[int, str], list[Tensor]] = defaultdict(list)

    def __enter__(self) -> Self:
        for i, layer in enumerate(self._encoder.layers):
            attn = layer.attn
            self._originals[i] = attn.forward
            attn.forward = MethodType(self._build(i), attn)  # ty:ignore[invalid-assignment]
        return self

    def __exit__(self, *exc: object) -> None:
        for i, layer in enumerate(self._encoder.layers):
            layer.attn.forward = self._originals[i]  # ty:ignore[invalid-assignment]

    def _build(self, layer_idx: int) -> Callable[..., Tensor]:
        bands = self._bands
        mass = self.mass
        tpf = self._encoder.tokens_per_frame
        window = self._encoder.window

        def forward(
            attn_self: CausalSelfAttention,
            x: Tensor,
            cos: Tensor,
            sin: Tensor,
            mask: object,
        ) -> Tensor:
            del mask  # rebuilt as a bool mask below, independent of attention_impl
            q, k, v = attn_self._qkv(x)  # noqa: SLF001
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
            num_frames = x.shape[1] // tpf
            readout_idx = torch.arange(num_frames, device=x.device) * tpf + (tpf - 1)
            bool_mask = frame_block_causal_mask(
                num_frames, tpf, window=window, device=x.device
            )
            q_readout = q[:, :, readout_idx, :]
            scale = q_readout.shape[-1] ** -0.5
            scores = (q_readout.float() @ k.float().transpose(-2, -1)) * scale
            scores = scores.masked_fill(bool_mask[readout_idx][None, None], MASK_BIAS)
            weights = scores.softmax(dim=-1)  # (b, h, t, s)
            for name, sl in bands.items():
                local = torch.arange(sl.start, sl.stop, device=x.device)
                band_idx = (
                    torch.arange(num_frames, device=x.device)[:, None] * tpf
                    + local[None, :]
                ).flatten()
                mass[layer_idx, name].append(
                    weights[..., band_idx].sum(dim=-1).mean(dim=0).detach().cpu()
                )  # (h, t)
            attn_full = F.scaled_dot_product_attention(q, k, v, attn_mask=~bool_mask)
            return attn_self._out(attn_full)  # noqa: SLF001

        return forward


@torch.no_grad()
def run_attention_mass(  # noqa: PLR0913
    model: PatchPolicy,
    shim: _CachedImageEncoder,
    loader: Iterable[Any],
    *,
    device: torch.device,
    max_batches: int,
    bands: dict[str, slice],
    arms: dict[str, Callable[[Tensor], Tensor] | None],
) -> dict[str, dict[tuple[int, str], Tensor]]:
    """One capture pass PER ARM (baseline + identical-frame by default), each
    against its own (smaller -- `max_batches`) subset, since the manual score
    recompute materializes a `(b, h, t, s)` tensor per layer."""
    encoder = _require_causal_frame_trunk(model)
    results: dict[str, dict[tuple[int, str], Tensor]] = {}
    for arm_name, manipulate in arms.items():
        shim.manipulate = manipulate
        with _AttentionMassCapture(encoder, bands) as capture:
            for b_idx, cpu_batch in enumerate(loader):
                if b_idx >= max_batches:
                    break
                batch = _to_device(cpu_batch, device)
                shim.generation = f"attn-{arm_name}-{b_idx}"
                model._features(batch)  # noqa: SLF001 -- mass captured as a side effect
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        results[arm_name] = {
            key: torch.stack(chunks).mean(dim=0) for key, chunks in capture.mass.items()
        }
    return results


# =============================================================================
# (c), part 2: identical-frame-controlled, drive-held-out linear probe with a
# depth curve
# =============================================================================


def _trunk_input(  # noqa: PLR0914
    model: PatchPolicy, encoder: CausalFrameTransformer, batch: Any
) -> tuple[Tensor, Tensor, Tensor, Tensor, int]:
    """`(x, cos, sin, bool_mask, num_frames)` -- replicates the first few lines
    of `CausalFrameTransformer.forward` so `_layer_outputs` can walk the
    blocks one at a time without re-deriving them per depth."""
    inputs = model.input_transform(batch)
    image_by_camera = PatchPolicy._get(inputs, ("image",))  # noqa: SLF001
    speed = PatchPolicy._get(inputs, model.speed)  # noqa: SLF001
    waypoints = PatchPolicy._get(inputs, model.waypoints)  # noqa: SLF001
    images = torch.stack([image_by_camera[c] for c in model.cameras], dim=2)
    tokens = model._frame_tokens(images, speed, waypoints)  # noqa: SLF001
    b, num_frames, tpf, d = tokens.shape
    flat = tokens.reshape(b, num_frames * tpf, d)
    x = flat + encoder._intra(num_frames, flat.device)  # noqa: SLF001
    frames = torch.arange(flat.shape[1], device=flat.device) // tpf
    cos, sin = frame_rope_cos_sin(
        frames, head_dim=encoder.head_dim, base=encoder.rope_base
    )
    cos, sin = cos.to(flat.dtype), sin.to(flat.dtype)
    bool_mask = frame_block_causal_mask(
        num_frames, tpf, window=encoder.window, device=flat.device
    )
    return x, cos, sin, bool_mask, num_frames


def _run_layer_sdpa(
    layer: nn.Module,  # actually CausalFrameTransformerBlock -- see `_layer_outputs`
    h: Tensor,
    cos: Tensor,
    sin: Tensor,
    bool_mask: Tensor,
) -> Tensor:
    """One block's forward, forcing the SDPA-equivalent math (see
    `_AttentionMassCapture`'s docstring for why that substitution is exact
    within `tests/test_causal_frame.py`'s tolerance) -- avoids threading
    `BlockMask` construction through a code path that only exists for this
    diagnostic. `drop_path`/`resid_dropout` are identity in `model.eval()`.
    """
    attn = layer.attn
    q, k, v = attn._qkv(layer.attn_norm(h))  # noqa: SLF001
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    attn_out = attn._out(F.scaled_dot_product_attention(q, k, v, attn_mask=~bool_mask))  # noqa: SLF001
    # NOT `h += attn_out`: `_layer_outputs` keeps every depth's `h` (same tensor
    # object) in its `outs` list, so an in-place update here would retroactively
    # corrupt every earlier depth already appended -- silently wrong probe
    # results, not a crash. See `causal_frame.py`'s own PLR6104 warning for the
    # same class of hazard (`ruff check` runs with `unsafe-fixes = true` and
    # will rewrite this back to `+=` if it's ever simplified without the noqa).
    h = h + attn_out  # noqa: PLR6104
    return h + layer.mlp(layer.mlp_norm(h))


@torch.no_grad()
def _layer_outputs(
    encoder: CausalFrameTransformer,
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    bool_mask: Tensor,
) -> Iterable[Tensor]:
    """Yields `x0, after_layer_0, ..., after_layer_{L-1}, after_final_norm` ONE
    AT A TIME (a generator, not a list): each is a full `(b, s, d)` tensor at
    `s = num_frames * tokens_per_frame`, and holding all `num_layers + 2` of
    them alive at once (a list) roughly multiplies this pass's memory by the
    depth count for no reason -- `_collect_probe_dataset` only needs a few
    subsampled patch vectors per depth, so it extracts-and-drops as it goes.

    Yields:
        Each depth's `(b, s, d)` activation, input first, final norm last.
    """
    h = x
    yield h
    for layer in encoder.layers:
        h = _run_layer_sdpa(layer, h, cos, sin, bool_mask)
        yield h
    yield encoder.norm(h)


def _infer_drive_ids(cpu_batch: Any) -> list[str] | None:
    """Best-effort drive-id extraction for the held-out split (§4.4.3): tries
    the plausible `input_id` locations in an rbyte batch dict. Returns `None`
    (the probe then falls back to a random split, with a one-time warning) if
    none match -- this schema guess isn't load-bearing for (a)/(b), so it
    fails soft here rather than raising.
    """
    for path in (("input_id",), ("meta", "input_id"), ("data", "input_id")):
        value = key_get_default(cpu_batch, tuple(map(MappingKey, path)), None)
        if value is None:
            continue
        if hasattr(value, "tolist"):
            value = value.tolist()
        flat: list[str] = []
        for row in value:
            leaf = row
            while isinstance(leaf, list):
                leaf = leaf[0]
            flat.append(str(leaf))
        return flat
    return None


@torch.no_grad()
def _collect_probe_dataset(  # noqa: PLR0913, PLR0914
    model: PatchPolicy,
    shim: _CachedImageEncoder,
    loader: Iterable[Any],
    *,
    device: torch.device,
    max_batches: int,
    bands: dict[str, slice],
    patches_per_camera_per_frame: int,
    manipulate: Callable[[Tensor], Tensor] | None,
    generator: torch.Generator,
) -> tuple[list[list[Tensor]], list[int], list[str | None]]:
    """`(xs, y, drive)`: `xs[depth]` collects one `(dim,)` patch-token vector
    per (subsampled patch, sample); `y`/`drive` are the parallel camera-id and
    drive-id labels. Patches are subsampled per camera per frame to keep the
    fit tractable (§4.4.3)."""
    encoder = _require_causal_frame_trunk(model)
    camera_bands = [
        (i, sl)
        for i, (_name, sl) in enumerate(
            (n, s) for n, s in bands.items() if n.startswith("patch:")
        )
    ]
    depths = encoder.num_layers + 2  # input + each block + final norm
    xs: list[list[Tensor]] = [[] for _ in range(depths)]
    ys: list[int] = []
    drives: list[str | None] = []

    shim.manipulate = manipulate
    for b_idx, cpu_batch in enumerate(loader):
        if b_idx >= max_batches:
            break
        batch = _to_device(cpu_batch, device)
        shim.generation = f"probe-{b_idx}"
        drive_ids = _infer_drive_ids(cpu_batch)
        x, cos, sin, bool_mask, num_frames = _trunk_input(model, encoder, batch)
        tpf = encoder.tokens_per_frame
        b = x.shape[0]
        drive_ids = drive_ids if drive_ids is not None else [None] * b

        # index sets don't depend on depth -- build them ONCE, then walk
        # `_layer_outputs` (a generator, single-pass) with depth as the OUTER
        # loop so only one depth's full `(b, s, d)` tensor is ever alive.
        tiles: list[tuple[int, Tensor]] = []
        for cam_idx, sl in camera_bands:
            local = torch.arange(sl.start, sl.stop, device=x.device)
            k = min(patches_per_camera_per_frame, local.numel())
            picked_local = local[torch.randperm(local.numel(), generator=generator)[:k]]
            tile = (
                torch.arange(num_frames, device=x.device)[:, None] * tpf
                + picked_local[None, :]
            ).flatten()  # (t * k,)
            tiles.append((cam_idx, tile))
            n_per_sample = tile.numel()
            ys.extend([cam_idx] * (n_per_sample * b))
            for did in drive_ids:
                drives.extend([did] * n_per_sample)

        for depth, out in enumerate(_layer_outputs(encoder, x, cos, sin, bool_mask)):
            for _cam_idx, tile in tiles:
                sub = out[:, tile]  # (b, t*k, d)
                xs[depth].extend(sub.reshape(-1, sub.shape[-1]).cpu().unbind(0))
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return xs, ys, drives


def _fit_linear_probe(  # noqa: PLR0913
    x_train: Tensor,
    y_train: Tensor,
    x_test: Tensor,
    y_test: Tensor,
    *,
    num_classes: int,
    steps: int,
    lr: float,
    seed: int,
) -> float:
    """Multinomial logistic regression, no sklearn dependency: a single
    standardized `nn.Linear` trained with Adam. Returns held-out accuracy."""
    torch.manual_seed(seed)
    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    probe = nn.Linear(x_train.shape[-1], num_classes)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        F.cross_entropy(probe(x_train), y_train).backward()
        optimizer.step()
    with torch.no_grad():
        pred = probe(x_test).argmax(dim=-1)
        return (pred == y_test).float().mean().item()


def run_probe(  # noqa: PLR0913, PLR0914
    model: PatchPolicy,
    shim: _CachedImageEncoder,
    loader_factory: Callable[[], Iterable[Any]],
    *,
    device: torch.device,
    max_batches: int,
    bands: dict[str, slice],
    identical_frame: Callable[[Tensor], Tensor],
    patches_per_camera_per_frame: int,
    steps: int,
    seed: int,
) -> dict[str, dict[int, float]]:
    """Held-out accuracy per depth, for the raw (content-confounded) arm AND
    the identical-frame control (§4.4.3's "arm that actually answers (c)'s
    question"): if the identical-frame accuracy tracks the raw one, the trunk
    input is decodable from POSITION/STRUCTURE alone, not merely content.
    """
    generator = torch.Generator().manual_seed(seed)
    results: dict[str, dict[int, float]] = {}
    for arm_name, manipulate in {
        "content": None,
        "identical_frame": identical_frame,
    }.items():
        xs, ys, drives = _collect_probe_dataset(
            model,
            shim,
            loader_factory(),
            device=device,
            max_batches=max_batches,
            bands=bands,
            patches_per_camera_per_frame=patches_per_camera_per_frame,
            manipulate=manipulate,
            generator=generator,
        )
        known_drives = sorted({d for d in drives if d is not None})
        if not known_drives:
            warnings.warn(
                "_infer_drive_ids found no input_id -- falling back to a random "
                "80/20 split for the probe (NOT drive-held-out, §4.4.3's own "
                "warning against a frame-level split still applies less strictly)",
                stacklevel=2,
            )
            n = len(ys)
            perm = torch.randperm(n, generator=generator).tolist()
            cut = int(0.8 * n)
            train_idx, test_idx = set(perm[:cut]), set(perm[cut:])
        else:
            cut = max(1, int(0.8 * len(known_drives)))
            train_drives = set(known_drives[:cut])
            train_idx = {i for i, d in enumerate(drives) if d in train_drives}
            test_idx = {i for i, d in enumerate(drives) if d not in train_drives}

        y = torch.tensor(ys, dtype=torch.long)
        num_classes = int(y.max().item()) + 1
        depth_acc: dict[int, float] = {}
        for depth, vectors in enumerate(xs):
            x = torch.stack(vectors)
            tr = torch.tensor(sorted(train_idx), dtype=torch.long)
            te = torch.tensor(sorted(test_idx), dtype=torch.long)
            depth_acc[depth] = _fit_linear_probe(
                x[tr],
                y[tr],
                x[te],
                y[te],
                num_classes=num_classes,
                steps=steps,
                lr=0.05,
                seed=seed + depth,
            )
        results[arm_name] = depth_acc
    return results


# =============================================================================
# Reporting
# =============================================================================


def _report_swap(acc: _Accumulator, *, n_boot: int, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    print("\n=== (a) camera-swap sensitivity: three-arm paired design ===")  # noqa: T201
    for arm in ("B_swap", "C_duplicate"):
        for suffix, label in (("", "all-frame mean"), ("@last", "last frame")):
            rel = acc.cat(f"{arm}/readout_rel_diff{suffix}")
            mean, lo, hi = _bootstrap_mean_ci(
                rel, n_boot=n_boot, alpha=0.05, generator=generator
            )
            print(  # noqa: T201
                f"  {arm:14s} readout_rel_diff ({label:14s}) = {mean:.4f} "
                f"[{lo:.4f}, {hi:.4f}]"
            )

    ratio, lo, hi = _bootstrap_ratio_ci(
        acc.cat("B_swap/readout_rel_diff"),
        acc.cat("C_duplicate/readout_rel_diff"),
        n_boot=n_boot,
        alpha=0.05,
        generator=generator,
    )
    print(f"\n  HEADLINE  Delta(A,B)/Delta(A,C) = {ratio:.4f}  [{lo:.4f}, {hi:.4f}]")  # noqa: T201
    fires = hi < SWAP_GATE_RATIO_THRESHOLD
    print(  # noqa: T201
        f"  gate (revised, §4.5): fires if this is < {SWAP_GATE_RATIO_THRESHOLD} "
        f"-> {'FIRES' if fires else 'does not fire'}"
    )

    for arm in ("B_swap", "C_duplicate"):
        clusters = acc.per_cluster.get(f"{arm}/recon_l1")
        if not clusters:
            continue
        print(f"\n  {arm} recon_l1 @ last frame, by cluster:")  # noqa: T201
        for label, values in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
            print(  # noqa: T201
                f"    {label:16s} n={len(values):5d}  mean={sum(values) / len(values):.4f}"
            )


def _report_importance(acc: _Accumulator, *, n_boot: int, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    print("\n=== (b) per-camera importance (permutation-importance family) ===")  # noqa: T201
    baseline = acc.cat("A_baseline/recon_l1_self@last")
    for cam in ("left", "right"):
        for kind in ("permute", "copy_from_center", "mean_frame", "zero"):
            key = f"importance/{cam}/{kind}/recon_l1@last"
            if key not in acc.per_clip:
                continue
            values = acc.cat(key)
            pct = (
                (values.mean() - baseline.mean())
                / baseline.mean().clamp_min(1e-12)
                * 100
            )
            mean, lo, hi = _bootstrap_mean_ci(
                values, n_boot=n_boot, alpha=0.05, generator=generator
            )
            print(  # noqa: T201
                f"  {cam:6s} {kind:18s} recon_l1@last = {mean:.4f} [{lo:.4f}, {hi:.4f}]  "
                f"({pct:+.1f}% vs baseline)"
            )
            if kind == "permute":
                verdict = "USED (>2% relative)" if pct > 2.0 else "NOT clearly used"  # noqa: PLR2004
                print(f"    -> §4.5 precondition: side camera {verdict}")  # noqa: T201


def _report_attention(attention: dict[str, dict[tuple[int, str], Tensor]]) -> None:
    print("\n=== (c) readout attention mass onto each camera band ===")  # noqa: T201
    for arm_name, mass in attention.items():
        print(f"\n  arm: {arm_name}")  # noqa: T201
        layers = sorted({layer for layer, _name in mass})
        bands = sorted({name for _layer, name in mass})
        for layer in layers:
            row = "  ".join(
                f"{name}={mass[layer, name].mean().item():.3f}"
                for name in bands
                if (layer, name) in mass
            )
            print(f"    layer {layer}: {row}")  # noqa: T201


def _report_probe(probe: dict[str, dict[int, float]]) -> None:
    print("\n=== (c) drive-held-out linear probe: patch token -> camera id ===")  # noqa: T201
    depths = sorted(next(iter(probe.values())))
    print("  depth  " + "  ".join(f"{arm:>16s}" for arm in probe))  # noqa: T201
    for depth in depths:
        row = "  ".join(f"{probe[arm][depth]:16.4f}" for arm in probe)
        print(f"  {depth:5d}  {row}")  # noqa: T201
    print(  # noqa: T201
        "\n  the 'identical_frame' column is the one that answers (c): any "
        "accuracy there is purely positional/structural, not content-confounded"
    )


# =============================================================================
# CLI
# =============================================================================


def main() -> None:  # noqa: PLR0914, PLR0915
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--artifact",
        help="wandb model artifact, e.g. yaak/alex-tmp/model-kughoqfi:latest",
    )
    group.add_argument("--ckpt", help="local checkpoint path")
    parser.add_argument(
        "--config-dir", required=True, help="absolute path to the hydra config dir"
    )
    parser.add_argument(
        "--experiment",
        default="yaak/patch_policy/dinov2_dinowm_causal_3cam",
        help="experiment supplying the (shared) val_3cam datamodule ONLY -- "
        "model geometry comes from the checkpoint hparams (§4.4.4)",
    )
    parser.add_argument("--camera-center", default="cam_front_left")
    parser.add_argument("--camera-left", default="cam_left_forward")
    parser.add_argument("--camera-right", default="cam_right_forward")
    parser.add_argument(
        "--batches", type=int, default=50, help="(a)/(b) matrix batches"
    )
    parser.add_argument(
        "--attention-batches",
        type=int,
        default=8,
        help="(c) attention-mass capture batches",
    )
    parser.add_argument(
        "--probe-batches", type=int, default=20, help="(c) probe batches"
    )
    parser.add_argument("--probe-patches-per-camera-per-frame", type=int, default=4)
    parser.add_argument("--probe-steps", type=int, default=300)
    parser.add_argument(
        "--include-mean-frame",
        action="store_true",
        help="add the mean-frame ablation to (b) (extra pre-pass, upper bound only, §4.4.1)",
    )
    parser.add_argument(
        "--skip-attention", action="store_true", help="skip (c)'s attention-mass table"
    )
    parser.add_argument(
        "--skip-probe", action="store_true", help="skip (c)'s linear probe"
    )
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out", help="dump the combined results as JSON to this path")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="extra hydra override for the datamodule composition (repeatable), e.g. "
        "--override datamodule.val.batch_size=4 -- the val batch_size default (32, "
        "`config/datamodule/yaak/train_3cam.yaml`) times a 32-frame causal episode "
        "in fp32 (this script never autocasts, §4.4.4) is much larger than the "
        "bf16+checkpointed training shapes the causal-arm docs benchmark",
    )
    args = parser.parse_args()

    pl.seed_everything(args.seed, workers=True)
    device = torch.device(args.device)

    with initialize_config_dir(config_dir=args.config_dir, version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[f"experiment={args.experiment}", *args.override],
        )
    datamodule = instantiate(cfg.datamodule)

    model = _load_model(artifact=args.artifact, ckpt=args.ckpt)
    model = model.to(device).eval()
    _require_causal_frame_trunk(model)  # fail loudly + early, before any data pass

    center, left, right = _camera_indices(
        model, center=args.camera_center, left=args.camera_left, right=args.camera_right
    )
    bands = _slot_layout(model)

    shim = _CachedImageEncoder(model.image_encoder)
    model.image_encoder = shim

    zero_vector: Tensor | None = None
    mean_vectors: dict[int, Tensor] = {}

    print(f"\ncheckpoint: {args.artifact or args.ckpt}")  # noqa: T201
    print(f"cameras: {model.cameras} (center={center}, left={left}, right={right})")  # noqa: T201

    def fresh_loader() -> Iterable[Any]:
        pl.seed_everything(args.seed, workers=True)
        return datamodule.val_dataloader()

    if args.include_mean_frame:
        print("computing mean-frame patch vectors (one-time pre-pass)...")  # noqa: T201
        mean_vectors = _mean_patch_vectors(
            shim, model, fresh_loader(), batches=args.batches, device=device
        )

    for cpu_batch in fresh_loader():
        batch = _to_device(cpu_batch, device)
        zero_vector = _zero_patch_embedding(shim, _stacked_images(model, batch))
        break

    print("\nrunning (a)+(b) matrix...")  # noqa: T201
    acc = run_matrix(
        model,
        shim,
        fresh_loader(),
        device=device,
        max_batches=args.batches,
        center=center,
        left=left,
        right=right,
        seed=args.seed,
        zero_vector=zero_vector,
        mean_vectors=mean_vectors or None,
    )
    _report_swap(acc, n_boot=args.n_boot, seed=args.seed)
    _report_importance(acc, n_boot=args.n_boot, seed=args.seed)

    attention: dict[str, dict[tuple[int, str], Tensor]] = {}
    if not args.skip_attention:
        print("\nrunning (c) attention-mass capture...")  # noqa: T201
        attention = run_attention_mass(
            model,
            shim,
            fresh_loader(),
            device=device,
            max_batches=args.attention_batches,
            bands=bands,
            arms={
                "baseline": None,
                "identical_frame": _duplicate(
                    center, tuple(i for i in range(len(model.cameras)) if i != center)
                ),
            },
        )
        _report_attention(attention)

    probe: dict[str, dict[int, float]] = {}
    if not args.skip_probe:
        print("\nrunning (c) linear probe...")  # noqa: T201
        probe = run_probe(
            model,
            shim,
            fresh_loader,
            device=device,
            max_batches=args.probe_batches,
            bands=bands,
            identical_frame=_duplicate(
                center, tuple(i for i in range(len(model.cameras)) if i != center)
            ),
            patches_per_camera_per_frame=args.probe_patches_per_camera_per_frame,
            steps=args.probe_steps,
            seed=args.seed,
        )
        _report_probe(probe)

    if args.out:
        dump = {
            "checkpoint": args.artifact or args.ckpt,
            "swap": {k: torch.cat(v).tolist() for k, v in acc.per_clip.items()},
            "cluster": acc.per_cluster,
            "attention": {
                arm: {
                    f"{layer}:{name}": t.tolist() for (layer, name), t in mass.items()
                }
                for arm, mass in attention.items()
            },
            "probe": probe,
        }
        with open(args.out, "w", encoding="utf-8") as f:  # noqa: PTH123
            json.dump(dump, f, indent=2)
        print(f"\nwrote {args.out}")  # noqa: T201


if __name__ == "__main__":
    main()
