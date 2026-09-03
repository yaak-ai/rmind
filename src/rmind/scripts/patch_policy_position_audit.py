"""Phase 1(d) diagnostics for the 3-camera causal PatchPolicy camera-identity
brief: does `encoder.intra_position_embedding` (the per-slot view/position
code shared by every camera's patches) carry any signal, and does it
causally reach what the trunk outputs.

Two checkpoint-only measurements -- no val data, no image forward pass:

1. **Table audit** (`audit_table`): row norms of `intra_position_embedding`
   broken out by slot type (speed / per-camera patch band / register /
   readout), pairwise cosine similarity between camera-band mean vectors (do
   the bands look like anything, or noise around zero), and -- with `--run`
   -- the ratio against the measured `quality/token_norm/train/patch`
   content scale logged during training (`PatchPolicy._frame_tokens`).

2. **Position-effect audit** (`audit_effect`): `cos(x, x + x_pos)` and the
   relative `||f(x + x_pos) - f(x)||` through every real trained trunk
   layer. Audit 1 is a *magnitude* comparison at the input; this checks the
   actual *causal* effect on the trunk's output, especially at the readout
   token the downstream heads consume -- self-attention aggregates the
   per-token position perturbation across all patches, so a token can end up
   far from "quiet" even when its own row norm is small relative to content.
   `x` is built from every real trained weight (fusion norm/gains,
   `patch_projection`, `speed_embedding`, register/readout tokens,
   `intra_position_embedding`, all trunk layers) except the raw pre-fusion
   image/goal features, which are synthetic per-patch-independent noise --
   `fusion_patch_norm`/`fusion_goal_gain` normalize away that upstream
   distribution by construction, so this substitutes for a real ViT forward
   pass. See `_synthetic_frame_tokens` for the exact substitution and its
   caveats (independent noise, not spatially-correlated real content).

Only supports the `CausalFrameTransformer` trunk (`intra_position_embedding`
tiled per-frame + `encoder.layers[i](x, cos, sin, mask)`), i.e. the
`*_causal*.yaml` PatchPolicy arms -- not `BlockCausalTransformer`'s
window-absolute table.

Usage:
    uv run python -m rmind.scripts.patch_policy_position_audit \
        --artifact yaak/alex-tmp/model-<run_id>:latest \
        [--run yaak/alex-tmp/<run_id>] [--samples 256] [--device cuda] \
        [--out results.json]
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import pytorch_lightning as pl
import torch
from torch import Tensor, nn
from torch.nn.functional import cosine_similarity

from rmind.components.transformer.causal_frame import (
    CausalFrameTransformer,
    frame_band_slices,
    frame_block_causal_block_mask,
    frame_block_causal_mask,
    frame_rope_cos_sin,
)
from rmind.models.patch_policy import PatchPolicy


def _require_causal_frame_trunk(model: PatchPolicy) -> CausalFrameTransformer:
    encoder = model.encoder
    if not isinstance(encoder, CausalFrameTransformer):
        msg = (
            f"patch_policy_position_audit only supports CausalFrameTransformer "
            f"(the *_causal*.yaml arms), got {type(encoder).__name__} -- "
            "BlockCausalTransformer's window-absolute table has no per-slot "
            "intra_position_embedding to audit"
        )
        raise TypeError(msg)
    return encoder


def _applied_position_table(encoder: CausalFrameTransformer) -> Tensor:
    """The intra-frame position table AS THE TRUNK ACTUALLY SEES IT.

    Since `b846a4f` (`fix(causal_frame): scale-balance the intra-frame position
    embedding`) `_intra()` does not add the raw table -- it adds
    `intra_position_norm(table) * intra_position_gain`, with the LayerNorm's
    `elementwise_affine=False`. Reading `.weight` directly understates the
    perturbation the trunk receives by `sqrt(dim_model) * gain / row_norm`
    (~11x on `kughoqfi`) and reports the position signal as having got *weaker*
    when it got stronger. Both audits must go through here.

    **Delegates to the trunk** when it exposes `intra_position_applied_table`,
    so there is exactly ONE definition of "what the model adds" and this cannot
    drift from it again -- that drift is precisely the bug above. The local
    reconstruction is kept as the fallback for trunks predating that method
    (and, within it, `getattr` guards the norm/gain so pre-`b846a4f` checkpoints
    still audit -- there the raw table IS the applied table).
    """
    applied = getattr(encoder, "intra_position_applied_table", None)
    if applied is not None:
        return applied().detach()
    table = encoder.intra_position_embedding.weight.detach()
    norm = getattr(encoder, "intra_position_norm", None)
    gain = getattr(encoder, "intra_position_gain", None)
    if norm is not None:
        table = norm(table)
    if gain is not None:
        table *= gain.detach()
    return table


def _raw_position_table(encoder: CausalFrameTransformer) -> Tensor:
    """The raw composed table, reported alongside the applied one for provenance.

    Composed, not `.weight`: on a factorized arm there is no single flat weight,
    and `camera_band_cosine_centered`/`patch_row_variance` must be computed on
    the same composed quantity in every arm for flat and factorized runs to be
    comparable on identical metrics.
    """
    table = getattr(encoder, "intra_position_table", None)
    if table is not None:
        return table().detach()
    return encoder.intra_position_embedding.weight.detach()


def _num_patches_per_camera(model: PatchPolicy, encoder: CausalFrameTransformer) -> int:
    """`(tokens_per_frame - non_patch_slots) / num_grid_cameras`, validated.

    `compress_cameras` cameras contribute `model.num_camera_latents` latents
    instead of raw patches and are excluded from both the non-patch count and
    the division (see `PatchPolicy._init_camera_compression`).

    Raises:
        ValueError: if that doesn't divide evenly.
    """
    num_register = (
        model.register_tokens.shape[0] if model.register_tokens is not None else 0
    )
    num_grid_cameras = len(model.cameras) - len(model.compress_cameras)
    non_patch = (
        1
        + num_register
        + (1 if model.readout_token is not None else 0)
        + len(model.compress_cameras) * model.num_camera_latents
    )
    num_patches, remainder = divmod(
        encoder.tokens_per_frame - non_patch, num_grid_cameras
    )
    if remainder:
        msg = (
            f"tokens_per_frame {encoder.tokens_per_frame} - {non_patch} non-patch "
            f"slots doesn't divide evenly across {num_grid_cameras} grid cameras"
        )
        raise ValueError(msg)
    return num_patches


def _slot_layout(model: PatchPolicy) -> tuple[dict[str, slice], int, int]:
    """Returns `(bands, num_patches, num_register)`. See `_num_patches_per_camera`
    for the divisibility check."""
    encoder = _require_causal_frame_trunk(model)
    num_register = (
        model.register_tokens.shape[0] if model.register_tokens is not None else 0
    )
    has_readout = model.readout_token is not None
    num_patches = _num_patches_per_camera(model, encoder)
    bands = frame_band_slices(
        cameras=model.cameras,
        num_patches=num_patches,
        num_register=num_register,
        has_readout=has_readout,
        compress_cameras=model.compress_cameras,
        num_camera_latents=model.num_camera_latents,
    )
    return bands, num_patches, num_register


@torch.no_grad()
def audit_table(  # noqa: PLR0914
    model: PatchPolicy, *, run: str | None = None
) -> dict[str, Any]:
    """Row norms and camera-band separation of `intra_position_embedding`,
    checkpoint only -- no data, no forward pass."""
    encoder = _require_causal_frame_trunk(model)
    bands, num_patches, _num_register = _slot_layout(model)

    # the APPLIED table -- what `_intra()` adds to the content tokens. The raw
    # `.weight` is reported alongside it for provenance only; every ratio and
    # cosine below is on the applied one (see `_applied_position_table`).
    table = _applied_position_table(encoder).cpu()
    raw_table = _raw_position_table(encoder).cpu()
    row_norms = table.norm(dim=-1)
    gain = getattr(encoder, "intra_position_gain", None)

    result: dict[str, Any] = {
        "tokens_per_frame": encoder.tokens_per_frame,
        "dim_model": encoder.dim_model,
        "cameras": list(model.cameras),
        "num_patches_per_camera": num_patches,
        "scale_balanced": gain is not None,
        "intra_position_gain": None if gain is None else float(gain),
        "row_norm_mean": row_norms.mean().item(),
        "row_norm_std": row_norms.std().item(),
        "raw_row_norm_mean": raw_table.norm(dim=-1).mean().item(),
        "raw_frobenius_norm": raw_table.norm().item(),
        "bands": {},
    }

    camera_bands = [name for name in bands if name.startswith("patch:")]
    band_means: dict[str, Tensor] = {}
    for name, sl in bands.items():
        rows = table[sl]
        band_means[name] = rows.mean(dim=0)
        result["bands"][name] = {
            "row_norm_mean": rows.norm(dim=-1).mean().item(),
            # population std (not sample): a 1-row band (speed/readout) is
            # legitimate, and unbiased std() on n=1 is NaN with a warning
            "row_norm_std": rows.norm(dim=-1).std(unbiased=False).item(),
            "mean_vector_norm": band_means[name].norm().item(),
        }

    # UNCENTERED cosines: kept for continuity with the earlier numbers, but they
    # are dominated by the slot-common component every band shares (which carries
    # no camera information), so they systematically understate the separation.
    result["camera_band_cosine"] = {
        f"{a}:{b}": cosine_similarity(band_means[a], band_means[b], dim=0).item()
        for i, a in enumerate(camera_bands)
        for b in camera_bands[i + 1 :]
    }

    # CENTERED: subtract the grand mean over ALL patch rows first, isolating the
    # per-camera component. This is the number that answers "does the table
    # encode camera identity"; the gate's `cosine > ~0.9` criterion (bands
    # COLLAPSING onto each other) is meant to be read off this one.
    patch_slice = slice(1, 1 + num_patches * len(model.cameras))
    grand_mean = table[patch_slice].mean(dim=0)
    identity = {name: band_means[name] - grand_mean for name in camera_bands}
    result["camera_band_cosine_centered"] = {
        f"{a}:{b}": cosine_similarity(identity[a], identity[b], dim=0).item()
        for i, a in enumerate(camera_bands)
        for b in camera_bands[i + 1 :]
    }
    result["camera_identity_norm"] = {
        name: vector.norm().item() for name, vector in identity.items()
    }
    # how much of the position table's variance is BETWEEN cameras vs WITHIN a
    # camera (per-patch spatial code). A large within/between ratio means the
    # uniform amplitude lift raised both, not camera identity preferentially.
    between = torch.stack([identity[name] for name in camera_bands]).pow(2).sum(-1)
    # NOTE: index `table`, not `table[patch_slice]` -- `bands[name]` are absolute
    # slot slices, so applying them to the already-patch-sliced view shifted every
    # camera's rows by one slot (the speed token) and mixed the last camera's band
    # with out-of-band rows
    within = torch.stack([
        (table[bands[name]] - band_means[name]).pow(2).sum(-1).mean()
        for name in camera_bands
    ])
    result["patch_row_variance"] = {
        "between_camera": between.mean().item(),
        "within_camera": within.mean().item(),
        "between_fraction": (between.mean() / (between.mean() + within.mean())).item(),
    }

    # The factorization arm, read off the trunk rather than re-derived, plus --
    # when the arm HAS a view embedding -- a DIRECT read of it. On a flat table
    # "is camera identity a rank-1 direction" can only be estimated (that is what
    # `camera_band_cosine_centered` above does); here it is a parameter, so its
    # row norms and pairwise cosines are the quantity itself, not a proxy.
    # Everything above stays computed on the COMPOSED table so flat and
    # factorized arms remain comparable on identical metrics.
    result["intra_position"] = {
        "scaling": getattr(encoder, "intra_position_scaling", "norm_gain"),
        "factorization": getattr(encoder, "intra_position_factorization", "flat"),
        "target_norm": getattr(encoder, "intra_position_target_norm", None),
        "patch_grid": getattr(encoder, "patch_grid", None),
        "camera_yaw_deg": getattr(encoder, "camera_yaw_deg", None),
        "panorama_camera_order": getattr(encoder, "panorama_camera_order", None),
    }
    view = getattr(encoder, "view_position_embedding", None)
    if view is not None:
        view_weight = view.weight.detach().cpu()
        names = list(model.cameras)
        result["view_position_embedding"] = {
            "row_norm": {
                name: view_weight[i].norm().item() for i, name in enumerate(names)
            },
            "cosine": {
                f"{names[i]}:{names[j]}": cosine_similarity(
                    view_weight[i], view_weight[j], dim=0
                ).item()
                for i in range(len(names))
                for j in range(i + 1, len(names))
            },
        }

    if run is not None:
        result["run"] = run
        result["content_scale"] = _fetch_content_scale(
            run, table_row_norm_mean=result["row_norm_mean"]
        )

    return result


def _fetch_content_scale(
    run: str, *, table_row_norm_mean: float
) -> dict[str, Any] | None:
    """Last logged `quality/token_norm/train/{patch,readout}` from the run's own
    wandb history -- the measured content scale to compare the table against,
    with no forward pass needed."""
    import wandb  # noqa: PLC0415

    api = wandb.Api()
    history = api.run(run).history(
        keys=["quality/token_norm/train/patch", "quality/token_norm/train/readout"],
        samples=10_000,
        pandas=False,
    )
    rows = [row for row in history if "quality/token_norm/train/patch" in row]
    if not rows:
        return None
    last = rows[-1]
    patch_norm = last["quality/token_norm/train/patch"]
    return {
        "patch_token_norm": patch_norm,
        "readout_token_norm": last.get("quality/token_norm/train/readout"),
        "patch_over_table_row_ratio": patch_norm / table_row_norm_mean,
    }


def _synthetic_patch_content(
    model: PatchPolicy, *, num_tokens: int, samples: int, device: torch.device
) -> Tensor:
    """`(samples, num_tokens, dim_model)` through the REAL trained
    `fusion_patch_norm`/`fusion_patch_gain`/`fusion_goal_gain`/`patch_projection`.
    Only the pre-fusion "raw ViT features"/"raw goal features" are synthetic
    (arbitrarily-scaled noise) -- `fusion_patch_norm` renormalizes them to
    ~unit RMS regardless of their upstream distribution, and `fusion_goal_gain`
    is calibrated the same way, so this substitutes for a real ViT forward pass.

    Raises:
        ValueError: if the checkpoint wasn't trained with `fusion_norm=True` --
            without that calibration there's no principled scale to synthesize.
    """
    fusion_patch_norm = model.fusion_patch_norm
    fusion_patch_gain = model.fusion_patch_gain
    fusion_goal_gain = model.fusion_goal_gain
    if (
        fusion_patch_norm is None
        or fusion_patch_gain is None
        or fusion_goal_gain is None
        or not isinstance(fusion_patch_norm, nn.LayerNorm)
    ):
        msg = (
            "patch_policy_position_audit requires a checkpoint trained with "
            "fusion_norm=True (fusion_patch_norm/fusion_patch_gain/"
            "fusion_goal_gain), got fusion_patch_norm="
            f"{type(fusion_patch_norm).__name__}"
        )
        raise ValueError(msg)

    patch_dim = fusion_patch_norm.normalized_shape[0]
    goal_dim = model.patch_projection.in_features - patch_dim
    raw_vit = torch.randn(samples, num_tokens, patch_dim, device=device) * 5.0 + 0.7
    patches = fusion_patch_norm(raw_vit) * fusion_patch_gain
    raw_goal = torch.randn(samples, 1, goal_dim, device=device)
    goal = (raw_goal * fusion_goal_gain).expand(-1, num_tokens, -1)
    return model.patch_projection(torch.cat([patches, goal], dim=-1))


def _synthetic_frame_tokens(
    model: PatchPolicy,
    encoder: CausalFrameTransformer,
    *,
    samples: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Builds `(x0, x1)`, both `(samples, tokens_per_frame, dim_model)`: `x0` is
    trunk-input content with every real trained weight EXCEPT the raw
    pre-fusion image/goal features (see `_synthetic_patch_content`); `x1 = x0 +
    intra_position_embedding`, i.e. what the model actually feeds the trunk.
    """
    dim_model = encoder.dim_model
    num_cameras = len(model.cameras)
    num_patches = _num_patches_per_camera(model, encoder)
    content_patches = _synthetic_patch_content(
        model, num_tokens=num_cameras * num_patches, samples=samples, device=device
    )

    speed_bin = torch.full(
        (samples, 1),
        int(model.speed_tokenizer.bins.item()) // 2,
        dtype=torch.long,
        device=device,
    )
    parts = [model.speed_embedding(speed_bin), content_patches]
    if model.register_tokens is not None:
        parts.append(
            model.register_tokens.reshape(1, -1, dim_model).expand(samples, -1, -1)
        )
    if model.readout_token is not None:
        parts.append(
            model.readout_token.reshape(1, 1, dim_model).expand(samples, -1, -1)
        )
    x0 = torch.cat(parts, dim=1)

    # must match `_intra()` exactly -- the raw embedding understates the real
    # perturbation ~11x post-`b846a4f` (see `_applied_position_table`)
    x1 = x0 + _applied_position_table(encoder).to(device=device, dtype=x0.dtype)
    return x0, x1


@torch.no_grad()
def audit_effect(
    model: PatchPolicy, *, samples: int = 256, device: torch.device, seed: int = 1337
) -> dict[str, Any]:
    """`cos(x, x+x_pos)` at the trunk input, then `||f(x+x_pos)-f(x)||` after
    every real trained trunk layer -- the causal counterpart to `audit_table`.
    """
    encoder = _require_causal_frame_trunk(model)
    bands, _num_patches, _num_register = _slot_layout(model)

    torch.manual_seed(seed)
    x0, x1 = _synthetic_frame_tokens(model, encoder, samples=samples, device=device)

    def band_stats(a: Tensor, b: Tensor) -> dict[str, dict[str, float]]:
        diff_norm = (b - a).norm(dim=-1)
        base_norm = a.norm(dim=-1)
        cos = cosine_similarity(a, b, dim=-1)
        return {
            name: {
                "cos": cos[:, sl].mean().item(),
                "rel_diff": (diff_norm[:, sl] / base_norm[:, sl].clamp_min(1e-8))
                .mean()
                .item(),
            }
            for name, sl in bands.items()
        }

    result: dict[str, Any] = {"input": band_stats(x0, x1), "layers": []}

    tokens_per_frame = encoder.tokens_per_frame
    frames = torch.zeros(
        tokens_per_frame, dtype=torch.long, device=device
    )  # single frame
    rope_cos, rope_sin = frame_rope_cos_sin(
        frames, head_dim=encoder.head_dim, base=encoder.rope_base
    )
    rope_cos, rope_sin = rope_cos.to(x0.dtype), rope_sin.to(x0.dtype)
    mask = (
        frame_block_causal_block_mask(
            1, tokens_per_frame, window=encoder.window, device=device
        )
        if encoder.attention_impl == "flex"
        else frame_block_causal_mask(
            1, tokens_per_frame, window=encoder.window, device=device
        )
    )

    h0, h1 = x0, x1
    for layer in encoder.layers:
        h0 = layer(h0, rope_cos, rope_sin, mask)
        h1 = layer(h1, rope_cos, rope_sin, mask)
        result["layers"].append(band_stats(h0, h1))

    h0, h1 = encoder.norm(h0), encoder.norm(h1)
    result["final_norm"] = band_stats(h0, h1)
    return result


def _report_table(result: dict[str, Any]) -> None:
    print(  # noqa: T201
        f"\n=== table audit: {result['tokens_per_frame']} slots x "
        f"{result['dim_model']} dim, cameras={result['cameras']} ==="
    )
    if result["scale_balanced"]:
        print(  # noqa: T201
            f"  scale-balanced (b846a4f): intra_position_gain = "
            f"{result['intra_position_gain']:.6f}; norms below are the APPLIED "
            f"table (raw row_norm mean = {result['raw_row_norm_mean']:.4f}, "
            f"raw frobenius = {result['raw_frobenius_norm']:.2f})"
        )
    else:
        print("  pre-b846a4f checkpoint: no norm/gain, raw table = applied table")  # noqa: T201
    for name, stats in result["bands"].items():
        print(  # noqa: T201
            f"  {name:24s} row_norm mean={stats['row_norm_mean']:.4f} "
            f"std={stats['row_norm_std']:.4f}  mean_vector_norm={stats['mean_vector_norm']:.4f}"
        )
    print("camera-band mean-vector cosine similarity (uncentered / CENTERED):")  # noqa: T201
    for pair, value in result["camera_band_cosine"].items():
        centered = result["camera_band_cosine_centered"][pair]
        print(f"  {pair}: {value:+.4f} / {centered:+.4f}")  # noqa: T201
    print("per-camera identity vector norm (band mean - grand patch mean):")  # noqa: T201
    for name, value in result["camera_identity_norm"].items():
        print(f"  {name}: {value:.4f}")  # noqa: T201
    variance = result["patch_row_variance"]
    print(  # noqa: T201
        f"patch-row variance: between-camera={variance['between_camera']:.3f} "
        f"within-camera={variance['within_camera']:.3f} "
        f"(between = {100 * variance['between_fraction']:.1f}%)"
    )
    if scale := result.get("content_scale"):
        print(f"vs measured content scale ({result['run']}):")  # noqa: T201
        print(f"  quality/token_norm/train/patch  = {scale['patch_token_norm']:.4f}")  # noqa: T201
        readout_norm = scale["readout_token_norm"]
        readout_str = f"{readout_norm:.4f}" if readout_norm is not None else "n/a"
        print(f"  quality/token_norm/train/readout = {readout_str}")  # noqa: T201
        print(f"  patch / table-row ratio = {scale['patch_over_table_row_ratio']:.2f}x")  # noqa: T201


def _report_effect(result: dict[str, Any]) -> None:
    print("\n=== position-effect audit: cos(x, x+x_pos) at trunk input ===")  # noqa: T201
    for name, stats in result["input"].items():
        print(f"  {name:24s} cos={stats['cos']:.4f}")  # noqa: T201

    print("\n=== ||f(x+x_pos)-f(x)|| / ||f(x)||, per layer ===")  # noqa: T201
    for i, layer_stats in enumerate(result["layers"]):
        row = " | ".join(
            f"{name}: rel={stats['rel_diff']:.4f} cos={stats['cos']:.4f}"
            for name, stats in layer_stats.items()
        )
        print(f"  layer {i}: {row}")  # noqa: T201

    print("\n=== after final encoder.norm (what code_head/offset_head consume) ===")  # noqa: T201
    for name, stats in result["final_norm"].items():
        print(f"  {name:24s} rel_diff={stats['rel_diff']:.4f} cos={stats['cos']:.4f}")  # noqa: T201


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--artifact", help="wandb model artifact, e.g. yaak/alex-tmp/model-<id>:latest"
    )
    group.add_argument("--ckpt", help="local checkpoint path")
    parser.add_argument(
        "--run",
        help="wandb run path (e.g. yaak/alex-tmp/<run_id>) to pull "
        "quality/token_norm/train/{patch,readout} history from -- optional",
    )
    parser.add_argument(
        "--samples", type=int, default=256, help="synthetic frames for the effect audit"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--skip-effect", action="store_true", help="table audit only")
    parser.add_argument("--out", help="dump the combined results as JSON to this path")
    args = parser.parse_args()

    pl.seed_everything(args.seed, workers=True)
    device = torch.device(args.device)

    model = (
        PatchPolicy.load_from_wandb_artifact(
            args.artifact, weights_only=False, map_location="cpu"
        )
        if args.artifact
        else PatchPolicy.load_from_checkpoint(
            args.ckpt, weights_only=False, map_location="cpu"
        )
    )
    model = model.to(device).eval()

    results: dict[str, Any] = {
        "checkpoint": args.artifact or args.ckpt,
        "table": audit_table(model, run=args.run),
    }
    _report_table(results["table"])

    if not args.skip_effect:
        results["effect"] = audit_effect(
            model, samples=args.samples, device=device, seed=args.seed
        )
        _report_effect(results["effect"])

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:  # noqa: PTH123
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.out}")  # noqa: T201


if __name__ == "__main__":
    main()
