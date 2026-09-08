"""Phase 1 of docs/action_tokenizer_repair_plan.md: measure the ActionTokenizer's
reconstruction ceiling on a val split. No training.

For the artifact currently wired into patch_policy (default
yaak/rmind/model-y74asdtd:v9, the `action_tokenizer_artifact` in
config/experiment/yaak/patch_policy/dinov3.yaml):

1. Per-field / per-chunk-step / per-speed-band reconstruction L1
   (encode -> quantize -> decode, compared against the tokenizer's own
   normalized target -- exactly ActionTokenizer._step's `recon` term, broken
   out instead of averaged).
2. Per-quantizer perplexity, and the per-channel decoded-distance share
   `d_q(c)` (mirrors patch_policy.py's `_neighbor_smoothing_targets`: hold the
   other quantizers at ground truth, substitute one candidate code, decode,
   and measure how much each channel moves).
3. Indicator-capacity check: how many distinct codes does quantizer 0 use per
   turn_signal state, and how much do those per-state code distributions
   overlap.

Fields are the tokenizer's own `_action_features` order: gas_pedal,
brake_pedal, steering_angle, turn_signal (config/model/yaak/action_tokenizer/raw.yaml
`targets:`).

Usage (from a repo checkout with the val rbyte cache built):
    nix develop --command uv run python scripts/action_tokenizer_ceiling.py \
        --artifact yaak/rmind/model-y74asdtd:v9 \
        --config-dir /abs/path/to/config \
        [--experiment yaak/action_tokenizer/pretrain] [--batches 200] [--device cuda]
"""

import argparse
from typing import TYPE_CHECKING

import numpy as np
import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch import Tensor
from torch.utils._pytree import tree_leaves, tree_map  # noqa: PLC2701

from rmind.models.action_tokenizer import ActionTokenizer

if TYPE_CHECKING:
    from collections.abc import Sequence

SPEED_BANDS = ((0, 5), (5, 10), (10, 20), (20, 35), (35, 60), (60, 130))
SPEED_KEY = "meta/VehicleMotion/speed"


def _to_device(batch: object, device: torch.device) -> object:
    return tree_map(
        lambda x: x.to(device, non_blocking=True) if isinstance(x, Tensor) else x, batch
    )


def _field_names(tokenizer: ActionTokenizer) -> "Sequence[str]":
    """`self.targets` leaves are the paths themselves (e.g. `(continuous,
    gas_pedal)`); `tree_leaves` order is insertion order (verified: torch
    pytree does not sort plain-dict keys), matching `_action_features`'s
    column order."""
    return [
        path[-1]
        for path in tree_leaves(tokenizer.targets, is_leaf=lambda x: isinstance(x, tuple))
    ]


@torch.no_grad()
def _neighbor_distance_shares(
    tokenizer: ActionTokenizer, target_codes: Tensor, base: Tensor, num_fields: int
) -> Tensor:
    """`d_q(c)` averaged per channel, `(num_fields,)`, mirroring
    PatchPolicy._neighbor_smoothing_targets's decomposition
    (patch_policy.py:679-729) but against the raw tokenizer's own codes."""
    quantizer = tokenizer.quantizer
    g, c = quantizer.num_quantizers, quantizer.codebook_size
    per_channel = base.new_zeros(num_fields)
    n = 0
    for q in range(g):
        for code in range(c):
            candidate = target_codes.clone()
            candidate[..., q] = code
            delta = (tokenizer.invert(candidate) - base).abs()  # (*b, action_dim)
            delta = delta.unflatten(-1, (-1, num_fields))  # (*b, step, field)
            per_channel += delta.mean(dim=tuple(range(delta.ndim - 1)))
            n += 1
    return per_channel / n


@torch.no_grad()
def evaluate(  # noqa: PLR0914
    tokenizer: ActionTokenizer,
    loader: object,
    *,
    device: torch.device,
    max_batches: int,
) -> dict[str, object]:
    fields = _field_names(tokenizer)
    nf = tokenizer._action_features  # noqa: SLF001
    action_clip = None  # inferred from the first batch

    # per-field / per-step accumulators
    step_sum: Tensor | None = None  # (step, field)
    step_count = 0

    # per-field / per-band accumulators
    band_sum: dict[str, Tensor] = {}
    band_count: dict[str, int] = {}

    # codebook-share accumulator (channel decomposition of d_q(c))
    channel_share_sum: Tensor | None = None
    channel_share_batches = 0

    # quantizer-0 usage split by turn_signal state (indicator is the last field).
    # Only meaningful for a tokenizer that still has turn_signal as a target --
    # the Phase 2a (3-feature) tokenizer drops it entirely, so this section is
    # skipped rather than mislabeling some other field's values as indicator
    # states.
    has_turn_signal = fields[-1] == "turn_signal"
    q0_size = tokenizer.quantizer.codebook_size
    q0_by_state: dict[int, Tensor] | None = (
        {s: torch.zeros(q0_size, dtype=torch.long) for s in (0, 1, 2)}
        if has_turn_signal
        else None
    )

    n = 0
    for i, cpu_batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = _to_device(cpu_batch, device)

        inputs = tokenizer.input_transform(batch)
        a = tokenizer._gather_actions(inputs)  # noqa: SLF001  (b, action_dim), normalized
        z = tokenizer.encoder(a)
        codes, z_q, _vq = tokenizer.quantizer(z)
        a_hat = tokenizer.decoder(z_q)

        err = (a_hat - a).abs().unflatten(-1, (-1, nf))  # (b, step, field)
        b, step, _ = err.shape
        action_clip = step

        if step_sum is None:
            step_sum = err.new_zeros(step, nf)
        step_sum += err.sum(dim=0)
        step_count += b

        speed = batch["data"][SPEED_KEY].reshape(b, step).float()  # (b, step) km/h
        for lo, hi in SPEED_BANDS:
            mask = (speed >= lo) & (speed < hi)  # (b, step)
            if not mask.any():
                continue
            key = f"{lo}-{hi}"
            masked_err = err[mask.unsqueeze(-1).expand_as(err)].reshape(-1, nf)
            band_sum[key] = band_sum.get(key, err.new_zeros(nf)) + masked_err.sum(dim=0)
            band_count[key] = band_count.get(key, 0) + masked_err.shape[0]

        # codebook capacity: run the (comparatively expensive) g*c decode sweep
        # on a bounded number of batches -- it is O(g*c) `invert` calls, same
        # cost class as patch_policy's per-step neighbor-smoothing target.
        if channel_share_batches < 20:  # noqa: PLR2004
            base = tokenizer.decoder(z_q)
            share = _neighbor_distance_shares(tokenizer, codes, base, nf)
            channel_share_sum = (
                share if channel_share_sum is None else channel_share_sum + share
            )
            channel_share_batches += 1

        # quantizer-0 usage per turn_signal state. `codes` is ONE code-set per
        # CHUNK (the RVQ quantizes the whole flattened window, not per step),
        # so turn_signal -- which CAN vary within the 6-step window -- is
        # reduced to a per-chunk majority vote across steps. turn_signal is
        # the last target field, scaled to {0.0, 0.5, 1.0}; round-trip to
        # {0,1,2}.
        if q0_by_state is not None:
            turn_signal_state = a[..., nf - 1 :: nf].mul(2).round().long().clamp(0, 2)
            majority = (
                torch.nn.functional.one_hot(turn_signal_state, num_classes=3)
                .sum(dim=1)
                .argmax(dim=-1)
            )  # (b,)
            q0 = codes[..., 0]  # (b,)
            for s in (0, 1, 2):
                mask = majority == s
                if mask.any():
                    q0_by_state[s] += torch.bincount(q0[mask].cpu(), minlength=q0_size)

        n += b

    if step_sum is None:
        msg = "no batches -- check --batches / dataloader"
        raise RuntimeError(msg)

    return {
        "fields": fields,
        "action_clip": action_clip,
        "step_l1": (step_sum / step_count).cpu().numpy(),  # (step, field)
        "band_l1": {
            k: (band_sum[k] / band_count[k]).cpu().numpy() for k in band_sum
        },
        "band_count": band_count,
        "channel_share": (
            None
            if channel_share_sum is None
            else (channel_share_sum / channel_share_batches).cpu().numpy()
        ),
        "q0_by_state": (
            None if q0_by_state is None else {s: v.numpy() for s, v in q0_by_state.items()}
        ),
        "num_samples": n,
    }


def _report(results: dict[str, object]) -> None:
    fields = results["fields"]
    step_l1 = results["step_l1"]

    print(f"\nval samples: {results['num_samples']}")  # noqa: T201

    print("\n[1] per-field / per-chunk-step reconstruction L1 (normalized units):")  # noqa: T201
    print(" | ".join(f"{h:>14s}" for h in ["step", *fields]))  # noqa: T201
    for step_idx, row in enumerate(step_l1):
        cells = [f"{step_idx:>14d}"] + [f"{v:14.4f}" for v in row]
        print(" | ".join(cells))  # noqa: T201
    mean_row = step_l1.mean(axis=0)
    print(" | ".join([f"{'mean':>14s}"] + [f"{v:14.4f}" for v in mean_row]))  # noqa: T201

    print("\n[1] per-field / per-speed-band reconstruction L1:")  # noqa: T201
    print(" | ".join(f"{h:>14s}" for h in ["band", "n", *fields]))  # noqa: T201
    for lo, hi in SPEED_BANDS:
        key = f"{lo}-{hi}"
        if key not in results["band_l1"]:
            continue
        row = results["band_l1"][key]
        n = results["band_count"][key]
        cells = [f"{key:>14s}", f"{n:>14d}"] + [f"{v:14.4f}" for v in row]
        print(" | ".join(cells))  # noqa: T201

    if results["channel_share"] is not None:
        share = results["channel_share"]
        pct = 100.0 * share / share.sum()
        print(  # noqa: T201
            "\n[2] per-channel share of decoded-distance d_q(c) "
            "(re-measured on this artifact; patch_policy.py:283-294 quoted "
            "38.9% turn_signal / 21.2% gas / 25.7% steering):"
        )
        for f, s, p in zip(fields, share, pct, strict=True):
            print(f"  {f:>16s}: mean d_q(c)={s:.4f}  share={p:5.1f}%")  # noqa: T201

    if results["q0_by_state"] is None:
        print(  # noqa: T201
            "\n[3] skipped -- this tokenizer has no turn_signal target "
            "(fields: " + ", ".join(fields) + ")"
        )
        return

    print("\n[3] quantizer-0 codebook usage by turn_signal state:")  # noqa: T201
    q0 = results["q0_by_state"]
    used = {s: set(np.nonzero(v)[0].tolist()) for s, v in q0.items()}
    for s, label in ((0, "OFF"), (1, "LEFT"), (2, "RIGHT")):
        total = int(q0[s].sum())
        print(  # noqa: T201
            f"  state={label:>5s} (n={total:>8d}): "
            f"{len(used[s])}/{tokenizer_codebook_size(q0)} codes used"
        )
    off_left = used[0] & used[1]
    off_right = used[0] & used[2]
    left_right = used[1] & used[2]
    print(  # noqa: T201
        f"  code overlap -- OFF∩LEFT={len(off_left)}  OFF∩RIGHT={len(off_right)}  "
        f"LEFT∩RIGHT={len(left_right)}"
    )


def tokenizer_codebook_size(q0_by_state: dict[int, np.ndarray]) -> int:
    return next(iter(q0_by_state.values())).shape[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--artifact",
        help="wandb model artifact, e.g. yaak/rmind/model-y74asdtd:v9",
    )
    group.add_argument("--ckpt", help="local checkpoint path")
    parser.add_argument(
        "--config-dir", required=True, help="absolute path to the hydra config dir"
    )
    parser.add_argument(
        "--experiment",
        default="yaak/action_tokenizer/pretrain",
        help="experiment supplying the (shared) val datamodule",
    )
    parser.add_argument("--batches", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    pl.seed_everything(args.seed, workers=True)
    device = torch.device(args.device)

    with initialize_config_dir(config_dir=args.config_dir, version_base=None):
        cfg = compose(config_name="train", overrides=[f"experiment={args.experiment}"])
    # instantiate ONLY the val branch -- `instantiate(cfg.datamodule)` would
    # also eagerly build (and cache) the much larger train dataset, which
    # Phase 1 never touches.
    val_loader = instantiate(cfg.datamodule.val)

    tokenizer = (
        ActionTokenizer.load_from_wandb_artifact(
            args.artifact, weights_only=False, map_location="cpu"
        )
        if args.artifact
        else ActionTokenizer.load_from_checkpoint(
            args.ckpt, weights_only=False, map_location="cpu"
        )
    )
    tokenizer = tokenizer.to(device).eval()

    results = evaluate(
        tokenizer,
        val_loader,
        device=device,
        max_batches=args.batches,
    )
    print(f"\nartifact: {args.artifact or args.ckpt}")  # noqa: T201
    _report(results)


if __name__ == "__main__":
    main()
