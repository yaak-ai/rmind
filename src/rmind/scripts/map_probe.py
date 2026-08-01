"""Counterfactual max-speed override probe for map-context PatchPolicy checkpoints.

The standing eval tool for the conditioning seam: runs IDENTICAL inputs
through a checkpoint under a sweep of `max_speed_override` values (km/h;
None = the batch input / all-UNKNOWN path) with deterministic argmax code
decoding, and reports, per override vs the None baseline:

- mean KL(softmax(logits_base) || softmax(logits_override)) over
  (sample, quantizer), plus max |logit delta|
- argmax code flip rate (fraction of (sample, quantizer) slots that change)
- decoded-action deltas: mean |delta| and mean signed delta of the
  denormalized chunk's gas / brake / steering over the action horizon

Sanity assertions built in:
- every override pair must produce DISTINCT logits (each class is live)
- override=None must be bitwise-identical to feeding an all-NaN
  `meta/MapContext/max_speed` input (missing input == all-UNKNOWN path)

Inputs come from the experiment's REAL val dataloader (needs the rbyte
cache); `--synthetic` falls back to seeded random raw-shaped tensors for
cache-less environments.

On a smoke checkpoint the magnitudes are meaningless -- the mechanism
verdict (overrides flow through, pairwise distinct, None == UNKNOWN) is the
deliverable. On a converged checkpoint the same numbers become the
counterfactual-sensitivity eval: a policy that READS the token should brake
(gas down / brake up) when the override drops.

Usage (box worktree, PYTHONPATH pointing at it):
  python -m rmind.scripts.map_probe --ckpt <model.ckpt> \
      --config-dir /abs/path/to/config \
      [--experiment yaak/patch_policy/dinov2_dinowm_maxspeed] \
      [--batches 4] [--device cuda] [--synthetic] \
      [--out diag_results/map_probe/override_probe.md]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils._pytree import tree_map  # noqa: PLC2701

from rmind.models.patch_policy import PatchPolicy

# None = UNKNOWN baseline; the rest cover WALK, city, rural, motorway, autobahn
DEFAULT_OVERRIDES: tuple[float | None, ...] = (None, 5.0, 10.0, 30.0, 50.0, 100.0, -1.0)
ACTION_NAMES = ("gas", "brake", "steer")


def override_name(v: float | None) -> str:
    if v is None:
        return "None(UNKNOWN)"
    if v < 0:
        return f"{v:g}(UNLIMITED)"
    if v <= 7:
        return f"{v:g}(WALK)"
    return f"{v:g}"


def _to_device(batch: object, device: torch.device) -> object:
    return tree_map(
        lambda x: x.to(device) if isinstance(x, Tensor) else x, batch
    )


def make_synthetic_batch(b: int = 8, t: int = 6, *, seed: int = 0) -> dict:
    """Raw-shaped deployment-style batch (no action series needed)."""
    g = torch.Generator().manual_seed(seed)
    return {
        "data": {
            "cam_front_left": torch.randint(
                0, 256, (b, t, 324, 576, 3), dtype=torch.uint8, generator=g
            ),
            "meta/VehicleMotion/speed": torch.rand((b, t, 1), generator=g) * 50.0,
            "waypoints/xy_normalized": torch.rand((b, t, 10, 2), generator=g) - 0.5,
        }
    }


@torch.no_grad()
def run_override(
    model: PatchPolicy, batch: dict, override: float | None
) -> dict[str, Tensor]:
    """Last-frame logits, argmax codes and decoded chunk under an override."""
    model.max_speed_override = override
    features, _ = model._features(batch, require_chunk=False)  # noqa: SLF001
    features = features[:, -1]
    code_logits, offsets = model._heads(features)  # noqa: SLF001
    codes = code_logits.argmax(dim=-1)
    offset = model._offset(offsets, codes)  # noqa: SLF001
    chunk = (model.tokenizer.invert(codes) + offset).unflatten(
        -1, (-1, model.tokenizer._action_features)  # noqa: SLF001
    )  # (b, horizon, action_features)
    return {
        "logits": code_logits.float().cpu(),
        "codes": codes.cpu(),
        "chunk": chunk.float().cpu(),
    }


def compare(base: dict[str, Tensor], other: dict[str, Tensor]) -> dict[str, float]:
    logp_b = base["logits"].log_softmax(dim=-1)
    logp_o = other["logits"].log_softmax(dim=-1)
    kl = F.kl_div(logp_o, logp_b, reduction="none", log_target=True).sum(-1)
    d_chunk = other["chunk"] - base["chunk"]  # (b, h, a)
    out = {
        "kl": kl.mean().item(),
        "max_dlogit": (other["logits"] - base["logits"]).abs().max().item(),
        "code_flips": (other["codes"] != base["codes"]).float().mean().item(),
    }
    for i, name in enumerate(ACTION_NAMES):
        out[f"d{name}_abs"] = d_chunk[..., i].abs().mean().item()
        out[f"d{name}_signed"] = d_chunk[..., i].mean().item()
    return out


def main() -> None:  # noqa: PLR0914, PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="local checkpoint path")
    ap.add_argument("--config-dir", required=True, help="abs path to hydra config dir")
    ap.add_argument("--experiment", default="yaak/patch_policy/dinov2_dinowm_maxspeed")
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument(
        "--overrides",
        type=float,
        nargs="*",
        default=None,
        help="override km/h values (None baseline is always included)",
    )
    ap.add_argument(
        "--synthetic",
        action="store_true",
        help="seeded random raw batches instead of the real val dataloader",
    )
    ap.add_argument(
        "--out", type=Path, default=Path("diag_results/map_probe/override_probe.md")
    )
    args = ap.parse_args()

    pl.seed_everything(args.seed, workers=True)
    device = torch.device(args.device)

    model = PatchPolicy.load_from_checkpoint(
        args.ckpt, map_location="cpu", weights_only=False
    )
    model.sample_codes = False  # deterministic argmax decoding
    if model.max_speed_tokenizer is None:
        raise SystemExit("not a map-context checkpoint (no max_speed_tokenizer)")
    vocab_size = model.max_speed_tokenizer.vocab_size
    model = model.to(device).eval()

    overrides: list[float | None] = [None]
    overrides += (
        list(args.overrides) if args.overrides else list(DEFAULT_OVERRIDES[1:])
    )

    if args.synthetic:
        batches = [
            make_synthetic_batch(seed=args.seed + i) for i in range(args.batches)
        ]
        source = f"synthetic seeded raw batches (b=8, t=6, seed={args.seed})"
    else:
        from hydra import compose, initialize_config_dir  # noqa: PLC0415
        from hydra.utils import instantiate  # noqa: PLC0415

        with initialize_config_dir(config_dir=args.config_dir, version_base=None):
            cfg = compose(
                config_name="train", overrides=[f"experiment={args.experiment}"]
            )
        datamodule = instantiate(cfg.datamodule)
        loader = datamodule.val_dataloader()
        batches = []
        for batch in loader:
            batches.append(batch)
            if len(batches) >= args.batches:
                break
        source = (
            f"real val batches from experiment={args.experiment} "
            f"(seed={args.seed}, {len(batches)} batches)"
        )

    # accumulate comparisons over batches
    sums: dict[str, dict[str, float]] = {}
    pairwise_min_dlogit: dict[tuple[str, str], float] = {}
    nan_check_max = 0.0
    n_samples = 0

    for batch in batches:
        batch = _to_device(batch, device)  # noqa: PLW2901
        outs = {override_name(v): run_override(model, batch, v) for v in overrides}
        base = outs[override_name(None)]
        n_samples += base["logits"].shape[0]

        for v in overrides[1:]:
            name = override_name(v)
            cmp_ = compare(base, outs[name])
            acc = sums.setdefault(name, dict.fromkeys(cmp_, 0.0))
            for k, val in cmp_.items():
                if k == "max_dlogit":
                    acc[k] = max(acc[k], val)
                else:
                    acc[k] += val

        names = [override_name(v) for v in overrides]
        for i, a in enumerate(names):
            for b_ in names[i + 1 :]:
                d = (outs[a]["logits"] - outs[b_]["logits"]).abs().max().item()
                key = (a, b_)
                pairwise_min_dlogit[key] = min(
                    pairwise_min_dlogit.get(key, float("inf")), d
                )

        # None == all-UNKNOWN input path (bitwise)
        speed = batch["data"]["meta/VehicleMotion/speed"]
        nan_batch = dict(batch)
        nan_batch["data"] = dict(batch["data"])
        nan_batch["data"]["meta/MapContext/max_speed"] = torch.full(
            (*speed.shape[:2], 1), float("nan"), device=device
        )
        out_nan = run_override(model, nan_batch, None)
        nan_check_max = max(
            nan_check_max,
            (base["logits"] - out_nan["logits"]).abs().max().item(),
        )

    nb = len(batches)
    rows = []
    for v in overrides[1:]:
        name = override_name(v)
        acc = sums[name]
        rows.append(
            {"override": name}
            | {
                k: (val if k == "max_dlogit" else val / nb)
                for k, val in acc.items()
            }
        )

    identical_pairs = [k for k, v in pairwise_min_dlogit.items() if v <= 0.0]
    distinct = not identical_pairs
    nan_ok = nan_check_max == 0.0

    verdict = (
        "PASS: overrides flow through (all override pairs pairwise-distinct"
        " in logits) and override=None is bitwise-identical to an all-NaN"
        " max_speed input (missing == all-UNKNOWN)."
        if distinct and nan_ok
        else f"FAIL: identical_pairs={identical_pairs}, "
        f"max|None - allNaN|={nan_check_max:g}"
    )

    def fmt(x: float) -> str:
        return f"{x:.3e}" if abs(x) < 1e-2 else f"{x:.4f}"

    cols = list(rows[0].keys())
    table = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    table += [
        "| "
        + " | ".join(r[c] if isinstance(r[c], str) else fmt(r[c]) for c in cols)
        + " |"
        for r in rows
    ]

    lines = [
        "# Counterfactual max-speed override probe",
        "",
        f"Checkpoint: `{args.ckpt}`",
        f"Inputs: {source}; {n_samples} samples/override; argmax decoding; "
        f"vocab_size={vocab_size}.",
        "",
        "All deltas vs the `None(UNKNOWN)` baseline, last-frame readout. "
        "`kl` = mean KL(base || override) per (sample, quantizer); "
        "`code_flips` = argmax code change rate; `d*` = decoded-action chunk "
        "deltas (denormalized units, mean over samples x horizon).",
        "",
        *table,
        "",
        "## Pairwise distinctness (min over batches of max |logit delta|)",
        "",
        *[
            f"- {a} vs {b}: {d:.3e}"
            for (a, b), d in sorted(pairwise_min_dlogit.items())
        ],
        "",
        f"## None == all-UNKNOWN check: max |logit delta| = {nan_check_max:g}",
        "",
        "## Verdict",
        "",
        verdict,
        "",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    print("\n".join(lines))

    if not (distinct and nan_ok):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
