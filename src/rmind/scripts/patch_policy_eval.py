"""Post-hoc per-frame-position evaluation of a PatchPolicy checkpoint.

The training/val losses logged by `PatchPolicy` average over all T frame
readouts, whose block-causal contexts span 1..T frames; `JointPolicyObjective`
(the FT baseline) only ever scores the newest frame. This script evaluates a
checkpoint on the val set and reports code/offset/sampled-recon losses PER
FRAME POSITION, plus the all-frames mean (what wandb shows) and the last-frame
value (comparable to the baseline).

Usage (from a repo checkout with the val rbyte cache built):
    uv run python -m rmind.scripts.patch_policy_eval \
        --artifact yaak/rmind/model-<run_id>:latest \
        --config-dir /abs/path/to/config \
        [--experiment yaak/patch_policy/dinov3] [--batches 200] [--device cuda]
"""

import argparse
import json
import re
from typing import TYPE_CHECKING

import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import Tensor
from torch.nn import functional as F
from torch.utils._pytree import tree_map  # noqa: PLC2701

from rmind.models.patch_policy import PatchPolicy

if TYPE_CHECKING:
    from rmind.utils import RuleBasedCluster

# how many leading batches also go through `_compute_metrics`, so this script's
# own reduction can be checked against exactly what training logs
_PARITY_BATCHES = 2


def _to_device(batch: object, device: torch.device) -> object:
    return tree_map(
        lambda x: x.to(device, non_blocking=True) if isinstance(x, Tensor) else x, batch
    )


def _focal_terms(
    logits: Tensor, targets: Tensor, *, gamma: float, label_smoothing: float
) -> Tensor:
    """Element-wise `rmind.components.loss.FocalLoss` (its `.mean()` is the loss).

    Element-wise so the same forward pass can be reduced per frame position, per
    quantizer and per window bucket, and so the loss can be reported BOTH at the
    checkpoint's own `label_smoothing` (comparable to the run's train curves) and
    at 0.0 (comparable to the pre-smoothing arms). Parity with
    `model.losses["code"]` is asserted against `_compute_metrics`.
    """
    ce = F.cross_entropy(
        logits.flatten(0, -2), targets.flatten(), reduction="none"
    ).reshape(targets.shape)
    pt = torch.exp(-ce)
    focal = (1 - pt).pow(gamma) * ce
    if not label_smoothing:
        return focal
    ce_uniform = -F.log_softmax(logits, dim=-1).mean(dim=-1)
    return (1 - label_smoothing) * focal + label_smoothing * ce_uniform


def _default_cluster_fn() -> "RuleBasedCluster":
    """The exact rule set from config/trainer/callbacks/patch_policy.yaml, so the
    per-cluster numbers here are comparable with the training-time predict metrics
    (which, however, were logged under SAMPLED decoding only)."""
    from rmind.utils import RuleBasedCluster  # noqa: PLC0415

    key = "meta/VehicleMotion/{}"
    return RuleBasedCluster(
        fields={
            "gas": {"key": key.format("gas_pedal_normalized"), "reduce": "last"},
            "brake": {"key": key.format("brake_pedal_normalized"), "reduce": "last"},
            "steer": {"key": key.format("steering_angle_normalized"), "reduce": "last"},
            "dp_g": {"key": key.format("gas_pedal_normalized"), "reduce": "last_diff"},
            "speed": {"key": key.format("speed"), "reduce": "last"},
        },
        rules=[
            {"name": "highway", "when": {"speed": {"ge": 70}}},
            {
                "name": "braking_turn",
                "when": {"brake": {"ge": 0.02}, "steer": {"abs_ge": 0.05}},
            },
            {
                "name": "braking",
                "when": {"brake": {"ge": 0.02}, "steer": {"abs_lt": 0.05}},
            },
            {
                "name": "cruise_turn",
                "when": {
                    "gas": {"ge": 0.05},
                    "brake": {"lt": 0.02},
                    "steer": {"abs_ge": 0.05},
                },
            },
            {
                "name": "acceleration",
                "when": {
                    "gas": {"ge": 0.05},
                    "brake": {"lt": 0.02},
                    "dp_g": {"ge": 0.01},
                },
            },
            {
                "name": "gas_release",
                "when": {
                    "gas": {"ge": 0.05},
                    "brake": {"lt": 0.02},
                    "dp_g": {"lt": -0.01},
                },
            },
            {
                "name": "cruise",
                "when": {
                    "gas": {"ge": 0.05},
                    "brake": {"lt": 0.02},
                    "steer": {"abs_lt": 0.05},
                    "dp_g": {"abs_lt": 0.01},
                },
            },
        ],
        default="idle_coast",
    )


@torch.no_grad()
def evaluate(  # noqa: C901, PLR0913, PLR0914, PLR0915, PLR0912
    model: PatchPolicy,
    loader: object,
    *,
    device: torch.device,
    max_batches: int,
    autocast: bool,
    num_readouts: int,
) -> dict[str, Tensor]:
    tokenizer = model.tokenizer
    num_quantizers = tokenizer.quantizer.num_quantizers
    gamma = model.losses["code"].gamma
    eps = model.losses["code"].label_smoothing

    sums: dict[str, Tensor] = {}
    count = 0
    # per-sample reconstruction L1 (H8 tails): kept per sample so p95/p99 are
    # exact rather than a mean of batch means
    tails: dict[str, list[Tensor]] = {}
    # {batch index: what `_compute_metrics` -- i.e. training -- would have logged}
    reference: dict[int, dict[str, float]] = {}

    for i, cpu_batch in enumerate(loader):
        if i >= max_batches:
            break

        batch = _to_device(cpu_batch, device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=autocast):
            if i < _PARITY_BATCHES:
                # inside the autocast block on purpose: training ran
                # `precision: bf16-mixed`, and a fp32 reference would differ from
                # the bf16 reconstruction at ~1e-2 on a gamma=2 focal, which is
                # large enough to mask a real reduction bug
                ref = model._compute_metrics(batch)["policy"]  # noqa: SLF001
                reference[i] = {
                    f"{group}/{key}": float(value)
                    for group in ("loss", "metric")
                    for key, value in ref[group].items()
                }
            features, chunk = model._features(batch)  # noqa: SLF001
            target_codes = tokenizer(chunk)  # (b, t, g)
            target = tokenizer._normalize(chunk.flatten(-2, -1))  # noqa: SLF001
            code_logits, offsets = model._heads(features)  # noqa: SLF001

            teacher_chunk = tokenizer.invert(target_codes) + model._offset(  # noqa: SLF001
                offsets, target_codes
            )
            # both decodings from the SAME logits: one multinomial draw (the
            # inference path with sample_codes=true) and argmax (deterministic)
            sampled_codes = model._sample_codes(code_logits)  # noqa: SLF001
            sampled_chunk = tokenizer.invert(sampled_codes) + model._offset(  # noqa: SLF001
                offsets, sampled_codes
            )
            argmax_codes = code_logits.argmax(dim=-1)
            argmax_chunk = tokenizer.invert(argmax_codes) + model._offset(  # noqa: SLF001
                offsets, argmax_codes
            )

        # per-quantizer code diagnostics: top-1 accuracy, probability mass on
        # the GT code, and sampling entropy (nats; uniform over 16 = 2.77)
        probs = code_logits.float().softmax(dim=-1)  # (b, t, g, c)
        p_gt = probs.gather(-1, target_codes[..., None]).squeeze(-1)  # (b, t, g)
        entropy = -(probs * probs.clamp_min(1e-10).log()).sum(dim=-1)  # (b, t, g)
        accuracy = (argmax_codes == target_codes).float()  # (b, t, g)
        # joint: all quantizer levels correct simultaneously ("exact behavior token")
        joint_accuracy = (argmax_codes == target_codes).all(dim=-1).float()  # (b, t)

        logits = code_logits.float()
        focal = {
            "code": _focal_terms(
                logits, target_codes, gamma=gamma, label_smoothing=eps
            ),
            "codeplain": _focal_terms(
                logits, target_codes, gamma=gamma, label_smoothing=0.0
            ),
        }  # (b, t, g) each
        # per-readout L1 over the whole action chunk, (b, t)
        l1 = {
            "offset": (teacher_chunk - target).abs().mean(dim=-1),
            "sampled_recon": (sampled_chunk - target).abs().mean(dim=-1),
            "argmax_recon": (argmax_chunk - target).abs().mean(dim=-1),
        }

        b, t = features.shape[:2]
        # the window bucket boundary is `window - 1` READOUTS in, so a t that is
        # not episode_length would silently move it
        assert t == num_readouts, (t, num_readouts)  # noqa: S101
        for pos in range(t):
            values: dict[str, Tensor] = {}
            for q in range(num_quantizers):
                for stem, terms in focal.items():
                    values[f"t{pos}/{stem}_{q}"] = terms[:, pos, q].mean()
                values[f"t{pos}/acc_{q}"] = accuracy[:, pos, q].mean()
                values[f"t{pos}/p_gt_{q}"] = p_gt[:, pos, q].mean()
                values[f"t{pos}/entropy_{q}"] = entropy[:, pos, q].mean()
            values[f"t{pos}/acc_joint"] = joint_accuracy[:, pos].mean()
            for name, per_readout in l1.items():
                values[f"t{pos}/{name}"] = per_readout[:, pos].mean()
            for key, value in values.items():
                sums[key] = sums.get(key, torch.zeros((), device=device)) + value * b

        # H8 tails: every sample's own recon L1, at the deployment position
        # (last frame) and averaged over the clip's readouts
        for name, per_readout in l1.items():
            tails.setdefault(f"{name}/last", []).append(
                per_readout[:, -1].float().cpu()
            )
            tails.setdefault(f"{name}/all", []).append(
                per_readout.mean(dim=-1).float().cpu()
            )

        # per-cluster x per-decoding, last frame only (deployment position),
        # per-field L1 mean over the action horizon. Skipped when the batch
        # lacks the raw meta series the cluster rules need (synthetic batches).
        try:
            labels = _default_cluster_fn()(batch, None)
        except (KeyError, TypeError):
            labels = None
        if labels is not None:
            action_features = tokenizer._action_features  # noqa: SLF001
            tgt = target[:, -1].unflatten(-1, (-1, action_features))
            per_field = {
                "sampled": (
                    sampled_chunk[:, -1].unflatten(-1, (-1, action_features)) - tgt
                )
                .abs()
                .mean(dim=1),  # (b, fields)
                "argmax": (
                    argmax_chunk[:, -1].unflatten(-1, (-1, action_features)) - tgt
                )
                .abs()
                .mean(dim=1),
            }
            for j, label in enumerate(labels):
                key = f"cluster_n/{label}"
                sums[key] = sums.get(key, torch.zeros((), device=device)) + 1
                for dec, l1 in per_field.items():
                    for f, field in enumerate(("gas", "brake", "steer")):
                        k = f"cluster/{label}/{dec}_{field}"
                        sums[k] = sums.get(k, torch.zeros((), device=device)) + l1[j, f]

        count += b

    # cluster sums are normalized by their own cluster counts at report time
    return (
        {
            k: (v if k.startswith(("cluster/", "cluster_n/")) else v / count)
            for k, v in sums.items()
        }
        | {f"tail/{k}": torch.cat(v) for k, v in tails.items()}
        | {"num_samples": torch.tensor(count)}
        | {
            f"ref{i}/{k}": torch.tensor(v)
            for i, ref in reference.items()
            for k, v in ref.items()
        }
    )


def main() -> None:  # noqa: C901, PLR0912, PLR0914, PLR0915
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--artifact", help="wandb model artifact, e.g. yaak/rmind/model-<id>:latest"
    )
    group.add_argument("--ckpt", help="local checkpoint path")
    parser.add_argument(
        "--config-dir", required=True, help="absolute path to the hydra config dir"
    )
    parser.add_argument(
        "--experiment",
        default="yaak/patch_policy/dinov3",
        help="experiment supplying the (shared) val datamodule",
    )
    parser.add_argument("--batches", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="override the val dataloader worker count",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="override val batch size"
    )
    parser.add_argument(
        "--cache-dir", default=None, help="absolute rbyte cache dir (cwd-independent)"
    )
    parser.add_argument(
        "--unshared-cache",
        action="store_true",
        help=(
            "pipefunc DiskCache lru_shared=false, so the in-memory LRU in front "
            "of the disk cache does not create a multiprocessing.Manager"
        ),
    )
    parser.add_argument(
        "--serial-build",
        action="store_true",
        help=(
            "build the rbyte sample index in-process. The configured forkserver "
            "ProcessPoolExecutor dies with BrokenProcessPool on the val drives; "
            "in-process takes ~40s for all 5. Implies --unshared-cache."
        ),
    )
    parser.add_argument("--json-out", default=None, help="write the raw numbers here")
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="fixes the shuffled val subset across invocations",
    )
    args = parser.parse_args()

    pl.seed_everything(args.seed, workers=True)
    device = torch.device(args.device)

    with initialize_config_dir(config_dir=args.config_dir, version_base=None):
        cfg = compose(config_name="train", overrides=[f"experiment={args.experiment}"])
    OmegaConf.set_struct(cfg, False)  # noqa: FBT003
    if args.cache_dir is not None:
        cfg.paths.rbyte.cache = args.cache_dir
    if args.num_workers is not None:
        cfg.datamodule.val.num_workers = args.num_workers
    if args.batch_size is not None:
        cfg.datamodule.val.batch_size = args.batch_size
    # only the VAL loader is needed: instantiating the datamodule wholesale also
    # builds the TRAIN sample index (~2M clip37 samples, hours) for nothing
    cfg.datamodule.train = None
    samples = cfg.datamodule.val.dataset.samples
    if args.unshared_cache or args.serial_build:
        samples.pipeline.cache_kwargs.lru_shared = False
    if args.serial_build:
        samples.executor = None
        # rbyte's config model is extra="allow", so these reach Pipeline.map()
        samples.parallel = False
        samples.scheduling_strategy = "generation"
    datamodule = instantiate(cfg.datamodule)

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

    results = evaluate(
        model,
        datamodule.val_dataloader(),
        device=device,
        max_batches=args.batches,
        autocast=device.type == "cuda",
        num_readouts=int(cfg.episode_length),
    )

    num_quantizers = model.tokenizer.quantizer.num_quantizers
    positions = sorted({
        int(m.group(1)) for k in results if (m := re.fullmatch(r"t(\d+)/.+", k))
    })

    def mean_over(prefixes: list[str], name: str) -> float:
        return torch.stack([results[f"{p}/{name}"] for p in prefixes]).mean().item()

    def q_mean(prefixes: list[str], stem: str) -> float:
        return (
            sum(mean_over(prefixes, f"{stem}_{q}") for q in range(num_quantizers))
            / num_quantizers
        )

    print(f"\ncheckpoint: {args.artifact or args.ckpt}")  # noqa: T201
    print(f"val samples: {int(results['num_samples'])}\n")  # noqa: T201

    eps = model.losses["code"].label_smoothing
    print(f"code focal label_smoothing={eps}; code_plain is the same focal at 0.0")  # noqa: T201
    print(f"sample_codes={bool(model.sample_codes)} (recon_sampled decoding)\n")  # noqa: T201
    header = [
        "pos(context)",
        "code_focal",
        "code_plain",
        "top1_acc",  # mean of per-quantizer marginal accuracies
        "joint_acc",  # all 4 levels correct simultaneously
        "p_gt",
        "entropy",
        "offset",
        "recon_sampled",
        "recon_argmax",
    ]
    print(" | ".join(f"{h:>13s}" for h in header))  # noqa: T201

    def cells_for(prefixes: list[str]) -> dict[str, float]:
        return {
            "code_focal": q_mean(prefixes, "code"),
            "code_plain": q_mean(prefixes, "codeplain"),
            "top1_acc": q_mean(prefixes, "acc"),
            "joint_acc": mean_over(prefixes, "acc_joint"),
            "p_gt": q_mean(prefixes, "p_gt"),
            "entropy": q_mean(prefixes, "entropy"),
            "offset": mean_over(prefixes, "offset"),
            "recon_sampled": mean_over(prefixes, "sampled_recon"),
            "recon_argmax": mean_over(prefixes, "argmax_recon"),
        }

    table: dict[str, dict[str, float]] = {}

    def row(label: str, prefixes: list[str]) -> None:
        values = cells_for(prefixes)
        table[label.strip()] = values
        print(  # noqa: T201
            " | ".join([f"{label:>13s}", *(f"{v:13.4f}" for v in values.values())])
        )

    for pos in positions:
        row(f"t={pos} ({pos + 1}f)", [f"t{pos}"])
    row("all (wandb)", [f"t{p}" for p in positions])
    last = [f"t{positions[-1]}"]
    row("last (=bsln)", last)

    # Readouts before `window - 1` see a PARTIAL context; from `window - 1` on
    # they see the full window inference serves. Same boundary as the
    # `code_partial_window` / `code_full_window` metrics `_compute_metrics` logs.
    window = getattr(model.encoder, "window", None)
    deltas: dict[str, float] = {}
    if window is not None and len(positions) > window - 1:
        print(f"\nwindow buckets (window={window}):")  # noqa: T201
        print(" | ".join(f"{h:>13s}" for h in header))  # noqa: T201
        partial = [f"t{p}" for p in positions if p < window - 1]
        full = [f"t{p}" for p in positions if p >= window - 1]
        row(f"partial 0-{window - 2}", partial)
        row(f"full {window - 1}-{positions[-1]}", full)
        print("\nfull vs partial (negative = the FULL window is better):")  # noqa: T201
        p_vals, f_vals = cells_for(partial), cells_for(full)
        for name in (
            "code_focal",
            "code_plain",
            "offset",
            "recon_sampled",
            "p_gt",
            "entropy",
        ):
            delta = (f_vals[name] - p_vals[name]) / p_vals[name] * 100
            deltas[name] = delta
            print(  # noqa: T201
                f"  {name:>14s}  partial {p_vals[name]:.4f}  "
                f"full {f_vals[name]:.4f}  {delta:+.2f}%"
            )

    print("\nper-sample recon L1 tails (mean / p50 / p95 / p99 / max):")  # noqa: T201
    tails: dict[str, dict[str, float]] = {}
    for key in sorted(k for k in results if k.startswith("tail/")):
        v = results[key]
        qs = torch.quantile(v, torch.tensor([0.5, 0.95, 0.99]))
        name = key.removeprefix("tail/")
        tails[name] = {
            "n": float(v.numel()),
            "mean": v.mean().item(),
            "p50": qs[0].item(),
            "p95": qs[1].item(),
            "p99": qs[2].item(),
            "max": v.max().item(),
        }
        t = tails[name]
        print(  # noqa: T201
            f"  {name:>26s}  {t['mean']:.4f}  {t['p50']:.4f}  "
            f"{t['p95']:.4f}  {t['p99']:.4f}  {t['max']:.4f}"
        )

    print("\n_compute_metrics parity (training's own reduction, first batches):")  # noqa: T201
    refs: dict[str, dict[str, float]] = {}
    for key in sorted(k for k in results if re.fullmatch(r"ref\d+/.+", k)):
        idx, name = key.split("/", 1)
        refs.setdefault(idx, {})[name] = results[key].item()
    for idx, ref in refs.items():
        print(f"  {idx}: " + "  ".join(f"{k}={v:.5f}" for k, v in ref.items()))  # noqa: T201

    cluster_labels = sorted(
        (k.removeprefix("cluster_n/") for k in results if k.startswith("cluster_n/")),
        key=lambda c: -results[f"cluster_n/{c}"].item(),
    )
    if cluster_labels:
        print(  # noqa: T201
            "\nper-cluster @ last frame (field L1 over horizon; both decodings):"
        )
        cols = ["cluster", "n"] + [
            f"{dec[:4]}_{f}"
            for dec in ("sampled", "argmax")
            for f in ("gas", "brake", "steer")
        ]
        print(" | ".join(f"{h:>12s}" for h in cols))  # noqa: T201
        for c in cluster_labels:
            n = results[f"cluster_n/{c}"].item()
            cells = [f"{c:>12s}", f"{int(n):12d}"] + [
                f"{results[f'cluster/{c}/{dec}_{f}'].item() / n:12.4f}"
                for dec in ("sampled", "argmax")
                for f in ("gas", "brake", "steer")
            ]
            print(" | ".join(cells))  # noqa: T201

    print("\nper-quantizer @ last frame:")  # noqa: T201
    print(  # noqa: T201
        " | ".join(
            f"{h:>13s}"
            for h in ["quantizer", "code_focal", "top1_acc", "p_gt", "entropy"]
        )
    )
    per_quantizer: dict[str, dict[str, float]] = {}
    for q in range(num_quantizers):
        per_quantizer[f"q{q}"] = {
            stem: mean_over(last, f"{stem}_{q}")
            for stem in ("code", "codeplain", "acc", "p_gt", "entropy")
        }
        cells = [f"{f'q{q}':>13s}"] + [
            f"{mean_over(last, f'{stem}_{q}'):13.4f}"
            for stem in ["code", "acc", "p_gt", "entropy"]
        ]
        print(" | ".join(cells))  # noqa: T201

    # what `val/policy/loss/*` would have been: the training losses mean over
    # EVERY (b, t) readout, so these are the all-position means
    all_pos = [f"t{p}" for p in positions]
    val_scalars = (
        {
            f"val/policy/loss/code_{q}": mean_over(all_pos, f"code_{q}")
            for q in range(num_quantizers)
        }
        | {
            f"val/policy/loss/code_{q}_unsmoothed": mean_over(all_pos, f"codeplain_{q}")
            for q in range(num_quantizers)
        }
        | {
            "val/policy/loss/offset": mean_over(all_pos, "offset"),
            "val/policy/metric/offset_sampled_recon": mean_over(
                all_pos, "sampled_recon"
            ),
            "val/policy/metric/offset_argmax_recon": mean_over(all_pos, "argmax_recon"),
            "val/policy/metric/offset_last": mean_over(last, "offset"),
            "val/policy/metric/offset_sampled_recon_last": mean_over(
                last, "sampled_recon"
            ),
        }
    )
    for q in range(num_quantizers):
        val_scalars[f"val/policy/metric/code_{q}_last"] = mean_over(last, f"code_{q}")
    if window is not None and len(positions) > window - 1:
        val_scalars["val/policy/metric/code_partial_window"] = q_mean(
            [f"t{p}" for p in positions if p < window - 1], "code"
        )
        val_scalars["val/policy/metric/code_full_window"] = q_mean(
            [f"t{p}" for p in positions if p >= window - 1], "code"
        )
        val_scalars["val/policy/metric/offset_partial_window"] = mean_over(
            [f"t{p}" for p in positions if p < window - 1], "offset"
        )
        val_scalars["val/policy/metric/offset_full_window"] = mean_over(
            [f"t{p}" for p in positions if p >= window - 1], "offset"
        )
    print("\nreconstructed val scalars (keys as training would have logged them):")  # noqa: T201
    for k, v in val_scalars.items():
        print(f"  {k:52s} {v:.6f}")  # noqa: T201

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:  # noqa: PTH123
            json.dump(
                {
                    "checkpoint": args.artifact or args.ckpt,
                    "experiment": args.experiment,
                    "num_samples": int(results["num_samples"]),
                    "batches_requested": args.batches,
                    "batch_size": args.batch_size,
                    "seed": args.seed,
                    "label_smoothing": eps,
                    "window": window,
                    "num_readouts": len(positions),
                    "sample_codes": bool(model.sample_codes),
                    "val_scalars": val_scalars,
                    "full_vs_partial_pct": deltas,
                    "by_position": table,
                    "per_quantizer_last": per_quantizer,
                    "tails": tails,
                    "compute_metrics_reference": refs,
                    "clusters": {
                        c: {
                            "n": int(results[f"cluster_n/{c}"].item()),
                            **{
                                f"{dec}_{fld}": results[
                                    f"cluster/{c}/{dec}_{fld}"
                                ].item()
                                / results[f"cluster_n/{c}"].item()
                                for dec in ("sampled", "argmax")
                                for fld in ("gas", "brake", "steer")
                            },
                        }
                        for c in cluster_labels
                    },
                },
                f,
                indent=1,
            )
        print(f"\nwrote {args.json_out}")  # noqa: T201


if __name__ == "__main__":
    main()
