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
from typing import TYPE_CHECKING

import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch import Tensor
from torch.utils._pytree import tree_map  # noqa: PLC2701

from rmind.models.patch_policy import PatchPolicy

if TYPE_CHECKING:
    from rmind.utils import RuleBasedCluster


def _to_device(batch: object, device: torch.device) -> object:
    return tree_map(
        lambda x: x.to(device, non_blocking=True) if isinstance(x, Tensor) else x, batch
    )


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
def evaluate(  # noqa: C901, PLR0914
    model: PatchPolicy,
    loader: object,
    *,
    device: torch.device,
    max_batches: int,
    autocast: bool,
) -> dict[str, Tensor]:
    tokenizer = model.tokenizer
    num_quantizers = tokenizer.quantizer.num_quantizers

    sums: dict[str, Tensor] = {}
    count = 0

    for i, cpu_batch in enumerate(loader):
        if i >= max_batches:
            break

        batch = _to_device(cpu_batch, device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=autocast):
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

        b, t = features.shape[:2]
        for pos in range(t):
            values: dict[str, Tensor] = {}
            for q in range(num_quantizers):
                values[f"t{pos}/code_{q}"] = model.losses["code"](
                    code_logits[:, pos, q, :].float(), target_codes[:, pos, q]
                )
                values[f"t{pos}/acc_{q}"] = accuracy[:, pos, q].mean()
                values[f"t{pos}/p_gt_{q}"] = p_gt[:, pos, q].mean()
                values[f"t{pos}/entropy_{q}"] = entropy[:, pos, q].mean()
            values[f"t{pos}/acc_joint"] = joint_accuracy[:, pos].mean()
            values[f"t{pos}/offset"] = (
                (teacher_chunk[:, pos] - target[:, pos]).abs().mean()
            )
            values[f"t{pos}/sampled_recon"] = (
                (sampled_chunk[:, pos] - target[:, pos]).abs().mean()
            )
            values[f"t{pos}/argmax_recon"] = (
                (argmax_chunk[:, pos] - target[:, pos]).abs().mean()
            )
            for key, value in values.items():
                sums[key] = sums.get(key, torch.zeros((), device=device)) + value * b

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
    return {
        k: (v if k.startswith(("cluster/", "cluster_n/")) else v / count)
        for k, v in sums.items()
    } | {"num_samples": torch.tensor(count)}


def main() -> None:  # noqa: PLR0914
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
    )

    num_quantizers = model.tokenizer.quantizer.num_quantizers
    positions = sorted({int(k[1]) for k in results if k.startswith("t")})

    def mean_over(prefixes: list[str], name: str) -> float:
        return torch.stack([results[f"{p}/{name}"] for p in prefixes]).mean().item()

    def q_mean(prefixes: list[str], stem: str) -> float:
        return (
            sum(mean_over(prefixes, f"{stem}_{q}") for q in range(num_quantizers))
            / num_quantizers
        )

    print(f"\ncheckpoint: {args.artifact or args.ckpt}")  # noqa: T201
    print(f"val samples: {int(results['num_samples'])}\n")  # noqa: T201

    header = [
        "pos(context)",
        "code_focal",
        "top1_acc",  # mean of per-quantizer marginal accuracies
        "joint_acc",  # all 4 levels correct simultaneously
        "p_gt",
        "entropy",
        "offset",
        "recon_sampled",
        "recon_argmax",
    ]
    print(" | ".join(f"{h:>13s}" for h in header))  # noqa: T201

    def row(label: str, prefixes: list[str]) -> None:
        cells = [f"{label:>13s}"] + [
            f"{v:13.4f}"
            for v in [
                q_mean(prefixes, "code"),
                q_mean(prefixes, "acc"),
                mean_over(prefixes, "acc_joint"),
                q_mean(prefixes, "p_gt"),
                q_mean(prefixes, "entropy"),
                mean_over(prefixes, "offset"),
                mean_over(prefixes, "sampled_recon"),
                mean_over(prefixes, "argmax_recon"),
            ]
        ]
        print(" | ".join(cells))  # noqa: T201

    for pos in positions:
        row(f"t={pos} ({pos + 1}f)", [f"t{pos}"])
    row("all (wandb)", [f"t{p}" for p in positions])
    last = [f"t{positions[-1]}"]
    row("last (=bsln)", last)

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
    for q in range(num_quantizers):
        cells = [f"{f'q{q}':>13s}"] + [
            f"{mean_over(last, f'{stem}_{q}'):13.4f}"
            for stem in ["code", "acc", "p_gt", "entropy"]
        ]
        print(" | ".join(cells))  # noqa: T201


if __name__ == "__main__":
    main()
