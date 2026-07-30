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

import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch import Tensor
from torch.utils._pytree import tree_map  # noqa: PLC2701

from rmind.models.patch_policy import PatchPolicy


def _to_device(batch: object, device: torch.device) -> object:
    return tree_map(
        lambda x: x.to(device, non_blocking=True) if isinstance(x, Tensor) else x, batch
    )


@torch.no_grad()
def evaluate(  # noqa: PLR0914
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
            codes = model._sample_codes(code_logits)  # noqa: SLF001
            sampled_chunk = tokenizer.invert(codes) + model._offset(offsets, codes)  # noqa: SLF001

        b, t = features.shape[:2]
        for pos in range(t):
            for q in range(num_quantizers):
                key = f"t{pos}/code_{q}"
                value = model.losses["code"](
                    code_logits[:, pos, q, :].float(), target_codes[:, pos, q]
                )
                sums[key] = sums.get(key, torch.zeros((), device=device)) + value * b
            for key, value in (
                (
                    f"t{pos}/offset",
                    (teacher_chunk[:, pos] - target[:, pos]).abs().mean(),
                ),
                (
                    f"t{pos}/sampled_recon",
                    (sampled_chunk[:, pos] - target[:, pos]).abs().mean(),
                ),
            ):
                sums[key] = sums.get(key, torch.zeros((), device=device)) + value * b

        count += b

    return {k: v / count for k, v in sums.items()} | {
        "num_samples": torch.tensor(count)
    }


def main() -> None:
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
    header = (
        ["pos(context)"]
        + [f"code_{q}" for q in range(num_quantizers)]
        + ["code_mean", "offset", "sampled_recon"]
    )
    print(f"\ncheckpoint: {args.artifact or args.ckpt}")  # noqa: T201
    print(f"val samples: {int(results['num_samples'])}\n")  # noqa: T201
    print(" | ".join(f"{h:>13s}" for h in header))  # noqa: T201

    def row(label: str, keys_prefix: list[str]) -> None:
        codes = [
            torch.stack([results[f"{p}/code_{q}"] for p in keys_prefix]).mean()
            for q in range(num_quantizers)
        ]
        offset = torch.stack([results[f"{p}/offset"] for p in keys_prefix]).mean()
        recon = torch.stack([results[f"{p}/sampled_recon"] for p in keys_prefix]).mean()
        cells = [f"{label:>13s}"] + [
            f"{v.item():13.4f}"
            for v in [*codes, torch.stack(codes).mean(), offset, recon]
        ]
        print(" | ".join(cells))  # noqa: T201

    for pos in positions:
        row(f"t={pos} ({pos + 1}f)", [f"t{pos}"])
    row("all (wandb)", [f"t{p}" for p in positions])
    row("last (=bsln)", [f"t{positions[-1]}"])


if __name__ == "__main__":
    main()
