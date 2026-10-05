#!/usr/bin/env python
"""Run the trained checkpoint over its own data and report what the heads actually emit.

The question this answers: the kit sees policy outputs under 0.1 and the truck barely moves.
Is that the model, or the on-kit input pipeline? Here the model is fed the SAME data it was
fitted on, through the SAME dataset code, so the pipeline is out of the picture.

  - heads small here too  -> the model. Nothing to debug on the kit.
  - heads large here      -> the model can emit large values, so the kit is feeding it
                             something it reads as "do nothing".

Ground truth rides along in the batch (the operator's own traction/steering/fork1), so the
predictions are also scored against what a human actually did at those frames.

    nix develop --command uv run python head_distribution.py [--batches N] [--split val]
"""
# ruff: noqa: T201, PLC0415, ANN001, ANN201

import argparse

import hydra
import rmind  # noqa: F401  — registers the `eval:` omegaconf resolver the configs use
import torch
from hydra.utils import instantiate


def quantiles(v: torch.Tensor) -> str:
    q = torch.tensor([0.01, 0.25, 0.5, 0.75, 0.99])
    p = torch.quantile(v, q)
    return "  ".join(f"p{int(x * 100):02d}={y:+.3f}" for x, y in zip(q.tolist(), p.tolist()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=12)
    ap.add_argument("--split", default="val", choices=["val", "train"])
    ap.add_argument(
        "--ckpt",
        default="lightning_logs/tfpakwx5/checkpoints/epoch=19-step=17580.ckpt",
        help="the run that produced the deployed policy_tfpakwx5_v18_fp32.trt",
    )
    args = ap.parse_args()

    with hydra.initialize(version_base=None, config_path="config"):
        cfg = hydra.compose(
            config_name="train",
            overrides=["experiment=palletjack/patch_policy/d12"],
            return_hydra_config=True,
        )

    print(f"instantiating model + datamodule ({args.split} split)")
    model = instantiate(cfg.model)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state["state_dict"], strict=False)
    if missing or unexpected:
        print(f"  state_dict: {len(missing)} missing, {len(unexpected)} unexpected")
    model.eval()

    # Instantiate the DATASET directly rather than the datamodule: the configured
    # `rbyte.dataloader.NodeDataLoader` requires a CUDA device, and this only needs
    # forward passes. Same dataset code, same transforms — just no GPU prefetch.
    ds = instantiate(cfg.datamodule[args.split].dataset)
    print(f"  dataset: {len(ds)} samples")

    heads = ("traction", "steering", "fork1")
    pred: dict[str, list[torch.Tensor]] = {h: [] for h in heads}
    true: dict[str, list[torch.Tensor]] = {h: [] for h in heads}
    aux: dict[str, list[torch.Tensor]] = {}

    step = max(1, len(ds) // args.batches)
    with torch.inference_mode():
        for i, idx in enumerate(range(0, len(ds), step)):
            if i >= args.batches:
                break
            batch = ds.get_batch([idx]).to_dict()
            out = model(batch)["policy"]["continuous"]
            for h in heads:
                pred[h].append(out[h].flatten().float().cpu())
            data = batch["data"]
            for h in heads:
                if h in data:
                    # last frame of the window: the one the readout acts on
                    true[h].append(data[h][:, -1].flatten().float().cpu())
            for k in ("speed", "relative_ego_pos", "relative_dropoff_pos", "fork_above_300"):
                if k in data:
                    aux.setdefault(k, []).append(data[k][:, -1].reshape(-1).float().cpu())
            print(f"  batch {i + 1}/{args.batches}", end="\r", flush=True)

    n = sum(len(t) for t in pred["traction"])
    print(f"\n\n{n} samples from the {args.split} split, checkpoint {args.ckpt}\n")

    print("PREDICTED (Gaussian mean — the tensor the .trt export emits and the kit acts on)")
    for h in heads:
        v = torch.cat(pred[h])
        print(f"  {h:<9} mean={v.mean():+.4f}  sd={v.std():.4f}  |max|={v.abs().max():.4f}")
        print(f"  {'':<9} {quantiles(v)}")
        print(f"  {'':<9} share with |value| < 0.1: {(v.abs() < 0.1).float().mean():.1%}")

    if any(true[h] for h in heads):
        print("\nGROUND TRUTH (what the operator actually did at those frames)")
        for h in heads:
            if not true[h]:
                continue
            t, p = torch.cat(true[h]), torch.cat(pred[h])
            print(f"  {h:<9} mean={t.mean():+.4f}  sd={t.std():.4f}  |max|={t.abs().max():.4f}")
            print(f"  {'':<9} {quantiles(t)}")
            if t.numel() == p.numel() and t.std() > 0 and p.std() > 0:
                r = torch.corrcoef(torch.stack([p, t]))[0, 1]
                print(f"  {'':<9} corr(pred, true)={r:+.3f}   sd ratio pred/true={p.std() / t.std():.3f}")

    if aux:
        print("\nAUX INPUTS (the same tensors the kit builds — for comparison against it)")
        for k, vs in aux.items():
            v = torch.cat(vs)
            print(f"  {k:<22} mean={v.mean():+.4f}  sd={v.std():.4f}  |max|={v.abs().max():.4f}")


if __name__ == "__main__":
    main()
