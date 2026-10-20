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
# ruff: file-ignore[print]

import argparse

import hydra
import torch
from hydra.utils import instantiate

import rmind  # ruff: ignore[unused-import]  — registers the `eval:` omegaconf resolver the configs use

SMALL_ACTION_THRESHOLD = 0.1
HEADS = ("traction", "steering", "fork1")


def quantiles(v: torch.Tensor) -> str:
    q = torch.tensor([0.01, 0.25, 0.5, 0.75, 0.99])
    p = torch.quantile(v, q)
    return "  ".join(
        f"p{int(x * 100):02d}={y:+.3f}"
        for x, y in zip(q.tolist(), p.tolist(), strict=True)
    )


def report_predictions(
    predictions: dict[str, list[torch.Tensor]], heads: tuple[str, ...]
) -> None:
    print(
        "PREDICTED (Gaussian mean — the tensor the .trt export emits and the kit acts on)"
    )
    for head in heads:
        values = torch.cat(predictions[head])
        print(
            f"  {head:<9} mean={values.mean():+.4f}  sd={values.std():.4f}  |max|={values.abs().max():.4f}"
        )
        print(f"  {'':<9} {quantiles(values)}")
        print(
            f"  {'':<9} share with |value| < 0.1: {(values.abs() < SMALL_ACTION_THRESHOLD).float().mean():.1%}"
        )


def report_ground_truth(
    predictions: dict[str, list[torch.Tensor]],
    targets: dict[str, list[torch.Tensor]],
    heads: tuple[str, ...],
) -> None:
    if not any(targets[head] for head in heads):
        return
    print("\nGROUND TRUTH (what the operator actually did at those frames)")
    for head in heads:
        if not targets[head]:
            continue
        target = torch.cat(targets[head])
        prediction = torch.cat(predictions[head])
        print(
            f"  {head:<9} mean={target.mean():+.4f}  sd={target.std():.4f}  |max|={target.abs().max():.4f}"
        )
        print(f"  {'':<9} {quantiles(target)}")
        if (
            target.numel() == prediction.numel()
            and target.std() > 0
            and prediction.std() > 0
        ):
            correlation = torch.corrcoef(torch.stack([prediction, target]))[0, 1]
            print(
                f"  {'':<9} corr(pred, true)={correlation:+.3f}   sd ratio pred/true={prediction.std() / target.std():.3f}"
            )


def report_aux_inputs(aux: dict[str, list[torch.Tensor]]) -> None:
    if not aux:
        return
    print("\nAUX INPUTS (the same tensors the kit builds — for comparison against it)")
    for name, tensors in aux.items():
        values = torch.cat(tensors)
        print(
            f"  {name:<22} mean={values.mean():+.4f}  sd={values.std():.4f}  |max|={values.abs().max():.4f}"
        )


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

    pred: dict[str, list[torch.Tensor]] = {h: [] for h in HEADS}
    true: dict[str, list[torch.Tensor]] = {h: [] for h in HEADS}
    aux: dict[str, list[torch.Tensor]] = {}

    with torch.inference_mode():
        for i, idx in enumerate(range(0, len(ds), max(1, len(ds) // args.batches))):
            if i >= args.batches:
                break
            batch = ds.get_batch([idx]).to_dict()
            out = model(batch)["policy"]["continuous"]
            for h in HEADS:
                pred[h].append(out[h].flatten().float().cpu())
            data = batch["data"]
            for h in HEADS:
                if h in data:
                    # last frame of the window: the one the readout acts on
                    true[h].append(data[h][:, -1].flatten().float().cpu())
            for k in (
                "speed",
                "relative_ego_pos",
                "relative_dropoff_pos",
                "fork_above_300",
            ):
                if k in data:
                    aux.setdefault(k, []).append(
                        data[k][:, -1].reshape(-1).float().cpu()
                    )
            print(f"  batch {i + 1}/{args.batches}", end="\r", flush=True)

    print(
        f"\n\n{sum(len(t) for t in pred['traction'])} samples from the {args.split} split, checkpoint {args.ckpt}\n"
    )

    report_predictions(pred, HEADS)
    report_ground_truth(pred, true, HEADS)
    report_aux_inputs(aux)


if __name__ == "__main__":
    main()
