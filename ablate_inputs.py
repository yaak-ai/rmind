#!/usr/bin/env python
"""Which input is the policy actually using? Permutation ablation over the real inputs.

Past actions are not observed by this model (they are targets only), so it cannot copy the
previous action outright. The live worry is the softer causal confusion: predicting traction
from SPEED ("still moving, so keep going") and ignoring the camera, which looks fine on a
held-out split drawn from the same drives and fails the moment the truck is stationary.

Each input group is destroyed in turn by PERMUTING it across samples — sample i is given
sample (i+k)'s speed, image, intention. Permutation keeps the marginal distribution the
model was trained on, unlike zeroing, which is off-distribution and collapses activations
for reasons that have nothing to do with reliance. Zeroing is reported too, as a cross-check.

Read the result as: how much of the prediction survives when this input is made meaningless?

    corr_vs_baseline ~ 1.0  ->  input IGNORED (destroying it changed nothing)
    corr_vs_baseline ~ 0.0  ->  input LOAD-BEARING

    nix develop --command uv run python ablate_inputs.py [--samples N]
"""
# ruff: noqa: T201, PLC0415, ANN001, ANN201, C901, PLR0912, PLR0914, PLR0915

import argparse
import copy

import hydra
import torch
from hydra.utils import instantiate

import rmind  # ruff: ignore[unused-import]  — registers the `eval:` omegaconf resolver the configs use

HEADS = ("traction", "steering", "fork1")
GROUPS = {
    "image (cam_fork)": ["cam_fork"],
    "speed": ["speed"],
    "relative_ego_pos": ["relative_ego_pos"],
    "relative_dropoff_pos": ["relative_dropoff_pos"],
    "fork_above_300": ["fork_above_300"],
    "ALL non-image": [
        "speed",
        "relative_ego_pos",
        "relative_dropoff_pos",
        "fork_above_300",
    ],
}


def corr(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.std() < 1e-8 or b.std() < 1e-8:
        return float("nan")
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--split", default="val", choices=["val", "train"])
    ap.add_argument(
        "--ckpt", default="lightning_logs/tfpakwx5/checkpoints/epoch=19-step=17580.ckpt"
    )
    args = ap.parse_args()

    with hydra.initialize(version_base=None, config_path="config"):
        cfg = hydra.compose(
            config_name="train",
            overrides=["experiment=palletjack/patch_policy/d12"],
            return_hydra_config=True,
        )

    model = instantiate(cfg.model)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"], strict=False)
    model.eval()

    ds = instantiate(cfg.datamodule[args.split].dataset)
    step = max(1, len(ds) // args.samples)
    idxs = list(range(0, len(ds), step))[: args.samples]
    print(f"loading {len(idxs)} samples of {len(ds)} from the {args.split} split")
    batches = [ds.get_batch([i]).to_dict() for i in idxs]
    n = len(batches)

    def run(bs) -> dict[str, torch.Tensor]:
        out: dict[str, list[torch.Tensor]] = {h: [] for h in HEADS}
        with torch.inference_mode():
            for j, b in enumerate(bs):
                o = model(b)["policy"]["continuous"]
                for h in HEADS:
                    out[h].append(o[h].flatten().float().cpu())
                print(f"    {j + 1}/{len(bs)}", end="\r", flush=True)
        return {h: torch.cat(v) for h, v in out.items()}

    print("  baseline")
    base = run(batches)
    truth = {
        h: torch.cat([b["data"][h][:, -1].flatten().float().cpu() for b in batches])
        for h in HEADS
        if h in batches[0]["data"]
    }

    print(f"\n{n} samples · checkpoint {args.ckpt}\n")
    print("BASELINE")
    for h in HEADS:
        line = f"  {h:<21} sd={base[h].std():.4f}  |max|={base[h].abs().max():.4f}"
        if h in truth:
            line += f"  corr_vs_truth={corr(base[h], truth[h]):+.3f}"
        print(line)

    for mode in ("permute", "zero"):
        print(
            f"\n{mode.upper()}D — corr vs baseline (~1.0 = input ignored, ~0.0 = load-bearing)"
        )
        print(
            f"  {'ablated input':<22}{'traction':>10}{'steering':>10}{'fork1':>10}   sd(traction)"
        )
        for label, keys in GROUPS.items():
            mod = []
            for i, b in enumerate(batches):
                nb = copy.deepcopy(b)
                for k in keys:
                    if k not in nb["data"]:
                        continue
                    if mode == "zero":
                        nb["data"][k] = torch.zeros_like(nb["data"][k])
                    else:
                        donor = batches[(i + n // 2) % n]["data"][k]
                        nb["data"][k] = donor.clone()
                mod.append(nb)
            out = run(mod)
            cs = [corr(out[h], base[h]) for h in HEADS]
            print(
                f"  {label:<22}{cs[0]:>10.3f}{cs[1]:>10.3f}{cs[2]:>10.3f}"
                f"   {out['traction'].std():.4f}"
            )


if __name__ == "__main__":
    main()
