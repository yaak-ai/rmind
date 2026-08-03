"""patch_policy_eval on cached val batches (bypasses rbyte entirely).

Same evaluate() + report as rmind.scripts.patch_policy_eval, but the batches
come from a torch.save'd list (caches/eval_v0_val_batches.pt) instead of a
fresh datamodule spin. Fallback for rbyte-cache corruption; subset is the
same seed-1337 val stream, just shorter (24 batches / 768 samples).

Usage: python ppe_cached.py <batch_cache.pt> <ckpt> [<device>]
"""

import sys

import pytorch_lightning as pl
import torch

from rmind.models.patch_policy import PatchPolicy
from rmind.scripts.patch_policy_eval import evaluate

cache, ckpt = sys.argv[1], sys.argv[2]
device = torch.device(sys.argv[3] if len(sys.argv) > 3 else "cuda")

pl.seed_everything(1337, workers=True)
batches = torch.load(cache, weights_only=False)
model = PatchPolicy.load_from_checkpoint(ckpt, weights_only=False, map_location="cpu")
model = model.to(device).eval()

results = evaluate(
    model, batches, device=device, max_batches=len(batches), autocast=True
)

num_quantizers = model.tokenizer.quantizer.num_quantizers
positions = sorted({int(k[1]) for k in results if k.startswith("t")})


def mean_over(prefixes, name):
    return torch.stack([results[f"{p}/{name}"] for p in prefixes]).mean().item()


def q_mean(prefixes, stem):
    return (
        sum(mean_over(prefixes, f"{stem}_{q}") for q in range(num_quantizers))
        / num_quantizers
    )


print(f"\ncheckpoint: {ckpt}")
print(f"val samples: {int(results['num_samples'])}\n")
header = [
    "pos(context)",
    "code_focal",
    "top1_acc",
    "joint_acc",
    "p_gt",
    "entropy",
    "offset",
    "recon_sampled",
    "recon_argmax",
]
print(" | ".join(f"{h:>13s}" for h in header))


def row(label, prefixes):
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
    print(" | ".join(cells))


for pos in positions:
    row(f"t={pos} ({pos + 1}f)", [f"t{pos}"])
row("all (wandb)", [f"t{p}" for p in positions])
row("last (=bsln)", [f"t{positions[-1]}"])

cluster_labels = sorted(
    (k.removeprefix("cluster_n/") for k in results if k.startswith("cluster_n/")),
    key=lambda c: -results[f"cluster_n/{c}"].item(),
)
if cluster_labels:
    print("\nper-cluster @ last frame (field L1 over horizon; both decodings):")
    cols = ["cluster", "n"] + [
        f"{dec[:4]}_{f}"
        for dec in ("sampled", "argmax")
        for f in ("gas", "brake", "steer")
    ]
    print(" | ".join(f"{h:>12s}" for h in cols))
    for c in cluster_labels:
        n = results[f"cluster_n/{c}"].item()
        cells = [f"{c:>12s}", f"{int(n):12d}"] + [
            f"{results[f'cluster/{c}/{dec}_{f}'].item() / n:12.4f}"
            for dec in ("sampled", "argmax")
            for f in ("gas", "brake", "steer")
        ]
        print(" | ".join(cells))
print("\ndone")
