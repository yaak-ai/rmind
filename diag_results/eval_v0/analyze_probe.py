"""Offline analysis of eval_v0 probe_dump.npz -> markdown tables.

Tasks covered: (1) counterfactual override probe with per-override decoded
means, deltas vs None, KL, code flips, and the speed>50 km/h 30-vs-100
directional contrast; (4) warm-start fidelity of each arm's None condition
vs the parent on the same batches.
"""

import sys

import numpy as np
import torch
import torch.nn.functional as F

NPZ, OUT = sys.argv[1], sys.argv[2]
d = np.load(NPZ)

ARMS = ["armM", "armMV"]
OVERRIDES = ["5(WALK)", "10", "30", "50", "100", "-1(UNLIMITED)"]
FIELDS = ["gas", "brake", "steer"]

speed = d["speed_last"]
n = speed.shape[0]
has_ms = "max_speed_last" in d.files
lines = []
lines.append("# eval_v0: counterfactual override probe + warm-start fidelity")
lines.append("")
lines.append(
    f"Samples: {n} (last-frame readout, argmax decoding, seed 1337, "
    "real val batches, shared across all checkpoints/conditions)."
)
lines.append(
    f"Vehicle speed (last frame): mean {speed.mean():.1f}, p50 "
    f"{np.median(speed):.1f}, p90 {np.percentile(speed, 90):.1f}, "
    f"max {speed.max():.1f} (units as stored; >50 subset n="
    f"{(speed > 50).sum()})."
)
if has_ms:
    ms = d["max_speed_last"]
    known = ~np.isnan(ms)
    lines.append(
        f"Batch map GT max_speed (last frame): known {known.sum()}/{n} "
        f"({100 * known.mean():.1f}%)"
        + (
            f"; known values p50 {np.median(ms[known]):.0f}" if known.any() else ""
        )
    )
else:
    lines.append("Batches carry NO meta/MapContext/max_speed key.")
lines.append("")


def kl(base_logits: np.ndarray, other_logits: np.ndarray) -> float:
    lb = torch.from_numpy(base_logits).log_softmax(-1)
    lo = torch.from_numpy(other_logits).log_softmax(-1)
    return F.kl_div(lo, lb, reduction="none", log_target=True).sum(-1).mean().item()


def chunk_means(chunk: np.ndarray) -> list[float]:
    # (n, horizon, action_features) -> per-field mean over samples x horizon
    return [chunk[..., i].mean() for i in range(3)]


for arm in ARMS:
    base = {k: d[f"{arm}/None/{k}"] for k in ("logits", "codes", "chunk")}
    lines.append(f"## {arm}: override sweep (deltas vs None baseline)")
    lines.append("")
    bm = chunk_means(base["chunk"])
    lines.append(
        f"None(UNKNOWN) decoded means: gas {bm[0]:.4f}, brake {bm[1]:.4f}, "
        f"steer {bm[2]:.4f}"
    )
    lines.append("")
    hdr = (
        "| override | gas | brake | steer | dgas | dbrake | dsteer | "
        "|dgas| | |dbrake| | KL(None‖ov) | code_flips | max|dlogit| |"
    )
    lines.append(hdr)
    lines.append("|" + "---|" * 12)
    for ov in OVERRIDES:
        o = {k: d[f"{arm}/{ov}/{k}"] for k in ("logits", "codes", "chunk")}
        m = chunk_means(o["chunk"])
        dchunk = o["chunk"] - base["chunk"]
        flips = (o["codes"] != base["codes"]).mean()
        row = [
            ov,
            *[f"{v:.4f}" for v in m],
            *[f"{dchunk[..., i].mean():+.4f}" for i in range(3)],
            f"{np.abs(dchunk[..., 0]).mean():.4f}",
            f"{np.abs(dchunk[..., 1]).mean():.4f}",
            f"{kl(base['logits'], o['logits']):.4f}",
            f"{flips:.4f}",
            f"{np.abs(o['logits'] - base['logits']).max():.3f}",
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # pairwise distinctness + NaN flood check
    conds = ["None", *OVERRIDES]
    ident = []
    for i, a in enumerate(conds):
        for b in conds[i + 1 :]:
            dmax = np.abs(d[f"{arm}/{a}/logits"] - d[f"{arm}/{b}/logits"]).max()
            if dmax <= 0:
                ident.append((a, b))
    nanmax = np.abs(d[f"{arm}/NaNflood/logits"] - base["logits"]).max()
    lines.append(
        f"Sanity: identical override pairs: {ident if ident else 'none (PASS)'}; "
        f"max|logit(None) - logit(all-NaN max_speed)| = {nanmax:g}"
        + (" (None == UNKNOWN flood, PASS)" if nanmax == 0 else
           " (val batches carry real map GT -> None != UNKNOWN)")
    )
    lines.append("")

    # speed-conditioned headline: override=30 vs override=100 on fast frames
    lines.append(f"### {arm}: speed-conditioned contrast (headline)")
    lines.append("")
    lines.append(
        "| subset | n | gas@30 | gas@100 | gas 30-100 | brake@30 | brake@100 "
        "| brake 30-100 |"
    )
    lines.append("|" + "---|" * 8)
    for label, mask in [
        ("all", np.ones(n, bool)),
        ("speed>50", speed > 50),
        ("speed>70", speed > 70),
        ("speed<=50", speed <= 50),
    ]:
        if mask.sum() == 0:
            continue
        c30 = d[f"{arm}/30/chunk"][mask]
        c100 = d[f"{arm}/100/chunk"][mask]
        g30, g100 = c30[..., 0].mean(), c100[..., 0].mean()
        b30, b100 = c30[..., 1].mean(), c100[..., 1].mean()
        lines.append(
            f"| {label} | {mask.sum()} | {g30:.4f} | {g100:.4f} | "
            f"{g30 - g100:+.4f} | {b30:.4f} | {b100:.4f} | {b30 - b100:+.4f} |"
        )
    lines.append("")

# Task 4: warm-start fidelity vs parent under None
lines.append("## Warm-start fidelity: arm None-condition vs parent (same batches)")
lines.append("")
pbase = {k: d[f"parent/None/{k}"] for k in ("logits", "codes", "chunk")}
pm = chunk_means(pbase["chunk"])
lines.append(
    f"Parent decoded means: gas {pm[0]:.4f}, brake {pm[1]:.4f}, steer {pm[2]:.4f}"
)
lines.append("")
lines.append(
    "| arm | dgas | dbrake | dsteer | |dgas| | |dbrake| | |dsteer| | "
    "code_agree | KL(parent‖arm) | max|dlogit| |"
)
lines.append("|" + "---|" * 10)
for arm in ARMS:
    a = {k: d[f"{arm}/None/{k}"] for k in ("logits", "codes", "chunk")}
    dchunk = a["chunk"] - pbase["chunk"]
    agree = (a["codes"] == pbase["codes"]).mean()
    row = [
        arm,
        *[f"{dchunk[..., i].mean():+.4f}" for i in range(3)],
        *[f"{np.abs(dchunk[..., i]).mean():.4f}" for i in range(3)],
        f"{agree:.4f}",
        f"{kl(pbase['logits'], a['logits']):.4f}",
        f"{np.abs(a['logits'] - pbase['logits']).max():.3f}",
    ]
    lines.append("| " + " | ".join(row) + " |")
lines.append("")

with open(OUT, "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
