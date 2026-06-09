#!/usr/bin/env python
"""Regenerate report figures + metric tables from the fan .npz caches."""
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 200, "font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
})
SHOT = Path("screenshots")
FIG = Path("report/figs")
FIG.mkdir(parents=True, exist_ok=True)
EPS = (0.05, 0.1, 0.2)
SPIKE_THR, PAD = 0.5, 50


def dilate(m, pad):
    return np.convolve(m.astype(float), np.ones(2 * pad + 1), "same") > 0 if pad else m


def metrics(path, slot0=True):
    d = np.load(SHOT / path)
    gt, s = d["gt"], d["sample"]
    if gt.ndim == 2:           # horizon cache -> first step
        gt, s = gt[:, 0], s[:, :, 0]
    valid = ~np.isnan(gt)
    gt, s = gt[valid], s[valid]
    err = np.abs(s - gt[:, None])
    spike = dilate(np.abs(gt) > SPIKE_THR, PAD)
    flat = ~spike
    out = {}
    for region, mask in (("spike", spike), ("flat", flat)):
        out[f"{region}_hit05"] = float((err[mask] <= 0.05).mean())
        out[f"{region}_hit10"] = float((err[mask] <= 0.10).mean())
        out[f"{region}_l1"] = float(np.nanmean(err[mask]))
        fin = mask & ~np.isnan(err).all(1)
        out[f"{region}_bo32"] = float(np.nanmin(err[fin], axis=1).mean())
    out["spike_frac"] = float(spike.mean())
    return out


# ---- integrator sweep (same image ckpt, varying sampler) -------------------
INTEG = [
    ("heun/8",     "fan_image.npz",            16),
    ("euler/16",   "fan_image_euler16.npz",    16),
    ("heun/16",    "fan_image_heun16.npz",     32),
    ("midpoint/16","fan_image_midpoint16.npz", 32),
    ("euler/32",   "fan_image_euler32.npz",    32),
    ("heun/32",    "fan_image_steps32.npz",    64),
    ("euler/64",   "fan_image_euler64.npz",    64),
    ("heun/128",   "fan_image_steps128.npz",  256),
]
integ = [{"name": n, "nfe": nfe, **metrics(f)} for n, f, nfe in INTEG]

# Figure 1: coarse-Euler contraction — flat hit@0.05 up while bo32 worsens
fig, ax1 = plt.subplots(figsize=(5.0, 3.2))
eul = [r for r in integ if r["name"].startswith("euler")]
heu = [r for r in integ if r["name"].startswith("heun")]
for grp, mk, lab in ((eul, "o-", "Euler"), (heu, "s--", "Heun")):
    grp = sorted(grp, key=lambda r: r["nfe"])
    ax1.plot([r["nfe"] for r in grp], [r["flat_hit05"] * 100 for r in grp],
             mk, label=f"{lab} hit@0.05", color="tab:blue" if lab == "Euler" else "tab:cyan")
ax1.set_xlabel("NFE"); ax1.set_ylabel("flat hit@0.05 (%)", color="tab:blue")
ax1.set_xscale("log", base=2)
ax2 = ax1.twinx(); ax2.grid(False)
for grp, mk, lab in ((eul, "o-", "Euler"), (heu, "s--", "Heun")):
    grp = sorted(grp, key=lambda r: r["nfe"])
    ax2.plot([r["nfe"] for r in grp], [r["flat_bo32"] * 1000 for r in grp],
             mk, color="tab:red" if lab == "Euler" else "salmon", alpha=0.8)
ax2.set_ylabel("flat best-of-32 L1 (×10⁻³)", color="tab:red")
ax1.legend(loc="center right", fontsize=7)
ax1.set_title("Coarser Euler: hit-rate ↑ but sample diversity ↓ (contraction)")
fig.tight_layout(); fig.savefig(FIG / "integrator.png"); plt.close(fig)

# ---- EMA (same v399 ckpt) + duration ---------------------------------------
ema = {k: metrics(v) for k, v in {
    "raw (v399, 400ep)": "fan_ema_raw.npz",
    "EMA (v399)": "fan_ema_ema.npz",
    "~100ep (legacy)": "fan_4l_legacy.npz",
}.items()}

# ---- conditioning null ------------------------------------------------------
cond = {k: metrics(v) for k, v in {  # integrator-matched (all heun/8)
    "summaries+wpts": "fan_4l_legacy.npz",
    "+image tokens": "fan_image.npz",
    "image ckpt, img ablated": "fan_image_ablated.npz",
}.items()}

# ---- horizon profile (fan_v100, 6-step chunk) ------------------------------
d = np.load(SHOT / "fan_v100.npz")
gt, s = d["gt"], d["sample"]                # (F,6), (F,32,6)
H = gt.shape[1]
gt0 = np.nan_to_num(gt[:, 0]); spike = dilate(np.abs(gt0) > SPIKE_THR, PAD)
per_h_l1 = [float(np.nanmean(np.abs(s[:, :, h] - gt[:, h][:, None]))) for h in range(H)]
per_h_l1_spike = [float(np.nanmean(np.abs(s[spike, :, h] - gt[spike, h][:, None]))) for h in range(H)]
# lag: shift of slot h mean prediction vs its own target on spikes (proves h-3 anchor)
def best_lag(h):
    p = np.convolve(np.nan_to_num(np.nanmean(s[:, :, h], 1)), np.ones(5) / 5, "same")
    t = gt[:, h]; errs = {}
    for sh in range(-4, 7):
        if sh > 0: a, b, m = p[:-sh], t[sh:], spike[sh:]
        elif sh < 0: a, b, m = p[-sh:], t[:sh], spike[:sh]
        else: a, b, m = p, t, spike
        errs[sh] = np.nanmean(np.abs(a - b)[m[:len(a)]])
    return min(errs, key=errs.get)
lags = [best_lag(h) for h in range(H)]

fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.2, 3.0))
axa.plot(range(1, H + 1), per_h_l1, "o-", label="all")
axa.plot(range(1, H + 1), per_h_l1_spike, "s--", label="spike", color="tab:orange")
axa.set_xlabel("horizon slot h"); axa.set_ylabel("mean-draw L1")
axa.set_title("Per-slot L1: flat U-shape (not copycat)"); axa.legend(fontsize=7)
axb.plot(range(1, H + 1), lags, "D-", color="tab:green")
axb.plot(range(1, H + 1), [3 - h for h in range(1, H + 1)], ":", color="0.5", label="lag = 3 − h")
axb.set_xlabel("horizon slot h"); axb.set_ylabel("best alignment shift (steps)")
axb.set_title("Constant chunk anchored at t+3"); axb.legend(fontsize=7)
fig.tight_layout(); fig.savefig(FIG / "horizon.png"); plt.close(fig)

# ---- sample-concentration fan at the big maneuver (legacy ckpt) ------------
d = np.load(SHOT / "fan_4l_legacy.npz")
fr, gt, s = d["frame"], d["gt"], d["sample"]
order = np.argsort(fr); fr, gt, s = fr[order], gt[order], s[order]
big = np.nanargmax(np.abs(np.nan_to_num(gt)))
lo, hi = max(0, big - 90), big + 90
fig, ax = plt.subplots(figsize=(7.2, 2.8))
xs = np.repeat(fr[lo:hi], s.shape[1])
ax.scatter(xs, s[lo:hi].ravel(), s=3, alpha=0.12, color="tab:blue", label="32 draws")
ax.plot(fr[lo:hi], gt[lo:hi], color="crimson", lw=1.6, label="ground truth")
ax.set_xlabel("frame"); ax.set_ylabel("steering")
ax.set_title("Sample concentration at a maneuver — mass on GT, not lucky tail")
ax.legend(fontsize=7, loc="upper right")
fig.tight_layout(); fig.savefig(FIG / "fan.png"); plt.close(fig)

print(json.dumps({"integ": integ, "ema": ema, "cond": cond,
                  "per_h_l1": per_h_l1, "per_h_l1_spike": per_h_l1_spike,
                  "lags": lags, "spike_frac": float(spike.mean())}, indent=1))
