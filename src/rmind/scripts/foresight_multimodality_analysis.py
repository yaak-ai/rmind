"""Phase-0 foresight multimodality analysis (D0-D4).

Consumes the val feature cache written per the cache-schema contract
(foresight_mm/cache/val: shard_*.pt + meta.json, optional val_grids.pt) and
computes:

  D0  k-NN future-entropy vs horizon (anti-leakage k-NN in FD-conditioning
      context space), overall / per motion stratum, + secondary-context check
  D1  prediction-vs-neighbor-centroid ratio r per entropy quintile
  D2  norm shrinkage per entropy quintile (+ per-patch variant if grids exist)
      and residual PCA / GMM bimodality in the top-entropy quintile
  D3  per-anchor fork detection (GMM(2) vs GMM(1) BIC on PCA'd neighbor
      futures) at H=1 vs H=5, + interpolation coefficient alpha of the FD
      prediction between fork cluster means
  D4  (proxy) future gas+brake conflict odds ratio in
      (top-entropy AND bottom-r) anchors vs rest

Outputs: foresight_mm/results/phase0_d0_d4.json, figures under
foresight_mm/results/figs/, printed summary.

Self-test: --synthetic generates a planted-structure fake cache
(foresight_mm/cache/synthetic_val) and asserts recovery; outputs go to
foresight_mm/results/synthetic/.

Usage:
  uv run python src/rmind/scripts/foresight_multimodality_analysis.py [--cache-dir ...] [--synthetic]

Design notes / deviations (also in foresight_mm/notes/analysis.md):
  - D3 per-anchor GMMs use covariance_type='tied' (a full per-component
    covariance is unidentifiable from k=32 points in 16 dims and its BIC
    penalty makes detection impossible); D2's PC1 GMM uses 'full' as specified
    (it is 1-D there). D3 projections are RMS-normalized per anchor and
    reg_covar=1e-3.
  - Motion strata defs are identical to foresight_leverage_l1.py::_strata_masks
    (stopped speed<0.5 > braking > turning |steer|>p80 > cruising); see
    foresight_phase0_common.motion_strata.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import foresight_phase0_common as com  # noqa: E402

C_BLUE, C_ORANGE, C_TEAL, C_PURPLE, C_GRAY, C_RED = (
    "#2563eb",
    "#ea580c",
    "#0d9488",
    "#7c3aed",
    "#6b7280",
    "#dc2626",
)
STRATUM_COLORS = [C_GRAY, C_RED, C_PURPLE, C_TEAL]
HORIZONS = [1, 2, 3, 4, 5]
PCTS = [5, 20, 40, 60, 80, 95]


def pick_device(min_free_gb: float = 4.0) -> str:
    if torch.cuda.is_available():
        try:
            free, _ = torch.cuda.mem_get_info()
            if free > min_free_gb * 2**30:
                return "cuda"
        except Exception:  # noqa: BLE001
            pass
    return "cpu"


def _style(ax):
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# --------------------------------------------------------------------------- #
# k-NN with anti-leakage eligibility
# --------------------------------------------------------------------------- #
def eligible_knn(
    x: torch.Tensor,
    drive: torch.Tensor,
    order: torch.Tensor,
    k: int,
    gap: float,
    device: str,
    chunk: int = 2048,
) -> tuple[torch.Tensor, torch.Tensor]:
    """k nearest neighbors under: eligible iff different drive OR
    |sample-order gap| > `gap`. Returns (nn_idx (n,k) long, n_excluded (n,) long
    where n_excluded counts ineligible candidates INCLUDING self)."""
    n = x.shape[0]
    xd = x.float().to(device)
    dr = drive.to(device)
    od = order.float().to(device)
    nn_idx = torch.empty(n, k, dtype=torch.long)
    n_excl = torch.empty(n, dtype=torch.long)
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        d = torch.cdist(xd[i0:i1], xd)  # (c, n) fp32
        inel = (dr[i0:i1, None] == dr[None, :]) & (
            (od[i0:i1, None] - od[None, :]).abs() <= gap
        )
        n_excl[i0:i1] = inel.sum(1).cpu()
        d.masked_fill_(inel, float("inf"))
        nn_idx[i0:i1] = d.topk(k, dim=1, largest=False).indices.cpu()
    beyond_self = int(n_excl.sum()) - n
    assert beyond_self > 0, (
        "anti-leakage eligibility mask excluded ZERO candidates beyond self — "
        "drive/order info is broken or samples are temporally disjoint"
    )
    return nn_idx, n_excl


# --------------------------------------------------------------------------- #
# D0 entropy maps
# --------------------------------------------------------------------------- #
def entropy_maps(
    nn_idx: torch.Tensor, fut: torch.Tensor, device: str, chunk: int = 4096
) -> tuple[torch.Tensor, torch.Tensor]:
    """mean-pairwise-L2 and cov-trace entropy proxies over the k neighbor
    futures, per anchor per horizon. fut: (n, 5, 384) fp32."""
    n, k = nn_idx.shape
    mpd = torch.empty(n, 5)
    ctr = torch.empty(n, 5)
    futd = fut.float().to(device)
    idxd = nn_idx.to(device)
    for h in range(5):
        yh = futd[:, h]
        for i0 in range(0, n, chunk):
            i1 = min(i0 + chunk, n)
            y = yh[idxd[i0:i1]]  # (c, k, 384)
            d = torch.cdist(y, y)
            mpd[i0:i1, h] = (d.sum(dim=(1, 2)) / (k * (k - 1))).cpu()
            ctr[i0:i1, h] = y.var(dim=1, unbiased=True).sum(dim=1).cpu()
    return mpd, ctr


def _curve(v: torch.Tensor) -> dict:
    """Summary of an (m, 5) entropy map: mean + percentiles per horizon."""
    if v.numel() == 0:
        return {"n": 0, "mean": None, "pct": None}
    q = torch.tensor([p / 100 for p in PCTS], dtype=v.dtype)
    pct = torch.quantile(v, q, dim=0)  # (len(PCTS), 5)
    return {
        "n": int(v.shape[0]),
        "mean": v.mean(0).tolist(),
        "pct": {str(p): pct[i].tolist() for i, p in enumerate(PCTS)},
    }


def quintile_assign(v: torch.Tensor) -> torch.Tensor:
    edges = torch.quantile(v.float(), torch.tensor([0.2, 0.4, 0.6, 0.8]))
    return torch.bucketize(v.float().contiguous(), edges)


# --------------------------------------------------------------------------- #
# D1
# --------------------------------------------------------------------------- #
def d1_ratio(
    nn_idx: torch.Tensor, fut1: torch.Tensor, yhat: torch.Tensor, device: str, chunk: int = 4096
) -> torch.Tensor:
    n = nn_idx.shape[0]
    r = torch.empty(n)
    f = fut1.float().to(device)
    p = yhat.float().to(device)
    idxd = nn_idx.to(device)
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        y = f[idxd[i0:i1]]  # (c, k, 384)
        num = (p[i0:i1] - y.mean(1)).norm(dim=1)
        den = (p[i0:i1, None, :] - y).norm(dim=2).min(dim=1).values
        r[i0:i1] = (num / den.clamp_min(1e-8)).cpu()
    return r


# --------------------------------------------------------------------------- #
# D3 fork detection
# --------------------------------------------------------------------------- #
def _gmm12(proj: np.ndarray) -> tuple[bool, float, float, np.ndarray]:
    """GMM(2) vs GMM(1) BIC on (k, p) projections. Returns
    (fork, bic1-bic2, min_weight, labels)."""
    from sklearn.mixture import GaussianMixture

    gm1 = GaussianMixture(1, covariance_type="tied", reg_covar=1e-3, random_state=0).fit(proj)
    gm2 = GaussianMixture(
        2, covariance_type="tied", reg_covar=1e-3, n_init=2, random_state=0, max_iter=200
    ).fit(proj)
    labels = gm2.predict(proj).astype(np.int8)
    wmin = float(gm2.weights_.min())
    dbic = float(gm1.bic(proj) - gm2.bic(proj))
    n0 = int((labels == 0).sum())
    fork = (dbic > 0.0) and (wmin >= 0.25) and 0 < n0 < len(labels)
    return fork, dbic, wmin, labels


def fork_scan(
    nn_idx: torch.Tensor,
    futh: torch.Tensor,
    yhat: torch.Tensor,
    device: str,
    n_jobs: int,
    pca_dims: int = 16,
    chunk: int = 2048,
) -> dict:
    """Per-anchor fork detection on the k neighbor futures at one horizon.
    Returns fork flags, dbic, and for forked anchors: alpha / orth ratio of the
    FD prediction relative to the fork cluster means (original pooled space)."""
    from joblib import Parallel, delayed

    n, k = nn_idx.shape
    p = min(pca_dims, k - 2)
    futd = futh.float().to(device)
    idxd = nn_idx.to(device)
    projs: list[np.ndarray] = []
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        y = futd[idxd[i0:i1]]  # (c, k, 384)
        yc = y - y.mean(dim=1, keepdim=True)
        # batched PCA to p dims via SVD; Vh rows are PCs
        _, _, vh = torch.linalg.svd(yc, full_matrices=False)
        pr = torch.einsum("ckd,cpd->ckp", yc, vh[:, :p, :])  # (c, k, p)
        pr = pr / pr.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        projs.append(pr.cpu())
    proj_all = torch.cat(projs).double().numpy()  # (n, k, p)

    results = Parallel(n_jobs=n_jobs, batch_size=256)(
        delayed(_gmm12)(proj_all[i]) for i in range(n)
    )
    fork = torch.tensor([r[0] for r in results])
    dbic = torch.tensor([r[1] for r in results])
    wmin = torch.tensor([r[2] for r in results])

    fut_cpu = futh.float()
    yhat_cpu = yhat.float()
    fork_idx, alphas, orths, cAs, cBs = [], [], [], [], []
    for i in torch.nonzero(fork).flatten().tolist():
        labels = torch.from_numpy(results[i][3].astype(np.int64))
        y = fut_cpu[nn_idx[i]]  # (k, 384)
        c_a = y[labels == 0].mean(0)
        c_b = y[labels == 1].mean(0)
        ab = c_b - c_a
        denom = ab.dot(ab).clamp_min(1e-12)
        alpha = float((yhat_cpu[i] - c_a).dot(ab) / denom)
        perp = yhat_cpu[i] - (c_a + alpha * ab)
        orth = float(perp.norm() / denom.sqrt())
        fork_idx.append(i)
        alphas.append(alpha)
        orths.append(orth)
        cAs.append(c_a)
        cBs.append(c_b)
    return {
        "fork": fork,
        "dbic": dbic,
        "wmin": wmin,
        "fork_idx": torch.tensor(fork_idx, dtype=torch.long),
        "alpha": torch.tensor(alphas),
        "orth": torch.tensor(orths),
        "labels": {i: results[i][3] for i in fork_idx},
    }


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def fig_entropy(mpd, mpd_s, strata, figs: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    m = mpd.mean(0)
    q = torch.quantile(mpd, torch.tensor([0.05, 0.2, 0.8, 0.95]), dim=0)
    ax.fill_between(HORIZONS, q[0], q[3], color=C_BLUE, alpha=0.10, label="p5-p95")
    ax.fill_between(HORIZONS, q[1], q[2], color=C_BLUE, alpha=0.25, label="p20-p80")
    ax.plot(HORIZONS, m, color=C_BLUE, lw=2, marker="o", ms=5, label="mean (context c)")
    ax.plot(
        HORIZONS, mpd_s.mean(0), color=C_ORANGE, lw=2, ls="--", marker="s", ms=4,
        label="mean (context cS)",
    )
    ax.set_xlabel("horizon H (steps of 1/3 s)")
    ax.set_ylabel("neighbor-future entropy (mean pairwise L2)")
    ax.set_title("D0: entropy vs horizon")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    ax = axes[1]
    for si, name in enumerate(com.STRATUM_NAMES):
        mask = strata == si
        if mask.any():
            ax.plot(
                HORIZONS, mpd[mask].mean(0), color=STRATUM_COLORS[si], lw=1.8,
                marker="o", ms=4, label=f"{name} (n={int(mask.sum())})",
            )
    ax.set_xlabel("horizon H")
    ax.set_ylabel("entropy (mean pairwise L2)")
    ax.set_title("D0: entropy vs horizon per motion stratum")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(figs / "d0_entropy_vs_h.png", dpi=150)
    plt.close(fig)


def fig_d1(med_r, figs: Path):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(range(1, 6), med_r, color=C_BLUE, width=0.62)
    for i, v in enumerate(med_r):
        ax.text(i + 1, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color="#374151")
    ax.axhline(1.0, color=C_GRAY, lw=1, ls=":")
    ax.set_xlabel("H=1 entropy quintile (1=low, 5=high)")
    ax.set_ylabel("median r = ||ŷ-ȳ|| / min ||ŷ-yᵢ||")
    ax.set_title("D1: prediction-centroid ratio per entropy quintile")
    _style(ax)
    fig.tight_layout()
    fig.savefig(figs / "d1_r_per_quintile.png", dpi=150)
    plt.close(fig)


def fig_d3_hists(alpha, orth, h_label, figs: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(alpha.numpy(), bins=40, color=C_BLUE, alpha=0.85)
    axes[0].axvline(0.5, color=C_RED, lw=1.2, ls="--")
    axes[0].set_xlabel("alpha (ŷ projected on [c_A, c_B])")
    axes[0].set_title(f"D3: interpolation coefficient, forked anchors ({h_label})")
    axes[1].hist(orth.numpy(), bins=40, color=C_TEAL, alpha=0.85)
    axes[1].set_xlabel("orthogonal residual / ||c_B - c_A||")
    axes[1].set_title(f"D3: orthogonal residual ratio ({h_label})")
    for ax in axes:
        ax.set_ylabel("# forked anchors")
        _style(ax)
    fig.tight_layout()
    fig.savefig(figs / "d3_alpha_orth_hist.png", dpi=150)
    plt.close(fig)


def fig_d3_examples(scan, nn_idx, futh, yhat, h_label, figs: Path, n_ex: int = 6):
    idx = scan["fork_idx"]
    if idx.numel() == 0:
        return
    order = torch.argsort(scan["dbic"][idx], descending=True)
    pick = idx[order[:n_ex]].tolist()
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
    for ax, i in zip(axes.flat, pick):
        y = futh[nn_idx[i]].float()
        labels = torch.from_numpy(scan["labels"][i].astype(np.int64))
        yc = y - y.mean(0)
        _, _, vh = torch.linalg.svd(yc, full_matrices=False)
        b = vh[:2]  # (2, 384)
        pts = yc @ b.T
        ph = (yhat[i].float() - y.mean(0)) @ b.T
        for lab, col in ((0, C_BLUE), (1, C_ORANGE)):
            m = labels == lab
            ax.scatter(pts[m, 0], pts[m, 1], s=22, color=col, alpha=0.8,
                       label=f"cluster {'AB'[lab]}")
            cm = pts[m].mean(0)
            ax.scatter([cm[0]], [cm[1]], s=90, color=col, marker="X",
                       edgecolor="white", linewidth=1.2, zorder=5)
        ax.scatter([ph[0]], [ph[1]], s=140, color=C_RED, marker="*",
                   edgecolor="white", linewidth=0.8, zorder=6, label="ŷ (FD pred)")
        ax.set_title(f"anchor {i} (ΔBIC={scan['dbic'][i]:.0f})", fontsize=9)
        _style(ax)
    handles, labels_ = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle(f"D3: example forked anchors ({h_label}), neighbor futures in local 2D PCA", fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(figs / "d3_fork_examples.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:  # noqa: PLR0915
    from scipy.stats import spearmanr
    from sklearn.mixture import GaussianMixture

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--grids", type=Path, default=None, help="val_grids.pt (default: <cache-dir>/../val_grids.pt or <cache-dir>/val_grids.pt)")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--gap", type=float, default=90.0, help="same-drive exclusion gap in sample-order units")
    ap.add_argument("--jobs", type=int, default=8, help="joblib workers for D3 GMMs")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--synthetic-n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.perf_counter()
    if args.synthetic:
        cache_dir = args.cache_dir or com.SYNTH_CACHE_DIR
        out_dir = args.out_dir or com.SYNTH_RESULTS_DIR
        com.generate_synthetic_cache(cache_dir, n=args.synthetic_n, seed=args.seed, force=True)
    else:
        cache_dir = args.cache_dir or com.DEFAULT_CACHE_DIR
        out_dir = args.out_dir or com.DEFAULT_RESULTS_DIR
    grids_path = args.grids
    if grids_path is None:
        for cand in (Path(cache_dir) / "val_grids.pt", Path(cache_dir).parent / "val_grids.pt"):
            if cand.exists():
                grids_path = cand
                break
    figs = Path(out_dir) / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    device = args.device or pick_device()
    print(f"[phase0] cache={cache_dir} grids={grids_path} out={out_dir} device={device}", flush=True)

    data = com.load_cache(cache_dir)
    n = data["sample_id"].shape[0]
    drives, drive_names = com.drive_codes(data["input_id"])
    order, order_src = com.sample_order(drives, data.get("frame_idx"))
    strata, strata_info = com.motion_strata(
        data["speed"][:, 5], data["brake"][:, 5], data["steer"][:, 5], data["turn"][:, 5]
    )
    print(f"[phase0] order source: {order_src}; strata: {strata_info['counts']}", flush=True)

    fut = data["fut_dino_pooled"].float()  # (n, 5, 384)
    yhat = data["pred_fd_pooled_h1"].float()
    c, _, _ = com.standardize(torch.cat([data["ctx_foresight_pooled"].float(),
                                         data["ctx_action_summary"].float()], dim=1))
    cs, _, _ = com.standardize(torch.cat([data["ctx_obs_summary"].float(),
                                          data["ctx_action_summary"].float()], dim=1))

    # ---- k-NN -------------------------------------------------------------- #
    try:
        nn_idx, n_excl = eligible_knn(c, drives, order, args.k, args.gap, device)
        nn_idx_s, n_excl_s = eligible_knn(cs, drives, order, args.k, args.gap, device)
    except torch.cuda.OutOfMemoryError:
        print("[phase0] OOM on GPU kNN, retrying on CPU", flush=True)
        device = "cpu"
        nn_idx, n_excl = eligible_knn(c, drives, order, args.k, args.gap, device)
        nn_idx_s, n_excl_s = eligible_knn(cs, drives, order, args.k, args.gap, device)
    mean_excl = float((n_excl - 1).float().mean())
    print(f"[phase0] anti-leakage: mean excluded candidates/anchor (beyond self) = {mean_excl:.1f}", flush=True)

    # ---- D0 ---------------------------------------------------------------- #
    mpd, ctr = entropy_maps(nn_idx, fut, device)
    mpd_s, _ = entropy_maps(nn_idx_s, fut, device)
    spear_cs = [float(spearmanr(mpd[:, h].numpy(), mpd_s[:, h].numpy()).statistic) for h in range(5)]
    d0 = {
        "entropy_mean_pairwise": {
            "overall": _curve(mpd),
            "per_stratum": {
                com.STRATUM_NAMES[s]: _curve(mpd[strata == s]) for s in range(4)
            },
            "low_entropy_floor_p5": torch.quantile(mpd, 0.05, dim=0).tolist(),
        },
        "entropy_cov_trace": {
            "overall": _curve(ctr),
            "low_entropy_floor_p5": torch.quantile(ctr, 0.05, dim=0).tolist(),
        },
        "context_secondary_cS": {
            "overall": _curve(mpd_s),
            "spearman_vs_primary_per_H": spear_cs,
        },
    }
    fig_entropy(mpd, mpd_s, strata, figs)

    # ---- D1 ---------------------------------------------------------------- #
    ent1 = mpd[:, 0]
    quint = quintile_assign(ent1)
    r = d1_ratio(nn_idx, fut[:, 0], yhat, device)
    med_r = [float(r[quint == q].median()) for q in range(5)]
    d1 = {
        "median_r_per_quintile": med_r,
        "mean_r_per_quintile": [float(r[quint == q].mean()) for q in range(5)],
        "spearman_entropy_r": float(spearmanr(ent1.numpy(), r.numpy()).statistic),
    }
    fig_d1(med_r, figs)

    # ---- D2 ---------------------------------------------------------------- #
    pn = yhat.norm(dim=1)
    gn = fut[:, 0].norm(dim=1)
    ratio_q = [float(pn[quint == q].mean() / gn[quint == q].mean()) for q in range(5)]
    per_patch = None
    if grids_path is not None and Path(grids_path).exists():
        grids = torch.load(grids_path, map_location="cpu", weights_only=False)
        m = grids["sample_id"].shape[0]
        if torch.equal(grids["sample_id"], data["sample_id"][:m]):
            gp = grids["pred_fd_grid_h1"].float().norm(dim=2)  # (m, 256)
            gg = grids["gt_grid_h1"].float().norm(dim=2)
            qg = quint[:m]
            per_patch = [
                float(gp[qg == q].mean() / gg[qg == q].mean()) if (qg == q).any() else None
                for q in range(5)
            ]
        else:
            print("[phase0] WARNING grids sample_id does not match cache head; skipping per-patch D2", flush=True)
    q5 = quint == 4
    res = (yhat - fut[:, 0])[q5]
    resc = (res - res.mean(0)).double().numpy()
    _, sv, vt = np.linalg.svd(resc, full_matrices=False)
    evr1 = float(sv[0] ** 2 / (sv**2).sum())
    proj = (resc @ vt[0])[:, None]
    gm1 = GaussianMixture(1, covariance_type="full", n_init=5, random_state=0).fit(proj)
    gm2 = GaussianMixture(2, covariance_type="full", n_init=5, random_state=0).fit(proj)
    d2 = {
        "note": "norm_ratio is the POOLED-vector variant of the brief's per-patch "
        "statistic; per_patch_ratio (grids subset) is the per-patch version",
        "norm_ratio_per_quintile": ratio_q,
        "per_patch_ratio_per_quintile": per_patch,
        "residual_pca_top_quintile": {
            "n": int(q5.sum()),
            "pc1_explained_variance_ratio": evr1,
            "gmm_pc1_bic1": float(gm1.bic(proj)),
            "gmm_pc1_bic2": float(gm2.bic(proj)),
            "gmm2_better": bool(gm2.bic(proj) < gm1.bic(proj)),
            "gmm2_weights": gm2.weights_.tolist(),
            "gmm2_means": gm2.means_.flatten().tolist(),
        },
    }

    # ---- D3 ---------------------------------------------------------------- #
    print(f"[phase0] D3 fork scan (n={n} anchors x 2 horizons, jobs={args.jobs}) ...", flush=True)
    scan1 = fork_scan(nn_idx, fut[:, 0], yhat, device, args.jobs)
    scan5 = fork_scan(nn_idx, fut[:, 4], yhat, device, args.jobs)
    f1, f5 = scan1["fork"], scan5["fork"]

    def _prev(f):
        return {
            "overall": float(f.float().mean()),
            "per_stratum": {
                com.STRATUM_NAMES[s]: (float(f[strata == s].float().mean()) if (strata == s).any() else None)
                for s in range(4)
            },
            "n_fork": int(f.sum()),
        }

    def _hist_stats(v):
        if v.numel() == 0:
            return {"n": 0}
        return {
            "n": int(v.numel()),
            "median": float(v.median()),
            "mean": float(v.mean()),
            "p10": float(torch.quantile(v, 0.10)),
            "p90": float(torch.quantile(v, 0.90)),
            "frac_in_0.25_0.75": float(((v > 0.25) & (v < 0.75)).float().mean()),
        }

    d3 = {
        "fork_prevalence_H1": _prev(f1),
        "fork_prevalence_H5": _prev(f5),
        "n_fork_H5_not_H1": int((f5 & ~f1).sum()),
        "n_fork_H1_not_H5": int((f1 & ~f5).sum()),
        "alpha_H1": _hist_stats(scan1["alpha"]),
        "alpha_H5": _hist_stats(scan5["alpha"]),
        "orth_ratio_H1": _hist_stats(scan1["orth"]),
        "orth_ratio_H5": _hist_stats(scan5["orth"]),
    }
    if scan1["fork_idx"].numel() > 0:
        fig_d3_hists(scan1["alpha"], scan1["orth"], "H=1", figs)
        fig_d3_examples(scan1, nn_idx, fut[:, 0], yhat, "H=1", figs)
    elif scan5["fork_idx"].numel() > 0:
        fig_d3_hists(scan5["alpha"], scan5["orth"], "H=5 (no H=1 forks)", figs)
        fig_d3_examples(scan5, nn_idx, fut[:, 4], yhat, "H=5 (no H=1 forks)", figs)

    # ---- D4 (proxy) --------------------------------------------------------- #
    gas = data["gas"].float()
    brake = data["brake"].float()
    conflict = ((gas[:, 6:11] > com.GAS_THRESH) & (brake[:, 6:11] > com.BRAKE_THRESH)).any(1)
    r_quint = quintile_assign(r)
    grp = (quint == 4) & (r_quint == 0)
    a = float((conflict & grp).sum()) + 0.5
    b = float((~conflict & grp).sum()) + 0.5
    cc = float((conflict & ~grp).sum()) + 0.5
    dd = float((~conflict & ~grp).sum()) + 0.5
    odds = (a * dd) / (b * cc)
    se = (1 / a + 1 / b + 1 / cc + 1 / dd) ** 0.5
    d4 = {
        "note": "SENSOR-SIGNAL PROXY (cached gas/brake at frames 6..10), not the "
        "full pedal_conflict_stats pipeline; Haldane +0.5 correction, Woolf CI",
        "conflict_rate_overall": float(conflict.float().mean()),
        "group_def": "top-entropy quintile (H=1, mean-pairwise) AND bottom-r quintile",
        "n_group": int(grp.sum()),
        "n_conflict_in_group": int((conflict & grp).sum()),
        "odds_ratio": odds,
        "ci95": [float(np.exp(np.log(odds) - 1.96 * se)), float(np.exp(np.log(odds) + 1.96 * se))],
    }

    out = {
        "config": {
            "cache_dir": str(cache_dir),
            "grids": str(grids_path) if grids_path else None,
            "n_samples": n,
            "k": args.k,
            "gap_samples": args.gap,
            "order_source": order_src,
            "device": device,
            "n_drives": len(drive_names),
            "mean_excluded_candidates_beyond_self": mean_excl,
            "mean_excluded_cS": float((n_excl_s - 1).float().mean()),
            "strata": strata_info,
            "synthetic": bool(args.synthetic),
        },
        "D0": d0,
        "D1": d1,
        "D2": d2,
        "D3": d3,
        "D4": d4,
    }

    # ---- synthetic self-test asserts ---------------------------------------- #
    if args.synthetic:
        truth = torch.load(Path(cache_dir) / "synth_truth.pt", map_location="cpu", weights_only=False)
        fork_gt = truth["fork"]
        means_planted = mpd[fork_gt].mean(0)
        diffs = means_planted[1:] - means_planted[:-1]
        checks = {}
        checks["D0_entropy_grows_with_H_on_planted"] = {
            "means": means_planted.tolist(),
            "pass": bool((diffs > 0).all()),
        }
        checks["D1_median_r_top_quintile_lt_0.5"] = {"value": med_r[4], "pass": med_r[4] < 0.5}
        recall5 = float(f5[fork_gt].float().mean())
        fp5 = float(f5[~fork_gt].float().mean())
        checks["D3_recall_planted_forks_H5_ge_0.6"] = {
            "recall": recall5, "false_pos_rate_nonplanted": fp5, "pass": recall5 >= 0.6,
        }
        a5 = scan5["alpha"][fork_gt[scan5["fork_idx"]]]
        med_a5 = float(a5.median()) if a5.numel() else float("nan")
        checks["D3_alpha_H5_centered_0.5"] = {
            "median": med_a5, "n": int(a5.numel()),
            "pass": bool(a5.numel() > 0 and 0.35 <= med_a5 <= 0.65),
        }
        out["synthetic_asserts"] = checks
        print("\n[phase0] SYNTHETIC SELF-TEST:")
        for k, v in checks.items():
            print(f"  {'PASS' if v['pass'] else 'FAIL'}  {k}: {json.dumps({kk: vv for kk, vv in v.items() if kk != 'pass'})}")
        assert all(v["pass"] for v in checks.values()), "synthetic self-test FAILED"

    out_path = Path(out_dir) / "phase0_d0_d4.json"
    out_path.write_text(json.dumps(com.to_jsonable(out), indent=2) + "\n")

    # ---- printed summary ----------------------------------------------------- #
    print(f"\n[phase0] ===== SUMMARY ({time.perf_counter() - t0:.0f}s) =====")
    print(f"  D0 entropy mean per H:        {[round(x, 3) for x in mpd.mean(0).tolist()]}")
    print(f"     floor (p5) per H:          {[round(x, 3) for x in d0['entropy_mean_pairwise']['low_entropy_floor_p5']]}")
    print(f"     spearman(c, cS) per H:     {[round(x, 3) for x in spear_cs]}")
    print(f"  D1 median r per quintile:     {[round(x, 3) for x in med_r]}  spearman(ent, r)={d1['spearman_entropy_r']:.3f}")
    print(f"  D2 norm ratio per quintile:   {[round(x, 3) for x in ratio_q]}")
    print(f"     residual PC1 EVR (Q5):     {evr1:.3f}  GMM2 better: {d2['residual_pca_top_quintile']['gmm2_better']}")
    print(f"  D3 fork prevalence H1/H5:     {d3['fork_prevalence_H1']['overall']:.3f} / {d3['fork_prevalence_H5']['overall']:.3f}"
          f"  (H5 not H1: {d3['n_fork_H5_not_H1']})")
    print(f"     alpha H1 median:           {d3['alpha_H1'].get('median')}")
    print(f"  D4 conflict OR (proxy):       {odds:.2f}  CI95={d4['ci95']}")
    print(f"  JSON -> {out_path}")
    print(f"  figs -> {figs}")


if __name__ == "__main__":
    import multiprocessing as mp
    import os

    mp.set_forkserver_preload(["rbyte", "polars"])
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
