"""G-1 / L2 incremental-content probes on the Phase-0 val feature cache.

Question: does the encoder's foresight content add near-future signal beyond
the present image (F5 vs F2) / beyond the policy summaries (F6 vs F4)?

Feature sets (pooled encoder/DINO features from the cache contract):
  F1 = ctx_foresight_pooled            F2 = img_dino_pooled_cur
  F3 = ctx_obs_summary                 F4 = ctx_obs_summary || ctx_action_summary
  F5 = F2 || F1                        F6 = F4 || F1

Targets (horizon limited to +5 steps ~ 1.67 s by clip11, not the brief's 2 s):
  - speed at clip frames 6..10 (5 ridge regressions, R^2 on probe-eval)
  - brake onset  = brake[5] <= thresh AND any(brake[6..10] > thresh),
    evaluated only on not-currently-braking samples (logistic, AUROC)
  - gas onset    analogous with GAS_THRESH

Leakage-free split BY DRIVE inside the val cache (~60/40 by samples,
deterministic greedy assignment, documented in the JSON output).

Outputs: foresight_mm/results/l2_content_probe.json + printed table.
Self-test: --synthetic (planted linear signal in the latent; F2 redundant with
F1 -> asserts R^2 > 0.5 and ~zero incremental delta).

Usage:
  uv run python src/rmind/scripts/foresight_leverage_l2.py [--cache-dir ...] [--synthetic]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import foresight_phase0_common as com  # noqa: E402

ALPHAS = [10.0**e for e in range(-2, 5)]  # 1e-2 .. 1e4
FEATURE_SETS = ["F1", "F2", "F3", "F4", "F5", "F6"]


# --------------------------------------------------------------------------- #
# split
# --------------------------------------------------------------------------- #
def split_by_drive_60_40(input_ids: list[str]) -> tuple[torch.Tensor, dict]:
    """Deterministic greedy drive assignment to a ~60/40 probe-train/probe-eval
    sample split. Returns (is_train bool mask, assignment doc)."""
    codes, names = com.drive_codes(input_ids)
    counts = {d: int((codes == i).sum()) for i, d in enumerate(names)}
    n_total = len(input_ids)
    # sort by (count desc, name) for determinism; assign a drive to train only
    # if that moves the train sample count closer to the 60% target
    target = 0.6 * n_total
    train_drives, eval_drives, n_train = [], [], 0
    for d in sorted(counts, key=lambda d: (-counts[d], d)):
        if abs(n_train + counts[d] - target) < abs(n_train - target):
            train_drives.append(d)
            n_train += counts[d]
        else:
            eval_drives.append(d)
    if not eval_drives:  # guarantee a non-empty eval side
        d = min(train_drives, key=lambda d: (counts[d], d))
        train_drives.remove(d)
        eval_drives.append(d)
    if not train_drives:
        d = min(eval_drives, key=lambda d: (counts[d], d))
        eval_drives.remove(d)
        train_drives.append(d)
    is_train = torch.tensor([i in train_drives for i in input_ids])
    doc = {
        "drive_counts": counts,
        "train_drives": train_drives,
        "eval_drives": eval_drives,
        "n_train": int(is_train.sum()),
        "n_eval": int((~is_train).sum()),
        "train_frac": float(is_train.float().mean()),
    }
    return is_train, doc


def fold_assignment(codes: torch.Tensor, is_train: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """3 CV folds inside probe-train: grouped by drive when >= 3 train drives,
    else shuffled sample-level KFold (documented)."""
    tr_idx = torch.nonzero(is_train).flatten()
    tr_codes = codes[tr_idx]
    uniq = tr_codes.unique().tolist()
    folds = torch.zeros(len(tr_idx), dtype=torch.long)
    if len(uniq) >= 3:
        sizes = [(int((tr_codes == u).sum()), u) for u in uniq]
        load = [0, 0, 0]
        for sz, u in sorted(sizes, reverse=True):
            f = int(np.argmin(load))
            folds[tr_codes == u] = f
            load[f] += sz
    else:
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(tr_idx), generator=g)
        for f in range(3):
            folds[perm[f::3]] = f
    return folds


# --------------------------------------------------------------------------- #
# ridge
# --------------------------------------------------------------------------- #
def ridge_fit(x: torch.Tensor, y: torch.Tensor, alpha: float) -> tuple[torch.Tensor, float]:
    xm, ym = x.mean(0), y.mean()
    xc, yc = x - xm, y - ym
    d = x.shape[1]
    a = xc.T @ xc + alpha * torch.eye(d, dtype=x.dtype)
    beta = torch.linalg.solve(a, xc.T @ yc)
    intercept = float(ym - xm @ beta)
    return beta, intercept


def r2_score(y: torch.Tensor, yp: torch.Tensor) -> float:
    ss_res = float(((y - yp) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def ridge_probe(
    x_tr: torch.Tensor, y_tr: torch.Tensor, x_ev: torch.Tensor, y_ev: torch.Tensor,
    folds: torch.Tensor,
) -> tuple[float, float]:
    """Sweep alpha via 3-fold CV on probe-train, refit, return (eval R^2, alpha)."""
    x_tr, y_tr = x_tr.double(), y_tr.double()
    x_ev, y_ev = x_ev.double(), y_ev.double()
    best_alpha, best_cv = ALPHAS[0], -1e18
    for alpha in ALPHAS:
        scores = []
        for f in range(3):
            m = folds != f
            if m.all() or not m.any():
                continue
            beta, b0 = ridge_fit(x_tr[m], y_tr[m], alpha)
            scores.append(r2_score(y_tr[~m], x_tr[~m] @ beta + b0))
        cv = float(np.mean(scores)) if scores else -1e18
        if cv > best_cv:
            best_cv, best_alpha = cv, alpha
    beta, b0 = ridge_fit(x_tr, y_tr, best_alpha)
    return r2_score(y_ev, x_ev @ beta + b0), best_alpha


def logistic_probe(
    x_tr: torch.Tensor, y_tr: torch.Tensor, x_ev: torch.Tensor, y_ev: torch.Tensor
) -> float:
    """Logistic regression (C=1.0, standardized inputs), returns eval AUROC."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    if y_tr.unique().numel() < 2 or y_ev.unique().numel() < 2:
        return float("nan")
    clf = LogisticRegression(C=1.0, max_iter=2000, random_state=0)
    clf.fit(x_tr.numpy(), y_tr.numpy().astype(np.int64))
    p = clf.predict_proba(x_ev.numpy())[:, 1]
    return float(roc_auc_score(y_ev.numpy().astype(np.int64), p))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--synthetic-n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.perf_counter()
    if args.synthetic:
        cache_dir = args.cache_dir or com.SYNTH_CACHE_DIR
        out_dir = args.out_dir or com.SYNTH_RESULTS_DIR
        com.generate_synthetic_cache(cache_dir, n=args.synthetic_n, seed=args.seed)
    else:
        cache_dir = args.cache_dir or com.DEFAULT_CACHE_DIR
        out_dir = args.out_dir or com.DEFAULT_RESULTS_DIR
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    data = com.load_cache(cache_dir)
    codes, _ = com.drive_codes(data["input_id"])
    is_train, split_doc = split_by_drive_60_40(data["input_id"])
    folds = fold_assignment(codes, is_train, seed=args.seed)
    print(f"[l2] split: train={split_doc['train_drives']} ({split_doc['n_train']}) | "
          f"eval={split_doc['eval_drives']} ({split_doc['n_eval']}) "
          f"train_frac={split_doc['train_frac']:.3f}", flush=True)

    fore = data["ctx_foresight_pooled"].float()
    img = data["img_dino_pooled_cur"].float()
    obs = data["ctx_obs_summary"].float()
    act = data["ctx_action_summary"].float()
    feats = {
        "F1": fore,
        "F2": img,
        "F3": obs,
        "F4": torch.cat([obs, act], dim=1),
        "F5": torch.cat([img, fore], dim=1),
        "F6": torch.cat([obs, act, fore], dim=1),
    }
    # standardize with train stats
    for k in feats:
        _, mu, sd = com.standardize(feats[k][is_train])
        feats[k] = (feats[k] - mu) / sd

    speed = data["speed"].float()
    brake = data["brake"].float()
    gas = data["gas"].float()

    targets: dict[str, dict] = {}
    for h in range(1, 6):
        targets[f"speed_h{h}"] = {"kind": "reg", "y": speed[:, 5 + h], "valid": torch.ones_like(is_train)}
    b_now = brake[:, 5] > com.BRAKE_THRESH
    targets["brake_onset"] = {
        "kind": "bin",
        "y": ((brake[:, 6:11] > com.BRAKE_THRESH).any(1) & ~b_now).float(),
        "valid": ~b_now,
    }
    g_now = gas[:, 5] > com.GAS_THRESH
    targets["gas_onset"] = {
        "kind": "bin",
        "y": ((gas[:, 6:11] > com.GAS_THRESH).any(1) & ~g_now).float(),
        "valid": ~g_now,
    }

    results: dict[str, dict] = {}
    for tname, t in targets.items():
        v = t["valid"]
        tr = is_train & v
        ev = ~is_train & v
        row: dict[str, float | None] = {}
        # folds vector is indexed over train samples; restrict to valid ones
        tr_pos_in_train = v[is_train]  # mask over the train-ordered folds vector
        folds_t = folds[tr_pos_in_train]
        for fname in FEATURE_SETS:
            x = feats[fname]
            if t["kind"] == "reg":
                r2, alpha = ridge_probe(x[tr], t["y"][tr], x[ev], t["y"][ev], folds_t)
                row[fname] = r2
                row[f"{fname}_alpha"] = alpha
            else:
                row[fname] = logistic_probe(x[tr], t["y"][tr], x[ev], t["y"][ev])
        row["delta_F5_vs_F2"] = (row["F5"] - row["F2"]) if row["F5"] is not None else None
        row["delta_F6_vs_F4"] = (row["F6"] - row["F4"]) if row["F6"] is not None else None
        row["n_train"] = int(tr.sum())
        row["n_eval"] = int(ev.sum())
        if t["kind"] == "bin":
            row["base_rate_train"] = float(t["y"][tr].mean())
            row["base_rate_eval"] = float(t["y"][ev].mean())
        results[tname] = row
        print(f"[l2] {tname}: " + " ".join(f"{f}={row[f]:.3f}" if isinstance(row[f], float) else f"{f}=NA"
                                           for f in FEATURE_SETS), flush=True)

    metric = {k: ("R2" if targets[k]["kind"] == "reg" else "AUROC") for k in targets}
    out = {
        "config": {
            "cache_dir": str(cache_dir),
            "n_samples": int(data["sample_id"].shape[0]),
            "alphas": ALPHAS,
            "cv": "3-fold grouped by drive if >=3 train drives else shuffled KFold(seed)",
            "logistic": "sklearn LogisticRegression C=1.0 max_iter=2000 (standardized inputs)",
            "horizon_note": "clip11 -> +5 steps ~ 1.67 s, not the brief's 2 s",
            "feature_sets": {
                "F1": "ctx_foresight_pooled", "F2": "img_dino_pooled_cur",
                "F3": "ctx_obs_summary", "F4": "ctx_obs_summary||ctx_action_summary",
                "F5": "F2||F1", "F6": "F4||F1",
            },
            "split": split_doc,
            "synthetic": bool(args.synthetic),
        },
        "metric_per_target": metric,
        "results": results,
        "headline": {
            "delta_F5_vs_F2": {k: results[k]["delta_F5_vs_F2"] for k in results},
            "delta_F6_vs_F4": {k: results[k]["delta_F6_vs_F4"] for k in results},
        },
    }

    # ---- printed table ------------------------------------------------------ #
    cols = [*FEATURE_SETS, "delta_F5_vs_F2", "delta_F6_vs_F4"]
    print("\n[l2] ===== TABLE (eval metric per feature set) =====")
    print(f"{'target':<14}{'metric':<8}" + "".join(f"{c:>16}" for c in cols))
    for tname, row in results.items():
        cells = "".join(
            f"{row[c]:>16.4f}" if isinstance(row[c], float) and not np.isnan(row[c]) else f"{'NA':>16}"
            for c in cols
        )
        print(f"{tname:<14}{metric[tname]:<8}{cells}")

    # ---- synthetic self-test ------------------------------------------------- #
    if args.synthetic:
        checks = {
            "planted_linear_signal_F1_speed_h1_R2_gt_0.5": {
                "value": results["speed_h1"]["F1"], "pass": results["speed_h1"]["F1"] > 0.5,
            },
            "planted_linear_signal_F2_speed_h1_R2_gt_0.5": {
                "value": results["speed_h1"]["F2"], "pass": results["speed_h1"]["F2"] > 0.5,
            },
            "redundant_feature_near_zero_delta_F5_vs_F2": {
                "value": float(np.mean([abs(results[f"speed_h{h}"]["delta_F5_vs_F2"]) for h in range(1, 6)])),
                "pass": float(np.mean([abs(results[f"speed_h{h}"]["delta_F5_vs_F2"]) for h in range(1, 6)])) < 0.05,
            },
        }
        out["synthetic_asserts"] = checks
        print("\n[l2] SYNTHETIC SELF-TEST:")
        for k, v in checks.items():
            print(f"  {'PASS' if v['pass'] else 'FAIL'}  {k}: value={v['value']:.4f}")
        assert all(v["pass"] for v in checks.values()), "l2 synthetic self-test FAILED"

    out_path = Path(out_dir) / "l2_content_probe.json"
    out_path.write_text(json.dumps(com.to_jsonable(out), indent=2) + "\n")
    print(f"\n[l2] JSON -> {out_path}  ({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    import multiprocessing as mp
    import os

    mp.set_forkserver_preload(["rbyte", "polars"])
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
