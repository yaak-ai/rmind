"""Analyze waypoint-probe predictions: permutation importance + mirror test.

Reads parquet written by the `*_wp_probe` inference configs and reports, per
checkpoint, how much steering depends on waypoints (permutation) and whether it
uses them *directionally* (mirror), overall and on turning frames.

Layout expected (one dir per condition, each with predictions/*.parquet):
    <root>/<model>/<permutation>-<reflection>/predictions/**/*.parquet

Conditions needed per model: baseline-none, waypoints-none, baseline-lateral
(optionally baseline-longitudinal as a control).

Usage:
    python analyze.py <root> --objective policy --models mean concat ...
    python analyze.py <root> --objective inverse_dynamics --models base pe
"""

import argparse
from pathlib import Path

import numpy as np
import polars as pl

WP = "batch/data/waypoints/xy_normalized"
TURN_THRESHOLD = 0.05
NDIM_FLAT = 2  # policy prediction is (N, 1); inverse-dynamics is (N, T-1, 1)


def cols(obj: str) -> tuple[str, str, str]:
    base = f"{obj}/{{}}/value/continuous/steering_angle"
    return (
        base.format("prediction_value"),
        base.format("score_l1"),
        base.format("ground_truth"),
    )


def load(root: str, model: str, cond: str) -> pl.DataFrame:
    fs = sorted(Path(root, model, cond, "predictions").rglob("*.parquet"))
    if not fs:
        msg = f"no parquet for {model}/{cond} under {root}"
        raise FileNotFoundError(msg)
    return pl.concat([pl.read_parquet(f) for f in fs])


def stack(df: pl.DataFrame, col: str) -> np.ndarray:
    return np.stack(df[col].to_numpy())  # ty:ignore[no-matching-overload]


def steer_pred(df: pl.DataFrame, col: str) -> np.ndarray:
    a = stack(df, col)
    return a[:, 0] if a.ndim == NDIM_FLAT else a[:, :, 0].ravel()


def steer_l1(df: pl.DataFrame, col: str) -> np.ndarray:
    a = stack(df, col)
    return a[:, 0] if a.ndim == NDIM_FLAT else a[:, :, 0].ravel()


def gt_aligned(df: pl.DataFrame, col: str, objective: str) -> np.ndarray:
    g = stack(df, col)  # (N, T, 1); T = full episode length
    # Align gt to the prediction's timesteps: the policy head predicts ONLY the
    # last step (-> N values); inverse-dynamics predicts steps 1..T-1 (-> N*(T-1)).
    return g[:, -1, 0] if objective == "policy" else g[:, 1:, 0].ravel()


def report(root: str, objective: str, models: list[str]) -> None:
    p_col, l1_col, gt_col = cols(objective)

    print("=" * 92)
    print("ROW-ALIGNMENT (lateral run: wp[...,1]==base, wp[...,0]==-base)")
    for m in models:
        b = stack(load(root, m, "baseline-none"), WP)
        lat = stack(load(root, m, "baseline-lateral"), WP)
        long_ok = np.allclose(b[..., 1], lat[..., 1], atol=1e-4)
        lat_ok = np.allclose(b[..., 0], -lat[..., 0], atol=1e-4)
        print(f"  {m:10s} long_match={long_ok} lat_negated={lat_ok}")

    print("=" * 92)
    print(f"PERMUTATION IMPORTANCE (shuffle wp; turn=|gt|>={TURN_THRESHOLD})")
    for m in models:
        b, p = load(root, m, "baseline-none"), load(root, m, "waypoints-none")
        lb, lp = steer_l1(b, l1_col), steer_l1(p, l1_col)
        tb = np.abs(gt_aligned(b, gt_col, objective)) >= TURN_THRESHOLD
        tp = np.abs(gt_aligned(p, gt_col, objective)) >= TURN_THRESHOLD
        da = 100 * (lp.mean() - lb.mean()) / lb.mean()
        dt = 100 * (lp[tp].mean() - lb[tb].mean()) / lb[tb].mean()
        print(
            f"  {m:10s} all {lb.mean():.4f}->{lp.mean():.4f} ({da:+.0f}%)   "
            f"turn {lb[tb].mean():.4f}->{lp[tp].mean():.4f} ({dt:+.0f}%)"
        )

    print("=" * 92)
    print("MIRROR (negate lateral wp; paired). corr<0 & flips => directional use")
    for m in models:
        b, lat = load(root, m, "baseline-none"), load(root, m, "baseline-lateral")
        pb, pm = steer_pred(b, p_col), steer_pred(lat, p_col)
        tm = np.abs(gt_aligned(b, gt_col, objective)) >= TURN_THRESHOLD
        ct = np.corrcoef(pb[tm], pm[tm])[0, 1]
        ft = (np.sign(pb[tm]) != np.sign(pm[tm])).mean()
        ca = np.corrcoef(pb, pm)[0, 1]
        fa = (np.sign(pb) != np.sign(pm)).mean()
        print(
            f"  {m:10s} TURN corr={ct:+.3f} flip={ft:.2f}   ALL corr={ca:+.3f} flip={fa:.2f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--objective", default="policy")
    ap.add_argument("--models", nargs="+", required=True)
    args = ap.parse_args()
    report(args.root, args.objective, args.models)


if __name__ == "__main__":
    main()
