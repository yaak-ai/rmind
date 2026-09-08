"""Read & visualize policy predictions (ground-truth vs prediction) for manual analysis.

Works on the parquet written by DataFramePredictionWriter with the
`policy_allfields` inference config, e.g.:

    outputs/<date>/<time>/predictions/yaak/alex-tmp/model-4vqatiom:v4.parquet

Each row is one 11-tick window. The model (4vqatiom) has history_steps=5 and
action_horizon=6, so the scored action target is at tick GT_IDX=5 and the
decision features are at tick FEAT_IDX=4 ("now"). The trajectory head predicts
6 future steps as chained *relative* poses (dx, dy, dyaw) — this script
integrates them back into an ego-local path for plotting.

Examples
--------
# 6 worst windows by gas loss -> PNGs (camera + trajectory + time-series + losses)
uv run python scripts/viz_policy_predictions.py PARQUET --worst gas_pedal --n 6 --out /tmp/viz

# a specific window
uv run python scripts/viz_policy_predictions.py PARQUET --sample "Niro131-HQ/2023-05-12--09-43-27#46976"

# just the bird's-eye trajectory (gt vs pred vs waypoints), no camera/time-series
uv run python scripts/viz_policy_predictions.py PARQUET --worst traj --n 8 --traj-only --out /tmp/traj

# quick loss summary, no plots
uv run python scripts/viz_policy_predictions.py PARQUET --summary
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

HISTORY = 5
HORIZON = 6
FEAT_IDX = HISTORY - 1  # 4  -> last history tick ("now")
GT_IDX = HISTORY  # 5  -> first scored horizon tick
DATA_ROOT = "/nasa/drives/yaak/data"
CAM = "cam_front_left"

CONT = ["gas_pedal", "brake_pedal", "steering_angle"]
HEADS = [("continuous", a) for a in CONT] + [("discrete", "turn_signal")]


# --------------------------------------------------------------------------- io
def arr(df: pl.DataFrame, name: str) -> np.ndarray:
    """Column -> dense np array with the row axis first."""
    return np.asarray(df[name].to_list(), dtype=float)


def scalar(df: pl.DataFrame, name: str) -> np.ndarray:
    return df[name].to_numpy().astype(float)


def loss_of(df: pl.DataFrame, head: str) -> np.ndarray:
    mod = "discrete" if head == "turn_signal" else "continuous"
    return scalar(df, f"policy/loss/value/{mod}/{head}")


def _integrate(dxy: np.ndarray, dyaw: np.ndarray) -> np.ndarray:
    """Vectorized chaining of per-step relative (dx,dy,dyaw) -> absolute ego
    positions. dxy: (n,H,2), dyaw: (n,H) -> (n,H,2) positions (origin dropped)."""
    th = np.concatenate(
        [np.zeros_like(dyaw[:, :1]), np.cumsum(dyaw, axis=1)[:, :-1]], axis=1
    )
    c, s = np.cos(th), np.sin(th)
    gx = dxy[..., 0] * c - dxy[..., 1] * s
    gy = dxy[..., 0] * s + dxy[..., 1] * c
    return np.stack([np.cumsum(gx, axis=1), np.cumsum(gy, axis=1)], axis=-1)


def traj_metrics(df: pl.DataFrame):
    """Chained-path ADE/FDE: L2 between integrated pred and gt ego positions
    (the conventional trajectory error, not per-step relative-vector error)."""
    pp = _integrate(
        arr(df, "policy/trajectory_value/value/xy"),
        arr(df, "policy/trajectory_value/value/yaw"),
    )
    gg = _integrate(
        arr(df, "policy/trajectory_gt/value/xy"),
        arr(df, "policy/trajectory_gt/value/yaw"),
    )
    l2 = np.linalg.norm(pp - gg, axis=-1)  # (n,H)
    return l2.mean(1), l2[:, -1]  # ADE, FDE


# ------------------------------------------------------------------ trajectory
def chain_relative(dxy: np.ndarray, dyaw: np.ndarray) -> np.ndarray:
    """Integrate per-step relative (dx, dy, dyaw) into an ego-local path.

    Returns (HORIZON+1, 2) starting at the origin (the ego pose at FEAT_IDX).
    dx/dy are expressed in the *previous* step's heading frame; yaw accumulates.
    x is forward along the reference heading, y is lateral.
    """
    pts = [(0.0, 0.0)]
    x = y = th = 0.0
    for i in range(len(dxy)):
        dx, dy = dxy[i]
        x += dx * np.cos(th) - dy * np.sin(th)
        y += dx * np.sin(th) + dy * np.cos(th)
        th += float(dyaw[i]) if dyaw is not None else 0.0
        pts.append((x, y))
    return np.asarray(pts)


def frame_path(drive: str, frame_idx: int, data_root: str) -> Path:
    return (
        Path(data_root)
        / drive
        / "frames"
        / f"{CAM}.pii.mp4"
        / "576x324"
        / f"{frame_idx:09d}.jpg"
    )


# ----------------------------------------------------------------- plot panels
def plot_bev(ax, row, *, title=True) -> None:
    pred_xy = np.asarray(row["policy/trajectory_value/value/xy"], dtype=float)
    gt_xy = np.asarray(row["policy/trajectory_gt/value/xy"], dtype=float)
    pred_yaw = np.asarray(row["policy/trajectory_value/value/yaw"], dtype=float)
    gt_yaw = np.asarray(row["policy/trajectory_gt/value/yaw"], dtype=float)

    gt = chain_relative(gt_xy, gt_yaw)
    pred = chain_relative(pred_xy, pred_yaw)

    # route waypoints at the reference tick (ego-local, stored /100 -> *100 = metres)
    wp = np.asarray(row["batch/data/waypoints/xy_normalized"], dtype=float)  # (11,10,2)
    wp_now = wp[FEAT_IDX] * 100.0

    # AXIS CONVENTION: because headings_denoised/heading is a compass heading
    # (0 ~ North) while positions are UTM (x=East, y=North), build_relative_
    # trajectory's rotation puts *forward* motion in component 1 (its "y_local")
    # and *lateral* in component 0 — the opposite of the "dx=forward" naming.
    # Verified empirically: forward-sum(comp1)/path-length = 0.999 on highway.
    # So we plot component 1 as forward (vertical) and component 0 as lateral.
    ax.plot(wp_now[:, 0], wp_now[:, 1], ".", color="0.7", ms=6, label="route waypoints")
    ax.plot(
        gt[:, 0], gt[:, 1], "-o", color="tab:green", ms=3, lw=2, label="gt trajectory"
    )
    ax.plot(
        pred[:, 0],
        pred[:, 1],
        "-o",
        color="tab:red",
        ms=3,
        lw=2,
        label="pred trajectory",
    )
    ax.plot(0, 0, "ks", ms=8)
    ax.annotate("ego", (0, 0), textcoords="offset points", xytext=(4, -12), fontsize=8)
    ade = np.linalg.norm(pred[1:] - gt[1:], axis=1).mean()
    fde = np.linalg.norm(pred[-1] - gt[-1])
    ax.set_aspect("equal", "datalim")
    ax.set_xlabel("lateral (m)")
    ax.set_ylabel("forward (m)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper right")
    if title:
        ax.set_title(f"trajectory  ADE={ade:.2f}m  FDE={fde:.2f}m", fontsize=9)


def plot_timeseries(ax, row) -> None:
    t = np.arange(11)
    speed = np.asarray(row["batch/data/meta/VehicleMotion/speed"], dtype=float)
    gas = np.asarray(
        row["batch/data/meta/VehicleMotion/gas_pedal_normalized"], dtype=float
    )
    brake = np.asarray(
        row["batch/data/meta/VehicleMotion/brake_pedal_normalized"], dtype=float
    )
    steer = np.asarray(
        row["batch/data/meta/VehicleMotion/steering_angle_normalized"], dtype=float
    )

    ax.plot(t, gas, "-o", ms=3, color="tab:green", label="gas gt")
    ax.plot(t, brake, "-o", ms=3, color="tab:red", label="brake gt")
    ax.plot(t, steer, "-o", ms=3, color="tab:blue", label="steer gt")
    ax.axvline(FEAT_IDX, color="0.6", ls=":", lw=1)
    ax.axvline(GT_IDX, color="k", ls="--", lw=1)
    ax.text(
        GT_IDX,
        1.02,
        "scored",
        fontsize=7,
        ha="center",
        transform=ax.get_xaxis_transform(),
    )

    # model predictions (single scored step) at GT_IDX
    def pv(a):
        return float(
            np.asarray(
                row[f"policy/prediction_value/value/continuous/{a}"], dtype=float
            )[0]
        )

    ax.plot(GT_IDX, pv("gas_pedal"), "*", color="darkgreen", ms=13, label="gas pred")
    ax.plot(GT_IDX, pv("brake_pedal"), "*", color="darkred", ms=13, label="brake pred")
    ax.plot(GT_IDX, pv("steering_angle"), "*", color="navy", ms=13, label="steer pred")

    ax2 = ax.twinx()
    ax2.plot(t, speed, "-", color="0.5", lw=1.5, label="speed")
    ax2.set_ylabel("speed (km/h)", color="0.4")
    ax.set_xlabel("tick (0..10);  : = features, -- = scored target")
    ax.set_ylabel("normalized pedal / steering")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, loc="upper left", ncol=2)


def loss_text(row) -> str:
    def g(path, i=0):
        v = row[path]
        return (
            float(np.asarray(v, dtype=float).reshape(-1)[i])
            if v is not None
            else float("nan")
        )

    lines = [
        f"{row['batch/meta/input_id']}",
        f"sample_id={row['batch/data/meta/sample_id']}",
        "",
    ]
    speed = np.asarray(row["batch/data/meta/VehicleMotion/speed"], dtype=float)
    lines.extend((
        f"speed(now)={speed[FEAT_IDX]:.1f}  speed(scored)={speed[GT_IDX]:.1f} km/h",
        "",
        f"{'head':14s}{'gt':>8s}{'pred':>8s}{'L1':>8s}{'std':>8s}{'NLL loss':>10s}",
    ))
    for mod, a in HEADS:
        gt = g(f"policy/ground_truth/value/{mod}/{a}", GT_IDX)
        pr = g(f"policy/prediction_value/value/{mod}/{a}")
        l1 = g(f"policy/score_l1/value/{mod}/{a}")
        std = g(f"policy/prediction_std/value/{mod}/{a}")
        ls = loss_of_row(row, a)
        lines.append(f"{a:14s}{gt:8.3f}{pr:8.3f}{l1:8.3f}{std:8.4f}{ls:10.3f}")
    ade, fde = row["_ade"], row["_fde"]
    lines.extend(("", f"trajectory   ADE={ade:.2f} m   FDE={fde:.2f} m"))
    return "\n".join(lines)


def loss_of_row(row, head):
    mod = "discrete" if head == "turn_signal" else "continuous"
    return float(row[f"policy/loss/value/{mod}/{head}"])


def render_window(row, out: Path, *, data_root: str, traj_only: bool):
    drive = row["batch/meta/input_id"]
    sid = row["batch/data/meta/sample_id"]
    fidx = int(
        np.asarray(
            row["batch/data/meta/ImageMetadata.cam_front_left/frame_idx"], dtype=int
        )[GT_IDX]
    )

    if traj_only:
        fig, ax = plt.subplots(figsize=(6, 6))
        plot_bev(ax, row)
        fig.suptitle(f"{drive}#{sid}", fontsize=9)
    else:
        fig = plt.figure(figsize=(16, 5))
        gs = fig.add_gridspec(1, 4, width_ratios=[1.25, 1.0, 1.35, 1.0])
        axc, axb, axt, axx = (fig.add_subplot(gs[0, i]) for i in range(4))

        # camera
        fp = frame_path(drive, fidx, data_root)
        try:
            axc.imshow(plt.imread(fp))
            axc.set_title(f"{CAM}  frame {fidx}", fontsize=8)
        except (FileNotFoundError, OSError, ValueError):
            axc.text(
                0.5,
                0.5,
                f"frame not found:\n{fp}",
                ha="center",
                va="center",
                fontsize=7,
            )
        axc.axis("off")

        plot_bev(axb, row)
        plot_timeseries(axt, row)
        axx.axis("off")
        axx.text(
            0.0,
            1.0,
            loss_text(row),
            family="monospace",
            fontsize=8,
            va="top",
            ha="left",
            transform=axx.transAxes,
        )
        fig.suptitle(f"{drive}#{sid}", fontsize=10)

    fig.tight_layout()
    safe = f"{drive.replace('/', '_')}__{sid}.png"
    out.mkdir(parents=True, exist_ok=True)
    dst = out / safe
    fig.savefig(dst, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return dst


# ------------------------------------------------------------------------ main
def attach_traj(df: pl.DataFrame) -> pl.DataFrame:
    ade, fde = traj_metrics(df)
    return df.with_columns(_ade=pl.Series(ade), _fde=pl.Series(fde))


def pick(df: pl.DataFrame, args) -> pl.DataFrame:
    if args.sample:
        drive, sid = args.sample.rsplit("#", 1)
        sub = df.filter(
            (pl.col("batch/meta/input_id") == drive)
            & (pl.col("batch/data/meta/sample_id") == int(sid))
        )
        if sub.height == 0:
            msg = f"no row for {args.sample}"
            raise SystemExit(msg)
        return sub
    key = args.worst
    if key == "traj":
        df = df.with_columns(_key=pl.col("_ade"))
    else:
        mod = "discrete" if key == "turn_signal" else "continuous"
        df = df.with_columns(_key=pl.col(f"policy/loss/value/{mod}/{key}"))
    return df.sort("_key", descending=True).head(args.n)


def print_summary(df: pl.DataFrame) -> None:
    for _, a in HEADS:
        loss_of(df, a)
    for _nm, _v in [
        ("traj_ADE(m)", df["_ade"].to_numpy()),
        ("traj_FDE(m)", df["_fde"].to_numpy()),
    ]:
        pass


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("parquet")
    ap.add_argument("--sample", help="DRIVE#SAMPLE_ID for a single window")
    ap.add_argument(
        "--worst",
        choices=[*CONT, "turn_signal", "traj"],
        help="rank windows by this loss head",
    )
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", default="/tmp/policy_viz")
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--traj-only", action="store_true")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()

    df = attach_traj(pl.read_parquet(args.parquet))

    if args.summary or (not args.sample and not args.worst):
        print_summary(df)
        if not (args.sample or args.worst):
            return

    sub = pick(df, args)
    out = Path(args.out)
    for row in sub.iter_rows(named=True):
        render_window(row, out, data_root=args.data_root, traj_only=args.traj_only)
        row.get("_key", row["_ade"])


if __name__ == "__main__":
    main()
