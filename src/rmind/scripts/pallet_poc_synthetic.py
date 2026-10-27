"""Synthetic accuracy study for the floor-constrained pallet pose.

Monte Carlo over the approach: put a EUR pallet at a range of distances ahead of
the fork tips (with random lateral offset and yaw), project its keypoints through
the `cam_fork` model, add Gaussian pixel noise, and solve with

    floor    `pose.solve` - (x, y, yaw) + camera-pitch offset, the detector's solver
    6dof     `pose.solve_6dof` - generic PnP, same keypoints, same start

It reports p95 error of lateral, forward and yaw against the insertion budget
(~±2 cm lateral, ~±1 deg yaw), how many keypoints stay in frame - which is where
the entry face drops out under the image - and the BIAS a small calibration error
causes, with the camera pitch fixed at its calibrated value and with it free -
which is the budget the `cam_fork` calibration has to meet.

Everything rides on `CAM_FORK`, which is not a calibration; re-run with the
real one. Keypoint noise is i.i.d. here; a real detector's errors are correlated
and heavier-tailed, so these are lower bounds.

    uv run python -m rmind.scripts.pallet_poc_synthetic --out outputs/pallet_poc
"""

import argparse
import dataclasses
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl

from rmind.pallet.geometry import (
    CAM_FORK,
    FORK_LENGTH,
    KEYPOINT_NAMES,
    KEYPOINTS_3D,
    FisheyeCamera,
    Pose2D,
)
from rmind.pallet.pose import solve, solve_6dof

# entry-face distance ahead of the fork tips (m); negative = forks inserted
DISTANCES: Final = np.round(np.arange(-0.6, 4.01, 0.1), 2)
LATERAL_RANGE: Final = 0.3
YAW_RANGE_DEG: Final = 10.0
ENTRY_FACE: Final = np.array([not n.startswith("pallet_far") for n in KEYPOINT_NAMES])

# the insertion budget from the plan
LATERAL_BUDGET_M: Final = 0.02
YAW_BUDGET_DEG: Final = 1.0

# calibration errors to probe: name -> perturbed solver camera
PERTURBATIONS: Final = {
    "height +1 cm": lambda c: dataclasses.replace(
        c, position=(c.position[0], c.position[1], c.position[2] + 0.01)
    ),
    "pitch +0.5 deg": lambda c: dataclasses.replace(c, pitch=c.pitch + np.deg2rad(0.5)),
    "yaw +0.5 deg": lambda c: dataclasses.replace(c, yaw=c.yaw + np.deg2rad(0.5)),
    "roll +0.5 deg": lambda c: dataclasses.replace(c, roll=c.roll + np.deg2rad(0.5)),
    "f +1 %": lambda c: dataclasses.replace(c, fx=c.fx * 1.01, fy=c.fy * 1.01),
}


def _wrap(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


def simulate(
    camera: FisheyeCamera,
    sigma_px: float,
    trials: int,
    rng: np.random.Generator,
    solver_camera: FisheyeCamera | None = None,
    *,
    with_6dof: bool = True,
    free_pitch: bool = True,
) -> pl.DataFrame:
    solver_camera = camera if solver_camera is None else solver_camera
    rows = []
    for d in DISTANCES:
        for _ in range(trials):
            truth = Pose2D(
                FORK_LENGTH + d,
                rng.uniform(-LATERAL_RANGE, LATERAL_RANGE),
                np.deg2rad(rng.uniform(-YAW_RANGE_DEG, YAW_RANGE_DEG)),
            )
            pixels, visible = camera.project(truth.apply(KEYPOINTS_3D))
            noisy = pixels + rng.normal(0.0, sigma_px, pixels.shape)
            row = {
                "distance": d,
                "visible": int(visible.sum()),
                "visible_face": int((visible & ENTRY_FACE).sum()),
            }
            estimate = solve(
                solver_camera,
                noisy,
                visible,
                sigma_px=max(sigma_px, 0.5),
                free_pitch=free_pitch,
            )
            if estimate is not None:
                row |= {
                    "floor_x": estimate.pose.x - truth.x,
                    "floor_y": estimate.pose.y - truth.y,
                    "floor_yaw": float(
                        np.rad2deg(_wrap(np.array(estimate.pose.yaw - truth.yaw)))
                    ),
                }
                if with_6dof:
                    p = solve_6dof(solver_camera, noisy, visible, estimate.pose)
                    if p is not None:
                        row |= {
                            "6dof_x": p[0] - truth.x,
                            "6dof_y": p[1] - truth.y,
                            "6dof_yaw": float(
                                np.rad2deg(_wrap(np.array(p[3] - truth.yaw)))
                            ),
                        }
            rows.append(row)
    return pl.DataFrame(rows)


def summarize(results: pl.DataFrame) -> pl.DataFrame:
    p95 = [
        (pl.col(c).abs().quantile(0.95)).alias(f"p95_{c}")
        for c in results.columns
        if c.split("_")[0] in {"floor", "6dof"}
    ]
    return (
        results
        .group_by("distance")
        .agg(
            pl.col("visible").median(),
            pl.col("visible_face").median(),
            pl.col("floor_x").is_not_null().mean().alias("solved"),
            *p95,
        )
        .sort("distance")
    )


def plot(summary: pl.DataFrame, sigma_px: float, path: Path) -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    blue, orange, aqua, yellow = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
    ink, muted, grid = "#2b2b2a", "#6b6a64", "#e4e3de"
    d = summary["distance"].to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True)
    panels = [
        (axes[0, 0], "Lateral error, p95 (cm)", "y", 100, LATERAL_BUDGET_M * 100),
        (axes[0, 1], "Yaw error, p95 (deg)", "yaw", 1, YAW_BUDGET_DEG),
        (axes[1, 0], "Forward error, p95 (cm)", "x", 100, None),
    ]
    for ax, title, key, scale, budget in panels:
        for name, color, label in (
            ("floor", blue, "floor-constrained + pitch"),
            ("6dof", orange, "generic PnP, 6 DoF"),
        ):
            column = f"p95_{name}_{key}"
            if column in summary.columns:
                ax.plot(
                    d,
                    summary[column].to_numpy() * scale,
                    color=color,
                    lw=2,
                    label=label,
                )
        if budget is not None:
            ax.axhline(budget, color=muted, lw=1, ls="--")
            ax.annotate(
                "budget",
                (d[-1], budget),
                color=muted,
                fontsize=8,
                ha="right",
                va="bottom",
            )
        ax.set_yscale("log")
        ax.set_title(title, color=ink, fontsize=11, loc="left")
    ax = axes[1, 1]
    ax.plot(d, summary["visible"].to_numpy(), color=aqua, lw=2, label="all keypoints")
    ax.plot(
        d,
        summary["visible_face"].to_numpy(),
        color=yellow,
        lw=2,
        label="entry-face keypoints",
    )
    ax.set_title("Keypoints in frame, median", color=ink, fontsize=11, loc="left")
    ax.set_ylim(0, len(KEYPOINT_NAMES) + 0.5)
    for ax in axes.ravel():
        ax.axvline(0, color=muted, lw=1)
        ax.grid(color=grid, lw=0.8)
        ax.tick_params(colors=muted, labelsize=9)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.legend(frameon=False, fontsize=8, labelcolor=ink)
    for ax in axes[1]:
        ax.set_xlabel(
            "entry face ahead of the fork tips (m); < 0 = forks inserted",
            color=muted,
            fontsize=9,
        )
    fig.suptitle(
        f"Pallet pose from keypoints, synthetic, {sigma_px:g} px noise, CAM_FORK (not calibrated)",
        color=ink,
        fontsize=12,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor="white")
    plt.close(fig)


def _p95(series: pl.Series) -> float:
    q = series.abs().quantile(0.95)
    return float("nan") if q is None else float(q)


def calibration_bias(
    camera: FisheyeCamera, rng: np.random.Generator, trials: int
) -> pl.DataFrame:
    """p95 |pose error| from a calibration error alone (no pixel noise)."""
    rows = []
    for name, perturb in PERTURBATIONS.items():
        for free_pitch in (False, True):
            results = simulate(
                camera,
                0.0,
                trials,
                rng,
                solver_camera=perturb(camera),
                with_6dof=False,
                free_pitch=free_pitch,
            )
            for d in (0.3, 1.0, 2.0):
                at = results.filter(pl.col("distance") == d)
                rows.append({
                    "calibration error": name,
                    "pitch": "free" if free_pitch else "fixed",
                    "distance": d,
                    "x (cm)": _p95(at["floor_x"]) * 100,
                    "y (cm)": _p95(at["floor_y"]) * 100,
                    "yaw (deg)": _p95(at["floor_yaw"]),
                })
    return pl.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sigma-px", type=float, nargs="+", default=[1.0, 2.0])
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_float_precision(3)

    for sigma in args.sigma_px:
        summary = summarize(simulate(CAM_FORK, sigma, args.trials, rng))
        summary.write_csv(args.out / f"synthetic_sigma{sigma:g}.csv")
        plot(summary, sigma, args.out / f"synthetic_sigma{sigma:g}.png")
        print(f"\n== {sigma:g} px keypoint noise")  # ruff: ignore[print]

    bias = calibration_bias(CAM_FORK, rng, max(args.trials // 4, 10))
    bias.write_csv(args.out / "calibration_bias.csv")
    print("\n== bias from calibration error alone")  # ruff: ignore[print]
    print(bias)  # ruff: ignore[print]


if __name__ == "__main__":
    main()
