#! /usr/bin/env python
"""Visualize the flow velocity field / probability-flow trajectories.

The flow decoder learns a velocity field v(x, t | condition) and sampling
integrates it from noise (t=0) to an action (t=1). This script makes that
field legible for the steering channel of one horizon slot, for a few
contrasting conditions (straight vs maneuver), revealing two things the
endpoint fan plot cannot:

  - multimodality: do trajectories funnel to one basin or split into branches?
  - path curvature: straight paths integrate in ~1 step; curved paths need many
    (the visual root of the euler-steps / Heun integrator findings).

Action space is action_horizon * action_dim (e.g. 18-D); a 2-D picture is
necessarily a slice. We integrate the FULL trajectory exactly and PROJECT onto
(flow_time, steering@slot). The trajectory bundle is exact; the faint
background field is an approximation (other dims frozen at the bundle mean at
each t) and labelled as such.

Usage (mirrors `just fan`):

    just field inference=yaak/control_transformer/policy \
        model.artifact=<entity/project/model-...:vX> \
        '+field.legacy_condition=true' '+field.out=field.png'

Options (via +field.*): num_traj, steps, slot, n_conditions, ema,
legacy_condition, out.
"""

import multiprocessing as mp
from pathlib import Path
from typing import TYPE_CHECKING, Any

import hydra
import matplotlib
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from structlog import get_logger

import rmind.components.objectives.flow_policy as flow_policy_module
from rmind.scripts.flow_sample_fan import (
    FRAME_IDX_KEY,
    LEGACY_CONDITION_TOKENS,
    _get_path,
    _load_ema_weights,
    _to_device,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logger = get_logger(__name__)

if TYPE_CHECKING:
    import pytorch_lightning as pl


def _integrate_trajectories(
    decoder: Any, condition: torch.Tensor, *, steps: int
) -> torch.Tensor:
    """Euler-integrate the full action trajectory, storing every step.

    Returns (n_traj, steps + 1, horizon, action_dim). Euler (not the decoder's
    configured method) on a fine grid so the path is rendered smoothly.
    """
    n = condition.shape[0]
    x = torch.randn(
        n, decoder.action_horizon, decoder.action_dim,
        dtype=condition.dtype, device=condition.device,
    )
    traj = [x]
    dt = 1.0 / steps
    for step in range(steps):
        t = torch.full((n,), step * dt, dtype=x.dtype, device=x.device)
        v = decoder(condition_tokens=condition, noised_actions=x, flow_time=t)
        x = x + v * dt
        traj.append(x)
    return torch.stack(traj, dim=1)


def _background_field(
    decoder: Any,
    condition_row: torch.Tensor,
    mean_traj: np.ndarray,
    *,
    slot: int,
    steer_idx: int,
    t_grid: np.ndarray,
    x_grid: np.ndarray,
) -> np.ndarray:
    """Approximate dx_steer/dt over the (t, x_steer) plane.

    The velocity of one component depends on the full action vector, so we
    freeze the other dims at the bundle mean at each t and sweep only the
    steering@slot component. Approximate — for visual context only.
    """
    device = condition_row.device
    field = np.zeros((len(x_grid), len(t_grid)), dtype=np.float32)
    for ti, t in enumerate(t_grid):
        base = torch.from_numpy(mean_traj[ti]).to(device)  # (horizon, action_dim)
        x = base.unsqueeze(0).repeat(len(x_grid), 1, 1).clone()
        x[:, slot, steer_idx] = torch.from_numpy(x_grid).to(device, x.dtype)
        t_t = torch.full((len(x_grid),), float(t), dtype=x.dtype, device=device)
        v = decoder(
            condition_tokens=condition_row.repeat(len(x_grid), 1, 1),
            noised_actions=x,
            flow_time=t_t,
        )
        field[:, ti] = v[:, slot, steer_idx].float().cpu().numpy()
    return field


def _plot_quiver(  # noqa: PLR0913
    decoder: Any,
    cond_row: torch.Tensor,
    mean_traj: np.ndarray,
    *,
    slot: int,
    ax_idx: tuple[int, int],
    ax_names: tuple[str, str],
    gt_point: tuple[float, float],
    t_snaps: tuple[float, ...],
    out_path: Path,
    title: str,
) -> None:
    """Row of 2-D velocity-field cross-sections at increasing flow time.

    Sweeps two action channels (e.g. steering, gas) of one horizon slot over a
    grid; all other dims are FROZEN at the bundle mean at each t (this is a
    cross-section, not the full field). Arrows = velocity projected onto the
    two swept channels; star = the GT action.
    """
    device = cond_row.device
    i, j = ax_idx
    g = np.linspace(-1.2, 1.2, 21)
    gx, gy = np.meshgrid(g, g)
    flat = np.stack([gx.ravel(), gy.ravel()], axis=1)

    fig, axes = plt.subplots(1, len(t_snaps), figsize=(4.2 * len(t_snaps), 4.2))
    for ax, t in zip(np.atleast_1d(axes), t_snaps, strict=True):
        ti = min(int(round(t * (mean_traj.shape[0] - 1))), mean_traj.shape[0] - 1)
        base = torch.from_numpy(mean_traj[ti]).to(device)  # (H, A)
        x = base.unsqueeze(0).repeat(len(flat), 1, 1).clone()
        x[:, slot, i] = torch.from_numpy(flat[:, 0]).to(device, x.dtype)
        x[:, slot, j] = torch.from_numpy(flat[:, 1]).to(device, x.dtype)
        with torch.inference_mode():
            v = decoder(
                condition_tokens=cond_row.repeat(len(flat), 1, 1),
                noised_actions=x,
                flow_time=torch.full((len(flat),), float(t), dtype=x.dtype, device=device),
            )
        u = v[:, slot, i].float().cpu().numpy().reshape(gx.shape)
        w = v[:, slot, j].float().cpu().numpy().reshape(gx.shape)
        mag = np.hypot(u, w)
        ax.quiver(gx, gy, u, w, mag, cmap="viridis", scale=30, width=0.004)
        ax.plot(
            gt_point[0], gt_point[1], "*", ms=20, color="crimson",
            markeredgecolor="black", label="gt action",
        )
        ax.plot(
            mean_traj[-1, slot, i], mean_traj[-1, slot, j], "P", ms=12,
            color="white", markeredgecolor="black", label="model endpoint",
        )
        ax.set_title(f"t = {t:.2f}")
        ax.set_xlabel(ax_names[0])
        ax.set_ylabel(ax_names[1])
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    logger.info("saved quiver figure", path=str(out_path))


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    fld = cfg.get("field") or {}
    num_traj = int(fld.get("num_traj", 200))
    steps = int(fld.get("steps", 50))
    slot = int(fld.get("slot", 0))
    n_conditions = int(fld.get("n_conditions", 3))
    out_path = Path(str(fld.get("out", "flow_field.png")))

    if bool(fld.get("legacy_condition", False)):
        flow_policy_module.POLICY_CONDITION_TOKENS = LEGACY_CONDITION_TOKENS
        logger.info("using legacy condition tokens")

    torch.set_float32_matmul_precision(cfg.matmul_precision)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model: pl.LightningModule = instantiate(cfg.model)
    if bool(fld.get("ema", False)):
        _load_ema_weights(model, cfg)
    model = model.to(device).eval()

    objective = model.objectives["policy"]
    decoder = objective.decoder
    steer_idx = next(
        i for i, k in enumerate(objective.action_keys) if "steering" in k
    )

    datamodule: pl.LightningDataModule = instantiate(cfg.datamodule)
    if hasattr(datamodule, "setup"):
        datamodule.setup("predict")

    # Scan the whole drive so condition selection can reach real maneuvers
    # (a single batch is often all straight driving). Keep condition tokens on
    # CPU; they are small (~tokens x dim per frame).
    conds: list[torch.Tensor] = []
    gts: list[np.ndarray] = []
    frames: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in datamodule.predict_dataloader():
            batch = _to_device(batch, device)
            episode = model.episode_builder(batch)
            embedding = model.encoder(
                src=episode.embeddings_flattened, mask=episode.attention_mask
            )
            conds.append(
                objective._condition_tokens(episode=episode, embedding=embedding).cpu()
            )
            gt = objective._target_actions(batch).float()
            if gt.shape[1] != decoder.action_horizon:
                gt = gt[:, objective._target_slice()]
            gts.append(gt[:, slot, :].cpu().numpy())  # all channels at the slot
            frame_idx = _get_path(batch, "data", FRAME_IDX_KEY)
            if frame_idx.ndim > 1:
                frame_idx = frame_idx[
                    :, min(objective.history_steps, frame_idx.shape[1] - 1)
                ]
            frames.append(frame_idx.cpu().numpy().flatten())
    condition = torch.cat(conds).to(device)
    gt_slot = np.concatenate(gts)  # (F, action_dim) GT action at the slot
    gt_steer = gt_slot[:, steer_idx]
    frame_idx = np.concatenate(frames)

    # Pick conditions spanning straight -> sharp maneuver by |GT steering|.
    valid = np.flatnonzero(~np.isnan(gt_steer))
    ranked = valid[np.argsort(np.abs(gt_steer[valid]))]
    picks = np.linspace(0, len(ranked) - 1, n_conditions).round().astype(int)
    rows = ranked[picks]

    fig, axes = plt.subplots(
        1, n_conditions, figsize=(6 * n_conditions, 5), squeeze=False
    )
    t_axis = np.linspace(0.0, 1.0, steps + 1)
    t_grid = np.linspace(0.0, 1.0, 25)
    x_grid = np.linspace(-1.3, 1.3, 41)

    with torch.inference_mode():
        for ax, row in zip(axes[0], rows, strict=True):
            cond_row = condition[row : row + 1]
            traj = _integrate_trajectories(
                decoder, cond_row.repeat(num_traj, 1, 1), steps=steps
            )  # (num_traj, steps+1, H, A)
            steer = traj[:, :, slot, steer_idx].float().cpu().numpy()  # (num_traj, T)
            mean_traj = traj.float().mean(dim=0).cpu().numpy()  # (steps+1, H, A)

            field = _background_field(
                decoder, cond_row, mean_traj,
                slot=slot, steer_idx=steer_idx, t_grid=t_grid, x_grid=x_grid,
            )
            tt, xx = np.meshgrid(t_grid, x_grid)
            ax.streamplot(
                tt, xx, np.ones_like(field), field,
                color="0.7", density=0.7, linewidth=0.5, arrowsize=0.6,
            )
            end = steer[:, -1]
            lo, hi = np.nanpercentile(end, [5, 95])
            norm = plt.Normalize(lo, hi if hi > lo else lo + 1e-6)
            for i in range(num_traj):
                ax.plot(
                    t_axis, steer[i], color=plt.cm.coolwarm(norm(end[i])),
                    alpha=0.18, lw=0.6,
                )
            ax.axhline(
                gt_steer[row], color="black", ls="--", lw=1.5,
                label=f"gt = {gt_steer[row]:.2f}",
            )
            ax.set_title(
                f"frame {int(frame_idx[row])} | |gt steer| = {abs(gt_steer[row]):.2f}"
            )
            ax.set_xlabel("flow time t  (0 = noise → 1 = action)")
            ax.set_ylabel(f"steering @ slot {slot}")
            ax.set_ylim(-1.4, 1.4)
            ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        "probability-flow trajectories (exact, projected) + approx field (grey)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    logger.info("saved field figure", path=str(out_path), conditions=rows.tolist())

    if bool(fld.get("quiver", False)):
        # 2-D velocity-field cross-section for the sharpest-maneuver condition,
        # in the (steering, gas) plane of the chosen slot, over flow time.
        gas_idx = next(
            (i for i, k in enumerate(objective.action_keys) if "gas" in k), None
        )
        if gas_idx is None:
            logger.warning("no gas channel found; skipping quiver")
            return
        row = int(rows[-1])  # largest |gt steering|
        cond_row = condition[row : row + 1]
        with torch.inference_mode():
            traj = _integrate_trajectories(
                decoder, cond_row.repeat(num_traj, 1, 1), steps=steps
            )
        mean_traj = traj.float().mean(dim=0).cpu().numpy()
        _plot_quiver(
            decoder, cond_row, mean_traj,
            slot=slot, ax_idx=(steer_idx, gas_idx), ax_names=("steering", "gas"),
            gt_point=(float(gt_slot[row, steer_idx]), float(gt_slot[row, gas_idx])),
            t_snaps=(0.1, 0.4, 0.7, 0.95),
            out_path=out_path.with_name(f"{out_path.stem}_quiver.png"),
            title=(
                f"velocity field cross-section, frame {int(frame_idx[row])} "
                f"(|gt steer|={abs(gt_steer[row]):.2f}); other dims frozen at "
                "bundle mean. red star = gt action, white = model endpoint"
            ),
        )


if __name__ == "__main__":
    mp.set_forkserver_preload(["rbyte", "polars"])

    main()
