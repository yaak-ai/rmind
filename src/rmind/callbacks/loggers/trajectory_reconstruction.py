# ruff: noqa: PLR2004  (manoeuvre thresholds are the category definitions)
from typing import Any, final

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from pydantic import validate_call
from torch import Tensor

from rmind.callbacks.safe import SafeCallback
from wandb import Image

from .common import _figure_to_rgba, _get_wandb_loggers

STEP_S = 1 / 3  # pose spacing (trajectory_step = 10 frames at 30 fps)

# (name, number of samples) -- 32 in total, covering the manoeuvres the codebook must learn
CATEGORIES: tuple[tuple[str, int], ...] = (
    ("stopped", 1),
    ("start from stop", 3),
    ("cruise <40 km/h", 2),
    ("cruise >80 km/h", 2),
    ("slowing down", 2),
    ("hard braking", 3),
    ("accelerating", 2),
    ("left turn >45°", 3),
    ("right turn >45°", 3),
    ("left curve 10-45°", 2),
    ("right curve 10-45°", 2),
    ("lane change", 3),
    ("roundabout-like", 2),
    ("other", 2),
)


def _categorize(poses: np.ndarray) -> np.ndarray:
    """`(n, P, 3)` ego-frame poses (x right, y forward, theta + = right) -> category index."""
    xy = np.concatenate([np.zeros_like(poses[:, :1, :2]), poses[..., :2]], axis=1)
    step = np.linalg.norm(np.diff(xy, axis=1), axis=-1) / STEP_S * 3.6  # km/h per step
    v0, v1 = step[:, 0], step[:, -1]
    th = np.degrees(poses[..., 2])
    th_end = th[:, -1]
    names = [name for name, _ in CATEGORIES]
    out = np.full(len(poses), names.index("other"))
    rules = [
        ("roundabout-like", (v0 > 8) & (v0 < 45) & (th.max(1) > 5) & (th.min(1) < -20) & (th.argmax(1) < th.argmin(1))),
        ("lane change", (np.abs(th_end) < 4) & (np.abs(poses[:, -1, 0]) > 2.5) & (v0 > 30)),
        ("right curve 10-45°", (th_end > 10) & (th_end <= 45)),
        ("left curve 10-45°", (th_end < -10) & (th_end >= -45)),
        ("right turn >45°", th_end > 45),
        ("left turn >45°", th_end < -45),
        ("accelerating", (np.abs(th_end) < 5) & (v1 > v0 + 12) & (v0 > 3)),
        ("hard braking", (np.abs(th_end) < 5) & (v0 > 20) & (v1 < 0.5 * v0)),
        ("slowing down", (np.abs(th_end) < 5) & (v0 > 20) & (v1 >= 0.5 * v0) & (v1 <= 0.8 * v0)),
        ("cruise >80 km/h", (np.abs(th_end) < 2) & (v0 > 80) & (np.abs(v1 - v0) < 5)),
        ("cruise <40 km/h", (np.abs(th_end) < 2) & (v0 > 15) & (v0 < 40) & (np.abs(v1 - v0) < 3)),
        ("start from stop", (v0 < 1.5) & (v1 > 8)),
        ("stopped", step.max(1) < 0.5),
    ]
    for name, mask in reversed(rules):  # first rule wins
        out[mask] = names.index(name)
    return out


@final
class WandbTrajectoryReconstructionLogger(SafeCallback):
    """Log ground-truth vs reconstructed trajectories for a fixed set of 32 validation
    samples (picked once, by manoeuvre type) at the end of every validation run, so
    the W&B image slider shows reconstruction improving over training. Also logs the
    final-pose error per manoeuvre type as `<key>/fde/<type>`.

    Expects a `TrajectoryTokenizer` (`_gather_poses`, `reconstruct`).
    """

    @validate_call
    def __init__(
        self,
        *,
        key: str = "val/reconstruction",
        max_collect_batches: int = 16,
        fail_gracefully: bool = True,
        disable_on_error: bool = False,
    ) -> None:
        super().__init__(fail_gracefully=fail_gracefully, disable_on_error=disable_on_error)
        self._key = key
        self._max_collect_batches = max_collect_batches
        self._pool: list[Tensor] = []
        self._fixed: Tensor | None = None
        self._labels: list[str] = []
        self.on_validation_batch_end = self._safe_hook(  # ty:ignore[invalid-assignment]
            "on_validation_batch_end", self._collect
        )
        self.on_validation_epoch_end = self._safe_hook(  # ty:ignore[invalid-assignment]
            "on_validation_epoch_end", self._log
        )

    def _collect(
        self,
        _trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        _outputs: Any,
        batch: Any,
        _batch_idx: int,
        _dataloader_idx: int = 0,
    ) -> None:
        if self._fixed is not None or len(self._pool) >= self._max_collect_batches:
            return
        with torch.no_grad():
            self._pool.append(pl_module._gather_poses(batch).detach().float().cpu())  # noqa: SLF001

    def _select(self) -> None:
        pool = torch.cat(self._pool).numpy()
        self._pool = []
        cat = _categorize(pool)
        rng = np.random.default_rng(0)
        picked: list[int] = []
        labels: list[str] = []
        for k, (name, n) in enumerate(CATEGORIES):
            idx = np.flatnonzero(cat == k)
            take = rng.choice(idx, min(n, len(idx)), replace=False) if len(idx) else []
            picked += list(take)
            labels += [name] * len(take)
        n_total = sum(n for _, n in CATEGORIES)
        rest = np.setdiff1d(np.arange(len(pool)), picked)
        fill = rng.choice(rest, n_total - len(picked), replace=False)
        picked += list(fill)
        labels += ["other"] * len(fill)
        self._fixed = torch.from_numpy(pool[picked])
        self._labels = labels

    def _log(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        loggers = _get_wandb_loggers(pl_module)
        if not loggers or not trainer.is_global_zero:
            return
        if trainer.sanity_checking:
            # the sanity check sees only the first couple of (urban, single-drive) batches;
            # pick the fixed set from the first full validation run instead
            self._pool = []
            return
        if self._fixed is None:
            if not self._pool:
                return
            self._select()
        if self._fixed is None:
            return

        with torch.no_grad():
            gt = self._fixed.to(pl_module.device)
            rec = pl_module.reconstruct(gt).float().cpu().numpy()  # ty:ignore[call-non-callable]
        gt = gt.cpu().numpy()
        fde = np.linalg.norm(rec[:, -1, :2] - gt[:, -1, :2], axis=-1)
        ade = np.linalg.norm(rec[..., :2] - gt[..., :2], axis=-1).mean(-1)

        fig, axes = plt.subplots(4, 8, figsize=(24, 13))
        for ax, g, r, label, f, a in zip(axes.ravel(), gt, rec, self._labels, fde, ade, strict=True):
            gx, gy = np.r_[0, g[:, 0]], np.r_[0, g[:, 1]]
            rx, ry = np.r_[0, r[:, 0]], np.r_[0, r[:, 1]]
            ax.plot(gx, gy, "-o", color="black", ms=3, lw=1.5, label="ground truth")
            ax.plot(rx, ry, "-o", color="tab:orange", ms=3, lw=1.5, label="reconstruction")
            ext = max(np.abs(np.r_[gx, rx]).max(), np.r_[gy, ry].max(), 1.0) * 1.15
            ax.set_xlim(-ext, ext)
            ax.set_ylim(-0.2 * ext, 1.8 * ext)
            ax.set_aspect("equal")
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=7)
            v0 = np.linalg.norm(g[0, :2]) / STEP_S * 3.6
            ax.set_title(f"{label}  {v0:.0f} km/h\nADE {a:.2f} m  FDE {f:.2f} m", fontsize=9)
        axes.ravel()[0].legend(fontsize=7, loc="upper left")
        fig.suptitle(
            f"step {trainer.global_step}: 32 fixed validation trajectories (x right, y forward, m); "
            f"mean ADE {ade.mean():.2f} m, FDE {fde.mean():.2f} m",
            fontsize=13,
        )
        fig.tight_layout()
        try:
            image = Image(_figure_to_rgba(fig))
        finally:
            plt.close(fig)

        per_type = {
            f"{self._key}/fde/{name}": float(fde[[lab == name for lab in self._labels]].mean())
            for name, _ in CATEGORIES
            if name in self._labels
        }
        for logger_ in loggers:
            logger_.log_image(key=self._key, images=[image], step=trainer.global_step)
            logger_.log_metrics(
                {f"{self._key}/ade": float(ade.mean()), f"{self._key}/fde": float(fde.mean()), **per_type},
                step=trainer.global_step,
            )
