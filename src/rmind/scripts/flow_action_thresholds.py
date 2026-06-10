"""Compute per-channel maneuver thresholds for the flow policy from data.

The maneuver-L1 val metric flags "active" frames per channel as `|GT_c| >
threshold_c`. gas/brake/steering have very different distributions (brake is
~always 0, gas is bounded well below steering's ±1), so a single threshold for
all channels is wrong — this prints a per-channel, data-derived tuple ready to
paste into `flow_maneuver_thresholds`.

It runs on CPU and never calls the model forward (only `_target_actions`, which
is pure data indexing), so it does not contend for the GPU. It DOES read the
dataset, so if a training run is live, give it an isolated rbyte cache to avoid
a cache collision:

    just thresholds inference=yaak/control_transformer/policy \\
        model.artifact=yaak/rmind/model-e61ycirr:v9 \\
        paths.rbyte.cache=/tmp/thresholds_cache \\
        '+thresholds.quantile=0.90'

Knobs (`+thresholds.*`): quantile (recommended threshold = this quantile of
|action|, default 0.90), active_eps (|x| above this counts as "active", default
1e-3), split (predict|val|train, default predict), merge (gas/brake ->
longitudinal, default false), lds (also fit + write per-chunk maneuver LDS
weighting stats into norm_out under "lds", default false), lds_bins (histogram
bins, default 64), lds_sigma (Gaussian smoothing width in bins, default 2.0).
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from structlog import get_logger

import rmind.components.objectives.flow_policy as flow_policy_module

logger = get_logger(__name__)

if TYPE_CHECKING:
    import pytorch_lightning as pl

QUANTILES = (0.80, 0.90, 0.95, 0.99)


def _to_device(obj: Any, device: torch.device) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if hasattr(obj, "to"):
        return obj.to(device)
    return obj


def _collect_targets(cfg: DictConfig, *, split: str) -> tuple[np.ndarray, list[str]]:
    """Return (chunks, action_keys): chunks has shape (frames, horizon, A).

    Per-chunk structure is kept (not flattened) so the LDS fit can compute a
    peak-over-horizon label; the maneuver-threshold stats flatten it themselves.
    """
    device = torch.device("cpu")

    model: pl.LightningModule = instantiate(cfg.model).to(device).eval()
    objective = model.objectives["policy"]
    if not isinstance(objective, flow_policy_module.FlowPolicyObjective):
        msg = f"policy objective is not a FlowPolicyObjective: {type(objective)}"
        raise TypeError(msg)
    horizon = objective.decoder.action_horizon

    datamodule: pl.LightningDataModule = instantiate(cfg.datamodule)
    if hasattr(datamodule, "setup"):
        datamodule.setup(split)
    match split:
        case "train":
            loader = datamodule.train_dataloader()
        case "val":
            loader = datamodule.val_dataloader()
        case _:
            loader = datamodule.predict_dataloader()

    chunks: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch in loader:
            batch = _to_device(batch, device)
            actions = objective._target_actions(batch).float()  # (B, H, A)
            if actions.shape[1] != horizon:
                actions = actions[:, objective._target_slice()]
            chunks.append(actions.cpu())

    actions = torch.cat(chunks).numpy()  # (frames, H, A)
    finite = np.isfinite(actions).all(axis=(1, 2))
    dropped = int((~finite).sum())
    if dropped:
        logger.warning("dropping non-finite chunks", count=dropped)
    return actions[finite], list(objective.action_keys)


def _write_norm_stats(
    actions: np.ndarray, keys: list[str], *, path: str, num_knots: int, merge: bool
) -> None:
    """Per-channel Gaussianize knots (in MODEL space): values at a quantile grid.

    With merge, model space is (longitudinal = gas - brake, steering) — knots are
    fit there. Knots are made strictly increasing (cummax + tiny ramp) so the
    transform's searchsorted is well-defined even where the marginal has a flat
    region (e.g. brake's point mass at 0, or longitudinal's coasting mass).
    """
    if merge:
        if tuple(keys) != ("gas_pedal", "brake_pedal", "steering_angle"):
            msg = f"merge requires keys gas/brake/steering, got {keys}"
            raise ValueError(msg)
        model = np.stack([actions[:, 0] - actions[:, 1], actions[:, 2]], axis=1)
    else:
        model = actions
    grid = np.linspace(0.0, 1.0, num_knots)
    knots: list[list[float]] = []
    for c in range(model.shape[1]):
        vals = np.quantile(model[:, c], grid)
        vals = np.maximum.accumulate(vals) + 1e-6 * np.arange(num_knots)
        knots.append([float(v) for v in vals])
    Path(path).write_text(
        json.dumps(
            {"action_keys": keys, "merge": merge, "grid": grid.tolist(), "knots": knots}
        )
    )
    logger.info("wrote Gaussianize knots", path=path, num_knots=num_knots, merge=merge)


def _gaussian_smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    """1-D Gaussian smoothing with edge padding (no scipy dependency)."""
    if sigma <= 0:
        return x
    radius = max(1, int(round(3 * sigma)))
    taps = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (taps / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(np.pad(x, radius, mode="edge"), kernel, mode="valid")


def _fit_lds(
    chunks: np.ndarray, keys: list[str], *, merge: bool, bins: int, sigma: float
) -> dict:
    """Per-chunk maneuver LDS stats (Yang et al. 2021), in physical MODEL space.

    The chunk label is the peak |action| over the horizon per channel — so the
    whole chunk (incl. its near-zero lead-in) is upweighted with its maneuver.
    Returns {edges, emp, smooth, model_keys}: per-channel bin edges + empirical
    and Gaussian-smoothed densities. alpha/cap are applied at load time (so they
    sweep without refitting); this only stores the densities.
    """
    if merge:
        # (frames, H, 3) gas/brake/steering -> (frames, H, 2) longitudinal/steering.
        model = np.stack([chunks[..., 0] - chunks[..., 1], chunks[..., 2]], axis=-1)
        model_keys = ["longitudinal", "steering_angle"]
    else:
        model = chunks
        model_keys = keys
    label = np.abs(model).max(axis=1)  # (frames, C): per-chunk peak per channel
    edges_all, emp_all, smooth_all = [], [], []
    for c in range(model.shape[-1]):
        col = label[:, c]
        hi = float(col.max())
        edges = np.linspace(0.0, hi if hi > 0 else 1.0, bins + 1)
        hist, _ = np.histogram(np.clip(col, 0.0, edges[-1]), bins=edges)
        emp = hist / max(hist.sum(), 1)
        smooth = _gaussian_smooth(emp, sigma)
        edges_all.append(edges.tolist())
        emp_all.append(emp.tolist())
        smooth_all.append(smooth.tolist())
    return {
        "edges": edges_all,
        "emp": emp_all,
        "smooth": smooth_all,
        "model_keys": model_keys,
    }


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    opts = cfg.get("thresholds") or {}
    quantile = float(opts.get("quantile", 0.90))
    active_eps = float(opts.get("active_eps", 1e-3))
    split = str(opts.get("split", "predict"))
    norm_out = str(opts.get("norm_out", "action_norm.json"))
    num_knots = int(opts.get("num_knots", 256))
    merge = bool(opts.get("merge", False))
    lds = bool(opts.get("lds", False))
    lds_bins = int(opts.get("lds_bins", 64))
    lds_sigma = float(opts.get("lds_sigma", 2.0))

    chunks, keys = _collect_targets(cfg, split=split)  # (frames, H, A)
    actions = chunks.reshape(-1, chunks.shape[-1])  # (frames*H, A) for threshold stats
    mag = np.abs(actions)
    n = actions.shape[0]
    logger.info("collected target actions", frames_x_horizon=n, channels=keys)

    recommended: list[float] = []
    print(f"\nPer-channel action stats over {n} (frame x horizon) targets:")  # noqa: T201
    print(  # noqa: T201
        f"{'channel':>22} {'mean':>8} {'std':>8} {'%active':>8} "
        + " ".join(f"q{int(q * 100):>2}".rjust(8) for q in QUANTILES)
        + f" {'rec@q' + str(int(quantile * 100)):>8} {'%flag':>7}"
    )
    for c, key in enumerate(keys):
        col, m = actions[:, c], mag[:, c]
        active = m > active_eps
        qs = [float(np.quantile(m, q)) for q in QUANTILES]
        # Recommended threshold = the chosen quantile of |action|. For a sparse
        # channel (e.g. brake) whose point mass at 0 exceeds (1 - quantile), that
        # quantile is ~0 and useless — fall back to the same quantile among the
        # ACTIVE values so the threshold still isolates real maneuvers.
        rec = float(np.quantile(m, quantile))
        if rec <= active_eps and active.any():
            rec = float(np.quantile(m[active], quantile))
        recommended.append(round(rec, 4))
        flagged = float((m > rec).mean())
        print(  # noqa: T201
            f"{key:>22} {col.mean():8.3f} {col.std():8.3f} "
            f"{active.mean() * 100:7.1f}% "
            + " ".join(f"{q:8.3f}" for q in qs)
            + f" {rec:8.3f} {flagged * 100:6.1f}%"
        )

    _write_norm_stats(actions, keys, path=norm_out, num_knots=num_knots, merge=merge)

    if lds:
        # Append per-chunk maneuver LDS densities to the same stats file.
        stats = json.loads(Path(norm_out).read_text())
        stats["lds"] = _fit_lds(
            chunks, keys, merge=merge, bins=lds_bins, sigma=lds_sigma
        )
        Path(norm_out).write_text(json.dumps(stats))
        logger.info(
            "wrote maneuver LDS densities",
            path=norm_out,
            bins=lds_bins,
            sigma=lds_sigma,
            model_keys=stats["lds"]["model_keys"],
        )

    print(  # noqa: T201
        "\nConfig-ready (order = action_keys above):\n"
        f"flow_maneuver_thresholds: {recommended}\n"
        f"flow_action_transform_stats: {norm_out!r}  # Gaussianize knots"
        f"{' (merge: longitudinal+steering, set flow_action_dim=2)' if merge else ''}"
        f"{chr(10) + f'flow_lds_stats: {norm_out!r}  # maneuver LDS (set flow_lds_alpha>0)' if lds else ''}\n"
    )


if __name__ == "__main__":
    main()
