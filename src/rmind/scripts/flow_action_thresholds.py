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
1e-3), split (predict|val, default predict), merge (gas/brake -> longitudinal,
default false), dct (also fit + write chunk-DCT per-coefficient mu/sigma into
norm_out, default false), dct_sigma_floor_frac (sigma-floor as a fraction of
the channel's sigma_0, default 0.05). The DCT sigma spectrum is always printed
as a diagnostic; `dct=true` additionally persists it for the transform.
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
    """Return (chunks, action_keys): chunks has shape (frames, horizon, A)."""
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
    loader = (
        datamodule.val_dataloader()
        if split == "val"
        else datamodule.predict_dataloader()
    )

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
        ), encoding="utf-8"
    )
    logger.info("wrote Gaussianize knots", path=path, num_knots=num_knots, merge=merge)


def _dct_spectrum(chunks: np.ndarray, *, norm_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-coefficient sigma spectrum of Gaussianized chunks (orthonormal DCT-II
    over the horizon axis).

    Pre-flight check for the chunk-frequency action transform: in slot space the
    flow MSE weights each orthogonal direction by its variance, so the variance
    share of coefficients k >= 1 IS the loss share that within-chunk structure
    receives — if it is ~1e-3, the constant/mid-anchored chunk is the
    gradient-rational optimum and per-coefficient standardization is the fix.
    sigma_k are the standardization scales; the table also informs the
    sigma-floor (don't unit-weight coefficients that are pure jitter).

    Returns (mu, sigma), each (horizon, model_dim).
    """
    from rmind.components.objectives.action_transform import (
        GaussianizeActionTransform,
        dct_basis,
    )

    transform = GaussianizeActionTransform.from_stats_file(norm_path)
    with torch.inference_mode():
        z = transform(torch.from_numpy(chunks).float()).numpy()  # (N, H, C) model
    horizon = z.shape[1]
    basis = dct_basis(horizon).numpy()
    coeff = np.einsum("kh,nhc->nkc", basis, z)  # (N, H, C)
    mu, sigma = coeff.mean(axis=0), coeff.std(axis=0)
    var = sigma**2

    print(  # noqa: T201
        f"\nDCT sigma spectrum over {z.shape[0]} Gaussianized chunks "
        f"(H={horizon}, model channels: {list(transform.model_action_keys)}):"
    )
    for c, key in enumerate(transform.model_action_keys):
        share = var[:, c] / var[:, c].sum()
        print(f"\n  {key}:")  # noqa: T201
        print(  # noqa: T201
            f"    {'k':>2} {'mu':>8} {'sigma':>8} {'sigma_k/sigma_0':>16} {'var share':>10}"
        )
        for ki in range(horizon):
            print(  # noqa: T201
                f"    {ki:>2} {mu[ki, c]:8.4f} {sigma[ki, c]:8.4f} "
                f"{sigma[ki, c] / sigma[0, c]:16.4f} {share[ki]:9.2e}"
            )
        print(  # noqa: T201
            f"    within-chunk (k>=1) variance share: {share[1:].sum():.2e}"
        )
    return mu, sigma


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    opts = cfg.get("thresholds") or {}
    quantile = float(opts.get("quantile", 0.90))
    active_eps = float(opts.get("active_eps", 1e-3))
    split = str(opts.get("split", "predict"))
    norm_out = str(opts.get("norm_out", "action_norm.json"))
    num_knots = int(opts.get("num_knots", 256))
    merge = bool(opts.get("merge", False))
    dct = bool(opts.get("dct", False))
    dct_sigma_floor_frac = float(opts.get("dct_sigma_floor_frac", 0.05))

    chunks, keys = _collect_targets(cfg, split=split)
    actions = chunks.reshape(-1, chunks.shape[-1])  # (frames*H, A)
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
    mu, sigma = _dct_spectrum(chunks, norm_path=norm_out)
    if dct:
        # Append the chunk-DCT standardization stats to the same file: the
        # transform Gaussianizes with the knots above, rotates with the DCT,
        # then standardizes each coefficient with these mu/sigma.
        stats = json.loads(Path(norm_out).read_text(encoding="utf-8"))
        stats["dct"] = {
            "mu": mu.tolist(),
            "sigma": sigma.tolist(),
            "sigma_floor_frac": dct_sigma_floor_frac,
        }
        Path(norm_out).write_text(json.dumps(stats), encoding="utf-8")
        logger.info(
            "wrote chunk-DCT stats",
            path=norm_out,
            horizon=mu.shape[0],
            sigma_floor_frac=dct_sigma_floor_frac,
        )

    print(  # noqa: T201
        "\nConfig-ready (order = action_keys above):\n"
        f"flow_maneuver_thresholds: {recommended}\n"
        f"flow_action_transform_stats: {norm_out!r}  # Gaussianize knots"
        f"{' (merge: longitudinal+steering, set flow_action_dim=2)' if merge else ''}"
        f"{chr(10) + 'flow_chunk_delta_weight: 0  # delta loss is meaningless across DCT coefficients' if dct else ''}\n"
    )


if __name__ == "__main__":
    main()
