"""Fit the flow policy's action-normalization stats from data.

Writes the action-norm JSON the PolicyObjective consumes at init:
  - Gaussianize knots (per-channel empirical-CDF -> N(0,1)), optionally on the
    gas/brake -> longitudinal merge (`+action_norm.merge`) — consumed by the
    Gaussianize stage (and, for the merge, GasBrakeMerge) of ActionTransform.
  - optional per-chunk maneuver LDS densities (`+action_norm.lds`) — used by
    LDSWeights for importance-weighted training.

These are a FROZEN normalization: fit ONCE on the full TRAIN split and applied
unchanged at train/val/inference (like input standardization). Refitting per
split breaks train/inference consistency, and the fit is tail-sensitive — the
rare, safety-critical maneuver quantiles need many samples, so a small val/
predict split yields a materially worse fit than the full train set. The result
is a (dataset x action-config) artifact: fit it once into the shared, IMMUTABLE
team dir (paths.action_norm_dir) and reference it everywhere, rather than
refitting per run. The fitter refuses to overwrite an existing file — bump the
name for a new variant — because checkpoints reference their stats by path, so a
silent rewrite would change the transform under existing checkpoints.

Builds NO model — it reads the dataset's action-target columns (the `targets`
mapping) straight from the dataloader, so it needs no checkpoint/artifact and
never touches the GPU. The dataset's windowing is parametrized by history_steps
/ action_horizon / sequence_length, so pass them to match the experiment you're
fitting for (the dataset SQL slices the *_normalized_target columns by them).
inference=... just pulls in /paths + /datamodule (the composed model is ignored);
set an isolated paths.rbyte.cache if a training run is live, to avoid a collision:

    just action-norm inference=yaak/control_transformer/policy \\
        datamodule=yaak/train '+action_norm.split=train' \\
        +history_steps=6 +action_horizon=6 +sequence_length=12 \\
        '+action_norm.merge=true' '+action_norm.lds=true' \\
        '+action_norm.norm_out=${paths.action_norm_dir}/action_norm_train.json'

Knobs (`+action_norm.*`): norm_out (output path, default
paths.action_norm_dir/action_norm.json), overwrite (replace an existing file,
default false), split (predict|val|train, default predict — pass split=train with
datamodule=yaak/train for the real fit), targets (name -> batch-path mapping of
the action-target columns, default the yaak gas/brake/steering *_normalized_target
columns), num_knots (Gaussianize grid resolution, default 256), merge (gas/brake
-> longitudinal, default false), lds (also fit + write per-chunk maneuver LDS
densities under "lds", default false), lds_bins (default 64), lds_sigma (Gaussian
smoothing width in bins, default 2.0).
"""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from structlog import get_logger

logger = get_logger(__name__)

# The dataset's action-target chunk columns (name -> batch path). This is the
# only thing the fit needs from "the model" — it mirrors the policy objective's
# `targets` (both just name the dataset's action columns). Override with
# +action_norm.targets if your columns differ.
DEFAULT_TARGETS: dict[str, tuple[str, ...]] = {
    "gas_pedal": ("data", "meta/VehicleMotion/gas_pedal_normalized_target"),
    "brake_pedal": ("data", "meta/VehicleMotion/brake_pedal_normalized_target"),
    "steering_angle": ("data", "meta/VehicleMotion/steering_angle_normalized_target"),
}


def _to_device(obj: Any, device: torch.device) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if hasattr(obj, "to"):
        return obj.to(device)
    return obj


def _collect_targets(
    cfg: DictConfig, *, split: str, targets: Mapping[str, tuple[str, ...]]
) -> tuple[np.ndarray, list[str]]:
    """Stack the dataset's action-target columns -> (frames, horizon, A).

    Pure data indexing: reads the *_normalized_target chunk columns named in
    `targets` straight from the dataloader — no model is built (the fit needs the
    dataset's action numbers, not the policy). Per-chunk structure is kept so the
    LDS fit can take a peak-over-horizon label; the Gaussianize fit flattens it.
    """
    device = torch.device("cpu")
    datamodule = instantiate(cfg.datamodule)
    if hasattr(datamodule, "setup"):
        datamodule.setup(split)
    match split:
        case "train":
            loader = datamodule.train_dataloader()
        case "val":
            loader = datamodule.val_dataloader()
        case _:
            loader = datamodule.predict_dataloader()

    paths = list(targets.values())
    chunks: list[torch.Tensor] = []
    with torch.inference_mode():
        for raw_batch in loader:
            batch = _to_device(raw_batch, device)
            cols: list[torch.Tensor] = []
            for path in paths:
                value: Any = batch
                for key in path:
                    value = value[key]
                cols.append(value)
            chunks.append(torch.stack(cols, dim=-1).float().cpu())  # (B, H, A)

    actions = torch.cat(chunks).numpy()  # (frames, H, A)
    finite = np.isfinite(actions).all(axis=tuple(range(1, actions.ndim)))
    dropped = int((~finite).sum())
    if dropped:
        logger.warning("dropping non-finite chunks", count=dropped)
    return actions[finite], list(targets)


def _write_norm_stats(
    actions: np.ndarray, keys: list[str], *, path: str, num_knots: int, merge: bool
) -> None:
    """Per-channel Gaussianize knots (in MODEL space): values at a quantile grid.

    With merge, model space is (longitudinal = gas - brake, steering) — knots are
    fit there. Knots are made strictly increasing (cummax + tiny ramp) so the
    transform's searchsorted is well-defined even where the marginal has a flat
    region (e.g. brake's point mass at 0, or longitudinal's coasting mass).

    Raises:
        ValueError: if `merge` is set but `keys` are not gas/brake/steering.
    """
    if merge:
        if tuple(keys) != ("gas_pedal", "brake_pedal", "steering_angle"):
            msg = f"merge requires keys gas/brake/steering, got {keys}"
            raise ValueError(msg)
        model = np.stack([actions[:, 0] - actions[:, 1], actions[:, 2]], axis=1)
        model_keys = ["longitudinal", "steering_angle"]
    else:
        model = actions
        model_keys = keys
    grid = np.linspace(0.0, 1.0, num_knots)
    knots: list[list[float]] = []
    for c in range(model.shape[1]):
        vals = np.quantile(model[:, c], grid)
        vals = np.maximum.accumulate(vals) + 1e-6 * np.arange(num_knots)
        knots.append([float(v) for v in vals])
    Path(path).write_text(
        # model_keys names the channel space the knots live in, so the Gaussianize
        # consumer is self-describing; action_keys/merge stay for provenance and
        # to load pre-split files (which had no model_keys).
        json.dumps({
            "action_keys": keys,
            "model_keys": model_keys,
            "merge": merge,
            "grid": grid.tolist(),
            "knots": knots,
        }),
        encoding="utf-8",
    )
    logger.info("wrote Gaussianize knots", path=path, num_knots=num_knots, merge=merge)


def _gaussian_smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    """1-D Gaussian smoothing with edge padding (no scipy dependency)."""
    if sigma <= 0:
        return x
    radius = max(1, round(3 * sigma))
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
    opts = cfg.get("action_norm") or {}
    split = str(opts.get("split", "predict"))
    # Default into the shared, immutable team artifacts dir (paths.action_norm_dir)
    # so the recorded path resolves on every load; name variants explicitly there.
    norm_out = str(
        opts.get("norm_out")
        or f"{cfg.get('paths', {}).get('action_norm_dir', 'artifacts')}/action_norm.json"
    )
    num_knots = int(opts.get("num_knots", 256))
    merge = bool(opts.get("merge", False))
    lds = bool(opts.get("lds", False))
    lds_bins = int(opts.get("lds_bins", 64))
    lds_sigma = float(opts.get("lds_sigma", 2.0))
    targets = (
        {k: tuple(v) for k, v in opts["targets"].items()}
        if opts.get("targets") is not None
        else DEFAULT_TARGETS
    )

    # Immutability: a checkpoint references its stats by path, so overwriting
    # silently rebuilds a different transform under existing checkpoints. Fail
    # fast (before the dataset pass) rather than clobber; bump the name instead.
    out_path = Path(norm_out)
    if out_path.exists() and not bool(opts.get("overwrite", False)):
        msg = (
            f"{norm_out} already exists; action-norm stats are immutable. Use a "
            "new (variant) name, or +action_norm.overwrite=true to replace it."
        )
        raise FileExistsError(msg)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    chunks, keys = _collect_targets(cfg, split=split, targets=targets)  # (frames, H, A)
    actions = chunks.reshape(-1, chunks.shape[-1])  # (frames*H, A)
    logger.info(
        "collected target actions", frames_x_horizon=actions.shape[0], channels=keys
    )
    # Heuristic guard: the maneuver-tail quantiles (the safety-critical part) need
    # many samples per knot, so a small split gives a noisy, tail-biased fit. Fit
    # on the full train split (datamodule=yaak/train +action_norm.split=train).
    if actions.shape[0] < 100 * num_knots:
        logger.warning(
            "fitting action-norm on a small sample; maneuver-tail quantiles will "
            "be unreliable — fit on the full train split",
            samples=actions.shape[0],
            num_knots=num_knots,
            split=split,
        )

    _write_norm_stats(actions, keys, path=norm_out, num_knots=num_knots, merge=merge)

    if lds:
        # Append per-chunk maneuver LDS densities to the same stats file.
        stats = json.loads(Path(norm_out).read_text(encoding="utf-8"))
        stats["lds"] = _fit_lds(
            chunks, keys, merge=merge, bins=lds_bins, sigma=lds_sigma
        )
        Path(norm_out).write_text(json.dumps(stats), encoding="utf-8")
        logger.info(
            "wrote maneuver LDS densities",
            path=norm_out,
            bins=lds_bins,
            sigma=lds_sigma,
            model_keys=stats["lds"]["model_keys"],
        )

    print(  # noqa: T201
        "\nConfig-ready — point paths.action_norm_stats at this one file "
        "(backs both the Gaussianize transform and, if present, the LDS weights):\n"
        f"  paths.action_norm_stats: {norm_out!r}"
        f"{'  # merged: set flow_action_dim=2' if merge else ''}"
        f"{'  (+ lds: set flow_lds_alpha>0)' if lds else ''}\n"
    )


if __name__ == "__main__":
    main()
