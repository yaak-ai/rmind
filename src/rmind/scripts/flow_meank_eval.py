#! /usr/bin/env python
"""Bias/variance decomposition of the flow policy via mean-of-K aggregation.

For each frame, draws K action chunks and compares:
  - single-draw L1   (mean over draws of per-draw L1) — what 1-sample deployment gets
  - mean-of-K L1     (L1 of the AVERAGED prediction)  — the flow's conditional-mean
                      error, i.e. ~the ceiling a deterministic regression head
                      could reach on the same conditioning ("bias")
  - variance tax     (single-draw minus mean-of-K)    — error attributable purely
                      to per-draw sampling spread

If the variance tax dominates (esp. on maneuvers), the flow's conditional mean
is fine and the "flow can't overfit" gap is a sampling artifact — fixable at
inference (aggregate draws) or irrelevant for a deterministic readout. If bias
dominates, the flow genuinely hasn't fit the conditional.

Usage (mirrors `just fan`):
    uv run python -m rmind.scripts.flow_meank_eval \
        --config-path <repo>/config --config-name predict.yaml \
        inference=yaak/control_transformer/policy \
        model.artifact=yaak/action-flow/model-<run>:vN \
        datamodule=yaak/predict_val '+meank.k=32'

Options (+meank.*): k (draws, default 32), spike_threshold (|gt steering|,
default 0.5), flat_threshold (default 0.05).
"""

import multiprocessing as mp
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


def _to_device(obj: Any, device: torch.device) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if hasattr(obj, "to"):
        return obj.to(device)
    return obj


def _collect(cfg: DictConfig, *, k: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Returns (gt (F,H,A), samples (F,K,H,A)) in RAW space, plus action keys."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model: pl.LightningModule = instantiate(cfg.model).to(device).eval()
    objective = model.objectives["policy"]
    if not isinstance(objective, flow_policy_module.FlowPolicyObjective):
        msg = f"policy objective is not a FlowPolicyObjective: {type(objective)}"
        raise TypeError(msg)
    horizon = objective.decoder.action_horizon

    datamodule: pl.LightningDataModule = instantiate(cfg.datamodule)
    if hasattr(datamodule, "setup"):
        datamodule.setup("predict")

    gts: list[torch.Tensor] = []
    samples: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch_idx, batch in enumerate(datamodule.predict_dataloader()):
            batch = _to_device(batch, device)
            with torch.autocast(
                device.type, torch.bfloat16, enabled=device.type == "cuda"
            ):
                episode = model.episode_builder(batch)
                embedding = model.encoder(
                    src=episode.embeddings_flattened, mask=episode.attention_mask
                )
                condition_tokens = objective._condition_tokens(
                    episode=episode, embedding=embedding
                )
                condition_rep = condition_tokens.repeat_interleave(k, dim=0)
                trajectories = objective.decoder.sample(
                    condition_tokens=condition_rep,
                    noise=objective._noise(
                        condition_tokens=condition_rep, generator=None
                    ),
                )
            b = condition_tokens.shape[0]
            trajectories = objective._to_raw_space(
                trajectories.float().reshape(b, k, horizon, -1)
            )
            gt = objective._target_actions(batch).float()
            if gt.shape[1] != horizon:
                gt = gt[:, objective._target_slice()]
            gts.append(gt.cpu())
            samples.append(trajectories.cpu())
            if batch_idx % 50 == 0:
                logger.debug("sampled", batch_idx=batch_idx)

    gt = torch.cat(gts).numpy()
    sample = torch.cat(samples).numpy()
    valid = np.isfinite(gt).all(axis=(1, 2)) & np.isfinite(sample).all(axis=(1, 2, 3))
    if (dropped := int((~valid).sum())) > 0:
        logger.warning("dropping non-finite frames", count=dropped)
    return gt[valid], sample[valid], list(objective.action_keys)


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    opts = cfg.get("meank") or {}
    k = int(opts.get("k", 32))
    spike_thr = float(opts.get("spike_threshold", 0.5))
    flat_thr = float(opts.get("flat_threshold", 0.05))

    torch.set_float32_matmul_precision(cfg.matmul_precision)
    gt, sample, keys = _collect(cfg, k=k)
    f = gt.shape[0]
    logger.info("collected", frames=f, draws=k)

    steer = keys.index("steering_angle")
    spike = np.abs(gt[..., steer]) > spike_thr  # (F, H)
    flat = np.abs(gt[..., steer]) < flat_thr

    mean_pred = sample.mean(axis=1)  # (F, H, A)
    per_draw_err = np.abs(sample - gt[:, None])  # (F, K, H, A)
    single = per_draw_err.mean(axis=1)  # (F, H, A) expected single-draw |err|
    agg = {1: single, k: np.abs(mean_pred - gt)}
    for kk in (2, 4, 8, 16):
        if kk < k:
            agg[kk] = np.abs(sample[:, :kk].mean(axis=1) - gt)

    print(f"\nmean-of-K decomposition over {f} frames (K draws averaged, raw space)")  # noqa: T201
    for c, key in enumerate(keys):
        print(f"\n  {key}:")  # noqa: T201
        print(f"    {'K':>4} {'overall L1':>11} {'flat L1':>9} {'spike L1':>9}")  # noqa: T201
        for kk in sorted(agg):
            e = agg[kk][..., c]
            sp = e[spike].mean() if spike.any() else float("nan")
            print(  # noqa: T201
                f"    {kk:>4} {e.mean():11.4f} {e[flat].mean():9.4f} {sp:9.4f}"
            )
        e1, ek = agg[1][..., c], agg[k][..., c]
        sp_tax = (
            (e1[spike].mean() - ek[spike].mean()) / e1[spike].mean()
            if spike.any()
            else float("nan")
        )
        print(  # noqa: T201
            f"    variance tax (1 - meanK/single): overall "
            f"{(e1.mean() - ek.mean()) / e1.mean():5.1%} | spike {sp_tax:5.1%}"
            f"  -> bias floor (meanK) = deterministic-head-reachable error"
        )


if __name__ == "__main__":
    mp.set_forkserver_preload(["rbyte", "polars"])
    main()
