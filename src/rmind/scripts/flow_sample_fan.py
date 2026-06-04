#! /usr/bin/env python
"""Sample-concentration diagnostic for the flow policy.

For each frame, draws N action trajectories from the flow decoder and measures
how the samples concentrate around the ground-truth first-step steering angle —
distinguishing "the distribution's mass is on the GT maneuver" from "best-of-N
fished a lucky tail sample". Produces an interactive fan plot (all N samples per
frame vs GT) with per-frame hit rates, plus spike-vs-flat aggregate stats.

Usage (mirrors rmind-predict; supply the same overrides you use for `just predict`):

    just fan inference=yaak/control_transformer/policy \
        model.artifact=<entity/project/model-...:vX> \
        '+fan.legacy_condition=true' '+fan.out=fan_steering.html'

Options (all via +fan.*):
    num_samples       draws per frame                       (default 32)
    eps               hit-rate thresholds                   (default [0.05, 0.1, 0.2])
    spike_threshold   |gt steering| defining a spike        (default 0.5)
    spike_pad         frames of context around spikes       (default 50)
    legacy_condition  condition on summaries+waypoints only, for checkpoints
                      trained before image tokens were added (default false)
    out               output figure path (.html)            (default flow_sample_fan.html)
    data              replot from a cached .npz (written next to `out` on every
                      sampling run) instead of re-running the model
"""

import multiprocessing as mp
from pathlib import Path
from typing import TYPE_CHECKING, Any

import hydra
import numpy as np
import plotly.graph_objects as go
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from plotly.subplots import make_subplots
from structlog import get_logger

import rmind.components.objectives.flow_policy as flow_policy_module
from rmind.components.base import Modality, SummaryToken

logger = get_logger(__name__)

if TYPE_CHECKING:
    import pytorch_lightning as pl

FRAME_IDX_KEY = "meta/ImageMetadata.cam_front_left/frame_idx"

# Conditioning used by checkpoints trained before image tokens were added.
LEGACY_CONDITION_TOKENS: tuple[tuple[Modality, str], ...] = (
    (Modality.SUMMARY, SummaryToken.OBSERVATION_SUMMARY),
    (Modality.SUMMARY, SummaryToken.OBSERVATION_HISTORY),
    (Modality.CONTEXT, "waypoints"),
)


def _get_path(obj: Any, *keys: str) -> Any:
    for key in keys:
        obj = getattr(obj, key, None) if hasattr(obj, key) else obj[key]
    return obj


def _to_device(obj: Any, device: torch.device) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if hasattr(obj, "to"):
        return obj.to(device)
    return obj


def _dilate(mask: np.ndarray, pad: int) -> np.ndarray:
    if pad <= 0:
        return mask
    kernel = np.ones(2 * pad + 1)
    return np.convolve(mask.astype(float), kernel, mode="same") > 0


def _intervals(frame: np.ndarray, mask: np.ndarray) -> list[tuple[float, float]]:
    """Contiguous True runs of `mask`, as (start_frame, end_frame) pairs."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return [
        (float(frame[start]), float(frame[end - 1]))
        for start, end in zip(edges[::2], edges[1::2], strict=True)
    ]


def _collect(
    cfg: DictConfig,
    *,
    num_samples: int,
    sampling_steps: int | None = None,
    sampling_method: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the model over the predict dataloader; return (frame, gt, sample)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.debug("instantiating model", target=cfg.model._target_)
    model: pl.LightningModule = instantiate(cfg.model)
    model = model.to(device).eval()

    objective = model.objectives["policy"]
    if not isinstance(objective, flow_policy_module.FlowPolicyObjective):
        msg = f"policy objective is not a FlowPolicyObjective: {type(objective)}"
        raise TypeError(msg)

    steer_idx = next(
        i for i, k in enumerate(objective.action_keys) if "steering" in k
    )
    horizon = objective.decoder.action_horizon

    logger.debug("instantiating datamodule", target=cfg.datamodule._target_)
    datamodule: pl.LightningDataModule = instantiate(cfg.datamodule)
    if hasattr(datamodule, "setup"):
        datamodule.setup("predict")

    frames: list[torch.Tensor] = []
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
                # One batched call: each frame's condition repeated N times.
                condition_rep = condition_tokens.repeat_interleave(
                    num_samples, dim=0
                )
                trajectories = objective.decoder.sample(
                    condition_tokens=condition_rep,
                    noise=objective._noise(
                        condition_tokens=condition_rep, generator=None
                    ),
                    steps=sampling_steps,
                    method=sampling_method,
                )

            batch_size = condition_tokens.shape[0]
            trajectories = trajectories.float().reshape(
                batch_size, num_samples, horizon, -1
            )

            gt = objective._target_actions(batch).float()
            if gt.shape[1] != horizon:
                gt = gt[:, objective._target_slice()]

            frame_idx = _get_path(batch, "data", FRAME_IDX_KEY)
            # x-axis: the first *predicted* step's frame when metadata covers
            # it, else the conditioning frame (predict batches may carry
            # metadata for the history window only).
            if frame_idx.ndim > 1:
                t = min(objective.history_steps, frame_idx.shape[1] - 1)
                frame_idx = frame_idx[:, t]

            frames.append(frame_idx.cpu().flatten())
            gts.append(gt[:, 0, steer_idx].cpu())  # first-step steering
            samples.append(trajectories[:, :, 0, steer_idx].cpu())  # (B, N)
            logger.debug("sampled batch", batch_idx=batch_idx, frames=batch_size)

    frame = torch.cat(frames).numpy()
    gt = torch.cat(gts).numpy()
    sample = torch.cat(samples).numpy()  # (F, N)

    valid = ~np.isnan(gt)
    if (dropped := int((~valid).sum())) > 0:
        logger.warning("dropping NaN ground-truth frames", count=dropped)
    frame, gt, sample = frame[valid], gt[valid], sample[valid]

    order = np.argsort(frame)
    return frame[order], gt[order], sample[order]


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    fan = cfg.get("fan") or {}
    num_samples = int(fan.get("num_samples", 32))
    eps_values = tuple(float(e) for e in fan.get("eps", (0.05, 0.1, 0.2)))
    spike_threshold = float(fan.get("spike_threshold", 0.5))
    spike_pad = int(fan.get("spike_pad", 50))
    out_path = Path(str(fan.get("out", "flow_sample_fan.html")))
    data_path = fan.get("data")
    legacy_condition = bool(fan.get("legacy_condition", False))

    if data_path is not None:
        cached = np.load(str(data_path))
        frame, gt, sample = cached["frame"], cached["gt"], cached["sample"]
        num_samples = sample.shape[1]
        logger.info("replotting from cache", path=str(data_path))
    else:
        if legacy_condition:
            flow_policy_module.POLICY_CONDITION_TOKENS = LEGACY_CONDITION_TOKENS
            logger.info(
                "using legacy condition tokens (summaries + waypoints, no image)"
            )
        sampling_steps = fan.get("sampling_steps")
        if sampling_steps is not None:
            sampling_steps = int(sampling_steps)
        sampling_method = fan.get("sampling_method")
        if sampling_steps is not None or sampling_method is not None:
            logger.info(
                "overriding flow sampling",
                steps=sampling_steps,
                method=sampling_method,
            )
        torch.set_float32_matmul_precision(cfg.matmul_precision)
        frame, gt, sample = _collect(
            cfg,
            num_samples=num_samples,
            sampling_steps=sampling_steps,
            sampling_method=sampling_method,
        )
        cache_path = out_path.with_suffix(".npz")
        np.savez_compressed(cache_path, frame=frame, gt=gt, sample=sample)
        logger.info("cached sampled data", path=str(cache_path))

    abs_err = np.abs(sample - gt[:, None])  # (F, N)
    spike = _dilate(np.abs(gt) > spike_threshold, spike_pad)
    flat = ~spike

    if (nan_frames := int(np.isnan(sample).any(axis=1).sum())) > 0:
        logger.warning(
            "frames with NaN samples (counted as misses)", count=nan_frames
        )

    logger.info(
        "sample concentration (fraction of N draws within eps of gt steering)",
        num_frames=len(frame),
        spike_frames=int(spike.sum()),
        num_samples=num_samples,
    )
    for eps in eps_values:
        hit = (abs_err <= eps).mean(axis=1)  # per-frame fraction; NaN -> miss
        logger.info(
            f"eps={eps:g}",
            spike=f"{hit[spike].mean():.1%}" if spike.any() else "n/a",
            flat=f"{hit[flat].mean():.1%}" if flat.any() else "n/a",
            overall=f"{hit.mean():.1%}",
        )
    # Exclude frames where every draw is NaN (poisoned condition embeddings)
    # so the best-of-N stat stays finite.
    has_draw = ~np.isnan(abs_err).all(axis=1)
    logger.info(
        "first-step steering L1 (nan-aware mean over draws)",
        spike=f"{np.nanmean(abs_err[spike]):.4f}" if spike.any() else "n/a",
        flat=f"{np.nanmean(abs_err[flat]):.4f}" if flat.any() else "n/a",
        best_of_n_spike=f"{np.nanmin(abs_err[spike & has_draw], axis=1).mean():.4f}"
        if (spike & has_draw).any()
        else "n/a",
        best_of_n_flat=f"{np.nanmin(abs_err[flat & has_draw], axis=1).mean():.4f}"
        if (flat & has_draw).any()
        else "n/a",
    )

    eps_plot = eps_values[min(1, len(eps_values) - 1)]
    hit_plot = (abs_err <= eps_plot).mean(axis=1)
    title = (
        f"{num_samples} draws/frame | spike hit@{eps_plot:g}: "
        f"{hit_plot[spike].mean():.1%} | flat: {hit_plot[flat].mean():.1%}"
        if spike.any()
        else f"{num_samples} draws/frame (no spikes above {spike_threshold:g})"
    )

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.04,
    )
    fig.add_trace(
        go.Scattergl(
            x=np.repeat(frame, num_samples),
            y=sample.ravel(),
            mode="markers",
            marker={"size": 2, "color": "rgba(31, 119, 180, 0.15)"},
            name="samples",
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scattergl(
            x=frame,
            y=gt,
            mode="lines",
            line={"color": "crimson", "width": 1.5},
            name="gt",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scattergl(
            x=frame,
            y=hit_plot,
            mode="lines",
            line={"color": "seagreen", "width": 1},
            name=f"hit@{eps_plot:g}",
        ),
        row=2,
        col=1,
    )
    for start, end in _intervals(frame, spike):
        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor="orange",
            opacity=0.12,
            line_width=0,
            row="all",
            col=1,
        )
    fig.update_yaxes(title_text="steering angle", row=1, col=1)
    fig.update_yaxes(title_text=f"hit@{eps_plot:g}", range=[0, 1], row=2, col=1)
    fig.update_xaxes(title_text="frame", row=2, col=1)
    fig.update_layout(
        title=title, height=700, legend={"orientation": "h", "y": 1.06}
    )

    fig.write_html(out_path, include_plotlyjs="cdn")
    logger.info("saved figure", path=str(out_path))


if __name__ == "__main__":
    mp.set_forkserver_preload(["rbyte", "polars"])

    main()
