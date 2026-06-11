#! /usr/bin/env python
"""Recursive action-feedback rollout: does open-loop error compound?

The action-history probe showed the policy depends HEAVILY on past actions
(perturbing them shifts predictions by 1.7-3.4x the baseline error — a copycat
signature). Open-loop metrics feed GT action history, hiding any compounding.
This eval feeds the model's OWN predictions back as action history (vision/
speed/waypoints stay GT — it isolates the action-feedback channel, the only
one measurable without a simulator) and compares against the open-loop control
as a function of rollout depth.

Readout:
  - recursive ~= open-loop: errors don't compound through the action channel;
    open-loop metrics are trustworthy for it (sim still needed for the
    vision-state channel).
  - recursive >> open-loop, growing with depth: open-loop metrics understate
    deployment error; copycat is biting; history-dropout training and/or
    closed-loop eval become priorities.

Usage:
    uv run python -m rmind.scripts.flow_rollout_eval \
        --config-path <repo>/config --config-name predict.yaml \
        inference=yaak/control_transformer/policy \
        model.artifact=yaak/action-flow/model-<run>:vN \
        datamodule=yaak/predict_val '+rollout.chain=64'

Options (+rollout.*): chain (frames per recursive chain before reset, default
64), max_frames (default 1500), device (default cuda:0).
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

FRAME_IDX_KEY = "meta/ImageMetadata.cam_front_left/frame_idx"
# Raw action-history batch fields (non-_target = observed history), in the
# objective's action_keys order (gas, brake, steering).
HIST_KEYS = (
    "meta/VehicleMotion/gas_pedal_normalized",
    "meta/VehicleMotion/brake_pedal_normalized",
    "meta/VehicleMotion/steering_angle_normalized",
)


def _to_device(obj: Any, device: torch.device) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if hasattr(obj, "to"):
        return obj.to(device)
    return obj


def _slice_row(batch: dict, i: int) -> dict:
    out: dict = {}
    for k, v in batch.items():
        if isinstance(v, dict):
            out[k] = _slice_row(v, i)
        elif isinstance(v, torch.Tensor):
            out[k] = v[i : i + 1]
        else:
            out[k] = v
    return out


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:  # noqa: PLR0915
    opts = cfg.get("rollout") or {}
    chain_len = int(opts.get("chain", 64))
    max_frames = int(opts.get("max_frames", 1500))
    device = torch.device(str(opts.get("device", "cuda:0")))

    torch.set_float32_matmul_precision(cfg.matmul_precision)
    model: pl.LightningModule = instantiate(cfg.model).to(device).eval()
    objective = model.objectives["policy"]
    if not isinstance(objective, flow_policy_module.FlowPolicyObjective):
        msg = f"policy objective is not a FlowPolicyObjective: {type(objective)}"
        raise TypeError(msg)
    horizon = objective.decoder.action_horizon
    hist_steps = objective.history_steps

    datamodule: pl.LightningDataModule = instantiate(cfg.datamodule)
    if hasattr(datamodule, "setup"):
        datamodule.setup("predict")

    # Materialize per-frame rows (CPU) in drive order.
    rows: list[dict] = []
    for batch in datamodule.predict_dataloader():
        b = batch if isinstance(batch, dict) else batch.to_dict()
        n = b["data"][HIST_KEYS[0]].shape[0]
        rows.extend(_slice_row(b, i) for i in range(n))
        if len(rows) >= max_frames:
            break
    rows = rows[:max_frames]
    logger.info("materialized rows", n=len(rows))

    t_field = rows[0]["data"][HIST_KEYS[0]].shape[1]
    fidx = np.array([
        int(r["data"][FRAME_IDX_KEY][0, min(hist_steps, t_field - 1)]) for r in rows
    ])
    stride = int(np.median(np.diff(fidx)))
    logger.info("field length / stride", t_field=t_field, stride=stride)

    gen = torch.Generator(device=device).manual_seed(0)

    def predict_first_action(row: dict) -> np.ndarray | None:
        row = _to_device(row, device)
        with torch.inference_mode(), torch.autocast(
            device.type, torch.bfloat16, enabled=device.type == "cuda"
        ):
            episode = model.episode_builder(row)
            embedding = model.encoder(
                src=episode.embeddings_flattened, mask=episode.attention_mask
            )
            cond = objective._condition_tokens(episode=episode, embedding=embedding)
            noise = torch.randn(
                1, horizon, objective.decoder.action_dim,
                device=device, generator=gen, dtype=cond.dtype,
            )
            sample = objective._to_raw_space(
                objective.decoder.sample(condition_tokens=cond, noise=noise)
            )
        a = sample[0, 0].float().cpu().numpy()  # first-step (gas, brake, steer)
        return a if np.isfinite(a).all() else None

    def gt_first_action(row: dict) -> np.ndarray:
        gt = objective._target_actions(row).float()
        if gt.shape[1] != horizon:
            gt = gt[:, objective._target_slice()]
        return gt[0, 0].cpu().numpy()

    results: dict[str, list[tuple[int, float]]] = {"open": [], "recursive": []}
    for mode in ("open", "recursive"):
        torch.manual_seed(0)
        gen.manual_seed(0)
        pred_hist: dict[int, np.ndarray] = {}  # frame_idx -> predicted action
        depth = 0
        for i, row in enumerate(rows):
            # chain bookkeeping: reset at gaps and every chain_len frames
            if i > 0 and (fidx[i] - fidx[i - 1] != stride or depth >= chain_len):
                pred_hist.clear()
                depth = 0
            if mode == "recursive" and pred_hist:
                row = {**row, "data": {**row["data"]}}
                for c, key in enumerate(HIST_KEYS):
                    field = row["data"][key].clone()
                    # history positions 0..hist_steps-1 hold actions at frames
                    # fidx[i] - (hist_steps-1-j)*stride
                    for j in range(min(hist_steps, t_field)):
                        f = fidx[i] - (hist_steps - 1 - j) * stride
                        if f in pred_hist:
                            field[0, j] = float(pred_hist[f][c])
                    row["data"][key] = field
            a = predict_first_action(row)
            if a is None:
                pred_hist.clear()
                depth = 0
                continue
            gt = gt_first_action(row)
            if np.isfinite(gt).all():
                err = float(np.abs(a - gt)[2])  # steering first-step |err|
                results[mode].append((depth, err))
            # the first predicted step targets frame fidx[i] + stride
            pred_hist[fidx[i] + stride] = a
            depth += 1
            if i % 200 == 0:
                logger.debug("rollout", mode=mode, i=i)

    print(f"\nrecursive action-feedback rollout (steering first-step |err|, chains of {chain_len})")  # noqa: T201
    print(f"{'depth bucket':>14} {'open-loop':>10} {'recursive':>10} {'ratio':>7} {'n':>6}")  # noqa: T201
    buckets = [(0, 4), (4, 8), (8, 16), (16, 32), (32, 64)]
    for lo, hi in buckets:
        sel = {}
        for mode in ("open", "recursive"):
            v = [e for d, e in results[mode] if lo <= d < hi]
            sel[mode] = (float(np.mean(v)), len(v)) if v else (float("nan"), 0)
        o, n_o = sel["open"]
        r, _ = sel["recursive"]
        print(f"{f'{lo}-{hi}':>14} {o:10.4f} {r:10.4f} {r / o if o > 0 else float('nan'):7.2f} {n_o:6d}")  # noqa: T201
    o_all = float(np.mean([e for _, e in results["open"]]))
    r_all = float(np.mean([e for _, e in results["recursive"]]))
    print(f"{'ALL':>14} {o_all:10.4f} {r_all:10.4f} {r_all / o_all:7.2f}")  # noqa: T201


if __name__ == "__main__":
    mp.set_forkserver_preload(["rbyte", "polars"])
    main()
