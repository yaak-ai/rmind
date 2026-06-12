#! /usr/bin/env python
"""Precompute frozen-encoder condition tokens for feature-cached training.

Every flow/regression finetune FREEZES the episode builder + encoder, so the 12
condition tokens per (drive, frame) are deterministic constants — yet the
training loop recomputes them (JPEG decode -> DINOv3 ViT -> 8-layer encoder)
every step of every epoch of every experiment, to feed a 3M-param decoder.
This script computes them ONCE per dataset split and saves them; training then
runs decoder-only from the cache (FlowFeatureTrainer), ~an order of magnitude
faster per step and with no image IO at all.

Two condition variants are saved per frame so ActionHistoryDropout-style
training stays EXACT under caching:
  - cond:       normal condition tokens
  - cond_hist0: condition tokens with the action-history fields zeroed
(the frozen encoder is a function, not a constant — zeroed inputs change the
summaries, so the variant must be precomputed, not patched post-hoc).

Validity: the cache is tied to the (frozen) encoder weights + the condition
spec (POLICY_CONDITION_TOKENS); both are recorded in the file's metadata. Any
flow checkpoint works as the model source — all finetunes share the pretrained
encoder unchanged. Invalidate by deleting the file if the encoder artifact or
condition spec ever changes.

Usage:
    uv run python -m rmind.scripts.flow_cache_features \
        --config-path <repo>/config --config-name train.yaml \
        experiment=yaak/control_transformer/finetune_flow_pilot_lds \
        model.artifact=yaak/action-flow/model-pg5lzmvk:v199 \
        '+cache.out_dir=artifacts/feature_cache/pilot'

Writes <out_dir>/train.pt and <out_dir>/val.pt.
Options (+cache.*): out_dir (required), splits (default [train, val]).
"""

import multiprocessing as mp
from pathlib import Path
from typing import TYPE_CHECKING, Any

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from structlog import get_logger

import rmind.components.objectives.flow_policy as flow_policy_module

logger = get_logger(__name__)

if TYPE_CHECKING:
    import pytorch_lightning as pl

FRAME_IDX_KEY = "meta/ImageMetadata.cam_front_left/frame_idx"
TIME_STAMP_KEY = "meta/ImageMetadata.cam_front_left/time_stamp"
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


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:  # noqa: PLR0915
    opts = cfg.get("cache") or {}
    out_dir = Path(str(opts["out_dir"]))
    splits = [str(s) for s in opts.get("splits", ["train", "val"])]
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision(cfg.matmul_precision)
    model: pl.LightningModule = instantiate(cfg.model).to(device).eval()
    objective = model.objectives["policy"]
    if not isinstance(objective, flow_policy_module.FlowPolicyObjective):
        msg = f"policy objective is not a FlowPolicyObjective: {type(objective)}"
        raise TypeError(msg)
    horizon = objective.decoder.action_horizon

    datamodule: pl.LightningDataModule = instantiate(cfg.datamodule)
    if hasattr(datamodule, "setup"):
        datamodule.setup("fit")

    def condition(batch: dict) -> torch.Tensor:
        with torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
            episode = model.episode_builder(batch)
            embedding = model.encoder(
                src=episode.embeddings_flattened, mask=episode.attention_mask
            )
            return objective._condition_tokens(
                episode=episode, embedding=embedding
            ).float()

    for split in splits:
        loader = (
            datamodule.train_dataloader() if split == "train" else datamodule.val_dataloader()
        )
        conds, conds0, targets, fidxs, ids = [], [], [], [], []
        tstamps: list[torch.Tensor] = []
        with torch.inference_mode():
            for batch_idx, batch in enumerate(loader):
                batch = _to_device(batch, device)
                cond = condition(batch)
                b0 = {**batch, "data": {**batch["data"]}}
                for k in HIST_KEYS:
                    b0["data"][k] = torch.zeros_like(batch["data"][k])
                cond0 = condition(b0)
                gt = objective._target_actions(batch).float()
                if gt.shape[1] != horizon:
                    gt = gt[:, objective._target_slice()]
                conds.append(cond.half().cpu())
                conds0.append(cond0.half().cpu())
                targets.append(gt.cpu())
                fidxs.append(batch["data"][FRAME_IDX_KEY].cpu())
                tstamps.append(batch["data"][TIME_STAMP_KEY].cpu())
                batch_ids = batch["meta"]["input_id"]
                ids.extend(str(x) for x in batch_ids)
                if batch_idx % 50 == 0:
                    logger.debug("cached", split=split, batch_idx=batch_idx)

        payload = {
            "cond": torch.cat(conds),          # (N, S, D) fp16
            "cond_hist0": torch.cat(conds0),   # (N, S, D) fp16
            "target_actions": torch.cat(targets),  # (N, H, A) fp32 raw
            "frame_idx": torch.cat(fidxs),          # (N, T) full history window
            "time_stamp": torch.cat(tstamps),       # (N, T) epoch stamps (reye)
            "input_id": ids,
            "meta": {
                "model_artifact": str(cfg.model.get("artifact", "?")),
                "condition_tokens": [
                    (str(m), str(k))
                    for m, k in flow_policy_module.POLICY_CONDITION_TOKENS
                ],
                "history_steps": objective.history_steps,
                "action_keys": list(objective.action_keys),
            },
        }
        path = out_dir / f"{split}.pt"
        torch.save(payload, path)
        logger.info(
            "wrote feature cache",
            path=str(path),
            frames=payload["cond"].shape[0],
            cond_shape=tuple(payload["cond"].shape[1:]),
            size_mb=round(path.stat().st_size / 1e6, 1),
        )


if __name__ == "__main__":
    mp.set_forkserver_preload(["rbyte", "polars"])
    main()
