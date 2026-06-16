#! /usr/bin/env python
"""Write reye-compatible prediction parquets from cached-track checkpoints.

Cached (FlowFeatureTrainer) checkpoints carry no encoder, so `just predict`
cannot run on them. This script produces the SAME parquet schema as the full
path's DataFramePredictionWriter — the format reye's dashboard consumes
(github.com/yaak-ai/reye: add the path under `models:` in
config/dashboard/default.yaml) — from the feature cache + objective weights:

    columns:
      batch/data/meta/ImageMetadata.cam_front_left/time_stamp   array[i64,T]
      batch/data/meta/ImageMetadata.cam_front_left/frame_idx    array[i32,T]
      batch/meta/input_id                                       str
      policy/{ground_truth,prediction_value,score_l1}/value/continuous/<key>
                                                                array[f32,H]

Works for both FlowPolicyObjective (readouts: single | meank | mode |
mode_medoid | ranker) and RegressionPolicyObjective (deterministic; readout
ignored). readout=ranker needs +cpredict.ranker_ckpt (a flow_ranker.py train
checkpoint) and uses its softmax-weighted readout (+cpredict.ranker_temp,
default 4.0).

Usage:
    uv run python -m rmind.scripts.flow_cached_predict \
        --config-path <repo>/config --config-name train.yaml \
        experiment=yaak/control_transformer/finetune_flow_pilot_lds_cached \
        '+cpredict.cache=artifacts/feature_cache/pilot/val.pt' \
        '+cpredict.cached_artifact=yaak/action-flow/model-<run>:vN' \
        '+cpredict.readout=mode' '+cpredict.out=outputs/cached_predictions/<run>_mode.parquet'

Options (+cpredict.*): cache, cached_artifact, out (required); readout
(default single), samples (K for meank/mode readouts, default 16).
"""

import glob
import multiprocessing as mp
import os
from pathlib import Path

import hydra
import polars as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from structlog import get_logger

from rmind.components.objectives.consensus import mode_aware_anchor
from rmind.components.objectives.flow_policy import FlowPolicyObjective

logger = get_logger(__name__)

TS_COL = "batch/data/meta/ImageMetadata.cam_front_left/time_stamp"
FI_COL = "batch/data/meta/ImageMetadata.cam_front_left/frame_idx"
ID_COL = "batch/meta/input_id"


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:  # noqa: PLR0915
    import wandb

    opts = cfg.get("cpredict") or {}
    cache_path = str(opts["cache"])
    artifact = str(opts["cached_artifact"])
    out = Path(str(opts["out"]))
    readout = str(opts.get("readout", "single"))
    k = int(opts.get("samples", 16))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ranker = None
    ranker_temp = float(opts.get("ranker_temp", 4.0))
    if readout == "ranker":
        from rmind.scripts.flow_ranker import DrawRanker

        rk = torch.load(
            str(opts["ranker_ckpt"]), map_location="cpu", weights_only=False
        )
        ranker = DrawRanker(**rk["config"]).to(device).eval()
        ranker.load_state_dict(rk["state_dict"])
    torch.set_float32_matmul_precision(cfg.get("matmul_precision", "high"))

    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    if "time_stamp" not in payload:
        msg = f"cache {cache_path} predates the time_stamp schema — rebuild it"
        raise KeyError(msg)
    cond = payload["cond"].float()
    gt = payload["target_actions"].float()

    trainer_module = instantiate(cfg.model)
    art = wandb.Api().artifact(artifact, type="model")
    ck = next(
        f for f in glob.glob(os.path.join(art.download(), "*")) if os.path.isfile(f)
    )
    trainer_module.load_state_dict(
        torch.load(ck, map_location="cpu", weights_only=False)["state_dict"]
    )
    objective = trainer_module.objective.to(device).eval()
    is_flow = isinstance(objective, FlowPolicyObjective)
    keys = list(objective.action_keys)
    steer_idx = next(
        (i for i, key in enumerate(keys) if "steering" in key), len(keys) - 1
    )

    preds: list[torch.Tensor] = []
    chunk = 512
    with torch.inference_mode():
        for lo in range(0, cond.shape[0], chunk):
            c = cond[lo : lo + chunk].to(device)
            with torch.autocast(
                device.type, torch.bfloat16, enabled=device.type == "cuda"
            ):
                if not is_flow:
                    p = objective._to_raw_space(
                        objective.decoder(condition_tokens=c).float()
                    )
                elif readout == "single":
                    p = objective._to_raw_space(
                        objective.decoder.sample(
                            condition_tokens=c,
                            noise=objective._noise(
                                condition_tokens=c, generator=None
                            ),
                        ).float()
                    )
                else:
                    c_rep = c.repeat_interleave(k, dim=0)
                    traj_m = (
                        objective.decoder.sample(
                            condition_tokens=c_rep,
                            noise=objective._noise(
                                condition_tokens=c_rep, generator=None
                            ),
                        )
                        .float()
                        .reshape(
                            c.shape[0],
                            k,
                            objective.decoder.action_horizon,
                            objective.decoder.action_dim,
                        )
                    )
                    draws = objective._to_raw_space(traj_m)
                    if readout == "meank":
                        p = draws.mean(dim=1)
                    elif readout == "ranker":
                        scores = ranker(c.float(), traj_m)  # (B, K)
                        w = torch.softmax(scores / ranker_temp, dim=1)
                        p = (draws * w[:, :, None, None]).sum(dim=1)
                    else:
                        p, _, _ = mode_aware_anchor(
                            draws,
                            steer_idx,
                            anchor="medoid" if readout == "mode_medoid" else "mean",
                        )
            preds.append(p.float().cpu())
            if (lo // chunk) % 10 == 0:
                logger.debug("predicted", frames=lo)

    pred = torch.cat(preds)
    err = (pred - gt).abs()
    horizon = gt.shape[1]

    cols: dict = {
        TS_COL: pl.Series(payload["time_stamp"].numpy(), dtype=pl.Array(pl.Int64, payload["time_stamp"].shape[1])),
        FI_COL: pl.Series(payload["frame_idx"].numpy(), dtype=pl.Array(pl.Int32, payload["frame_idx"].shape[1])),
        ID_COL: pl.Series(list(payload["input_id"])),
    }
    for c, key in enumerate(keys):
        cols[f"policy/ground_truth/value/continuous/{key}"] = pl.Series(
            gt[..., c].numpy(), dtype=pl.Array(pl.Float32, horizon)
        )
        cols[f"policy/prediction_value/value/continuous/{key}"] = pl.Series(
            pred[..., c].numpy(), dtype=pl.Array(pl.Float32, horizon)
        )
        cols[f"policy/score_l1/value/continuous/{key}"] = pl.Series(
            err[..., c].numpy(), dtype=pl.Array(pl.Float32, horizon)
        )
    df = pl.DataFrame(cols)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out)
    logger.info(
        "wrote cached predictions",
        path=str(out),
        rows=df.height,
        readout=readout if is_flow else "deterministic",
    )


if __name__ == "__main__":
    mp.set_forkserver_preload(["rbyte", "polars"])
    main()
