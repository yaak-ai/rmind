#! /usr/bin/env python
"""Learned draw ranker — close the selection gap of the flow action expert.

The residual decomposition (flow_meank_eval) shows the bottleneck is SELECTION,
not coverage: on held-out spike frames meanK steering chunk-L1 is ~0.33, the
oracle mode anchor is ~0.26-0.28 and the best single draw ~0.15. The decoder
already produces good trajectories; the readout can't tell which one to trust.
This script learns that readout: a small scorer s(cond, draw) trained to rank
the K frozen-flow draws of a frame by their ground-truth distance, used at
inference as a top-M weighted readout (M=K recovers meanK, M=1 is argmax).

Stages (+ranker.stage=...):

  draws  — sample a K-draw bank per cached frame from a frozen FlowFeatureTrainer
           checkpoint (same sampler/seed discipline as flow_meank_eval) and store
           model-space draws + per-channel raw-space chunk-L1 labels:
             artifacts/ranker/<split>_draws.pt
               {draws (N,K,H,A) fp16 model-space, l1 (N,K,C) fp32 raw-space,
                steer_idx, artifact, k}
  train  — train the ranker on the train bank (cond comes from the feature
           cache); listwise softmax-CE over K draws against soft targets
           softmax(-L1/tau), loss per-frame weighted by label spread so flat
           frames (where all draws tie) don't drown the maneuvers.
  eval   — selection table on the val bank: spike/flat/overall steering chunk-L1
           for ranker top-M mean (M in 1,2,4,8,16,K), softmax-weighted mean,
           vs meanK / oracle-draw / oracle-mode reference rows.

Usage (cached experiment config supplies the objective/decoder graph):
    uv run python -m rmind.scripts.flow_ranker --config-path $PWD/config \
        --config-name train.yaml \
        experiment=yaak/control_transformer/finetune_flow_pilot_lds_cached \
        '+ranker.stage=draws' '+ranker.cache=artifacts/feature_cache/pilot/train.pt' \
        '+ranker.cached_artifact=yaak/action-flow/model-<run>:v4' \
        '+ranker.out=artifacts/ranker/train_draws.pt'

Options (+ranker.*): stage (draws|train|eval); cache, cached_artifact, out, k
(draws stage); cache, bank, val_cache, val_bank, ckpt_out, epochs, lr, tau,
batch, target_channel, spread_weight, pointwise (train stage); cache, bank,
ckpt (eval stage).
"""

import glob
import multiprocessing as mp
import os
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from structlog import get_logger
from torch import Tensor, nn
from torch.nn import functional as F

from rmind.components.objectives.consensus import split_modes
from rmind.components.objectives.flow_policy import FlowPolicyObjective

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# stage: draws
# --------------------------------------------------------------------------- #
def _load_cached_objective(cfg: DictConfig, artifact: str, device) -> FlowPolicyObjective:
    import wandb

    trainer_module = instantiate(cfg.model)
    art = wandb.Api().artifact(artifact, type="model")
    ck = next(
        f for f in glob.glob(os.path.join(art.download(), "*")) if os.path.isfile(f)
    )
    sd = torch.load(ck, map_location="cpu", weights_only=False)["state_dict"]
    trainer_module.load_state_dict(sd)
    objective = trainer_module.objective.to(device).eval()
    if not isinstance(objective, FlowPolicyObjective):
        msg = f"artifact objective is not a FlowPolicyObjective: {type(objective)}"
        raise TypeError(msg)
    return objective


def _stage_draws(cfg: DictConfig, opts) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_path, artifact = str(opts["cache"]), str(opts["cached_artifact"])
    out, k = Path(str(opts["out"])), int(opts.get("k", 32))

    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    cond = payload["cond"].float()
    gt = payload["target_actions"].float()  # raw space (N, H, C)

    objective = _load_cached_objective(cfg, artifact, device)
    horizon = objective.decoder.action_horizon
    keys = list(objective.action_keys)
    steer_idx = next((i for i, key in enumerate(keys) if "steering" in key), -1)

    draws_model: list[Tensor] = []
    l1: list[Tensor] = []
    chunk = 256
    torch.manual_seed(0)  # same discipline as flow_meank_eval: reproducible bank
    with torch.inference_mode():
        for lo in range(0, cond.shape[0], chunk):
            c = cond[lo : lo + chunk].to(device)
            c_rep = c.repeat_interleave(k, dim=0)
            with torch.autocast(
                device.type, torch.bfloat16, enabled=device.type == "cuda"
            ):
                traj = objective.decoder.sample(
                    condition_tokens=c_rep,
                    noise=objective._noise(condition_tokens=c_rep, generator=None),
                )
            traj = traj.float().reshape(c.shape[0], k, horizon, -1)
            raw = objective._to_raw_space(traj)
            g = gt[lo : lo + chunk].to(device)
            # per-draw, per-channel chunk L1 in raw space: (B, K, C)
            err = (raw - g[:, None]).abs().mean(dim=2)
            draws_model.append(traj.half().cpu())
            l1.append(err.float().cpu())
            if (lo // chunk) % 20 == 0:
                logger.info("sampled", frames=lo, total=cond.shape[0])

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "draws": torch.cat(draws_model),
            "l1": torch.cat(l1),
            "steer_idx": steer_idx,
            "keys": keys,
            "artifact": artifact,
            "cache": cache_path,
            "k": k,
        },
        out,
    )
    logger.info("wrote draw bank", path=str(out), frames=cond.shape[0], k=k)


# --------------------------------------------------------------------------- #
# ranker model
# --------------------------------------------------------------------------- #
class DrawRanker(nn.Module):
    """s(cond, draw) -> score. Higher = expected closer to ground truth.

    Condition: the 12 per-frame encoder tokens are flattened (roles are
    positional in the cache, so flatten preserves them) and projected; each
    model-space draw (H*A floats) is projected separately; the head scores the
    joint + elementwise-product features. ~1.4M params.
    """

    def __init__(self, *, cond_tokens: int = 12, cond_dim: int = 384,
                 horizon: int = 6, action_dim: int = 2, width: int = 256) -> None:
        super().__init__()
        self.cond_net = nn.Sequential(
            nn.LayerNorm(cond_tokens * cond_dim),
            nn.Linear(cond_tokens * cond_dim, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.draw_net = nn.Sequential(
            nn.Linear(horizon * action_dim, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.head = nn.Sequential(
            nn.Linear(3 * width, width), nn.GELU(), nn.Linear(width, 1)
        )

    def forward(self, cond: Tensor, draws: Tensor) -> Tensor:
        """cond (B, S, D), draws (B, K, H, A) -> scores (B, K)."""
        c = self.cond_net(cond.flatten(1))  # (B, W)
        d = self.draw_net(draws.flatten(2))  # (B, K, W)
        c = c[:, None].expand_as(d)
        return self.head(torch.cat([c, d, c * d], dim=-1)).squeeze(-1)


# --------------------------------------------------------------------------- #
# stage: train
# --------------------------------------------------------------------------- #
def _stage_train(opts) -> None:  # noqa: PLR0915
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bank = torch.load(str(opts["bank"]), map_location="cpu", weights_only=False)
    cache = torch.load(str(opts["cache"]), map_location="cpu", weights_only=False)
    cond_all = cache["cond"]  # (N, S, D) fp16
    draws_all, l1_all = bank["draws"], bank["l1"]  # (N,K,H,A) fp16, (N,K,C)
    target_channel = int(opts.get("target_channel", bank["steer_idx"]))
    labels_all = l1_all[..., target_channel]  # (N, K) raw-space steering chunk L1

    epochs = int(opts.get("epochs", 8))
    lr = float(opts.get("lr", 3e-4))
    tau = float(opts.get("tau", 0.05))
    batch = int(opts.get("batch", 512))
    spread_weight = bool(opts.get("spread_weight", True))
    pointwise = bool(opts.get("pointwise", False))
    ckpt_out = Path(str(opts.get("ckpt_out", "artifacts/ranker/ranker.pt")))

    n = cond_all.shape[0]
    valid = torch.isfinite(labels_all).all(1) & torch.isfinite(
        draws_all.float()
    ).flatten(1).all(1)
    idx_all = torch.nonzero(valid).squeeze(1)
    logger.info("train frames", n=int(idx_all.numel()), dropped=int(n - idx_all.numel()))

    model = DrawRanker(
        cond_tokens=cond_all.shape[1], cond_dim=cond_all.shape[2],
        horizon=draws_all.shape[2], action_dim=draws_all.shape[3],
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps_per_epoch = (idx_all.numel() + batch - 1) // batch
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * steps_per_epoch)

    g = torch.Generator().manual_seed(0)
    for ep in range(epochs):
        perm = idx_all[torch.randperm(idx_all.numel(), generator=g)]
        tot, cnt = 0.0, 0
        for lo in range(0, perm.numel(), batch):
            ix = perm[lo : lo + batch]
            cond = cond_all[ix].to(device).float()
            draws = draws_all[ix].to(device).float()
            lab = labels_all[ix].to(device)  # (B, K)
            scores = model(cond, draws)
            if pointwise:
                loss_f = F.mse_loss(-scores, lab / tau, reduction="none").mean(1)
            else:
                # listwise CE against soft targets: best draws get the mass
                target = F.softmax(-lab / tau, dim=1)
                loss_f = -(target * F.log_softmax(scores, dim=1)).sum(1)
            if spread_weight:
                # frames where all draws tie (flat road) carry ~no signal;
                # weight by label spread, mean-1 normalized per batch
                w = lab.std(dim=1)
                w = w / w.mean().clamp_min(1e-8)
                loss = (loss_f * w).mean()
            else:
                loss = loss_f.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot, cnt = tot + float(loss) * ix.numel(), cnt + ix.numel()
        logger.info("epoch", ep=ep, loss=tot / cnt)

    ckpt_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "cond_tokens": int(cond_all.shape[1]),
                "cond_dim": int(cond_all.shape[2]),
                "horizon": int(draws_all.shape[2]),
                "action_dim": int(draws_all.shape[3]),
            },
            "target_channel": target_channel,
            "tau": tau,
            "bank": str(opts["bank"]),
        },
        ckpt_out,
    )
    logger.info("wrote ranker", path=str(ckpt_out))


# --------------------------------------------------------------------------- #
# stage: eval
# --------------------------------------------------------------------------- #
def _stage_eval(cfg: DictConfig, opts) -> None:  # noqa: PLR0915
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bank = torch.load(str(opts["bank"]), map_location="cpu", weights_only=False)
    cache = torch.load(str(opts["cache"]), map_location="cpu", weights_only=False)
    ck = torch.load(str(opts["ckpt"]), map_location="cpu", weights_only=False)

    model = DrawRanker(**ck["config"]).to(device).eval()
    model.load_state_dict(ck["state_dict"])

    # raw-space draws for the selection readouts: invert via the objective
    objective = _load_cached_objective(cfg, str(bank["artifact"]), device)
    steer = int(bank["steer_idx"])
    gt = cache["target_actions"].float()  # (N, H, C) raw
    draws_m = bank["draws"]  # (N, K, H, A) model space fp16
    cond_all = cache["cond"]
    n, k = draws_m.shape[0], draws_m.shape[1]

    scores = torch.empty(n, k)
    raw = torch.empty_like(draws_m, dtype=torch.float32)
    with torch.inference_mode():
        for lo in range(0, n, 1024):
            d = draws_m[lo : lo + 1024].to(device).float()
            c = cond_all[lo : lo + 1024].to(device).float()
            scores[lo : lo + 1024] = model(c, d).cpu()
            raw[lo : lo + 1024] = objective._to_raw_space(d).cpu()

    gt_np, raw_np = gt.numpy(), raw.numpy()
    valid = np.isfinite(gt_np).all(axis=(1, 2)) & np.isfinite(raw_np).all(
        axis=(1, 2, 3)
    )
    gt_np, raw_np, sc = gt_np[valid], raw_np[valid], scores.numpy()[valid]
    spike = (np.abs(gt_np[..., steer]) > 0.5).any(axis=1)
    flat = (np.abs(gt_np[..., steer]) < 0.05).all(axis=1)

    def steer_l1(pred):  # (F, H, A) -> (F,)
        return np.abs(pred[..., steer] - gt_np[..., steer]).mean(axis=1)

    def gas_l1(pred):
        return np.abs(pred[..., 0] - gt_np[..., 0]).mean(axis=1)

    def row(name, pred):
        fe, ge = steer_l1(pred), gas_l1(pred)
        print(  # noqa: T201
            f"  {name:24s} overall {fe.mean():.4f}  flat {fe[flat].mean():.4f}  "
            f"SPIKE {fe[spike].mean():.4f}  | gas spike {ge[spike].mean():.4f}"
        )

    print(f"\nranker selection table ({valid.sum()} frames, K={k}):")  # noqa: T201
    row("meanK (baseline)", raw_np.mean(axis=1))
    order = np.argsort(-sc, axis=1)  # best score first
    for m in (1, 2, 4, 8, 16):
        if m > k:
            continue
        top = np.take_along_axis(
            raw_np, order[:, :m, None, None], axis=1
        ).mean(axis=1)
        row(f"ranker top-{m} mean", top)
    w = np.exp(sc - sc.max(axis=1, keepdims=True))
    w /= w.sum(axis=1, keepdims=True)
    row("ranker softmax-wt", (raw_np * w[:, :, None, None]).sum(axis=1))

    # references: oracle draw (coverage), oracle mode (selection ceiling)
    per_draw = np.abs(raw_np[..., steer] - gt_np[:, None, :, steer]).mean(axis=2)
    best = np.take_along_axis(
        raw_np, per_draw.argmin(axis=1)[:, None, None, None], axis=1
    ).squeeze(1)
    row("oracle best draw", best)
    bimodal, order_m, left_n, _ = split_modes(torch.from_numpy(raw_np), steer)
    bimodal_np, order_np, left_np = bimodal.numpy(), order_m.numpy(), left_n.numpy()
    anchors = []
    for i in range(raw_np.shape[0]):
        if not bimodal_np[i]:
            anchors.append(raw_np[i].mean(axis=0))
            continue
        a = raw_np[i][order_np[i, : left_np[i]]].mean(axis=0)
        b = raw_np[i][order_np[i, left_np[i] :]].mean(axis=0)
        ea = np.abs(a[:, steer] - gt_np[i, :, steer]).mean()
        eb = np.abs(b[:, steer] - gt_np[i, :, steer]).mean()
        anchors.append(a if ea <= eb else b)
    row("oracle mode anchor", np.stack(anchors))

    # how good is the ranker as a classifier of the best draw?
    top1 = order[:, 0]
    oracle1 = per_draw.argmin(axis=1)
    hit = (top1 == oracle1).mean()
    rank_of_pick = np.take_along_axis(
        per_draw.argsort(axis=1).argsort(axis=1), top1[:, None], axis=1
    ).squeeze(1)
    print(  # noqa: T201
        f"\n  top1==oracle: {hit:.1%} overall, "
        f"{(top1[spike] == oracle1[spike]).mean():.1%} on spikes "
        f"(chance {1 / k:.1%}); median oracle-rank of pick: "
        f"{np.median(rank_of_pick):.0f} overall, {np.median(rank_of_pick[spike]):.0f} spikes"
    )


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    opts = cfg.get("ranker") or {}
    stage = str(opts.get("stage", "draws"))
    torch.set_float32_matmul_precision(cfg.get("matmul_precision", "high"))
    if stage == "draws":
        _stage_draws(cfg, opts)
    elif stage == "train":
        _stage_train(opts)
    elif stage == "eval":
        _stage_eval(cfg, opts)
    else:
        msg = f"unknown ranker stage: {stage}"
        raise ValueError(msg)


if __name__ == "__main__":
    mp.set_forkserver_preload(["rbyte", "polars"])
    main()
