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

Also computes a MODE-AWARE readout and a multimodality census. Plain mean-of-K
is a mode-averaging estimator: on a genuinely bimodal conditional (turn left XOR
right) it splits the difference — the exact regression-to-the-mean failure flow
exists to avoid. The mode-aware readout clusters the K draws on a 1-D maneuver
signature (chunk-mean steering), commits to the DOMINANT cluster (draw count ~
probability mass), and averages within it — identical to mean-of-K on unimodal
frames, mode-committing on bimodal ones. This is sample-then-select as practiced
in motion forecasting (MultiPath anchors, MTR/DenseTNT NMS, Trajectron++ "most
likely" deployment; MDN take-dominant-component) and is ~Minimum-Bayes-Risk
consensus decoding (Kumar & Byrne 2004) with hard clustering. The census reports
how often the conditional is actually multimodal — with route waypoints in the
conditioning, discrete modes should be rare at current scale; this measures it.

Usage (mirrors `just fan`):
    uv run python -m rmind.scripts.flow_meank_eval \
        --config-path <repo>/config --config-name predict.yaml \
        inference=yaak/control_transformer/policy \
        model.artifact=yaak/action-flow/model-<run>:vN \
        datamodule=yaak/predict_val '+meank.k=32'

Options (+meank.*): k (draws, default 32), spike_threshold (|gt steering|,
default 0.5), flat_threshold (default 0.05), mode_gap (min separation in
chunk-mean steering to declare two modes, default 0.15), mode_min_frac
(minority cluster must hold at least this fraction of draws, default 0.125).
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


def _collect(
    cfg: DictConfig, *, k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
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
    drive_ids: list[str] = []
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
            ids = batch["meta"]["input_id"]
            drive_ids.extend(str(x) for x in (ids if isinstance(ids, list) else ids))
            if batch_idx % 50 == 0:
                logger.debug("sampled", batch_idx=batch_idx)

    gt = torch.cat(gts).numpy()
    sample = torch.cat(samples).numpy()
    drives = np.array(drive_ids)
    valid = np.isfinite(gt).all(axis=(1, 2)) & np.isfinite(sample).all(axis=(1, 2, 3))
    if (dropped := int((~valid).sum())) > 0:
        logger.warning("dropping non-finite frames", count=dropped)
    return gt[valid], sample[valid], drives[valid], list(objective.action_keys)


def _mode_aware_anchor(
    sample: np.ndarray, steer_idx: int, *, gap_thr: float, min_frac: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Numpy wrapper over the shared torch winner-take-all consensus
    (rmind.components.objectives.consensus — the same code path predict() uses
    with predict_readout=mode). Returns (anchor, bimodal, mode_sep)."""
    from rmind.components.objectives.consensus import mode_aware_anchor

    anchor, bimodal, mode_sep = mode_aware_anchor(
        torch.from_numpy(sample).float(),
        steer_idx,
        gap_thr=gap_thr,
        min_frac=min_frac,
    )
    return anchor.numpy(), bimodal.numpy(), mode_sep.numpy()


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    opts = cfg.get("meank") or {}
    k = int(opts.get("k", 32))
    spike_thr = float(opts.get("spike_threshold", 0.5))
    flat_thr = float(opts.get("flat_threshold", 0.05))
    mode_gap = float(opts.get("mode_gap", 0.15))
    mode_min_frac = float(opts.get("mode_min_frac", 0.125))

    torch.set_float32_matmul_precision(cfg.matmul_precision)
    gt, sample, drives, keys = _collect(cfg, k=k)
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

    anchor, bimodal, mode_sep = _mode_aware_anchor(
        sample, steer, gap_thr=mode_gap, min_frac=mode_min_frac
    )
    mode_err = np.abs(anchor - gt)  # (F, H, A)
    from rmind.components.objectives.consensus import (
        _members,
        mode_aware_anchor,
        split_modes,
    )

    s_t = torch.from_numpy(sample).float()
    medoid_anchor, _, _ = mode_aware_anchor(
        s_t, steer, gap_thr=mode_gap, min_frac=mode_min_frac, anchor="medoid"
    )
    medoid_err = np.abs(medoid_anchor.numpy() - gt)  # (F, H, A)

    # Residual decomposition on the steering channel, per FRAME (mode choice is
    # a per-frame decision): dominant- vs oracle-mode selection + coverage.
    bm_t, order_t, left_t, _sep = split_modes(
        s_t, steer, gap_thr=mode_gap, min_frac=mode_min_frac
    )
    kk_ = s_t.shape[1]
    dom_anchor = anchor.copy()
    min_anchor = anchor.copy()
    for i in torch.nonzero(bm_t).flatten().tolist():
        mm = _members(order_t, left_t, i, dominant=False, k=kk_)
        min_anchor[i] = s_t[i, mm].mean(dim=0).numpy()
    def frame_steer_l1(pred):  # (F, H, A) -> (F,) chunk-mean steering L1
        return np.abs(pred[..., steer] - gt[..., steer]).mean(axis=1)
    fe_dom = frame_steer_l1(dom_anchor)
    fe_min = frame_steer_l1(min_anchor)
    fe_oracle = np.where(bimodal, np.minimum(fe_dom, fe_min), fe_dom)
    fe_single = np.abs(sample[..., steer] - gt[:, None, :, steer]).mean(axis=2).mean(axis=1)
    fe_meank = frame_steer_l1(sample.mean(axis=1))
    fe_medoid = frame_steer_l1(medoid_anchor.numpy())
    fe_best = np.abs(sample[..., steer] - gt[:, None, :, steer]).mean(axis=2).min(axis=1)

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
        em = mode_err[..., c]
        spm = em[spike].mean() if spike.any() else float("nan")
        print(  # noqa: T201
            f"    mode {em.mean():11.4f} {em[flat].mean():9.4f} {spm:9.4f}"
            "   <- winner-take-all consensus (MBR-style)"
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

    # Multimodality census: how often is the conditional ACTUALLY multimodal?
    spike_frame = spike.any(axis=1)  # (F,) frame contains a spike slot
    flat_frame = flat.all(axis=1)
    print(  # noqa: T201
        f"\nmultimodality census (gap>{mode_gap:g} in chunk-mean steering, "
        f"minority>={mode_min_frac:.0%} of {k} draws):"
    )
    for name, m in (
        ("all frames", np.ones(f, dtype=bool)),
        ("flat frames", flat_frame),
        ("spike frames", spike_frame),
    ):
        if not m.any():
            continue
        bi = bimodal & m
        print(  # noqa: T201
            f"  {name:14s} bimodal {bi.sum():5d}/{int(m.sum()):5d} "
            f"({bi.mean() / max(m.mean(), 1e-12):6.2%})"
            + (
                f" | mode separation {mode_sep[bi].mean():.3f}"
                if bi.any()
                else ""
            )
        )
    if bimodal.any():
        es = mode_err[..., steer]
        em_k = agg[k][..., steer]
        print(  # noqa: T201
            f"  on bimodal frames, steering L1: mean-of-K "
            f"{em_k[bimodal].mean():.4f} vs mode-aware {es[bimodal].mean():.4f}"
            f" (single-draw {agg[1][..., steer][bimodal].mean():.4f})"
        )

    # Residual decomposition (steering, spike FRAMES): where does the
    # remaining error live? selection regret (dominant - oracle mode choice),
    # within-mode error (oracle), coverage floor (best single draw).
    print("\nresidual decomposition (steering chunk L1 on spike frames):")  # noqa: T201
    def dec_row(name, fe, m):
        print(f"  {name:16s} {fe[m].mean():.4f}" if m.any() else f"  {name:16s} n/a")  # noqa: T201
    m_sp = spike_frame
    for name, fe in (
        ("single", fe_single), ("mean-of-K", fe_meank), ("mode (WTA)", fe_dom),
        ("mode-medoid", fe_medoid), ("oracle-mode", fe_oracle),
        ("best-draw", fe_best),
    ):
        dec_row(name, fe, m_sp)
    if m_sp.any():
        sel = fe_dom[m_sp].mean() - fe_oracle[m_sp].mean()
        print(  # noqa: T201
            f"  -> selection regret (dominant-oracle): {sel:.4f} | "
            f"within-mode residual (oracle): {fe_oracle[m_sp].mean():.4f} | "
            f"coverage floor (best draw): {fe_best[m_sp].mean():.4f}"
        )

    # Per-drive breakdown (held-out evals span multiple drives).
    uniq = sorted(set(drives.tolist()))
    if len(uniq) > 1 or True:
        print("\nper-drive (steering, spike frames):")  # noqa: T201
        print(  # noqa: T201
            f"  {'drive':36s} {'frames':>6} {'spikeF':>6} {'bimod%':>7} "
            f"{'single':>7} {'meanK':>7} {'mode':>7} {'oracle':>7}"
        )
        for d in uniq:
            dm_ = drives == d
            msp = dm_ & spike_frame
            bi_pct = (bimodal & msp).sum() / max(msp.sum(), 1)
            def v(fe):
                return f"{fe[msp].mean():7.4f}" if msp.any() else "    n/a"
            print(  # noqa: T201
                f"  {d[:36]:36s} {int(dm_.sum()):6d} {int(msp.sum()):6d} "
                f"{bi_pct:7.2%} {v(fe_single)} {v(fe_meank)} {v(fe_dom)} {v(fe_oracle)}"
            )


if __name__ == "__main__":
    mp.set_forkserver_preload(["rbyte", "polars"])
    main()
