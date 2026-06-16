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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
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
    wpts: list[torch.Tensor] = []
    torch.manual_seed(0)  # deterministic eval: same ckpt -> same numbers
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
            # current-frame route waypoints (the conditioning timestep is the
            # last history index)
            wpts.append(batch["data"]["waypoints/xy_normalized"][:, -1].float().cpu())
            if batch_idx % 50 == 0:
                logger.debug("sampled", batch_idx=batch_idx)

    gt = torch.cat(gts).numpy()
    sample = torch.cat(samples).numpy()
    drives = np.array(drive_ids)
    wp = torch.cat(wpts).numpy()  # (F, n_wpts, 2); may contain NaN (route gaps)
    valid = np.isfinite(gt).all(axis=(1, 2)) & np.isfinite(sample).all(axis=(1, 2, 3))
    if (dropped := int((~valid).sum())) > 0:
        logger.warning("dropping non-finite frames", count=dropped)
    return gt[valid], sample[valid], drives[valid], wp[valid], list(objective.action_keys)


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


def _collect_cached(
    cfg: DictConfig, *, cache_path: str, artifact: str, k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Cached-checkpoint path: condition tokens from the feature cache (the
    frozen encoder makes them checkpoint-independent), objective weights from a
    FlowFeatureTrainer wandb artifact. Returns the same tuple as _collect, with
    waypoints as NaN (not stored in the cache — the wpt-selector diagnostics
    degrade to their meanK fallback)."""
    import glob
    import os

    import wandb
    from hydra.utils import instantiate

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    cond = payload["cond"].float()  # (N, S, D)
    gt = payload["target_actions"].float()
    drives = np.array(payload["input_id"])

    trainer_module = instantiate(cfg.model)
    art = wandb.Api().artifact(artifact, type="model")
    ck = next(
        f for f in glob.glob(os.path.join(art.download(), "*")) if os.path.isfile(f)
    )
    sd = torch.load(ck, map_location="cpu", weights_only=False)["state_dict"]
    trainer_module.load_state_dict(sd)
    objective = trainer_module.objective.to(device).eval()
    if not isinstance(objective, flow_policy_module.FlowPolicyObjective):
        msg = f"cached artifact objective is not a FlowPolicyObjective: {type(objective)}"
        raise TypeError(msg)
    horizon = objective.decoder.action_horizon

    samples: list[torch.Tensor] = []
    chunk = 512
    torch.manual_seed(0)  # deterministic eval: same ckpt -> same numbers
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
            samples.append(
                objective._to_raw_space(
                    traj.float().reshape(c.shape[0], k, horizon, -1)
                ).cpu()
            )
            if (lo // chunk) % 10 == 0:
                logger.debug("sampled", frames=lo)

    sample = torch.cat(samples).numpy()
    gt = gt.numpy()
    wp = np.full((gt.shape[0], 10, 2), np.nan, dtype=np.float32)
    valid = np.isfinite(gt).all(axis=(1, 2)) & np.isfinite(sample).all(axis=(1, 2, 3))
    if (dropped := int((~valid).sum())) > 0:
        logger.warning("dropping non-finite frames", count=dropped)
    return (
        gt[valid],
        sample[valid],
        drives[valid],
        wp[valid],
        list(objective.action_keys),
    )


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    opts = cfg.get("meank") or {}
    k = int(opts.get("k", 32))
    spike_thr = float(opts.get("spike_threshold", 0.5))
    flat_thr = float(opts.get("flat_threshold", 0.05))
    mode_gap = float(opts.get("mode_gap", 0.15))
    mode_min_frac = float(opts.get("mode_min_frac", 0.125))

    torch.set_float32_matmul_precision(cfg.get("matmul_precision", "high"))
    if opts.get("cache"):
        gt, sample, drives, wp, keys = _collect_cached(
            cfg,
            cache_path=str(opts["cache"]),
            artifact=str(opts["cached_artifact"]),
            k=k,
        )
    else:
        gt, sample, drives, wp, keys = _collect(cfg, k=k)
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

    # Waypoint-consistency selection — MEASURED NEGATIVE on the 5 val drives,
    # kept as a documented diagnostic. The route-turn signal correlates with GT
    # chunk steering at only ~-0.29 on spike frames (~-0.13 overall; per-
    # waypoint lateral offsets ~0), far too weak to arbitrate between modes
    # ~0.5 apart: wpt-mode 0.457 vs mass-selection 0.341. The model itself uses
    # waypoints heavily (zeroing them shifts predictions ~3x baseline error),
    # so the ROUTE info is real but a hand-crafted scalar extraction is too
    # crude — selection needs a learned ranker or mass recalibration
    # (history-dropout training).
    seg = np.diff(wp, axis=1)  # (F, n-1, 2)
    cross = seg[:, :-1, 0] * seg[:, 1:, 1] - seg[:, :-1, 1] * seg[:, 1:, 0]
    dot = (seg[:, :-1] * seg[:, 1:]).sum(-1)
    ang = np.arctan2(cross, dot)  # (F, n-2) signed turn per joint
    route_near = ang[:, :3].sum(axis=1)  # near-term curvature (chunk-scale)
    route_full = ang.sum(axis=1)
    gt_ms = gt[..., steer].mean(axis=1)  # (F,) GT chunk-mean steering
    ok_r = np.isfinite(route_near) & np.isfinite(gt_ms)
    def corr(a, b, m):
        return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 10 else float("nan")
    c_near = corr(route_near, gt_ms, ok_r)
    c_full = corr(route_full, gt_ms, ok_r & np.isfinite(route_full))
    route = route_near if abs(c_near) >= abs(c_full) else route_full
    ok_r = np.isfinite(route) & np.isfinite(gt_ms)
    if int(ok_r.sum()) > 10:
        a_fit, b_fit = np.polyfit(route[ok_r], gt_ms[ok_r], 1)
    else:  # no waypoints (cached mode) -> selector falls back to meanK
        a_fit, b_fit = 0.0, float("nan")
    steer_hat = a_fit * route + b_fit  # (F,) route-implied chunk-mean steering
    print(  # noqa: T201
        f"\nroute-turn signal: corr(near3, gt_steer)={c_near:.3f} "
        f"corr(full, gt_steer)={c_full:.3f} | using "
        f"{'near3' if abs(c_near) >= abs(c_full) else 'full'}, fit a={a_fit:.3f} b={b_fit:.3f}"
    )

    ms_dom = dom_anchor[..., steer].mean(axis=1)
    ms_min = min_anchor[..., steer].mean(axis=1)
    pick_min = (
        bimodal
        & np.isfinite(steer_hat)
        & (np.abs(ms_min - steer_hat) < np.abs(ms_dom - steer_hat))
    )
    wpt_mode_anchor = dom_anchor.copy()
    wpt_mode_anchor[pick_min] = min_anchor[pick_min]
    fe_wpt_mode = frame_steer_l1(wpt_mode_anchor)
    # aggressive variant: pick the single draw matching the route signal, on
    # bimodal frames only (elsewhere selection noise just hurts; use meanK).
    draw_ms = sample[..., steer].mean(axis=2)  # (F, K)
    pick_draw = np.abs(draw_ms - steer_hat[:, None]).argmin(axis=1)
    fe_draw_pick = np.abs(
        np.take_along_axis(sample[..., steer], pick_draw[:, None, None], axis=1)[:, 0]
        - gt[..., steer]
    ).mean(axis=1)
    fe_wpt_draw = np.where(bimodal & np.isfinite(steer_hat), fe_draw_pick, fe_meank)

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
        ("mode-medoid", fe_medoid), ("wpt-mode", fe_wpt_mode),
        ("wpt-draw", fe_wpt_draw), ("oracle-mode", fe_oracle),
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
            f"{'single':>7} {'meanK':>7} {'mode':>7} {'wptM':>7} {'oracle':>7}"
        )
        for d in uniq:
            dm_ = drives == d
            msp = dm_ & spike_frame
            bi_pct = (bimodal & msp).sum() / max(msp.sum(), 1)
            def v(fe):
                return f"{fe[msp].mean():7.4f}" if msp.any() else "    n/a"
            print(  # noqa: T201
                f"  {d[:36]:36s} {int(dm_.sum()):6d} {int(msp.sum()):6d} "
                f"{bi_pct:7.2%} {v(fe_single)} {v(fe_meank)} {v(fe_dom)} {v(fe_wpt_mode)} {v(fe_oracle)}"
            )


if __name__ == "__main__":
    mp.set_forkserver_preload(["rbyte", "polars"])
    main()
