"""Shared helpers for the Phase-0 foresight multimodality analysis suite.

Consumed by:
  - rmind.scripts.foresight_multimodality_analysis  (D0-D4)
  - rmind.scripts.foresight_leverage_l2              (G-1/L2 content probes)

Implements the cache-contract loader (adapted from rmind-rqv vjepa_probe.load_cache),
drive/order parsing for the anti-leakage k-NN rule, motion strata, and the
synthetic-cache generator used by the --synthetic self-tests.

NOTE: this module never touches the dataloader / GPU-heavy model code.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE_DIR = REPO_ROOT / "foresight_mm" / "cache" / "val"
DEFAULT_RESULTS_DIR = REPO_ROOT / "foresight_mm" / "results"
SYNTH_CACHE_DIR = REPO_ROOT / "foresight_mm" / "cache" / "synthetic_val"
SYNTH_RESULTS_DIR = REPO_ROOT / "foresight_mm" / "results" / "synthetic"

# Same thresholds as src/rmind/scripts/pedal_conflict_stats.py L39-40.
GAS_THRESH: float = 1.0 / 255 + 0.001
BRAKE_THRESH: float = 1.0 / 164 + 0.001

STRATUM_NAMES = ["stopped", "braking", "turning", "cruising"]

# Contract keys that must exist in every shard (input_id is a python list).
CONTRACT_TENSOR_KEYS = [
    "ctx_foresight_pooled",
    "ctx_action_summary",
    "ctx_obs_summary",
    "ctx_obs_history",
    "img_dino_pooled_cur",
    "fut_dino_pooled",
    "pred_fd_pooled_h1",
    "speed",
    "gas",
    "brake",
    "steer",
    "turn",
    "sample_id",
]


# --------------------------------------------------------------------------- #
# cache IO
# --------------------------------------------------------------------------- #
def load_cache(cache_dir: str | Path) -> dict:
    """Load and concat all shard_*.pt files (vjepa_probe.load_cache pattern)."""
    cache_dir = Path(cache_dir)
    shards = sorted(cache_dir.glob("shard_*.pt"))
    if not shards:
        msg = f"no shard_*.pt files under {cache_dir}"
        raise FileNotFoundError(msg)
    parts = [torch.load(s, map_location="cpu", weights_only=False) for s in shards]
    tensor_keys = [k for k in parts[0] if k != "input_id"]
    data = {k: torch.cat([p[k] for p in parts]) for k in tensor_keys}
    ids: list[str] = []
    for p in parts:
        ids.extend(p["input_id"])
    data["input_id"] = ids
    missing = [k for k in CONTRACT_TENSOR_KEYS if k not in data]
    if missing:
        msg = f"cache at {cache_dir} missing contract keys: {missing}"
        raise KeyError(msg)
    n = data["sample_id"].shape[0]
    if len(ids) != n:
        msg = f"input_id length {len(ids)} != n_samples {n}"
        raise ValueError(msg)
    meta_path = cache_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("n_samples") not in (None, n):
            print(
                f"[load_cache] WARNING meta.json n_samples={meta.get('n_samples')} "
                f"!= loaded {n}",
                flush=True,
            )
    print(f"[load_cache] {n} samples from {len(shards)} shards in {cache_dir}", flush=True)
    return data


def drive_codes(input_ids: list[str]) -> tuple[torch.Tensor, list[str]]:
    """Map input_id strings to integer drive codes (order of first appearance)."""
    uniq: dict[str, int] = {}
    codes = []
    for i in input_ids:
        if i not in uniq:
            uniq[i] = len(uniq)
        codes.append(uniq[i])
    return torch.tensor(codes, dtype=torch.long), list(uniq)


def drive_num(input_id: str) -> str:
    m = re.match(r"Niro(\d+)", input_id.split("/")[0])
    return m.group(1) if m else input_id


def sample_order(codes: torch.Tensor, frame_idx: torch.Tensor | None) -> tuple[torch.Tensor, str]:
    """Per-sample temporal order in *sample-order units* (1 unit = 1 dataloader
    sample = 10 raw frames at stride 10). Used by the anti-leakage gap rule.

    If frame_idx is cached, normalize by the per-drive median positive diff so a
    gap threshold of 90 always means "90 samples ~ 30 s" regardless of whether
    frame_idx stores raw frame numbers or timestamps. Else fall back to the
    per-drive dataloader order (cumcount).
    """
    n = codes.shape[0]
    order = torch.zeros(n, dtype=torch.float64)
    if frame_idx is not None:
        fi = frame_idx.to(torch.float64)
        for c in codes.unique():
            m = codes == c
            f = fi[m]
            diffs = (f[1:] - f[:-1]).abs()
            diffs = diffs[diffs > 0]
            stride = diffs.median().item() if diffs.numel() else 1.0
            order[m] = f / max(stride, 1e-9)
        return order, "frame_idx (normalized by per-drive median stride)"
    for c in codes.unique():
        m = codes == c
        order[m] = torch.arange(int(m.sum()), dtype=torch.float64)
    return order, "per-drive dataloader cumcount"


# Same values as foresight_leverage_l1.py (SPEED_STOPPED, STEER_TURN_QUANTILE).
SPEED_STOPPED: float = 0.5
STEER_TURN_QUANTILE: float = 0.80


def motion_strata(
    speed5: torch.Tensor,
    brake5: torch.Tensor,
    steer5: torch.Tensor,
    turn5: torch.Tensor | None = None,  # unused; kept for call-site compat
) -> tuple[torch.Tensor, dict]:
    """Motion stratum at the current frame (clip index 5).

    SAME definitions as foresight_leverage_l1.py::_strata_masks (L500-515):
      stopped : speed < SPEED_STOPPED (0.5; speed is meta/VehicleMotion/speed)
      braking : ~stopped & brake > BRAKE_THRESH
      turning : ~stopped & ~braking & |steer| > p80(|steer|)
      cruising: everything else
    """
    del turn5
    speed5 = speed5.float()
    brake5 = brake5.float()
    steer5 = steer5.float()
    steer_p80 = float(steer5.abs().quantile(STEER_TURN_QUANTILE))
    stopped = speed5 < SPEED_STOPPED
    braking = ~stopped & (brake5 > BRAKE_THRESH)
    turning = ~stopped & ~braking & (steer5.abs() > steer_p80)
    strata = torch.full_like(speed5, 3, dtype=torch.long)  # cruising
    strata[turning] = 2
    strata[braking] = 1
    strata[stopped] = 0
    info = {
        "definitions": {
            "stopped": f"speed[5] < {SPEED_STOPPED}",
            "braking": f"~stopped & brake[5] > {BRAKE_THRESH:.4f}",
            "turning": f"~stopped & ~braking & |steer[5]| > p80(|steer|)={steer_p80:.4f}",
            "cruising": "else",
            "note": "identical to foresight_leverage_l1.py::_strata_masks "
            "(SPEED_STOPPED=0.5, STEER_TURN_QUANTILE=0.80)",
        },
        "steer_p80_abs": steer_p80,
        "counts": {STRATUM_NAMES[i]: int((strata == i).sum()) for i in range(4)},
    }
    return strata, info


def standardize(x: torch.Tensor, mean: torch.Tensor | None = None, std: torch.Tensor | None = None):
    x = x.float()
    if mean is None:
        mean = x.mean(0)
    if std is None:
        std = x.std(0).clamp_min(1e-6)
    return (x - mean) / std, mean, std


def to_jsonable(obj):
    if isinstance(obj, torch.Tensor):
        obj = obj.detach().cpu()
        return obj.item() if obj.ndim == 0 else obj.tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


# --------------------------------------------------------------------------- #
# synthetic cache (self-test)
# --------------------------------------------------------------------------- #
def _smooth_walk(g: torch.Generator, n: int, d: int, width: int = 61) -> torch.Tensor:
    """Smooth latent trajectory: boxcar-smoothed white noise, (n, d)."""
    w = torch.randn(n + 2 * width, d, generator=g)
    k = torch.ones(1, 1, width) / width
    s = torch.nn.functional.conv1d(w.T.unsqueeze(1), k, padding=0).squeeze(1).T
    return s[:n]


def generate_synthetic_cache(
    out_dir: str | Path = SYNTH_CACHE_DIR,
    n: int = 4000,
    seed: int = 0,
    shard_size: int = 1500,
    n_grid: int = 128,
    force: bool = False,
) -> Path:
    """Write a fake cache obeying the contract, with planted structure:

    - contexts live on a 3-dim smooth latent manifold z (per-drive smooth walk,
      so temporal neighbors ARE context neighbors -> anti-leakage rule matters)
    - futures bimodal for the ~30% subset with z0 > q70 ("fork region"); mode
      gap = GAP0 * H grows with horizon; mode sign s = +-1 iid per sample
    - pred_fd_pooled_h1 = mode mean (the between-modes centroid)
    - speed frames 6..10 are linear in (z1, z2) -> L2 planted linear signal
    - img_dino_pooled_cur encodes the same latent as ctx_foresight_pooled
      (redundant feature -> ~zero incremental delta in L2)

    Ground truth saved to out_dir/synth_truth.pt.
    """
    out = Path(out_dir)
    if out.exists() and any(out.glob("shard_*.pt")) and not force:
        print(f"[synth] cache already exists at {out} (use force=True to regen)", flush=True)
        return out
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.pt"):
        old.unlink()

    g = torch.Generator().manual_seed(seed)
    d_lat, d_emb, n_pat = 3, 384, 256
    GAP0 = 2.0

    drives = [("Niro900-HQ/2026-01-01--00-00-00", 2200), ("Niro901-HQ/2026-01-02--00-00-00", n - 2200)]
    z_parts, ids, fidx = [], [], []
    for name, nd in drives:
        z_parts.append(_smooth_walk(g, nd, d_lat))
        ids.extend([name] * nd)
        fidx.append(torch.arange(nd, dtype=torch.long) * 10 + 1000)
    z = torch.cat(z_parts)
    z = (z - z.mean(0)) / z.std(0).clamp_min(1e-6)
    frame_idx = torch.cat(fidx)

    fork = z[:, 0] > z[:, 0].quantile(0.70)  # ~30% planted forks
    s = (torch.randint(0, 2, (n,), generator=g) * 2 - 1).float()

    def lin_map():
        return torch.randn(d_lat, d_emb, generator=g) / d_lat**0.5

    A_f, A_a, A_o, A_h, A_img, C = (lin_map() for _ in range(6))
    # keep the future's sensitivity to the latent small relative to the planted
    # mode gap, so neighborhood spread does not drown the fork structure
    C = C * 0.1
    mode_dir = torch.randn(d_emb, generator=g)
    Cq, _ = torch.linalg.qr(C.T)  # (384,3) orthonormal basis of rowspace
    mode_dir = mode_dir - Cq @ (Cq.T @ mode_dir)
    mode_dir = mode_dir / mode_dir.norm()

    def emb(A, noise=0.05):
        return z @ A + noise * torch.randn(n, d_emb, generator=g)

    ctx_fore, ctx_act, ctx_obs, ctx_hist, img_cur = (emb(A) for A in (A_f, A_a, A_o, A_h, A_img))
    base = z @ C
    fut = torch.zeros(n, 5, d_emb)
    for h in range(1, 6):
        sigma = 0.05 + 0.02 * h
        offset = fork.float() * s * (GAP0 * h / 2.0)
        fut[:, h - 1] = base + offset[:, None] * mode_dir + sigma * torch.randn(n, d_emb, generator=g)
    pred = base + 0.02 * torch.randn(n, d_emb, generator=g)

    # ---- signals (n, 11) -------------------------------------------------- #
    t_rel = torch.arange(11).float() - 5.0
    speed = 30.0 + 8.0 * z[:, 1:2] + 1.5 * t_rel[None, :] * z[:, 2:3]
    speed = (speed + 0.5 * torch.randn(n, 11, generator=g)).clamp(0.0, 130.0)
    stopped = z[:, 1] < z[:, 1].quantile(0.10)
    speed[stopped] = 0.0

    brake = torch.zeros(n, 11)
    braking_now = z[:, 2] > z[:, 2].quantile(0.85)
    brake[braking_now, :6] = 0.3
    p_on = torch.sigmoid(2.0 * z[:, 2])
    b_onset = (torch.rand(n, generator=g) < 0.4 * p_on) & ~braking_now
    starts = torch.randint(6, 11, (n,), generator=g)
    gas = torch.zeros(n, 11)
    gas_now = z[:, 1] > z[:, 1].quantile(0.40)
    gas[gas_now, :6] = 0.2
    g_onset = (torch.rand(n, generator=g) < 0.5 * torch.sigmoid(2.0 * z[:, 1])) & ~gas_now
    g_starts = torch.randint(6, 11, (n,), generator=g)
    for i in range(n):
        if b_onset[i]:
            brake[i, starts[i] :] = 0.25
        if g_onset[i]:
            gas[i, g_starts[i] :] = 0.2

    steer = 0.5 + 0.06 * torch.tanh(z[:, 0:1]) + 0.01 * torch.randn(n, 11, generator=g)
    turn = torch.zeros(n, 11)
    turn[torch.rand(n, generator=g) < 0.05] = 1.0

    # planted pedal conflicts, mildly enriched in fork region
    p_conf = torch.where(fork, torch.full((n,), 0.06), torch.full((n,), 0.02))
    conf = torch.rand(n, generator=g) < p_conf
    conf_frame = torch.randint(6, 11, (n,), generator=g)
    for i in torch.nonzero(conf).flatten().tolist():
        gas[i, conf_frame[i]] = max(gas[i, conf_frame[i]].item(), 0.05)
        brake[i, conf_frame[i]] = max(brake[i, conf_frame[i]].item(), 0.05)

    full = {
        "ctx_foresight_pooled": ctx_fore.half(),
        "ctx_action_summary": ctx_act.half(),
        "ctx_obs_summary": ctx_obs.half(),
        "ctx_obs_history": ctx_hist.half(),
        "img_dino_pooled_cur": img_cur.half(),
        "fut_dino_pooled": fut.half(),
        "pred_fd_pooled_h1": pred.half(),
        "speed": speed.half(),
        "gas": gas.half(),
        "brake": brake.half(),
        "steer": steer.half(),
        "turn": turn.half(),
        "sample_id": torch.arange(n, dtype=torch.int64),
        "frame_idx": frame_idx,
    }
    n_shards = 0
    for i0 in range(0, n, shard_size):
        i1 = min(i0 + shard_size, n)
        shard = {k: v[i0:i1].clone() for k, v in full.items()}
        shard["input_id"] = ids[i0:i1]
        torch.save(shard, out / f"shard_{n_shards:05d}.pt")
        n_shards += 1
    (out / "meta.json").write_text(
        json.dumps(
            {
                "n_samples": n,
                "n_shards": n_shards,
                "fd_loss_check": None,
                "notes": f"SYNTHETIC self-test cache, seed={seed}, gap0={GAP0}, "
                "fork region z0>q70 (~30%), pred=mode mean",
            },
            indent=2,
        )
        + "\n"
    )

    # grids file (first n_grid samples) — exercises the D2 per-patch path
    def grid(pooled):
        return (
            pooled[:n_grid, None, :].expand(n_grid, n_pat, d_emb)
            + 0.05 * torch.randn(n_grid, n_pat, d_emb, generator=g)
        ).half()

    torch.save(
        {
            "pred_fd_grid_h1": grid(pred),
            "gt_grid_cur": grid(base),
            "gt_grid_h1": grid(fut[:, 0]),
            "gt_grid_h5": grid(fut[:, 4]),
            "sample_id": full["sample_id"][:n_grid].clone(),
            "input_id": ids[:n_grid],
        },
        out / "val_grids.pt",
    )
    torch.save(
        {"fork": fork, "s": s, "z": z, "gap0": GAP0, "mode_dir": mode_dir, "seed": seed},
        out / "synth_truth.pt",
    )
    print(f"[synth] wrote {n} samples / {n_shards} shards + val_grids.pt -> {out}", flush=True)
    return out
