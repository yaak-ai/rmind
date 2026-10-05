"""Training/eval health metrics for the robot-native nero patch policy (P9).

All functions are gradient-free and shape-generic over `(n, H, A)` per-side
chunk rows (`n` = valid (batch, frame, side) rows, `H` = chunk steps, `A` = 13
axes: 7 arm joints then 6 fingers in `/1000` command units). They are ports of
what the team found it needed elsewhere:

* code confidence / entropy / margin / usage (#276, `PatchPolicy._confidence_metrics`)
  -- cross-entropy punishes confident errors, accuracy does not;
* per-axis explained variance per horizon bucket -- never an aggregate L1;
* grasp-event metrics (nutron_act): close/open onset timing per finger, the
  steady-state grasp level during holds, and the under-grip rate (the reason
  P4 trains on COMMANDED targets);
* alarms as 0/1 metrics so a dashboard (or a test) can assert on them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor

from rmind.data.nero_robot import ARM_AXES, FINGER_AXES, FINGER_NAMES

__all__ = [
    "HORIZON_BUCKETS",
    "TOKEN_NORM_BAND",
    "alarm_metrics",
    "code_confidence_metrics",
    "grasp_event_metrics",
    "grasp_window_mask",
    "horizon_ev_metrics",
    "per_axis_ev",
]

#: chunk-step buckets at 30 Hz: the first ~0.3 s (what n_next_actions executes),
#: up to 1 s, and the rest of the 3.3 s chunk
HORIZON_BUCKETS: dict[str, tuple[int, int]] = {
    "h00_09": (0, 10),
    "h10_29": (10, 30),
    "h30_99": (30, 100),
}
#: hand/patch and state/patch input-norm ratio band; outside it a vector token
#: is drowned (the speed-token 20x problem) or dominates
TOKEN_NORM_BAND = (0.3, 3.0)
#: 50% of the Revo2 command range (counts / 1000)
GRASP_THRESHOLD = 0.5
#: under-grip: predicted closed level below the commanded one by > 50 counts
UNDERGRIP_MARGIN = 0.05
DEAD_RATIO = 0.10
FPS = 30


def per_axis_ev(pred: Tensor, target: Tensor, weight: Tensor | None = None) -> Tensor:
    """EV = 1 - MSE/Var per axis over `(n, H, A)` with an optional `(n, H)` weight."""
    w = (
        torch.ones(target.shape[:-1], device=target.device)
        if weight is None
        else weight.to(target.device)
    )
    w = w.unsqueeze(-1).to(target.dtype)
    total = w.sum(dim=(0, 1)).clamp_min(1.0)
    mean = (target * w).sum(dim=(0, 1)) / total
    var = (((target - mean) ** 2) * w).sum(dim=(0, 1)) / total
    mse = (((pred - target) ** 2) * w).sum(dim=(0, 1)) / total
    return 1.0 - mse / var.clamp_min(1e-12)


def horizon_ev_metrics(
    pred: Tensor, target: Tensor, real: Tensor, *, prefix: str = "ev"
) -> dict[str, Tensor]:
    """Per-axis-group EV per horizon bucket, plus every axis over the full chunk."""
    out: dict[str, Tensor] = {}
    h = target.shape[1]
    for name, (lo, hi) in HORIZON_BUCKETS.items():
        hi = min(hi, h)  # noqa: PLW2901
        if lo >= hi:
            continue
        ev = per_axis_ev(pred[:, lo:hi], target[:, lo:hi], real[:, lo:hi])
        out[f"{prefix}/arm/{name}"] = ev[list(ARM_AXES)].mean()
        out[f"{prefix}/finger/{name}"] = ev[list(FINGER_AXES)].mean()
    ev = per_axis_ev(pred, target, real)
    for axis, value in enumerate(ev):
        out[f"{prefix}/axis_{axis:02d}"] = value
    out[f"{prefix}/arm"] = ev[list(ARM_AXES)].mean()
    out[f"{prefix}/finger"] = ev[list(FINGER_AXES)].mean()
    return out


def code_confidence_metrics(
    code_logits: Tensor, argmax_codes: Tensor
) -> dict[str, Tensor]:
    """`(n, g, c)` logits -> entropy/confidence/margin/usage per quantizer (#276)."""
    log_probs = code_logits.log_softmax(dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)  # (n, g)
    confidence = probs.amax(dim=-1)
    top2 = code_logits.topk(2, dim=-1).values
    margin = top2[..., 0] - top2[..., 1]
    g, c = code_logits.shape[-2], code_logits.shape[-1]
    flat = argmax_codes.reshape(-1, g).transpose(0, 1)  # (g, n)
    hit = torch.zeros(g, c, dtype=torch.bool, device=flat.device).scatter_(
        -1, flat, torch.ones_like(flat, dtype=torch.bool)
    )
    used = hit.sum(dim=-1).float()
    out: dict[str, Tensor] = {}
    for q in range(g):
        out[f"code_entropy_{q}"] = entropy[..., q].mean()
        out[f"code_confidence_{q}"] = confidence[..., q].mean()
        out[f"code_margin_{q}"] = margin[..., q].mean()
        out[f"code_usage_{q}"] = used[q]
    out["code_usage_min_frac"] = used.min() / c
    return out


def _first_crossing(x: Tensor, *, rising: bool) -> Tensor:
    """`(n, H)` -> first step index crossing `GRASP_THRESHOLD`, or -1."""
    above = x > GRASP_THRESHOLD
    cross = (above[:, 1:] & ~above[:, :-1]) if rising else (~above[:, 1:] & above[:, :-1])
    has = cross.any(dim=1)
    idx = cross.float().argmax(dim=1) + 1
    return torch.where(has, idx, torch.full_like(idx, -1))


def grasp_window_mask(target: Tensor, real: Tensor, *, half_width: int = 15) -> Tensor:
    """`(n, H)`: steps within +-`half_width` (0.5 s) of any finger crossing 50%."""
    fingers = target[..., list(FINGER_AXES)]  # (n, H, 6)
    above = fingers > GRASP_THRESHOLD
    flip = torch.zeros_like(above)
    flip[:, 1:] = above[:, 1:] ^ above[:, :-1]
    event = flip.any(dim=-1).float()  # (n, H)
    kernel = torch.ones(1, 1, 2 * half_width + 1, device=target.device)
    near = torch.nn.functional.conv1d(event[:, None], kernel, padding=half_width)[:, 0]
    return (near > 0) & real


def grasp_event_metrics(
    pred: Tensor, target: Tensor, real: Tensor, *, names: Sequence[str] = FINGER_NAMES
) -> dict[str, Tensor]:
    """Onset timing, steady-state grasp level and under-grip rate, per finger.

    `pred`/`target` are ABSOLUTE `(n, H, A)` chunks (fingers in counts/1000).
    Onset error: |first close (open) crossing of 50% in pred - in target| in ms,
    over rows where both cross; `miss` = the target crosses and pred does not.
    Steady state: real steps where the target is closed (> 50%) and flat; mean
    (pred - target) in COUNTS; under-grip = pred < target - 50 counts there.
    """
    out: dict[str, Tensor] = {}
    real = real.bool()
    for j, (axis, name) in enumerate(zip(FINGER_AXES, names, strict=True)):
        del j
        p, t = pred[..., axis], target[..., axis]
        p = torch.where(real, p, t)  # padded steps cannot create a crossing
        for kind, rising in (("close", True), ("open", False)):
            kp, kt = _first_crossing(p, rising=rising), _first_crossing(t, rising=rising)
            both = (kp >= 0) & (kt >= 0)
            target_has = kt >= 0
            if bool(both.any()):
                out[f"grasp/{kind}_onset_ms/{name}"] = (
                    (kp[both] - kt[both]).abs().float().mean() * 1000.0 / FPS
                )
            if bool(target_has.any()):
                out[f"grasp/{kind}_miss/{name}"] = (kp[target_has] < 0).float().mean()
        flat = torch.zeros_like(real)
        flat[:, 1:-1] = ((t[:, 2:] - t[:, :-2]).abs() < 0.005) & real[:, 1:-1]  # noqa: PLR2004
        hold = flat & (t > GRASP_THRESHOLD)
        if bool(hold.any()):
            err = (p - t)[hold]
            out[f"grasp/hold_error_counts/{name}"] = err.mean() * 1000.0
            out[f"grasp/undergrip_rate/{name}"] = (err < -UNDERGRIP_MARGIN).float().mean()
    return out


def alarm_metrics(
    *,
    code_usage: Mapping[str, Tensor] | None = None,
    codebook_size: int | None = None,
    pred: Tensor | None = None,
    target: Tensor | None = None,
    real: Tensor | None = None,
    token_norms: Mapping[str, Tensor] | None = None,
    features: Tensor | None = None,
) -> dict[str, Tensor]:
    """0/1 training-health alarms (P9). Everything optional; absent -> no key."""
    out: dict[str, Tensor] = {}
    if code_usage and codebook_size:
        used = torch.stack([
            v for k, v in code_usage.items() if k.startswith("code_usage_") and k[-1].isdigit()
        ])
        out["alarm/code_usage_below_half"] = (used.min() < 0.5 * codebook_size).float()
    if pred is not None and target is not None:
        w = (real if real is not None else torch.ones(target.shape[:-1], dtype=torch.bool))
        w = w.bool()
        sd_p = pred[w].std(dim=0)
        sd_t = target[w].std(dim=0)
        live = sd_t > 1e-6  # noqa: PLR2004
        ratio = torch.where(live, sd_p / sd_t.clamp_min(1e-12), torch.ones_like(sd_t))
        dead = ratio[list(FINGER_AXES)] < DEAD_RATIO
        out["alarm/dead_finger_channel"] = dead.any().float()
        out["dead_finger_channels"] = dead.sum().float()
    if token_norms:
        lo, hi = TOKEN_NORM_BAND
        patch = token_norms.get("patch")
        for name in ("state", "hand"):
            value = token_norms.get(name)
            if patch is not None and value is not None:
                ratio = value / patch.clamp_min(1e-12)
                out[f"token_ratio/{name}_patch"] = ratio
                out[f"alarm/token_ratio_{name}"] = ((ratio < lo) | (ratio > hi)).float()
    if features is not None:
        out["alarm/nonfinite_features"] = (~torch.isfinite(features)).any().float()
    return out
