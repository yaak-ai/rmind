"""Invertible action transform for the flow policy: gas/brake merge + Gaussianize.

Flow matching regresses a velocity whose target is matched to N(0, 1) noise, so
the action *targets* should be well-conditioned and ~Gaussian. Raw driving
actions are not: gas/brake/steering have very different, non-zero-mean marginals
and steering's maneuvers are extreme tails. Linear standardization was tried and
reverted (a sigma set by the near-zero bulk maps a 0.98 steering spike to ~5
sigma; see flow_action_expert_improvements.md, suspect 5).

Two composable steps, fit from data offline (flow_action_thresholds.py writes
the stats); this module only consumes them, adding no train-time data pass:

1. **gas/brake -> signed longitudinal merge** (optional). gas and brake are
   physically mutually exclusive and brake is a point mass at 0 — pathological
   for a continuous Gaussian flow. Merge to one signed channel
   `longitudinal = gas - brake` (steering passes through), so model space is 2-d
   `(longitudinal, steering)`. Inverse splits `gas = relu(long)`,
   `brake = relu(-long)` (exact when gas*brake == 0, ~always true).

2. **per-channel Gaussianize** (empirical CDF -> N(0, 1)) in model space, via
   stored quantile knots. Monotonic, invertible, non-clipping (unlike pi0's
   q01/q99 normalization, which would squash the steering maneuvers that are the
   signal here), making each marginal ~exactly Gaussian (matched to the prior).

Train in model space; invert samples back to raw before any reported metric
(flow-space loss is not comparable across transforms — raw-space metrics decide).
"""

import json
from pathlib import Path
from typing import Self

import torch
from torch import Tensor, nn

_EPS = 1e-6
_GAS, _BRAKE, _STEER = "gas_pedal", "brake_pedal", "steering_angle"
MERGE_RAW_KEYS = (_GAS, _BRAKE, _STEER)
MERGE_MODEL_KEYS = ("longitudinal", _STEER)


def _interp1d(query: Tensor, xp: Tensor, yp: Tensor) -> Tensor:
    """Piecewise-linear interpolation of (xp -> yp) at `query`, ends clamped.

    xp must be strictly increasing (1-D); yp matches its length; query is any
    shape. Queries outside [xp[0], xp[-1]] saturate at the end knots.
    """
    idx = torch.searchsorted(xp, query.contiguous(), right=True).clamp(1, xp.numel() - 1)
    x0, x1 = xp[idx - 1], xp[idx]
    y0, y1 = yp[idx - 1], yp[idx]
    t = (query - x0) / (x1 - x0).clamp_min(_EPS)
    return torch.clamp(y0 + t * (y1 - y0), min=yp[0], max=yp[-1])


class GaussianizeActionTransform(nn.Module):
    """gas/brake merge (optional) + per-channel empirical-CDF -> N(0,1).

    `grid` (K,) are the quantile levels in [0, 1]; `knots[c]` (model_dim, K) are
    the channel-c MODEL-space action values at those levels (strictly increasing
    per channel). With `merge=True`, model space is `(longitudinal, steering)`
    and raw space is `(gas, brake, steering)`; otherwise raw == model.
    """

    def __init__(
        self,
        *,
        grid: Tensor,
        knots: Tensor,
        action_keys: tuple[str, ...],
        merge: bool = False,
    ) -> None:
        super().__init__()
        model_keys = MERGE_MODEL_KEYS if merge else action_keys
        if merge and action_keys != MERGE_RAW_KEYS:
            msg = f"merge requires raw action_keys == {MERGE_RAW_KEYS}, got {action_keys}"
            raise ValueError(msg)
        if knots.shape[0] != len(model_keys):
            msg = f"knots has {knots.shape[0]} channels, expected {len(model_keys)} (merge={merge})"
            raise ValueError(msg)
        if knots.shape[1] != grid.shape[0]:
            msg = f"knots width {knots.shape[1]} != grid length {grid.shape[0]}"
            raise ValueError(msg)
        self.action_keys = action_keys  # raw I/O keys
        self.model_action_keys = model_keys
        self.merge = merge
        self.raw_dim = len(action_keys)
        self.model_dim = len(model_keys)
        # Non-persistent: fit from data, not learned; keeps old checkpoints
        # loadable and lets the stats be re-fit independently.
        self.register_buffer("grid", grid.float(), persistent=False)
        self.register_buffer("knots", knots.float(), persistent=False)

    @classmethod
    def from_stats_file(cls, path: str | Path) -> Self:
        stats = json.loads(Path(path).read_text())
        return cls(
            grid=torch.tensor(stats["grid"]),
            knots=torch.tensor(stats["knots"]),
            action_keys=tuple(stats["action_keys"]),
            merge=bool(stats.get("merge", False)),
        )

    def _merge_raw(self, raw: Tensor) -> Tensor:
        # (..., 3) gas/brake/steering -> (..., 2) longitudinal/steering.
        longitudinal = raw[..., 0] - raw[..., 1]
        return torch.stack([longitudinal, raw[..., 2]], dim=-1)

    def _split_model(self, model: Tensor) -> Tensor:
        # (..., 2) longitudinal/steering -> (..., 3) gas/brake/steering.
        longitudinal, steering = model[..., 0], model[..., 1]
        gas = longitudinal.clamp_min(0.0)
        brake = (-longitudinal).clamp_min(0.0)
        return torch.stack([gas, brake, steering], dim=-1)

    def _gaussianize(self, x: Tensor) -> Tensor:
        cols = [
            _interp1d(x[..., c], self.knots[c], self.grid).clamp(_EPS, 1.0 - _EPS)
            for c in range(self.model_dim)
        ]
        return torch.special.ndtri(torch.stack(cols, dim=-1))

    def _inverse_gaussianize(self, z: Tensor) -> Tensor:
        u = torch.special.ndtr(z)
        cols = [
            _interp1d(u[..., c], self.grid, self.knots[c]) for c in range(self.model_dim)
        ]
        return torch.stack(cols, dim=-1)

    def forward(self, raw: Tensor) -> Tensor:
        """Raw actions (..., raw_dim) -> Gaussianized model space (..., model_dim)."""
        model = self._merge_raw(raw) if self.merge else raw
        return self._gaussianize(model)

    def inverse(self, z: Tensor) -> Tensor:
        """Gaussianized model space (..., model_dim) -> raw actions (..., raw_dim)."""
        model = self._inverse_gaussianize(z)
        return self._split_model(model) if self.merge else model
