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

3. **chunk DCT + per-coefficient standardization** (optional). Horizon slots are
   ~90%+ correlated, so in slot space the flow MSE gives within-chunk structure
   only its variance share (measured ~5-12% — the constant/mid-anchored chunk is
   the gradient-rational optimum; every architectural lever for slot identity
   fails because the incentive, not the information, is missing). Rotate the
   Gaussianized chunk with an orthonormal DCT-II over the horizon axis (slot
   index becomes frequency index: k=0 chunk mean, k=1 slope, ...) and
   standardize each (coefficient, channel) with data-fit mu/sigma — every
   coefficient is then unit-variance and within-chunk shape gets equal loss
   pressure to the level. A sigma-floor (fraction of the channel's sigma_0)
   keeps near-empty coefficients from being amplified to full weight. The
   inverse multiplies errors back DOWN by sigma_k, so raw-space samples are
   robust to bad high-coefficient predictions (the opposite of mu-law's
   amplifying inverse). Precedent: pi0-FAST tokenizes action chunks via
   normalize + time-axis DCT for the same correlation pathology
   (arXiv:2501.09747); trajectory-DCT is standard in human-motion prediction
   (arXiv:1908.05436).

Train in model space; invert samples back to raw before any reported metric
(flow-space loss is not comparable across transforms — raw-space metrics decide).
NB: with the DCT stage on, the model-space horizon axis is frequency, not time —
the within-chunk delta loss is meaningless there and must be disabled (the
objective guards this via `mixes_horizon`).
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
DCT_SIGMA_FLOOR_FRAC = 0.05


def dct_basis(horizon: int) -> Tensor:
    """Orthonormal DCT-II basis (horizon, horizon): basis @ basis.T == I.

    Row k is the k-th frequency; basis @ chunk projects a (horizon,) signal
    onto (mean, slope, curvature, ...) coefficients.
    """
    h = torch.arange(horizon, dtype=torch.float64)
    basis = ((2.0 / horizon) ** 0.5) * torch.cos(
        torch.pi * (h[None, :] + 0.5) * h[:, None] / horizon
    )
    basis[0] *= 0.5**0.5
    return basis.float()


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

    def __init__(  # noqa: PLR0913
        self,
        *,
        grid: Tensor,
        knots: Tensor,
        action_keys: tuple[str, ...],
        merge: bool = False,
        dct_mu: Tensor | None = None,
        dct_sigma: Tensor | None = None,
        dct_sigma_floor_frac: float = DCT_SIGMA_FLOOR_FRAC,
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
        if (dct_mu is None) != (dct_sigma is None):
            msg = "dct_mu and dct_sigma must be given together"
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
        # Chunk DCT + per-coefficient standardization (optional, step 3 above).
        self.mixes_horizon = dct_mu is not None
        self.action_horizon: int | None = None
        if dct_mu is not None and dct_sigma is not None:
            if dct_mu.shape != dct_sigma.shape or dct_mu.ndim != 2:  # noqa: PLR2004
                msg = (
                    "dct_mu/dct_sigma must both be (horizon, model_dim), got "
                    f"{tuple(dct_mu.shape)} / {tuple(dct_sigma.shape)}"
                )
                raise ValueError(msg)
            if dct_mu.shape[1] != self.model_dim:
                msg = f"dct stats have {dct_mu.shape[1]} channels, expected {self.model_dim}"
                raise ValueError(msg)
            if not 0.0 <= dct_sigma_floor_frac < 1.0:
                msg = f"dct_sigma_floor_frac must be in [0, 1), got {dct_sigma_floor_frac}"
                raise ValueError(msg)
            self.action_horizon = dct_mu.shape[0]
            # sigma-floor: never standardize a coefficient harder than
            # floor_frac of its channel's sigma_0 — near-empty coefficients
            # stay below unit variance instead of amplifying jitter to full
            # loss weight.
            sigma_eff = dct_sigma.float().clamp_min(
                dct_sigma_floor_frac * dct_sigma[0].float()
            )
            self.register_buffer("dct", dct_basis(self.action_horizon), persistent=False)
            self.register_buffer("dct_mu", dct_mu.float(), persistent=False)
            self.register_buffer("dct_sigma_eff", sigma_eff, persistent=False)

    @classmethod
    def from_stats_file(cls, path: str | Path) -> Self:
        stats = json.loads(Path(path).read_text(encoding="utf-8"))
        dct = stats.get("dct")
        return cls(
            grid=torch.tensor(stats["grid"]),
            knots=torch.tensor(stats["knots"]),
            action_keys=tuple(stats["action_keys"]),
            merge=bool(stats.get("merge", False)),
            dct_mu=torch.tensor(dct["mu"]) if dct else None,
            dct_sigma=torch.tensor(dct["sigma"]) if dct else None,
            dct_sigma_floor_frac=float(
                dct.get("sigma_floor_frac", DCT_SIGMA_FLOOR_FRAC)
            )
            if dct
            else DCT_SIGMA_FLOOR_FRAC,
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

    def _dct_standardize(self, z: Tensor) -> Tensor:
        # (..., H, C) time-domain Gaussianized -> standardized DCT coefficients.
        if z.shape[-2] != self.action_horizon:
            msg = f"chunk horizon {z.shape[-2]} != dct stats horizon {self.action_horizon}"
            raise ValueError(msg)
        coeff = torch.einsum("kh,...hc->...kc", self.dct, z)
        return (coeff - self.dct_mu) / self.dct_sigma_eff

    def _inverse_dct_standardize(self, z: Tensor) -> Tensor:
        # standardized DCT coefficients -> (..., H, C) time-domain Gaussianized.
        coeff = z * self.dct_sigma_eff + self.dct_mu
        return torch.einsum("kh,...kc->...hc", self.dct, coeff)

    def forward(self, raw: Tensor) -> Tensor:
        """Raw actions (..., H, raw_dim) -> model space (..., H, model_dim).

        With the DCT stage on, the model-space H axis is frequency (k-th
        standardized DCT coefficient), not time.
        """
        model = self._merge_raw(raw) if self.merge else raw
        z = self._gaussianize(model)
        return self._dct_standardize(z) if self.mixes_horizon else z

    def inverse(self, z: Tensor) -> Tensor:
        """Model space (..., H, model_dim) -> raw actions (..., H, raw_dim)."""
        if self.mixes_horizon:
            z = self._inverse_dct_standardize(z)
        model = self._inverse_gaussianize(z)
        return self._split_model(model) if self.merge else model
