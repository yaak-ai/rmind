"""Invertible action transform for the flow policy: gas/brake merge + Gaussianize.

Flow matching regresses a velocity whose target is matched to N(0, 1) noise, so
the action *targets* should be well-conditioned and ~Gaussian. Raw driving
actions are not: gas/brake/steering have very different, non-zero-mean marginals
and steering's maneuvers are extreme tails. Plain linear standardization is a
poor fit here: a sigma set by the near-zero bulk maps a 0.98 steering spike to
~5 sigma, distorting exactly the maneuvers that matter.

The map is two independent, composable, invertible stages — kept as separate
modules because they change for different reasons (and one needs no file):

1. **`GasBrakeMerge`** — structural, fixed algebra, NO fitted params/file. gas and
   brake are physically mutually exclusive and brake is a point mass at 0 —
   pathological for a continuous Gaussian flow. Merge to one signed channel
   `longitudinal = gas - brake` (steering passes through), so model space is 2-d
   `(longitudinal, steering)`. Inverse splits `gas = relu(long)`,
   `brake = relu(-long)` (exact when gas*brake == 0, ~always true). Usable on its
   own (e.g. a "merge but no warp" ablation) straight from config.

2. **`Gaussianize`** — statistical, fit from data offline (flow_action_norm.py
   writes the knots; this module only consumes them). Per-channel empirical
   CDF -> N(0, 1) via stored quantile knots, dimension-preserving. Monotonic,
   invertible, non-clipping (unlike pi0's q01/q99 normalization, which would
   squash the steering maneuvers that are the signal here), making each marginal
   ~exactly Gaussian (matched to the prior).

`ActionTransform` composes them into the fixed pipeline the policy uses
(raw --merge--> physical model --gaussianize--> flow model), exposing
`physical_model` (the merge boundary, where the LDS labels are binned). The two
stages are coupled by the coordinate system — the Gaussianize knots are fit in
the merge's OUTPUT space — so the composite cross-checks their channel keys.

Train in model space; invert samples back to raw before any reported metric
(flow-space loss is not comparable across transforms — raw-space metrics decide).
"""

import json
from pathlib import Path
from typing import Self, override

import torch
from pydantic import ConfigDict, InstanceOf, validate_call
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
    idx = torch.searchsorted(xp, query.contiguous(), right=True).clamp(
        1, xp.numel() - 1
    )
    x0, x1 = xp[idx - 1], xp[idx]
    y0, y1 = yp[idx - 1], yp[idx]
    t = (query - x0) / (x1 - x0).clamp_min(_EPS)
    return torch.clamp(y0 + t * (y1 - y0), min=yp[0], max=yp[-1])


class GasBrakeMerge(nn.Module):
    """Structural, file-less, invertible gas/brake <-> signed-longitudinal merge.

    raw (gas, brake, steering) <-> model (longitudinal = gas - brake, steering).
    Fixed algebra, no fitted params or file; ONNX-clean (relu only). Reusable on
    its own — it defines the (longitudinal, steering) coordinate system that the
    Gaussianize knots, the LDS bins, and the decoder action_dim are all relative
    to, so it always runs FIRST when present.
    """

    raw_keys: tuple[str, ...] = MERGE_RAW_KEYS
    model_keys: tuple[str, ...] = MERGE_MODEL_KEYS

    @property
    def raw_dim(self) -> int:
        return len(self.raw_keys)

    @property
    def model_dim(self) -> int:
        return len(self.model_keys)

    @override
    def forward(self, raw: Tensor) -> Tensor:
        """(..., 3) gas/brake/steering -> (..., 2) longitudinal/steering."""
        longitudinal = raw[..., 0] - raw[..., 1]
        return torch.stack([longitudinal, raw[..., 2]], dim=-1)

    @staticmethod
    def inverse(model: Tensor) -> Tensor:
        """(..., 2) longitudinal/steering -> (..., 3) gas/brake/steering."""
        longitudinal, steering = model[..., 0], model[..., 1]
        gas = longitudinal.clamp_min(0.0)
        brake = (-longitudinal).clamp_min(0.0)
        return torch.stack([gas, brake, steering], dim=-1)


class Gaussianize(nn.Module):
    """File-backed, dimension-preserving per-channel empirical-CDF -> N(0,1).

    `grid` (K,) are the quantile levels in [0, 1]; `knots[c]` (model_dim, K) are
    the channel-c MODEL-space action values at those levels (strictly increasing
    per channel). `model_keys` name the channels in the space the knots were fit
    in (post-merge when a GasBrakeMerge precedes it).
    """

    grid: Tensor
    knots: Tensor

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self, *, grid: Tensor, knots: Tensor, model_keys: tuple[str, ...]
    ) -> None:
        super().__init__()
        if knots.shape[0] != len(model_keys):
            msg = f"knots has {knots.shape[0]} channels, expected {len(model_keys)}"
            raise ValueError(msg)
        if knots.shape[1] != grid.shape[0]:
            msg = f"knots width {knots.shape[1]} != grid length {grid.shape[0]}"
            raise ValueError(msg)
        # inverse() maps u in [0,1] to a knot index via u*(K-1), which is exact
        # only for a uniform grid linspace(0, 1, K). The fitter always writes
        # that, but guard it so a non-uniform grid can't silently produce a wrong
        # inverse (which would then ship to deployment unnoticed).
        expected_grid = torch.linspace(0.0, 1.0, grid.shape[0], dtype=grid.dtype)
        if not torch.allclose(grid.float(), expected_grid.float(), atol=1e-5):
            msg = "grid must be a uniform linspace(0, 1, K); the inverse assumes it"
            raise ValueError(msg)
        self.model_keys = tuple(model_keys)
        self.model_dim = len(model_keys)
        # Non-persistent: fit from data, not learned; keeps old checkpoints
        # loadable and lets the stats be re-fit independently.
        self.register_buffer("grid", grid.float(), persistent=False)
        self.register_buffer("knots", knots.float(), persistent=False)

    @classmethod
    def from_stats_file(cls, path: str | Path) -> Self:
        stats = json.loads(Path(path).read_text(encoding="utf-8"))
        # The file names the channel space its knots live in. Older files (fit
        # before the merge/Gaussianize split) stored raw `action_keys` + a `merge`
        # flag instead of `model_keys`; derive it for backward compatibility.
        model_keys = stats.get("model_keys")
        if model_keys is None:
            model_keys = (
                MERGE_MODEL_KEYS if stats.get("merge") else tuple(stats["action_keys"])
            )
        return cls(
            grid=torch.tensor(stats["grid"]),
            knots=torch.tensor(stats["knots"]),
            model_keys=tuple(model_keys),
        )

    def forward(self, model: Tensor) -> Tensor:
        """Model-space actions (..., model_dim) -> Gaussianized (..., model_dim)."""
        cols = [
            _interp1d(model[..., c], self.knots[c], self.grid).clamp(_EPS, 1.0 - _EPS)
            for c in range(self.model_dim)
        ]
        return torch.special.ndtri(torch.stack(cols, dim=-1))

    def inverse(self, z: Tensor) -> Tensor:
        """Gaussianized (..., model_dim) -> model-space actions (..., model_dim).

        Interpolates over the FIXED uniform grid linspace(0, 1, K), so the knot
        index is floor(u * (K - 1)) — no searchsorted. ndtr lowers to Erf, so this
        whole path is ONNX-clean, which the deployment/export graph relies on; it
        matches a searchsorted lookup to float precision. (forward() keeps
        searchsorted: it interpolates over the NON-uniform knots and is not on the
        deploy path.)
        """
        u = torch.special.ndtr(z)  # (..., model_dim) in [0, 1]
        k = self.knots.shape[1]
        position = (u * (k - 1)).clamp(0.0, k - 1.0)
        lo = position.floor().clamp(0.0, k - 2.0).long()
        frac = position - lo.to(position.dtype)
        cols = [
            self.knots[c][lo[..., c]]
            + frac[..., c] * (self.knots[c][lo[..., c] + 1] - self.knots[c][lo[..., c]])
            for c in range(self.model_dim)
        ]
        return torch.stack(cols, dim=-1)


class ActionTransform(nn.Module):
    """Fixed two-stage invertible map raw <-> flow (model) space for the policy.

    raw --[merge]--> physical model --[gaussianize]--> flow model. Either stage is
    optional (at least one required; full identity is `None` at the call site, see
    `build_action_transform`). `physical_model` exposes the merge boundary — the
    interpretable (longitudinal, steering) physical units the LDS labels are
    binned in. The exposed contract (`action_keys`, `model_action_keys`,
    `raw_dim`, `model_dim`) is what PolicyObjective._Wiring validates against the
    decoder and LDS weights.
    """

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        *,
        merge: InstanceOf[GasBrakeMerge] | None = None,
        gaussianize: InstanceOf[Gaussianize] | None = None,
    ) -> None:
        super().__init__()
        # The Gaussianize knots are fit in the merge's OUTPUT space, so when both
        # stages are present their channel keys must agree — else the warp is
        # being applied in the wrong coordinate system (e.g. a merged file with
        # merge off, or vice versa).
        if (
            merge is not None
            and gaussianize is not None
            and merge.model_keys != gaussianize.model_keys
        ):
            msg = (
                f"gaussianize channels {gaussianize.model_keys} != merge output "
                f"{merge.model_keys}: the knots must be fit in post-merge space"
            )
            raise ValueError(msg)
        self.merge = merge
        self.gaussianize = gaussianize

        # Resolve the exposed contract once (what PolicyObjective._Wiring checks).
        # action_keys is the raw I/O space (the merge input, or the gaussianize
        # channels when there's no merge); model_action_keys is the physical model
        # space (the merge boundary, where LDS labels are binned); model_dim is
        # the flow space the decoder sees (Gaussianize preserves the merge's dim).
        if merge is not None:
            self.action_keys = merge.raw_keys
            self.model_action_keys = merge.model_keys
            self.model_dim = (
                gaussianize.model_dim if gaussianize is not None else merge.model_dim
            )
        elif gaussianize is not None:
            self.action_keys = gaussianize.model_keys
            self.model_action_keys = gaussianize.model_keys
            self.model_dim = gaussianize.model_dim
        else:
            msg = "ActionTransform needs at least one stage (merge and/or gaussianize)"
            raise ValueError(msg)
        self.raw_dim = len(self.action_keys)

    def physical_model(self, raw: Tensor) -> Tensor:
        """Raw actions (..., raw_dim) -> physical model space (..., model_dim).

        The gas/brake merge WITHOUT the Gaussianize step: the interpretable
        (longitudinal, steering) values in physical units. Used for maneuver
        importance labels, which must be binned in physical (not Gaussianized
        z) space. Identity when there is no merge stage.
        """
        return self.merge(raw) if self.merge is not None else raw

    def forward(self, raw: Tensor) -> Tensor:
        """Raw actions (..., raw_dim) -> flow (model) space (..., model_dim)."""
        model = self.physical_model(raw)
        return self.gaussianize(model) if self.gaussianize is not None else model

    def inverse(self, z: Tensor) -> Tensor:
        """Flow (model) space (..., model_dim) -> raw actions (..., raw_dim)."""
        model = self.gaussianize.inverse(z) if self.gaussianize is not None else z
        return self.merge.inverse(model) if self.merge is not None else model


def build_action_transform(
    *, merge: bool = False, stats_path: str | None = None
) -> ActionTransform | None:
    """Hydra factory: compose the configured stages, or None (identity) when off.

    `merge` toggles the file-less GasBrakeMerge (a structural choice, so it lives
    in config — usable without any stats file); `stats_path`, when set, loads the
    Gaussianize warp from a fitted stats file. Returns None when neither is on, so
    the PolicyObjective is just handed a ready transform (or None) to validate.
    """
    merge_stage = GasBrakeMerge() if merge else None
    gaussianize_stage = Gaussianize.from_stats_file(stats_path) if stats_path else None
    if merge_stage is None and gaussianize_stage is None:
        return None
    return ActionTransform(merge=merge_stage, gaussianize=gaussianize_stage)
