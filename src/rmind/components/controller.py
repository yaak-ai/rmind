"""A small trajectory -> action-sequence inverse-dynamics controller.

This is a **learned inverse-dynamics / trajectory-tracking controller**, not
full MPC. It shares MPC's interface -- take a plan over a prediction horizon,
emit a control sequence over a (shorter) control horizon, apply index 0, and
re-solve next tick against a replanned trajectory -- but not its machinery:
there is no forward dynamics model, no cost function, and no search over
candidate action sequences. It answers "what actions realize this
trajectory" by direct regression, in one forward pass. True MPC would add a
forward model (action-sequence -> resulting trajectory) and optimize
(CEM/gradient) over action sequences at each step, using this inverse model
at most as a fast initializer for that search.

Continuous action fields (gas/brake/steering) follow the same `(mean,
logvar)` convention as `rmind.components.objectives.policy.PolicyObjective`
(`x[..., 0]` = mean, `x[..., 1]` = logvar, scored with
`rmind.components.loss.GaussianNLLLoss`), for consistency with how the rest
of the repo represents continuous actions.
"""

from typing import Final, NamedTuple, override

import torch
from pydantic import validate_call
from torch import Tensor, nn
from torch.nn import Module

#: Order matches `policy/ground_truth/value/continuous/*` field naming in the
#: predict parquet this controller is trained from.
DEFAULT_CONTINUOUS_FIELDS: Final[tuple[str, ...]] = (
    "gas_pedal",
    "brake_pedal",
    "steering_angle",
)
NUM_TURN_SIGNAL_CLASSES: Final[int] = 3  # {OFF, LEFT, RIGHT}


def build_features(*, position: Tensor, heading: Tensor, speed: Tensor) -> Tensor:
    """Flatten a dead-reckoned future trajectory + current speed into the
    controller's input vector.

    Args:
        position: `(*batch, num_poses, 2)` ego-centric `(x, y)`, meters --
            the `position` output of
            `rmind.components.dead_reckoning.dead_reckon_future_trajectory`.
        heading: `(*batch, num_poses)` ego-centric heading, radians -- the
            `heading` output of the same function.
        speed: `(*batch,)` or `(*batch, 1)` current speed (same units used at
            training time; the parquet's `VehicleMotion/speed` is km/h).

    Returns:
        `(*batch, num_poses * 3 + 1)`.
    """
    flat_position = position.flatten(start_dim=-2)  # (*batch, num_poses*2)
    speed = speed if speed.shape[-1:] == (1,) else speed.unsqueeze(-1)
    return torch.cat([flat_position, heading, speed], dim=-1)


class TrajectoryToActionMLP(Module):
    """`build_features(...)` output -> the action sequence that realizes it.

    Input: a flattened future trajectory (position + heading) plus current
    speed, as produced by `build_features`. Output: a dict with one `(mean,
    logvar)` pair per entry in `continuous_fields`, and turn-signal class
    logits.

    `action_steps` is the receding-horizon part of the MPC analogy: the model
    emits the actions for the next `action_steps` ticks in one shot (an
    open-loop control sequence over the plan), of which a controller would
    apply only index 0 before re-solving against a freshly replanned
    trajectory. It is deliberately decoupled from the trajectory horizon
    encoded in `in_features` -- an MPC's prediction horizon is routinely
    longer than its control horizon, and here the asymmetry is extreme (a
    10s trajectory is worth conditioning on even when only the next action is
    applied; see the near-stop identifiability finding this was built for).

    Shapes are `(*batch, action_steps, 2)` per continuous field and
    `(*batch, action_steps, num_turn_signal_classes)` for the turn signal --
    except at `action_steps == 1`, where that axis is squeezed out so the
    outputs (and the head weight shapes, hence saved checkpoints) are
    identical to the single-action model this generalizes.
    """

    @validate_call
    def __init__(
        self,
        *,
        in_features: int,
        hidden_size: int = 128,
        continuous_fields: tuple[str, ...] = DEFAULT_CONTINUOUS_FIELDS,
        num_turn_signal_classes: int = NUM_TURN_SIGNAL_CLASSES,
        action_steps: int = 1,
    ) -> None:
        super().__init__()

        if action_steps < 1:
            msg = f"action_steps must be >= 1, got {action_steps}"
            raise ValueError(msg)

        self.continuous_fields: tuple[str, ...] = continuous_fields
        self.action_steps: int = action_steps
        self.num_turn_signal_classes: int = num_turn_signal_classes

        self.trunk = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.continuous_head = nn.Linear(
            hidden_size, len(continuous_fields) * action_steps * 2
        )
        self.turn_signal_head = nn.Linear(
            hidden_size, num_turn_signal_classes * action_steps
        )

    @override
    def forward(self, features: Tensor) -> dict[str, Tensor | dict[str, Tensor]]:
        hidden = self.trunk(features)

        continuous_out = self.continuous_head(hidden).unflatten(
            -1, (len(self.continuous_fields), self.action_steps, 2)
        )
        turn_signal = self.turn_signal_head(hidden).unflatten(
            -1, (self.action_steps, self.num_turn_signal_classes)
        )
        if self.action_steps == 1:
            # squeeze the step axis so single-action models keep the exact
            # shapes (and head weight layout) they had before this generalized.
            continuous_out = continuous_out.squeeze(-2)
            turn_signal = turn_signal.squeeze(-2)

        continuous = {
            field: continuous_out[..., i, :]
            if self.action_steps == 1
            else continuous_out[..., i, :, :]
            for i, field in enumerate(self.continuous_fields)
        }

        return {"continuous": continuous, "turn_signal": turn_signal}


class LongitudinalDynamicsParams(NamedTuple):
    """Linear longitudinal dynamics: `dv ~= gas_gain*gas - brake_gain*brake -
    drag_coeff*speed - offset`. One set of 4 params per vehicle (or per
    drive), fit by ordinary least squares -- an actual (if minimal) forward
    dynamics model, as opposed to `TrajectoryToActionMLP`'s pooled black-box
    regression or a post-hoc additive bias correction on top of it (both of
    which failed to generalize to held-out drives; see the investigation
    this was built to follow up on).

    `dv` and `speed` share whatever tick/unit convention they were fit with
    (this repo's convention throughout: km/h per one ~0.333s tick, i.e. NOT
    normalized by dt -- the params absorb dt implicitly, so don't reuse them
    at a different tick rate without refitting).
    """

    gas_gain: float
    brake_gain: float
    drag_coeff: float
    offset: float


def fit_longitudinal_dynamics(
    *, gas: Tensor, brake: Tensor, speed: Tensor, dv: Tensor
) -> LongitudinalDynamicsParams:
    """OLS fit of the linear model above from matched `(n,)` 1D samples
    (e.g. one vehicle's, or one drive's, rows). Needs at least 4 samples;
    dozens-to-hundreds recommended for a stable fit.
    """
    design = torch.stack([gas, -brake, -speed, -torch.ones_like(gas)], dim=-1)
    solution = torch.linalg.lstsq(design, dv.unsqueeze(-1)).solution.squeeze(-1)
    return LongitudinalDynamicsParams(*solution.tolist())


def invert_longitudinal_dynamics(
    *, target_dv: Tensor, speed: Tensor, params: LongitudinalDynamicsParams
) -> tuple[Tensor, Tensor]:
    """Analytic inverse: the `(gas, brake)` the fitted model predicts would
    realize `target_dv` at `speed` -- one pedal at a time, never both (gas
    when the model says positive net drive force is needed, brake
    otherwise), matching how the pedals are actually used. Both outputs
    clamped to `[0, 1]`.
    """
    needed = target_dv + params.drag_coeff * speed + params.offset  # a*gas - b*brake
    is_positive = needed >= 0
    gas = torch.where(
        is_positive,
        (needed / params.gas_gain).clamp(0.0, 1.0),
        torch.zeros_like(needed),
    )
    brake = torch.where(
        is_positive,
        torch.zeros_like(needed),
        (-needed / params.brake_gain).clamp(0.0, 1.0),
    )
    return gas, brake


class LaggedLongitudinalDynamicsParams(NamedTuple):
    """Generalizes `LongitudinalDynamicsParams` with `num_lags` recent ticks
    of gas/brake instead of just the current one, to capture throttle/brake
    response inertia: pressing the pedal doesn't produce an instantaneous
    speed change (engine spool-up, CVT/torque-converter response, hybrid
    ICE-engagement lag). `gas_gains[0]`/`brake_gains[0]` weight the CURRENT
    tick; `gas_gains[k]`/`brake_gains[k]` weight the tick `k` steps in the
    past. `num_lags=1` reduces exactly to `LongitudinalDynamicsParams`.
    """

    gas_gains: tuple[float, ...]
    brake_gains: tuple[float, ...]
    drag_coeff: float
    offset: float


def fit_lagged_longitudinal_dynamics(
    *, gas_lags: Tensor, brake_lags: Tensor, speed: Tensor, dv: Tensor
) -> LaggedLongitudinalDynamicsParams:
    """OLS fit. `gas_lags`/`brake_lags` are `(n, num_lags)`, MUST share the
    same `num_lags` (gas and brake are fit with the same lag depth; there's
    no support for asymmetric lag counts): column 0 is the current tick,
    column `k` is `k` ticks in the past -- same tick convention as
    `LongitudinalDynamicsParams`. `num_lags=1` (both `(n, 1)`) reduces to the
    plain model.

    Raises:
        ValueError: if `gas_lags` and `brake_lags` have different `num_lags`.
    """
    num_lags = gas_lags.shape[-1]
    if brake_lags.shape[-1] != num_lags:
        msg = (
            f"gas_lags and brake_lags must have the same num_lags, got "
            f"{num_lags} and {brake_lags.shape[-1]}"
        )
        raise ValueError(msg)
    design = torch.cat(
        [
            gas_lags,
            -brake_lags,
            -speed.unsqueeze(-1),
            -torch.ones_like(speed).unsqueeze(-1),
        ],
        dim=-1,
    )
    solution = torch.linalg.lstsq(design, dv.unsqueeze(-1)).solution.squeeze(-1)
    return LaggedLongitudinalDynamicsParams(
        gas_gains=tuple(solution[:num_lags].tolist()),
        brake_gains=tuple(solution[num_lags : 2 * num_lags].tolist()),
        drag_coeff=solution[-2].item(),
        offset=solution[-1].item(),
    )


def invert_lagged_longitudinal_dynamics(
    *,
    target_dv: Tensor,
    speed: Tensor,
    past_gas: Tensor,
    past_brake: Tensor,
    params: LaggedLongitudinalDynamicsParams,
) -> tuple[Tensor, Tensor]:
    """Analytic inverse for the CURRENT tick's `(gas, brake)`, netting out
    the known contribution of past pedal inputs first.

    `past_gas`/`past_brake` are `(n, num_lags - 1)`: the ALREADY-OBSERVED
    pedal values 1..num_lags-1 ticks before the tick being solved for --
    known at inference time in a real controller (you know what you did
    last tick). Pass `(n, 0)` tensors (or reuse
    `invert_longitudinal_dynamics`) when `num_lags == 1`.
    """
    num_lags = len(params.gas_gains)
    gas_gains = torch.as_tensor(params.gas_gains, dtype=target_dv.dtype)
    brake_gains = torch.as_tensor(params.brake_gains, dtype=target_dv.dtype)

    past_contribution = (
        (past_gas * gas_gains[1:]).sum(-1) - (past_brake * brake_gains[1:]).sum(-1)
        if num_lags > 1
        else torch.zeros_like(target_dv)
    )
    needed = target_dv - past_contribution + params.drag_coeff * speed + params.offset
    is_positive = needed >= 0
    gas = torch.where(
        is_positive, (needed / gas_gains[0]).clamp(0.0, 1.0), torch.zeros_like(needed)
    )
    brake = torch.where(
        is_positive,
        torch.zeros_like(needed),
        (-needed / brake_gains[0]).clamp(0.0, 1.0),
    )
    return gas, brake


class LateralDynamicsParams(NamedTuple):
    """Linear-in-parameters lateral (yaw-rate) dynamics, bicycle-model style:
    `dheading ~= steer_gain*(steering_angle*speed) + camber_bias`.

    The `steering_angle*speed` coupling (not `steering_angle` alone) is the
    physically-motivated part: real bicycle-model kinematics give yaw rate
    proportional to `speed * tan(steer)/wheelbase`, so a given steering input
    produces MORE heading change per tick at higher speed, and -- the
    testable, falsifiable consequence -- essentially none at a standstill (a
    parked car doesn't yaw from steering-wheel input alone). `steer_gain`
    absorbs wheelbase, dt, and whatever units `steering_angle` is normalized
    in; it is NOT a physical wheelbase and shouldn't be read as one.
    `camber_bias` is a speed-independent heading drift (road camber pull,
    steering-center miscalibration, sensor bias) -- the lateral analog of
    `LongitudinalDynamicsParams.offset`.

    Same unit convention as `LongitudinalDynamicsParams`: `dheading`/`speed`
    are whatever tick rate they were fit at (this repo: km/h and radians per
    one ~0.333s tick); the params absorb dt implicitly.
    """

    steer_gain: float
    camber_bias: float


def fit_lateral_dynamics(
    *, steering_angle: Tensor, speed: Tensor, dheading: Tensor
) -> LateralDynamicsParams:
    """OLS fit of the linear model above from matched `(n,)` 1D samples. As
    with `fit_longitudinal_dynamics`, needs at least a few dozen samples
    spanning a range of speeds for a stable fit -- `steering_angle*speed`
    only identifies `steer_gain` where speed actually varies.
    """
    design = torch.stack([steering_angle * speed, torch.ones_like(speed)], dim=-1)
    solution = torch.linalg.lstsq(design, dheading.unsqueeze(-1)).solution.squeeze(-1)
    return LateralDynamicsParams(*solution.tolist())


def invert_lateral_dynamics(
    *,
    target_dheading: Tensor,
    speed: Tensor,
    params: LateralDynamicsParams,
    min_speed: float = 1.0,
) -> Tensor:
    """Analytic inverse: the `steering_angle` the fitted model predicts would
    realize `target_dheading` at `speed`, clamped to `[-1, 1]`.

    Undefined/unstable as `speed -> 0` (the model itself says steering has no
    effect there, so `target_dheading` isn't achievable by steering alone at
    a standstill regardless of how the equation is solved) -- rows with
    `|speed| < min_speed` (same units as the fit) return `steering_angle=0`
    rather than dividing by a near-zero denominator.
    """
    denominator = params.steer_gain * speed
    raw = (target_dheading - params.camber_bias) / denominator
    steering_angle = torch.where(speed.abs() < min_speed, torch.zeros_like(raw), raw)
    return steering_angle.clamp(-1.0, 1.0)
