import operator
from collections.abc import Set as AbstractSet
from itertools import starmap
from typing import Any, Literal, cast, final, override

import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from pydantic import InstanceOf, validate_call
from tensordict import TensorDict
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F
from torch.utils._pytree import tree_map  # noqa: PLC2701

from rmind.components.base import Modality, SummaryToken, TensorTree
from rmind.components.containers import ModuleDict
from rmind.components.episode import Episode
from rmind.components.objectives.base import (
    Metrics,
    Objective,
    ObjectivePredictionKey,
    Prediction,
    Targets,
)
from rmind.utils.functional import (
    build_local_trajectory,
    build_relative_trajectory,
    gauss_prob,
    non_zero_signal_with_threshold,
)

# longitudinal mode classes: gas and brake share one softmax classifier so the
# press decision is mutually exclusive by construction (drivers never press both;
# independent heads let brake ride during launches)
MODE_COAST, MODE_GAS, MODE_BRAKE = 0, 1, 2
_LONGITUDINAL_MODES: dict[str, int] = {"gas_pedal": MODE_GAS, "brake_pedal": MODE_BRAKE}


@final
class PolicyObjective(Objective):
    @validate_call
    def __init__(
        self,
        *,
        norm: InstanceOf[Module] | None = None,
        heads: InstanceOf[ModuleDict],
        losses: InstanceOf[ModuleDict] | None = None,
        targets: Targets | None = None,
        history_steps: int = 1,
        action_horizon: int = 1,
        trajectory_head: InstanceOf[Module] | None = None,
        trajectory_loss: InstanceOf[Module] | None = None,
        trajectory_loss_weight: float = 1.0,
        # P1 motion rebalancing: scale each window's trajectory loss by a
        # band-pass triangular window on arc_len (distance the car travels over
        # the horizon), peaking at the low-speed/launch regime. This targets the
        # exact band the closed-loop stall lives in: stopped-stays-stopped
        # windows (arc_len ~ 0) AND the cruising majority (arc_len large) both
        # sit at baseline weight 1, while the launch band is up-weighted.
        #   signal = clamp(1 - |arc_len - motion_peak| / motion_width, min=0)
        #   weight = 1 + trajectory_motion_weight * signal   (mean-normalized)
        # motion_peak/width are in meters over the horizon (~2 s @ 6 steps):
        # a ~5-10 km/h launch that accelerates covers roughly 3-6 m, hence the
        # 4 m default peak. 0.0 weight == disabled (plain mean, identical to
        # before). Keeping stopped windows at baseline (not zero) preserves
        # stop-holding; keeping cruising at baseline stops it from dominating.
        trajectory_motion_weight: float = 0.0,
        trajectory_motion_peak: float = 4.0,
        trajectory_motion_width: float = 4.0,
        trajectory_target: Literal["absolute", "relative"] = "absolute",
        # winner-takes-all multi-modal trajectory: pair a K-mode trajectory_head
        # (MultiModalGRUTrajectoryHead) with trajectory_mode_head, a classifier
        # over head1's K candidates ("which one will be chosen"). Both None ==
        # today's single-mode behavior, unchanged. See _compute_logits.
        trajectory_mode_head: InstanceOf[Module] | None = None,
        trajectory_mode_loss: InstanceOf[Module] | None = None,
        trajectory_mode_loss_weight: float = 1.0,
        xy_key: tuple[str, ...] = ("input", "trajectory", "xy"),
        heading_key: tuple[str, ...] = ("input", "trajectory", "heading"),
        feature_keys: list[tuple[str, ...]] = [  # noqa: B006
            (Modality.SUMMARY, SummaryToken.OBSERVATION_SUMMARY),
            (Modality.SUMMARY, SummaryToken.OBSERVATION_HISTORY),
            (Modality.CONTEXT, "waypoints"),
        ],
        feature_pool: InstanceOf[ModuleDict] | None = None,
        # multi-tick variant of feature_keys/feature_pool: instead of reading
        # ONE tick's post-encoder embedding, attends (via a learned-query
        # cross-attention pool, e.g. AttentionPoolHead) over the
        # `history_steps` ticks ending at (and including) the same tick
        # feature_keys reads. Opt-in: None/empty by default == identical
        # behavior to today. A missing history_feature_pool entry for a given
        # key falls back to plain mean-pooling (_pool_feature's existing
        # behavior) -- but note that silently mean-pooling across a
        # *multi-timestep* window collapses temporal order entirely, which
        # defeats the point of this feature; always pair a
        # history_feature_keys entry with a matching history_feature_pool
        # entry in practice.
        history_feature_keys: list[tuple[str, ...]] | None = None,
        history_feature_pool: InstanceOf[ModuleDict] | None = None,
        # raw (pre-encoder, pre-fusion) features read straight off episode_builder's
        # frozen embeddings — e.g. DINOv3 patch tokens before the (also frozen)
        # cross-modal encoder gets to attend/pool over them. Observation audit
        # (2026-07-09/10) found legible, static cues (e.g. a green light unchanged
        # across the whole history window) that the gate still misses — this lets
        # heads read the un-fused patches directly, in case the frozen encoder's
        # pooling is what's discarding the cue, not DINOv3 itself.
        raw_feature_keys: list[tuple[str, ...]] | None = None,
        raw_feature_pool: InstanceOf[ModuleDict] | None = None,
        # trainable side-copy vision encoder: an independently-initialized backbone
        # (e.g. a second TimmBackbone) run on raw pixels from episode.input, kept
        # OUTSIDE episode_builder/encoder so ModuleFreezer's frozen paths never touch
        # it — trains purely off this objective's losses without risking the shared,
        # frozen DINOv3+fusion-encoder path every other head still depends on.
        trainable_image_key: tuple[str, ...] = (Modality.IMAGE, "cam_front_left"),
        trainable_image_encoder: InstanceOf[Module] | None = None,
        trainable_image_pool: InstanceOf[Module] | None = None,
        # raw (un-embedded) waypoints, read straight off episode.input instead of
        # through an embeddings/projections Linear -- for checkpoints whose
        # episode_builder has no ("context", "waypoints") token at all (e.g.
        # alex/pt-no-speed-experiments) there's nothing to pool via feature_keys,
        # but the raw route xy is still useful signal. Flattened (num_points,
        # xy) -> (num_points * 2) and concatenated onto `features` alongside the
        # feature_keys parts, so it reaches heads/longitudinal_mode_head/
        # trajectory_head exactly like a feature_keys entry would (adjust their
        # in_features by +num_points*2). None (default) == disabled.
        raw_waypoints_key: tuple[str, ...] | None = None,
        # modality dropout, not elementwise: zeroes the *entire* flattened
        # waypoints vector for a fraction of samples each training step, so the
        # heads can't just always copy the route instead of learning from
        # observation/history -- same rationale as forward_dynamics'
        # action_dependency_dropout, applied here instead since that dropout
        # lives on a different objective. No-op at eval (self.training is False).
        raw_waypoints_dropout: float = 0.0,
        # truncates raw_waypoints_key to just the nearest `raw_waypoints_horizon`
        # points (of the num_points axis) before flattening, dropping the rest --
        # e.g. 5 of 10 halves the raw waypoints contribution to `features` from
        # num_points*2 to raw_waypoints_horizon*2 dims. None (default) keeps all
        # num_points, unchanged from before this option existed.
        raw_waypoints_horizon: int | None = None,
        # raw (un-embedded) speed, same idea as raw_waypoints_key -- for
        # checkpoints with no ("continuous", "speed") token at all, the raw
        # km/h value is read straight off episode.input, normalized by
        # speed_scale (same as speed_head/condition_steering_on_speed), and
        # concatenated onto `features` (+1 to every consumer's in_features).
        # None (default) == disabled.
        raw_speed_key: tuple[str, ...] | None = None,
        raw_speed_dropout: float = 0.0,
        prediction_std_scale: dict[str, float] | None = None,
        gate_horizon_aggregate: Literal["first", "max"] = "first",
        gate_fire_threshold: float = 0.5,
        longitudinal_mode_head: InstanceOf[Module] | None = None,
        longitudinal_mode_loss: InstanceOf[Module] | None = None,
        longitudinal_mode_loss_weight: float = 1.0,
        longitudinal_press_threshold: float = 0.01,
        speed_head: InstanceOf[Module] | None = None,
        speed_loss: InstanceOf[Module] | None = None,
        speed_loss_weight: float = 0.1,
        speed_key: tuple[str, ...] = ("input", "continuous", "speed"),
        speed_scale: float = 50.0,
        speed_delta_scale: float = 10.0,
        pedal_heads: InstanceOf[ModuleDict] | None = None,
        condition_steering_on_speed: bool = False,
        detach_h_traj: bool = False,
        # steering-angle GT correction paired with HistoryViewpointPerturbation
        # (rmind.components.augment): that augmentation's implicit trajectory
        # correction (via the perturbed heading feeding build_relative_trajectory)
        # only reaches the trajectory head. Left alone, the steering_angle action
        # head is trained on the real, unperturbed recorded continuation for the
        # same sample -- i.e. "don't correct" -- directly contradicting the
        # trajectory head. dyaw_key=None (default) disables this entirely.
        dyaw_key: tuple[str, ...] | None = None,
        # normalized-steering-units * km/h per degree of dyaw; sign convention:
        # dyaw follows `heading` (positive = compass/rightward), so a positive
        # dyaw subtracts from steering_angle (steer left to recover) -- verified
        # empirically (real-drive steering_angle/heading-rate correlation is
        # positive, i.e. positive steering_angle == rightward turn). Divides by
        # current speed (bicycle-model yaw rate ~ v * steer): empirically the
        # degrees-of-heading-change-per-unit-steering slope scaled ~14x between
        # ~15 km/h and ~120 km/h on real fleet data, so a speed-independent
        # constant would be far too weak on the highway and far too strong at
        # parking-lot speed. 1.5 is a rough starting point from that same fit
        # (gain * speed_kmh landed around 1.2-2.0 across the 30-140 km/h range)
        # -- treat as a tunable, not a derived constant.
        steering_correction_gain: float = 0.0,
        # future ticks over which the correction linearly decays to 0 (full
        # strength at the first predicted step, since that's the step
        # build_relative_trajectory's chain assigns the "undo the offset" delta
        # to; zero by the time the model should be back on the recorded path)
        steering_correction_decay_steps: int = 2,
        # avoids a division blowup near-stationary, where the real-data fit was
        # noisy/inconsistent anyway (see steering_correction_gain)
        steering_correction_speed_floor: float = 5.0,
    ) -> None:
        super().__init__()

        self.norm: Module | None = norm
        self.heads: ModuleDict = heads
        self.losses: ModuleDict | None = losses
        self.targets: Targets | None = targets
        self.history_steps: int = history_steps
        self.action_horizon: int = action_horizon
        self.trajectory_head: Module | None = trajectory_head
        self.trajectory_loss: Module | None = trajectory_loss
        self.trajectory_loss_weight: float = trajectory_loss_weight
        self.trajectory_motion_weight: float = trajectory_motion_weight
        self.trajectory_motion_peak: float = trajectory_motion_peak
        self.trajectory_motion_width: float = trajectory_motion_width
        # "absolute": every horizon step vs one fixed frame anchored at
        # [history_steps - 1] (build_local_trajectory) — foreshortens forward
        # progress during a turn since it's measured against a stale heading.
        # "relative": each step vs the previous step's own pose
        # (build_relative_trajectory) — requires trajectory_head(predict_yaw=True).
        self.trajectory_target: Literal["absolute", "relative"] = trajectory_target
        # head2: classifies which of head1's K candidate trajectories wins.
        # Trained against the GT-closest candidate (see _trajectory_nll_per_mode);
        # at inference (no GT), its own argmax picks the winner instead.
        self.trajectory_mode_head: Module | None = trajectory_mode_head
        self.trajectory_mode_loss: Module | None = trajectory_mode_loss
        self.trajectory_mode_loss_weight: float = trajectory_mode_loss_weight
        self.xy_key: tuple[str, ...] = xy_key
        self.heading_key: tuple[str, ...] = heading_key
        self.feature_keys: list[tuple[str, ...]] = feature_keys
        self.feature_pool: ModuleDict | None = feature_pool
        self.history_feature_keys: list[tuple[str, ...]] = history_feature_keys or []
        self.history_feature_pool: ModuleDict | None = history_feature_pool
        self.raw_feature_keys: list[tuple[str, ...]] = raw_feature_keys or []
        self.raw_feature_pool: ModuleDict | None = raw_feature_pool
        self.trainable_image_key: tuple[str, ...] = trainable_image_key
        self.trainable_image_encoder: Module | None = trainable_image_encoder
        self.trainable_image_pool: Module | None = trainable_image_pool
        self.raw_waypoints_key: tuple[str, ...] | None = raw_waypoints_key
        self.raw_waypoints_dropout: float = raw_waypoints_dropout
        self.raw_waypoints_horizon: int | None = raw_waypoints_horizon
        self.raw_speed_key: tuple[str, ...] | None = raw_speed_key
        self.raw_speed_dropout: float = raw_speed_dropout
        self.prediction_std_scale: dict[str, float] = prediction_std_scale or {}
        # hurdle-gate decode: per-tick onset classification is weak (AUC ~0.72) and
        # fires in 1-tick bursts; "press within the predicted horizon" is easier and
        # holding the gate over the approach gives temporal persistence for free
        self.gate_horizon_aggregate: Literal["first", "max"] = gate_horizon_aggregate
        self.gate_fire_threshold: float = gate_fire_threshold
        # coupled longitudinal decode: a 3-class coast/gas/brake classifier makes
        # the press decision (CE, trained on all samples); the per-pedal Gaussian
        # heads model magnitude only (pair them with MaskedGaussianNLLLoss)
        self.longitudinal_mode_head: Module | None = longitudinal_mode_head
        self.longitudinal_mode_loss: Module | None = longitudinal_mode_loss
        self.longitudinal_mode_loss_weight: float = longitudinal_mode_loss_weight
        self.longitudinal_press_threshold: float = longitudinal_press_threshold
        # desired-speed inverse-dynamics decode: speed_head forecasts multi-step
        # Δspeed (diff vs the conditioning tick, not absolute/relative — see
        # gas_stall_report.md §8) from the same pooled features as the other
        # heads; pedal_heads then map {Δspeed_pred (detached), current_speed} ->
        # gas/brake, replacing the generic full-context `heads` entry for those
        # two keys. Δspeed is a smooth, non-bimodal target (unlike raw gas/brake),
        # aimed at sidestepping the AUC ceiling from the bimodal press decision.
        self.speed_head: Module | None = speed_head
        self.speed_loss: Module | None = speed_loss
        self.speed_loss_weight: float = speed_loss_weight
        self.speed_key: tuple[str, ...] = speed_key
        # raw km/h values (~0-130 speed, ~±30 delta) are wildly out of scale next
        # to the rest of the network's ~unit-scale, LayerNorm'd activations; a
        # freshly-initialized head fed them directly explodes (loss ~1e7, seen
        # in the first speed_delta training run, 2026-07-09). Divide down to a
        # roughly unit-scale range before they touch any learned weights.
        self.speed_scale: float = speed_scale
        self.speed_delta_scale: float = speed_delta_scale
        self.pedal_heads: ModuleDict | None = pedal_heads
        # current_speed may not survive well into the pooled observation_summary
        # (it's one scalar among many tokens); steering-angle-to-curvature
        # mapping is speed-dependent (slip/dynamics), so give it explicitly.
        self.condition_steering_on_speed: bool = condition_steering_on_speed
        # heads/longitudinal_mode_head are conditioned on h_traj (see forward);
        # detaching stops their losses from also shaping the trajectory_head,
        # isolating it to trajectory_loss only (same idea as the pedal_heads
        # detach above, applied to the trajectory coupling instead of speed).
        self.detach_h_traj: bool = detach_h_traj
        self.dyaw_key: tuple[str, ...] | None = dyaw_key
        self.steering_correction_gain: float = steering_correction_gain
        self.steering_correction_decay_steps: int = steering_correction_decay_steps
        self.steering_correction_speed_floor: float = steering_correction_speed_floor

    def _apply_steering_correction(
        self, targets: TensorTree, *, episode: Episode
    ) -> TensorTree:
        """Nudges the steering-angle GT to agree with the trajectory head's
        implicit "recover from the injected heading offset" target instead of
        contradicting it with the real, unperturbed recorded continuation --
        see `dyaw_key`/`steering_correction_gain` and
        `rmind.components.augment.HistoryViewpointPerturbation`.
        """
        if self.dyaw_key is None or self.steering_correction_gain == 0.0:
            return targets

        dyaw = episode.get(self.dyaw_key, default=None)
        speed = episode.get(self.speed_key, default=None)
        if dyaw is None or speed is None:
            return targets

        steering = targets[Modality.CONTINUOUS]["steering_angle"]
        single_step = steering.dim() == 2  # (b, 1): the last-tick-only path
        if single_step:
            steering = steering.unsqueeze(1)  # (b, 1, 1)
        h = steering.shape[1]

        n = min(self.steering_correction_decay_steps, h)
        decay = torch.zeros(h, device=steering.device, dtype=steering.dtype)
        if n > 0:
            decay[:n] = torch.linspace(
                1.0, 0.0, n + 1, device=steering.device, dtype=steering.dtype
            )[:-1]

        current_speed = (
            speed[:, self.history_steps - 1].reshape(-1, 1).to(steering.dtype)
        )
        correction = (
            -self.steering_correction_gain
            * dyaw[:, 0].to(steering.dtype).reshape(-1, 1)
            / current_speed.clamp(min=self.steering_correction_speed_floor)
            * decay.unsqueeze(0)
        ).unsqueeze(-1)  # (b, h, 1)

        corrected = (steering + correction).clamp(-1.0, 1.0)
        if single_step:
            corrected = corrected.squeeze(1)

        targets = dict(targets)
        targets[Modality.CONTINUOUS] = dict(targets[Modality.CONTINUOUS])
        targets[Modality.CONTINUOUS]["steering_angle"] = corrected
        return targets

    @override
    def forward(self, episode: Episode, embedding: Tensor) -> TensorDict:
        if self.norm is not None:
            embedding = self.norm(embedding)

        logits, _, _, mode_logits, _, _, _, _ = self._compute_logits(
            episode=episode, embedding=embedding, keep_horizon=True
        )
        mode = (
            self._longitudinal_mode(mode_logits)[0] if mode_logits is not None else None
        )

        def fn(nk: tuple[str, ...], x: Tensor) -> Tensor:
            # x is (b, h, d); emit the first predicted step -> (b, 1)
            first = x[:, :1, :]
            match nk:
                case (Modality.CONTINUOUS, name):
                    mean = first[..., 0]
                    # optimistic decoding: shift zero-inflated controls (e.g. gas)
                    # off the between-modes mean by k standard deviations
                    if (k := self.prediction_std_scale.get(name)) is not None:
                        mean += k * torch.sqrt(torch.exp(first[..., 1]))
                    # coupled longitudinal decode: the shared mode classifier
                    # decides which pedal (if any) emits its magnitude
                    if (
                        mode is not None
                        and (m := _LONGITUDINAL_MODES.get(name)) is not None
                    ):
                        return torch.where(mode == m, mean, torch.zeros_like(mean))
                    # hurdle head ([mean, log_var, gate], see HurdleGaussianNLLLoss):
                    # emit the press magnitude only when the gate fires
                    if first.shape[-1] == 3:  # noqa: PLR2004
                        gate = self._gate_prob(x) > self.gate_fire_threshold
                        return torch.where(gate, mean, torch.zeros_like(mean))
                    return mean
                case (Modality.DISCRETE, "turn_signal"):
                    return non_zero_signal_with_threshold(first).class_idx
                case _:
                    raise NotImplementedError

        return TensorDict(logits).named_apply(fn, nested_keys=True)  # ty:ignore[invalid-return-type, invalid-argument-type]

    def _gate_prob(self, x: Tensor) -> Tensor:
        """Hurdle gate probability aggregated over predicted horizon steps.

        x: (b, h, 3) hurdle logits. Returns (b, 1). "max" fires if a press is
        predicted anywhere in the horizon, not just at the immediate next tick.
        """
        probs = torch.sigmoid(x[..., 2])  # (b, h)
        if self.gate_horizon_aggregate == "max":
            return probs.max(dim=1, keepdim=True).values
        return probs[:, :1]

    def _longitudinal_mode(self, mode_logits: Tensor) -> tuple[Tensor, Tensor]:
        """Decode coast/gas/brake from (b, h, 3) mode logits → ((b, 1), (b, 1, 3)).

        With gate_horizon_aggregate="max" each class takes its max probability
        over the horizon before the argmax, so an approaching press outweighs
        transient per-tick coast confidence (same rationale as _gate_prob).
        """
        probs = torch.softmax(mode_logits, dim=-1)  # (b, h, 3)
        if self.gate_horizon_aggregate == "max":
            probs = probs.max(dim=1, keepdim=True).values
        else:
            probs = probs[:, :1]
        return probs.argmax(dim=-1), probs

    def _pool_feature(
        self, key: tuple[str, ...], x: Tensor, *, pool_dict: ModuleDict | None
    ) -> Tensor:
        if pool_dict is not None and (pool := pool_dict.get(key, default=None)) is not None:
            return pool(x)
        return x.mean(dim=1, keepdim=True)

    def _raw_dropout_feature(
        self,
        episode: Episode,
        key: tuple[str, ...],
        idx: int,
        dropout: float,
        *,
        scale: float = 1.0,
        horizon: int | None = None,
    ) -> Tensor:
        """Reads a raw (un-embedded) feature straight off episode.input at
        `idx`, flattens it to (b, 1, d), and -- in training only -- zeroes it
        for `dropout` fraction of samples (modality dropout, not elementwise,
        so heads can't just always copy this input). Shared by
        raw_waypoints_key/raw_speed_key; `scale` divides out-of-scale raw
        physical units (e.g. km/h) down to roughly the rest of the network's
        unit-scale activations before it's ever concatenated onto `features`.
        `horizon` (raw_waypoints_key only) keeps just the nearest `horizon`
        points of the second-to-last axis (b, num_points, 2), dropping the
        rest before flattening -- None keeps every point, unchanged.
        """
        x = episode.input.get(key)[:, idx] / scale  # (b, ...)
        if horizon is not None:
            x = x[:, :horizon]
        flat = x.reshape(x.shape[0], 1, -1)  # (b, 1, d)
        if self.training and dropout > 0.0:
            keep = (
                torch.rand(flat.shape[0], 1, 1, device=flat.device) >= dropout
            ).to(flat.dtype)
            flat *= keep
        return flat

    def _history_window_slice(self, idx: int, window: int) -> slice:
        """Slice selecting the `window` ticks ending at (and including) tick `idx`.

        `idx` here is always either a small non-negative int (history_steps - 1,
        the fixed training/predict tick) or -1 (inference default) — the only
        two values `_compute_logits`/`predict()` ever pass as `feature_idx`/`idx`.

        - idx == -1: as a slice *stop*, -1 + 1 == 0 means "up to but excluding
          index 0" (EMPTY), not "through the last tick" — Python's negative-
          slice convention has no way to say "through the end" except
          `stop=None`, so this is special-cased.
        - idx <= -2: `idx + 1` stays negative and is still "N from the end" —
          correct as-is, no special case needed.
        - idx >= 0: `idx - window + 1` must be clamped to 0 ourselves. If left
          negative, it gets reinterpreted as "N from the end" instead of
          "clamp to sequence start" (e.g. idx=1, window=3 -> naive start=-1,
          and index[-1:2] on a 5-tick index is EMPTY, not the 2 ticks
          available) — so we clamp explicitly rather than let it silently
          produce an empty/wrong window. With window == history_steps and
          idx == history_steps - 1 (the only fixed-idx case this is ever
          called with), start is always exactly 0 — no clamping actually
          triggers in practice; the clamp is a defensive correctness
          guarantee, not a normally-exercised path.
        """
        if idx >= 0:
            start = max(idx - window + 1, 0)
            stop = idx + 1
        else:
            start = idx - window + 1
            stop = None if idx == -1 else idx + 1
        return slice(start, stop)

    def _trajectory_gt(self, episode: Episode) -> Tensor | None:
        """(b, num_steps, 2) absolute xy, or (b, num_steps, 3) relative (dx,
        dy, dyaw), per self.trajectory_target. None if xy/heading unavailable.
        """
        xy = episode.get(self.xy_key, default=None)
        heading = episode.get(self.heading_key, default=None)
        if xy is None or heading is None:
            return None
        if self.trajectory_target == "relative":
            return build_relative_trajectory(
                xy=xy, heading_deg=heading, history_steps=self.history_steps
            )
        return build_local_trajectory(
            xy=xy, heading_deg=heading, history_steps=self.history_steps
        )

    def _trajectory_nll_per_mode(self, traj_logits: Tensor, traj_gt: Tensor) -> Tensor:
        """Per-(batch, mode) trajectory NLL, for winner-takes-all mode selection.

        traj_logits: (b, num_modes, num_steps, 4|6). traj_gt: (b, num_steps, 2|3).
        Returns (b, num_modes): self.trajectory_loss summed over steps and
        channels (a constant per-mode normalization -- argmin over modes is
        unaffected by summing vs. averaging).
        """
        gt = traj_gt.unsqueeze(1)  # (b, 1, num_steps, 2|3), broadcasts over modes
        if self.trajectory_target == "relative":
            channels = [
                (traj_logits[..., :2], gt[..., 0]),
                (traj_logits[..., 2:4], gt[..., 1]),
                (traj_logits[..., 4:6], gt[..., 2]),
            ]
        else:
            channels = [
                (traj_logits[..., :2], gt[..., 0]),
                (traj_logits[..., 2:], gt[..., 1]),
            ]

        loss = cast("Module", self.trajectory_loss)
        prev_reduction = loss.reduction  # type: ignore[attr-defined]
        loss.reduction = "none"  # type: ignore[attr-defined]
        try:
            per_step = sum(
                starmap(loss, channels)
            )  # (b, num_modes, num_steps)
        finally:
            loss.reduction = prev_reduction  # type: ignore[attr-defined]
        return per_step.sum(dim=-1)  # (b, num_modes)

    def _compute_logits(  # noqa: PLR0914
        self,
        *,
        episode: Episode,
        embedding: Tensor,
        feature_idx: int | None = None,
        keep_horizon: bool = False,
    ) -> tuple[
        TensorTree,
        Tensor | None,
        Tensor | None,
        Tensor | None,
        Tensor | None,
        Tensor | None,
        Tensor | None,
        Tensor | None,
    ]:
        """Returns (action_logits, traj_logits, h_traj, mode_logits, speed_logits,
        traj_logits_modes, mode_select_logits, mode_winner).

        traj_logits: (b, num_steps, 4) = [mean_x, logvar_x, mean_y, logvar_y], or
            (b, num_steps, 6) with trailing [mean_yaw, logvar_yaw] when the
            trajectory_head was built with predict_yaw=True, or None. When
            trajectory_head is multi-modal (MultiModalGRUTrajectoryHead), this
            is already the winner-selected single candidate (see mode_winner
            below) -- shape/semantics match the single-mode case exactly, so
            every existing consumer of traj_logits/h_traj needs no changes.
        h_traj:      (b, num_steps, hidden_size) GRU hidden states or None.
            Winner-selected in the multi-modal case, same reasoning as above.
        mode_logits: (b, num_steps, 3) coast/gas/brake logits or None.
        speed_logits: (b, num_steps, 2) = [mean_dv, logvar_dv] Δspeed forecast,
            normalized by speed_delta_scale (diff vs. the conditioning tick —
            see gas_stall_report.md §8), or None when speed_head is unset.
        traj_logits_modes: (b, num_modes, num_steps, 4|6), all of head1's
            candidate trajectories, or None when trajectory_head is absent or
            single-mode.
        mode_select_logits: (b, num_modes) head2's raw classification logits
            over which candidate wins, or None.
        mode_winner: (b,) winning mode index — the GT-closest candidate while
            training (teacher-forced hard winner-takes-all; see
            _trajectory_nll_per_mode), or argmax(mode_select_logits) otherwise
            (matching what a real deployment does, with no GT available). None
            unless trajectory_head is multi-modal.
        keep_horizon: at inference, return all predicted steps (b, h, d) instead of
        only the first (b, 1, d); ignored in the training branch (already full).
        """
        _b, _ = episode.input.batch_size

        if feature_idx is not None:
            idx = feature_idx
        else:
            idx = (
                self.history_steps - 1
                if (self.training and self.action_horizon > 1)
                else -1
            )
        embeddings = episode.index[idx].select(*self.feature_keys).parse(embedding)

        parts = [
            self._pool_feature(k, embeddings.get(k), pool_dict=self.feature_pool)
            for k in self.feature_keys
        ]

        if self.raw_feature_keys:
            # pre-encoder, pre-fusion features (e.g. frozen DINOv3 patch tokens)
            # read directly off episode.input_embeddings — bypasses whatever the
            # (also frozen) cross-modal encoder does to them
            parts += [
                self._pool_feature(
                    k,
                    episode.input_embeddings.get(k)[:, idx],
                    pool_dict=self.raw_feature_pool,
                )
                for k in self.raw_feature_keys
            ]

        if self.history_feature_keys:
            history_embeddings = (
                episode.index[self._history_window_slice(idx, self.history_steps)]
                .select(*self.history_feature_keys)
                .parse(embedding)
            )
            parts += [
                self._pool_feature(
                    k,
                    # (b, t', n, d) -> (b, t'*n, d): t' <= history_steps ticks,
                    # n tokens/tick for this key (n=1 for most SUMMARY keys).
                    rearrange(history_embeddings.get(k), "b t n d -> b (t n) d"),
                    pool_dict=self.history_feature_pool,
                )
                for k in self.history_feature_keys
            ]

        if self.trainable_image_encoder is not None:
            # independent, trainable vision side-branch: runs on raw pixels from
            # episode.input, entirely outside episode_builder/encoder, so it trains
            # off this objective's losses alone without touching the frozen shared
            # DINOv3+encoder path every other head still relies on
            pixels = episode.input.get(self.trainable_image_key)[:, idx]
            patch_tokens = self.trainable_image_encoder(pixels)
            parts.append(
                self.trainable_image_pool(patch_tokens)
                if self.trainable_image_pool is not None
                else patch_tokens.mean(dim=1, keepdim=True)
            )

        features = rearrange(parts, "i b 1 d -> b 1 (i d)")

        if self.raw_waypoints_key is not None:
            features = torch.cat(
                [
                    features,
                    self._raw_dropout_feature(
                        episode,
                        self.raw_waypoints_key,
                        idx,
                        self.raw_waypoints_dropout,
                        horizon=self.raw_waypoints_horizon,
                    ),
                ],
                dim=-1,
            )

        if self.raw_speed_key is not None:
            features = torch.cat(
                [
                    features,
                    # raw km/h values would dwarf the rest of the network's
                    # ~unit-scale, LayerNorm'd activations -- normalize the
                    # same way as speed_head/condition_steering_on_speed
                    self._raw_dropout_feature(
                        episode,
                        self.raw_speed_key,
                        idx,
                        self.raw_speed_dropout,
                        scale=self.speed_scale,
                    ),
                ],
                dim=-1,
            )

        current_speed: Tensor | None = None
        if self.speed_head is not None or self.condition_steering_on_speed:
            # normalized (~unit scale); raw km/h would dwarf the rest of the
            # network's LayerNorm'd activations (see speed_scale docstring)
            current_speed = (
                episode.get(self.speed_key).squeeze(-1)[:, idx] / self.speed_scale
            )  # (b,)

        speed_logits: Tensor | None = None
        if self.speed_head is not None:
            speed_logits = self.speed_head(features)  # (b, steps, 2), normalized Δspeed

        heads_features = features
        if self.condition_steering_on_speed:
            heads_features = torch.cat(
                [features, cast("Tensor", current_speed).view(-1, 1, 1)], dim=-1
            )  # (b, 1, d+1)

        traj_logits: Tensor | None = None
        h_traj: Tensor | None = None
        traj_logits_modes: Tensor | None = None
        mode_select_logits: Tensor | None = None
        mode_winner: Tensor | None = None
        if self.trajectory_head is not None:
            traj_logits, h_traj = self.trajectory_head(
                features
            )  # (b, steps, 4|6), (b, steps, H); multi-modal head instead gives
            # (b, num_modes, steps, 4|6), (b, num_modes, steps, H)

            if traj_logits.dim() == 4:  # noqa: PLR2004
                # winner-takes-all: head2 (trajectory_mode_head) decides which
                # of head1's num_modes candidates the rest of the network
                # commits to -- both the trajectory output itself and the
                # h_traj hidden state that conditions the action heads below.
                traj_logits_modes = traj_logits
                if self.trajectory_mode_head is not None:
                    mode_select_logits = self.trajectory_mode_head(features)[
                        :, 0
                    ]  # (b, num_modes)

                traj_gt = self._trajectory_gt(episode)
                if (
                    self.training
                    and self.action_horizon > 1
                    and self.trajectory_loss is not None
                    and traj_gt is not None
                ):
                    # oracle winner: the GT-closest candidate. Teacher-forces
                    # each mode's gradient toward the samples it already fits
                    # best (hard WTA) instead of every mode chasing one mean.
                    nll = self._trajectory_nll_per_mode(traj_logits_modes, traj_gt)
                    mode_winner = nll.argmin(dim=-1)
                elif mode_select_logits is not None:
                    # no GT (inference): defer to head2's own prediction, same
                    # as a real deployment would.
                    mode_winner = mode_select_logits.argmax(dim=-1)
                else:
                    mode_winner = torch.zeros(
                        traj_logits_modes.shape[0],
                        dtype=torch.long,
                        device=traj_logits_modes.device,
                    )

                b_idx = torch.arange(mode_winner.shape[0], device=mode_winner.device)
                traj_logits = traj_logits_modes[b_idx, mode_winner]  # (b, steps, 4|6)
                h_traj = h_traj[b_idx, mode_winner]  # (b, steps, H)

            if self.detach_h_traj:
                h_traj = h_traj.detach()

        logits = self.heads(heads_features, h_traj)
        mode_logits = (
            self.longitudinal_mode_head(heads_features, h_traj)
            if self.longitudinal_mode_head is not None
            else None
        )

        if self.pedal_heads is not None:
            speed_pred_mean = cast("Tensor", speed_logits)[..., :1].detach()  # (b, steps, 1)
            steps = speed_pred_mean.shape[1]
            current_speed_bcast = (
                cast("Tensor", current_speed).view(-1, 1, 1).expand(-1, steps, 1)
            )  # (b, steps, 1)
            pedal_input = torch.cat(
                [speed_pred_mean, current_speed_bcast], dim=-1
            )  # (b, steps, 2)
            pedal_logits = self.pedal_heads(pedal_input)  # {"continuous": {gas_pedal, brake_pedal}}
            logits = {
                **logits,
                Modality.CONTINUOUS: {
                    **logits.get(Modality.CONTINUOUS, {}),
                    **pedal_logits[Modality.CONTINUOUS],
                },
            }

        def _to_steps(x: Tensor) -> Tensor:
            # MLP: (b, 1, h*d) → (b, h, d);  GRU: already (b, h, d)
            if x.shape[1] == 1 and self.action_horizon > 1:
                return rearrange(x, "b 1 (h d) -> b h d", h=self.action_horizon)
            return x

        if self.training and self.action_horizon > 1:
            return (
                tree_map(_to_steps, logits),
                traj_logits,
                h_traj,
                _to_steps(mode_logits) if mode_logits is not None else None,
                _to_steps(speed_logits) if speed_logits is not None else None,
                traj_logits_modes,
                mode_select_logits,
                mode_winner,
            )
        # Inference: normalize to (b, h, d) per-step layout, then keep all steps
        # or only the first depending on keep_horizon

        def _norm_steps(x: Tensor) -> Tensor:
            x = _to_steps(x)
            return x if keep_horizon else x[:, :1]

        return (
            tree_map(_norm_steps, logits),
            traj_logits,
            h_traj,
            _norm_steps(mode_logits) if mode_logits is not None else None,
            _norm_steps(speed_logits) if speed_logits is not None else None,
            traj_logits_modes,
            mode_select_logits,
            mode_winner,
        )

    @override
    def compute_metrics(self, *, episode: Episode, embedding: Tensor) -> Metrics:  # noqa: PLR0914
        if self.norm is not None:
            embedding = self.norm(embedding)

        (
            logits,
            traj_logits,
            _,
            mode_logits,
            speed_logits,
            _,
            mode_select_logits,
            mode_winner,
        ) = self._compute_logits(episode=episode, embedding=embedding)

        if self.training and self.action_horizon > 1:
            targets = tree_map(
                lambda k: episode.get(k)[:, self.history_steps :],
                self.targets,
                is_leaf=lambda x: isinstance(x, tuple),
            )
            targets = self._apply_steering_correction(targets, episode=episode)
            losses = self.losses(
                tree_map(Rearrange("b h d -> (b h) d"), logits),
                tree_map(Rearrange("b h 1 -> (b h)"), targets),
            )  # ty:ignore[call-non-callable]
        else:
            targets = tree_map(
                lambda k: episode.get(k)[:, -1],
                self.targets,
                is_leaf=lambda x: isinstance(x, tuple),
            )
            targets = self._apply_steering_correction(targets, episode=episode)
            # logits is (b, 1, out_features) after normalization in _compute_logits
            losses = self.losses(
                tree_map(Rearrange("b 1 d -> b d"), logits),
                tree_map(Rearrange("b 1 -> b"), targets),
            )  # ty:ignore[call-non-callable]

        all_losses: dict = dict(losses)

        if mode_logits is not None and self.longitudinal_mode_loss is not None:
            # shared coast/gas/brake decision, trained on all samples (the pedal
            # magnitude losses see pressed samples only — MaskedGaussianNLLLoss)
            gas_t = targets[Modality.CONTINUOUS]["gas_pedal"].reshape(-1)
            brake_t = targets[Modality.CONTINUOUS]["brake_pedal"].reshape(-1)
            mode_target = torch.zeros_like(gas_t, dtype=torch.long)
            mode_target[gas_t.abs() > self.longitudinal_press_threshold] = MODE_GAS
            # brake wins if both pressed (rare; safety)
            mode_target[brake_t.abs() > self.longitudinal_press_threshold] = MODE_BRAKE
            all_losses["longitudinal_mode"] = (
                self.longitudinal_mode_loss(
                    mode_logits.reshape(-1, mode_logits.shape[-1]), mode_target
                )
                * self.longitudinal_mode_loss_weight
            )

        xy = episode.get(self.xy_key)
        heading = episode.get(self.heading_key)

        if (
            traj_logits is not None
            and self.trajectory_loss is not None
            and self.training
            and self.action_horizon > 1
            and xy is not None
            and heading is not None
        ):
            flat = rearrange(traj_logits, "b h d -> (b h) d")  # (b*steps, 4 or 6)
            num_steps = traj_logits.shape[1]
            if self.trajectory_target == "relative":
                traj_gt = build_relative_trajectory(
                    xy=xy, heading_deg=heading, history_steps=self.history_steps
                )  # (b, num_steps, 3) = dx, dy, dyaw
                channels = [(flat[..., :2], traj_gt[..., 0]),
                            (flat[..., 2:4], traj_gt[..., 1]),
                            (flat[..., 4:6], traj_gt[..., 2])]
                # distance actually travelled over the horizon: sum of per-step
                # relative displacements (rotation-invariant), meters
                motion = traj_gt[..., :2].norm(dim=-1).sum(dim=1)  # (b,)
            else:
                traj_gt = build_local_trajectory(
                    xy=xy, heading_deg=heading, history_steps=self.history_steps
                )  # (b, num_steps, 2)
                channels = [(flat[..., :2], traj_gt[..., 0]),
                            (flat[..., 2:], traj_gt[..., 1])]
                # absolute frame: positions vs the fixed anchor, so the terminal
                # step's displacement is the horizon's net forward progress (m)
                motion = traj_gt[..., :2].norm(dim=-1)[:, -1]  # (b,)

            def _traj_nll() -> Tensor:
                return sum(  # ty:ignore[invalid-return-type]
                    self.trajectory_loss(pred, rearrange(g, "b h -> (b h)"))  # ty:ignore[call-non-callable]
                    for pred, g in channels
                ) / len(channels)

            if self.trajectory_motion_weight > 0.0:
                # P1 rebalancing: band-pass triangular window peaking at the
                # low-speed/launch regime (see __init__); stopped and cruising
                # both fall to baseline weight 1.
                signal = (
                    1.0
                    - (motion - self.trajectory_motion_peak).abs()
                    / self.trajectory_motion_width
                ).clamp(min=0.0)  # (b,), triangle: 1 at peak, 0 outside peak±width
                w = 1.0 + self.trajectory_motion_weight * signal  # (b,)
                w = (w / w.mean()).repeat_interleave(num_steps)  # (b*steps,)
                prev_reduction = self.trajectory_loss.reduction  # type: ignore[attr-defined]
                self.trajectory_loss.reduction = "none"  # type: ignore[attr-defined]
                try:
                    per_step = _traj_nll()  # (b*steps,)
                finally:
                    self.trajectory_loss.reduction = prev_reduction  # type: ignore[attr-defined]
                traj_loss = (w * per_step).mean()
            else:
                traj_loss = _traj_nll()
            all_losses["trajectory"] = traj_loss * self.trajectory_loss_weight

        if (
            mode_select_logits is not None
            and self.trajectory_mode_loss is not None
            and mode_winner is not None
        ):
            # head2: classifies which of head1's candidates wins (see
            # _trajectory_nll_per_mode for how mode_winner is picked)
            all_losses["trajectory_mode"] = (
                self.trajectory_mode_loss(mode_select_logits, mode_winner)
                * self.trajectory_mode_loss_weight
            )

        if (
            speed_logits is not None
            and self.speed_loss is not None
            and self.training
            and self.action_horizon > 1
        ):
            speed_raw = episode.get(self.speed_key).squeeze(-1)  # (b, t), km/h
            current_speed = speed_raw[:, self.history_steps - 1]  # (b,)
            future_speed = speed_raw[
                :, self.history_steps : self.history_steps + self.action_horizon
            ]  # (b, steps)
            # diff repr. (see gas_stall_report.md §8), normalized to ~unit scale
            # to match speed_head's output and what pedal_heads/steering consume
            speed_delta_gt = (
                future_speed - current_speed.unsqueeze(-1)
            ) / self.speed_delta_scale  # (b, steps)
            all_losses["speed"] = (
                self.speed_loss(
                    rearrange(speed_logits, "b h d -> (b h) d"),
                    rearrange(speed_delta_gt, "b h -> (b h)"),
                )
                * self.speed_loss_weight
            )  # ty:ignore[call-non-callable]

        return {"loss": all_losses}

    @override
    def predict(  # noqa: C901, PLR0912, PLR0914, PLR0915
        self,
        episode: Episode,
        *,
        embedding: Tensor,
        keys: AbstractSet[ObjectivePredictionKey],
        **kwargs: Any,
    ) -> TensorDict:
        predictions: dict[ObjectivePredictionKey, Prediction] = {}

        b, _t = episode.input.batch_size

        if self.norm is not None:
            embedding = self.norm(embedding)

        if (key := ObjectivePredictionKey.GROUND_TRUTH) in keys:
            ground_truth_paths = self.heads.tree_paths() + (
                self.pedal_heads.tree_paths() if self.pedal_heads is not None else ()
            )
            predictions[key] = Prediction(
                value=episode.input.select(*ground_truth_paths).squeeze(-1),
                timestep_indices=slice(None),
            )

        if keys & {
            ObjectivePredictionKey.PREDICTION_VALUE,
            ObjectivePredictionKey.PREDICTION_STD,
            ObjectivePredictionKey.PREDICTION_PROBS,
            ObjectivePredictionKey.SCORE_LOGPROB,
            ObjectivePredictionKey.SCORE_L1,
            ObjectivePredictionKey.SUMMARY_EMBEDDINGS,
            ObjectivePredictionKey.SCORE_L1_REL,
            ObjectivePredictionKey.PREDICTION_DIFF_PREV,
            ObjectivePredictionKey.GROUND_TRUTH_DIFF_PREV,
            ObjectivePredictionKey.PREDICTION_DIFF_HIST,
            ObjectivePredictionKey.GROUND_TRUTH_DIFF_HIST,
            ObjectivePredictionKey.SCORE_SIGNED_ERROR,
            ObjectivePredictionKey.LOSS,
            ObjectivePredictionKey.TRAJECTORY_VALUE,
            ObjectivePredictionKey.TRAJECTORY_GT,
            ObjectivePredictionKey.TRAJECTORY_MODES_VALUE,
            ObjectivePredictionKey.TRAJECTORY_MODE_PROBS,
            ObjectivePredictionKey.SPEED_VALUE,
            ObjectivePredictionKey.SPEED_GT,
        }:
            # train-aligned scoring for horizon models: features at the last history
            # tick, gt at the first horizon tick — the action that drives the car.
            # (scoring against [:, -1] with features at -1 lags the target by one
            # tick and leaks the target action into the model's context)
            if self.action_horizon > 1:
                feature_idx = self.history_steps - 1
                gt_idx = self.history_steps
            else:
                feature_idx = -1
                gt_idx = -1

            if (key := ObjectivePredictionKey.SUMMARY_EMBEDDINGS) in keys:
                predictions[key] = episode.index.select(Modality.SUMMARY)[
                    [feature_idx]
                ].parse(embedding)

            (
                raw_logits_full,
                traj_logits,
                _,
                mode_logits_full,
                speed_logits_full,
                traj_logits_modes,
                mode_select_logits,
                _mode_winner,
            ) = self._compute_logits(
                episode=episode,
                embedding=embedding,
                feature_idx=feature_idx,
                keep_horizon=True,
            )
            # scoring keys use the first predicted step; the full horizon is kept
            # only for gate/mode aggregation (see _gate_prob, _longitudinal_mode)
            raw_logits = tree_map(
                operator.itemgetter((slice(None), slice(1))), raw_logits_full
            )
            logits = TensorDict(raw_logits, batch_size=[b, 1])
            logits_full = TensorDict(raw_logits_full, batch_size=[b])
            mode, mode_probs = (
                self._longitudinal_mode(mode_logits_full)
                if mode_logits_full is not None
                else (None, None)
            )

            timestep_indices = (
                slice(gt_idx, gt_idx + 1) if gt_idx >= 0 else slice(-1, None)
            )

            if (key := ObjectivePredictionKey.PREDICTION_VALUE) in keys:

                def fn(
                    action_type: tuple[Modality, str], x: torch.Tensor
                ) -> torch.Tensor:
                    match action_type:
                        case (Modality.CONTINUOUS, name):
                            # coupled longitudinal decode: the shared mode
                            # classifier decides which pedal emits its magnitude
                            if (
                                mode is not None
                                and (m := _LONGITUDINAL_MODES.get(name)) is not None
                            ):
                                return torch.where(
                                    mode == m, x[..., 0], torch.zeros_like(x[..., 0])
                                )
                            # hurdle head ([mean, log_var, gate]): gated magnitude
                            if x.shape[-1] == 3:  # noqa: PLR2004
                                gate = (
                                    self._gate_prob(logits_full[action_type])
                                    > self.gate_fire_threshold
                                )
                                return torch.where(
                                    gate, x[..., 0], torch.zeros_like(x[..., 0])
                                )
                            return x[..., 0]
                        case (Modality.DISCRETE, "turn_signal"):
                            return non_zero_signal_with_threshold(x).class_idx
                        case _:
                            msg = f"Invalid action type: {action_type}"
                            raise NotImplementedError(msg)

                predictions[key] = Prediction(
                    value=logits.named_apply(fn, nested_keys=True),
                    timestep_indices=timestep_indices,
                )

            if (key := ObjectivePredictionKey.PREDICTION_STD) in keys:

                def fn(
                    action_type: tuple[Modality, str], x: torch.Tensor
                ) -> torch.Tensor:
                    match action_type:
                        case (Modality.CONTINUOUS, _):
                            return torch.sqrt(torch.exp(x[..., 1]))

                        case (Modality.DISCRETE, "turn_signal"):
                            return torch.zeros_like(x[..., 0])  # placeholder
                        case _:
                            msg = f"Invalid action type: {action_type}"
                            raise NotImplementedError(msg)

                predictions[key] = Prediction(
                    value=logits.named_apply(fn, nested_keys=True),
                    timestep_indices=timestep_indices,
                )

            if (key := ObjectivePredictionKey.PREDICTION_PROBS) in keys:

                def fn(
                    action_type: tuple[Modality, str], x: torch.Tensor
                ) -> torch.Tensor:
                    match action_type:
                        case (Modality.CONTINUOUS, name):
                            # coupled longitudinal decode: this pedal's mode
                            # probability, aggregated per gate_horizon_aggregate
                            if (
                                mode_probs is not None
                                and (m := _LONGITUDINAL_MODES.get(name)) is not None
                            ):
                                return mode_probs[..., m]
                            # hurdle head ([mean, log_var, gate]): gate probability,
                            # aggregated over the horizon per gate_horizon_aggregate
                            if x.shape[-1] == 3:  # noqa: PLR2004
                                return self._gate_prob(logits_full[action_type])
                            mean = x[..., 0]
                            std = torch.sqrt(torch.exp(x[..., 1]))
                            return gauss_prob(mean, mean=mean, std=std)

                        case (Modality.DISCRETE, "turn_signal"):
                            return non_zero_signal_with_threshold(x).prob

                        case _:
                            msg = f"Invalid action type: {action_type}"
                            raise NotImplementedError(msg)

                predictions[key] = Prediction(
                    value=logits.named_apply(fn, nested_keys=True),
                    timestep_indices=timestep_indices,
                )

            if (key := ObjectivePredictionKey.SCORE_LOGPROB) in keys:

                def fn(
                    action_type: tuple[Modality, str], x: torch.Tensor
                ) -> torch.Tensor:
                    match action_type:
                        case (Modality.CONTINUOUS, _):
                            mean = x[..., 0]
                            std = torch.sqrt(torch.exp(x[..., 1]))
                            gt = episode.input[action_type][:, gt_idx]
                            return -torch.log(gauss_prob(gt, mean=mean, std=std))

                        case (Modality.DISCRETE, "turn_signal"):
                            gt = episode.input[action_type][:, gt_idx]
                            return F.cross_entropy(
                                x.squeeze(1),
                                gt.squeeze(1).long(),  # ty:ignore[unresolved-attribute]
                                reduction="none",
                            ).unsqueeze(1)

                        case _:
                            msg = f"Invalid action type: {action_type}"
                            raise NotImplementedError(msg)

                predictions[key] = Prediction(
                    value=logits.named_apply(fn, nested_keys=True),
                    timestep_indices=timestep_indices,
                )

            if (key := ObjectivePredictionKey.SCORE_L1) in keys:

                def fn(
                    action_type: tuple[Modality, str], x: torch.Tensor
                ) -> torch.Tensor:
                    gt = episode.input[action_type][:, gt_idx]
                    match action_type:
                        case (Modality.CONTINUOUS, _):
                            return F.l1_loss(x[..., 0], gt, reduction="none")  # ty:ignore[invalid-argument-type]

                        case (Modality.DISCRETE, "turn_signal"):
                            return F.l1_loss(
                                non_zero_signal_with_threshold(x).class_idx.float(),
                                gt.float(),
                                reduction="none",
                            )

                        case _:
                            msg = f"Invalid action type: {action_type}"
                            raise NotImplementedError(msg)

                predictions[key] = Prediction(
                    value=logits.named_apply(fn, nested_keys=True),
                    timestep_indices=timestep_indices,
                )

            if (key := ObjectivePredictionKey.SCORE_L1_REL) in keys:

                def fn(
                    action_type: tuple[Modality, str], x: torch.Tensor
                ) -> torch.Tensor:
                    gt_episode = cast(
                        "Tensor", episode.input[action_type]
                    )  # (b, ep_length, 1)
                    gt = gt_episode[:, gt_idx]  # (b, 1)
                    match action_type:
                        case (Modality.CONTINUOUS, _):
                            prediction = x[..., 0]  # (b, 1)
                            return (prediction - gt).abs() / (gt.abs() + 1e-4)

                        case (Modality.DISCRETE, "turn_signal"):
                            return F.l1_loss(
                                non_zero_signal_with_threshold(x).class_idx.float(),
                                gt.float(),
                                reduction="none",
                            )

                        case _:
                            msg = f"Invalid action type: {action_type}"
                            raise NotImplementedError(msg)

                predictions[key] = Prediction(
                    value=logits.named_apply(fn, nested_keys=True),
                    timestep_indices=timestep_indices,
                )

            if (key := ObjectivePredictionKey.SCORE_SIGNED_ERROR) in keys:

                def fn(
                    action_type: tuple[Modality, str], x: torch.Tensor
                ) -> torch.Tensor:
                    gt = episode.input[action_type][:, gt_idx]
                    match action_type:
                        case (Modality.CONTINUOUS, _):
                            return x[..., 0] - gt  # ty:ignore[unsupported-operator]

                        case (Modality.DISCRETE, "turn_signal"):
                            return (
                                non_zero_signal_with_threshold(x).class_idx.float()
                                - gt.float()
                            )

                        case _:
                            msg = f"Invalid action type: {action_type}"
                            raise NotImplementedError(msg)

                predictions[key] = Prediction(
                    value=logits.named_apply(fn, nested_keys=True),
                    timestep_indices=timestep_indices,
                )

            if (key := ObjectivePredictionKey.PREDICTION_DIFF_PREV) in keys:

                def fn(
                    action_type: tuple[Modality, str], x: torch.Tensor
                ) -> torch.Tensor:
                    gt_episode = cast(
                        "Tensor", episode.input[action_type]
                    )  # (b, ep_length, 1)
                    gt_prev = gt_episode[:, gt_idx - 1]  # (b, 1)
                    match action_type:
                        case (Modality.CONTINUOUS, _):
                            prediction = x[..., 0]  # (b, 1)
                            return prediction - gt_prev

                        case (Modality.DISCRETE, "turn_signal"):
                            prediction = non_zero_signal_with_threshold(
                                x
                            ).class_idx  # (b, 1)
                            return (prediction != gt_prev).float()

                        case _:
                            msg = f"Invalid action type: {action_type}"
                            raise NotImplementedError(msg)

                predictions[key] = Prediction(
                    value=logits.named_apply(fn, nested_keys=True),
                    timestep_indices=timestep_indices,
                )

            if (key := ObjectivePredictionKey.GROUND_TRUTH_DIFF_PREV) in keys:

                def fn(
                    action_type: tuple[Modality, str], _x: torch.Tensor
                ) -> torch.Tensor:
                    gt_episode = cast(
                        "Tensor", episode.input[action_type]
                    )  # (b, ep_length, 1)
                    gt_prev = gt_episode[:, gt_idx - 1]  # (b, 1)
                    gt = gt_episode[:, gt_idx]  # (b, 1)
                    match action_type:
                        case (Modality.CONTINUOUS, _):
                            return gt - gt_prev

                        case (Modality.DISCRETE, "turn_signal"):
                            return (gt != gt_prev).float()

                        case _:
                            msg = f"Invalid action type: {action_type}"
                            raise NotImplementedError(msg)

                predictions[key] = Prediction(
                    value=logits.named_apply(fn, nested_keys=True),
                    timestep_indices=timestep_indices,
                )

            if (key := ObjectivePredictionKey.PREDICTION_DIFF_HIST) in keys:

                def fn(
                    action_type: tuple[Modality, str], x: torch.Tensor
                ) -> torch.Tensor:
                    gt_episode = cast(
                        "Tensor", episode.input[action_type]
                    )  # (b, ep_length, 1)
                    match action_type:
                        case (Modality.CONTINUOUS, _):
                            # history ticks only — never include the target itself
                            gt_hist = gt_episode[:, :gt_idx].mean(dim=1)  # (b, 1)
                            prediction = x[..., 0]  # (b, 1)
                            return prediction - gt_hist

                        case (Modality.DISCRETE, "turn_signal"):
                            gt_hist = torch.mode(
                                gt_episode[:, :gt_idx], dim=1
                            ).values  # (b, 1)
                            prediction = non_zero_signal_with_threshold(
                                x
                            ).class_idx  # (b, 1)
                            return (prediction != gt_hist).float()

                        case _:
                            msg = f"Invalid action type: {action_type}"
                            raise NotImplementedError(msg)

                predictions[key] = Prediction(
                    value=logits.named_apply(fn, nested_keys=True),
                    timestep_indices=timestep_indices,
                )

            if (key := ObjectivePredictionKey.GROUND_TRUTH_DIFF_HIST) in keys:

                def fn(
                    action_type: tuple[Modality, str], _x: torch.Tensor
                ) -> torch.Tensor:
                    gt_episode = cast(
                        "Tensor", episode.input[action_type]
                    )  # (b, ep_length, 1)
                    gt = gt_episode[:, gt_idx]  # (b, 1)
                    match action_type:
                        case (Modality.CONTINUOUS, _):
                            # history ticks only — never include the target itself
                            gt_hist = gt_episode[:, :gt_idx].mean(dim=1)  # (b, 1)
                            return gt - gt_hist

                        case (Modality.DISCRETE, "turn_signal"):
                            gt_hist = torch.mode(gt_episode[:, :gt_idx], dim=1).values
                            return (gt != gt_hist).float()

                        case _:
                            msg = f"Invalid action type: {action_type}"
                            raise NotImplementedError(msg)

                predictions[key] = Prediction(
                    value=logits.named_apply(fn, nested_keys=True),
                    timestep_indices=timestep_indices,
                )

            if (key := ObjectivePredictionKey.LOSS) in keys:
                if self.losses is not None and self.targets is not None:
                    targets = tree_map(
                        lambda k: episode.get(k)[:, gt_idx],
                        self.targets,
                        is_leaf=lambda x: isinstance(x, tuple),
                    )
                    reductions = {
                        name: m.reduction  # type: ignore[attr-defined]
                        for name, m in self.losses.named_modules()
                        if hasattr(m, "reduction")
                    }
                    for name, m in self.losses.named_modules():
                        if name in reductions:
                            m.reduction = "none"  # type: ignore[attr-defined]
                    try:
                        losses = self.losses(
                            tree_map(Rearrange("b 1 d -> b d"), raw_logits),
                            tree_map(Rearrange("b 1 -> b"), targets),
                        )  # ty:ignore[call-non-callable]
                    finally:
                        for name, m in self.losses.named_modules():
                            if name in reductions:
                                m.reduction = reductions[name]  # type: ignore[attr-defined]
                    predictions[key] = Prediction(
                        value=TensorDict(losses, batch_size=[b]),
                        timestep_indices=timestep_indices,
                    )

            if (key := ObjectivePredictionKey.TRAJECTORY_VALUE) in keys:
                if traj_logits is not None:
                    # traj_logits: (b, steps, 4) = [mean_x, logvar_x, mean_y, logvar_y],
                    # or (b, steps, 6) with trailing [mean_yaw, logvar_yaw] when
                    # trajectory_target == "relative"; xy means are always at [0, 2]
                    value = {"xy": traj_logits[..., [0, 2]]}  # (b, steps, 2) means
                    if self.trajectory_target == "relative":
                        value["yaw"] = traj_logits[..., 4]  # (b, steps) mean dyaw
                    predictions[key] = Prediction(
                        value=TensorDict(value, batch_size=[b]),
                        timestep_indices=timestep_indices,
                    )

            if (key := ObjectivePredictionKey.TRAJECTORY_GT) in keys:
                xy = episode.get(self.xy_key, default=None)
                heading = episode.get(self.heading_key, default=None)
                if xy is not None and heading is not None:
                    if self.trajectory_target == "relative":
                        traj_gt = build_relative_trajectory(
                            xy=xy, heading_deg=heading, history_steps=self.history_steps
                        )  # (b, steps, 3) = dx, dy, dyaw
                        value = {"xy": traj_gt[..., :2], "yaw": traj_gt[..., 2]}
                    else:
                        traj_gt = build_local_trajectory(
                            xy=xy, heading_deg=heading, history_steps=self.history_steps
                        )  # (b, steps, 2)
                        value = {"xy": traj_gt}
                    predictions[key] = Prediction(
                        value=TensorDict(value, batch_size=[b]),
                        timestep_indices=timestep_indices,
                    )

            if (key := ObjectivePredictionKey.TRAJECTORY_MODES_VALUE) in keys:
                if traj_logits_modes is not None:
                    # traj_logits_modes: (b, num_modes, steps, 4|6) -- all of
                    # head1's candidates, same xy/yaw column convention as
                    # TRAJECTORY_VALUE (which only keeps the winner)
                    value = {"xy": traj_logits_modes[..., [0, 2]]}
                    if self.trajectory_target == "relative":
                        value["yaw"] = traj_logits_modes[..., 4]
                    predictions[key] = Prediction(
                        value=TensorDict(value, batch_size=[b]),
                        timestep_indices=timestep_indices,
                    )

            if (key := ObjectivePredictionKey.TRAJECTORY_MODE_PROBS) in keys:
                if mode_select_logits is not None:
                    predictions[key] = Prediction(
                        value=TensorDict(
                            {"probs": mode_select_logits.softmax(dim=-1)},
                            batch_size=[b],
                        ),
                        timestep_indices=timestep_indices,
                    )

            if (key := ObjectivePredictionKey.SPEED_VALUE) in keys:
                if speed_logits_full is not None:
                    # denormalized back to km/h for direct interpretability
                    value = {"delta": speed_logits_full[..., 0] * self.speed_delta_scale}
                    predictions[key] = Prediction(
                        value=TensorDict(value, batch_size=[b]),
                        timestep_indices=timestep_indices,
                    )

            if (key := ObjectivePredictionKey.SPEED_GT) in keys:
                speed_raw = episode.get(self.speed_key, default=None)
                if speed_raw is not None:
                    speed_raw = speed_raw.squeeze(-1)  # (b, t), km/h
                    # only gt_idx itself is a genuine future tick in this window
                    # (history_steps ticks of history + this single scored target,
                    # confirmed against score_l1's own gt_idx-aligned computation —
                    # ticks beyond gt_idx are NOT present here); a true multi-step
                    # comparison against speed_head's full horizon needs joining
                    # consecutive (1-tick-apart) predict rows' own gt_idx values.
                    current_speed = speed_raw[:, feature_idx]
                    target_speed = speed_raw[:, gt_idx]
                    value = {"delta": (target_speed - current_speed).unsqueeze(-1)}  # (b, 1), km/h
                    predictions[key] = Prediction(
                        value=TensorDict(value, batch_size=[b]),
                        timestep_indices=timestep_indices,
                    )

        return TensorDict(predictions).auto_batch_size_(2)  # ty:ignore[invalid-argument-type]
