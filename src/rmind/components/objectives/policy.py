import operator
from collections.abc import Set as AbstractSet
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
        xy_key: tuple[str, ...] = ("input", "trajectory", "xy"),
        heading_key: tuple[str, ...] = ("input", "trajectory", "heading"),
        feature_keys: list[tuple[str, ...]] = [  # noqa: B006
            (Modality.SUMMARY, SummaryToken.OBSERVATION_SUMMARY),
            (Modality.SUMMARY, SummaryToken.OBSERVATION_HISTORY),
            (Modality.CONTEXT, "waypoints"),
        ],
        feature_pool: InstanceOf[ModuleDict] | None = None,
        prediction_std_scale: dict[str, float] | None = None,
        gate_horizon_aggregate: Literal["first", "max"] = "first",
        gate_fire_threshold: float = 0.5,
        longitudinal_mode_head: InstanceOf[Module] | None = None,
        longitudinal_mode_loss: InstanceOf[Module] | None = None,
        longitudinal_mode_loss_weight: float = 1.0,
        longitudinal_press_threshold: float = 0.01,
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
        self.xy_key: tuple[str, ...] = xy_key
        self.heading_key: tuple[str, ...] = heading_key
        self.feature_keys: list[tuple[str, ...]] = feature_keys
        self.feature_pool: ModuleDict | None = feature_pool
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

    @override
    def forward(self, episode: Episode, embedding: Tensor) -> TensorDict:
        if self.norm is not None:
            embedding = self.norm(embedding)

        logits, _, _, mode_logits = self._compute_logits(
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

    def _pool_feature(self, key: tuple[str, ...], x: Tensor) -> Tensor:
        if (
            self.feature_pool is not None
            and (pool := self.feature_pool.get(key, default=None)) is not None
        ):
            return pool(x)
        return x.mean(dim=1, keepdim=True)

    def _compute_logits(
        self,
        *,
        episode: Episode,
        embedding: Tensor,
        feature_idx: int | None = None,
        keep_horizon: bool = False,
    ) -> tuple[TensorTree, Tensor | None, Tensor | None, Tensor | None]:
        """Returns (action_logits, traj_logits, h_traj, mode_logits).

        traj_logits: (b, num_steps, 4) = [mean_x, logvar_x, mean_y, logvar_y] or None.
        h_traj:      (b, num_steps, hidden_size) GRU hidden states or None.
        mode_logits: (b, num_steps, 3) coast/gas/brake logits or None.
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

        parts = [self._pool_feature(k, embeddings.get(k)) for k in self.feature_keys]

        features = rearrange(parts, "i b 1 d -> b 1 (i d)")

        traj_logits: Tensor | None = None
        h_traj: Tensor | None = None
        if self.trajectory_head is not None:
            traj_logits, h_traj = self.trajectory_head(
                features
            )  # (b, steps, 4), (b, steps, H)

        logits = self.heads(features, h_traj)
        mode_logits = (
            self.longitudinal_mode_head(features, h_traj)
            if self.longitudinal_mode_head is not None
            else None
        )

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
        )

    @override
    def compute_metrics(self, *, episode: Episode, embedding: Tensor) -> Metrics:  # noqa: PLR0914
        if self.norm is not None:
            embedding = self.norm(embedding)

        logits, traj_logits, _, mode_logits = self._compute_logits(
            episode=episode, embedding=embedding
        )

        if self.training and self.action_horizon > 1:
            targets = tree_map(
                lambda k: episode.get(k)[:, self.history_steps :],
                self.targets,
                is_leaf=lambda x: isinstance(x, tuple),
            )
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
            traj_gt = build_local_trajectory(
                xy=xy, heading_deg=heading, history_steps=self.history_steps
            )  # (b, num_steps, 2)
            flat = rearrange(traj_logits, "b h d -> (b h) d")  # (b*steps, 4)
            gt_x = rearrange(traj_gt[..., 0], "b h -> (b h)")  # (b*steps,)
            gt_y = rearrange(traj_gt[..., 1], "b h -> (b h)")  # (b*steps,)
            traj_loss = (
                self.trajectory_loss(flat[..., :2], gt_x)
                + self.trajectory_loss(flat[..., 2:], gt_y)
            ) / 2  # ty:ignore[call-non-callable]
            all_losses["trajectory"] = traj_loss * self.trajectory_loss_weight

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
            predictions[key] = Prediction(
                value=episode.input.select(*self.heads.tree_paths()).squeeze(-1),
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

            raw_logits_full, traj_logits, _, mode_logits_full = self._compute_logits(
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
                    # traj_logits: (b, steps, 4) = [mean_x, logvar_x, mean_y, logvar_y]
                    predictions[key] = Prediction(
                        value=TensorDict(
                            {"xy": traj_logits[..., [0, 2]]},  # (b, steps, 2) means
                            batch_size=[b],
                        ),
                        timestep_indices=timestep_indices,
                    )

            if (key := ObjectivePredictionKey.TRAJECTORY_GT) in keys:
                xy = episode.get(self.xy_key, default=None)
                heading = episode.get(self.heading_key, default=None)
                if xy is not None and heading is not None:
                    traj_gt = build_local_trajectory(
                        xy=xy, heading_deg=heading, history_steps=self.history_steps
                    )  # (b, steps, 2)
                    predictions[key] = Prediction(
                        value=TensorDict({"xy": traj_gt}, batch_size=[b]),
                        timestep_indices=timestep_indices,
                    )

        return TensorDict(predictions).auto_batch_size_(2)  # ty:ignore[invalid-argument-type]
