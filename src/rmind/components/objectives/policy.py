from collections.abc import Set as AbstractSet
from typing import Any, cast, final, override

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

    @override
    def forward(self, episode: Episode, embedding: Tensor) -> TensorDict:
        if self.norm is not None:
            embedding = self.norm(embedding)

        logits, _, _ = self._compute_logits(episode=episode, embedding=embedding)

        def fn(nk: tuple[str, ...], x: Tensor) -> Tensor:
            # always return the first predicted step (b, 1, d) or (b, h, d) -> (b, 1)
            first = x[:, :1, :]
            match nk:
                case (Modality.CONTINUOUS, _):
                    return first[..., 0]
                case (Modality.DISCRETE, "turn_signal"):
                    return non_zero_signal_with_threshold(first).class_idx
                case _:
                    raise NotImplementedError

        return TensorDict(logits).named_apply(fn, nested_keys=True)  # ty:ignore[invalid-return-type, invalid-argument-type]

    def _compute_logits(
        self, *, episode: Episode, embedding: Tensor
    ) -> tuple[TensorTree, Tensor | None, Tensor | None]:
        """Returns (action_logits, traj_logits, h_traj).

        traj_logits: (b, num_steps, 4) = [mean_x, logvar_x, mean_y, logvar_y] or None.
        h_traj:      (b, num_steps, hidden_size) GRU hidden states or None.
        """
        _b, _ = episode.input.batch_size

        idx = self.history_steps - 1 if (self.training and self.action_horizon > 1) else -1
        embeddings = (
            episode
            .index[idx]
            .select(
                (Modality.SUMMARY, SummaryToken.OBSERVATION_HISTORY),
                (Modality.SUMMARY, SummaryToken.OBSERVATION_SUMMARY),
                (Modality.CONTEXT, "waypoints"),
            )
            .parse(embedding)
        )

        observation_history = embeddings.get((
            Modality.SUMMARY,
            SummaryToken.OBSERVATION_HISTORY,
        ))

        observation_summary = embeddings.get((
            Modality.SUMMARY,
            SummaryToken.OBSERVATION_SUMMARY,
        ))

        waypoints = embeddings.get((Modality.CONTEXT, "waypoints")).mean(
            dim=1, keepdim=True
        )

        features = rearrange(
            [observation_summary, observation_history, waypoints],
            "i b 1 d -> b 1 (i d)",
        )

        traj_logits: Tensor | None = None
        h_traj: Tensor | None = None
        if self.trajectory_head is not None:
            traj_logits, h_traj = self.trajectory_head(features)  # (b, steps, 4), (b, steps, H)

        logits = self.heads(features, h_traj)
        if self.training and self.action_horizon > 1:
            # MLP: (b, 1, h*d) → (b, h, d);  GRU: already (b, h, d)
            return (
                tree_map(
                    lambda x: (
                        rearrange(x, "b 1 (h d) -> b h d", h=self.action_horizon)
                        if x.shape[1] == 1
                        else x
                    ),
                    logits,
                ),
                traj_logits,
                h_traj,
            )
        # Inference: normalize to (b, 1, out_features) — first predicted step only
        def _first_step(x: Tensor) -> Tensor:
            if x.shape[1] > 1:  # GRU: (b, h, d) → (b, 1, d)
                return x[:, :1]
            if self.action_horizon > 1:  # MLP: (b, 1, h*d) → (b, 1, d)
                return rearrange(x, "b 1 (h d) -> b h d", h=self.action_horizon)[:, :1]
            return x
        return tree_map(_first_step, logits), traj_logits, h_traj

    @override
    def compute_metrics(self, *, episode: Episode, embedding: Tensor) -> Metrics:
        if self.norm is not None:
            embedding = self.norm(embedding)

        logits, traj_logits, _ = self._compute_logits(episode=episode, embedding=embedding)

        if self.training and self.action_horizon > 1:
            targets = tree_map(
                lambda k: episode.get(k)[:, self.history_steps:],
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
                xy=xy,
                heading_deg=heading,
                history_steps=self.history_steps,
            )  # (b, num_steps, 2)
            flat = rearrange(traj_logits, "b h d -> (b h) d")         # (b*steps, 4)
            gt_x = rearrange(traj_gt[..., 0], "b h -> (b h)")         # (b*steps,)
            gt_y = rearrange(traj_gt[..., 1], "b h -> (b h)")         # (b*steps,)
            traj_loss = (
                self.trajectory_loss(flat[..., :2], gt_x)
                + self.trajectory_loss(flat[..., 2:], gt_y)
            ) / 2  # ty:ignore[call-non-callable]
            all_losses["trajectory"] = traj_loss * self.trajectory_loss_weight

        return {"loss": all_losses}

    @override
    def predict(  # noqa: C901, PLR0912, PLR0915
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
            if (key := ObjectivePredictionKey.SUMMARY_EMBEDDINGS) in keys:
                predictions[key] = episode.index.select(Modality.SUMMARY)[[-1]].parse(
                    embedding
                )

            raw_logits, traj_logits, _ = self._compute_logits(
                episode=episode, embedding=embedding
            )
            logits = TensorDict(raw_logits, batch_size=[b, 1])

            timestep_indices = slice(-1, None)

            if (key := ObjectivePredictionKey.PREDICTION_VALUE) in keys:

                def fn(
                    action_type: tuple[Modality, str], x: torch.Tensor
                ) -> torch.Tensor:
                    match action_type:
                        case (Modality.CONTINUOUS, _):
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
                        case (Modality.CONTINUOUS, _):
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
                            gt = episode.input[action_type][:, -1]
                            return -torch.log(gauss_prob(gt, mean=mean, std=std))

                        case (Modality.DISCRETE, "turn_signal"):
                            gt = episode.input[action_type][:, -1]
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
                    gt = episode.input[action_type][:, -1]
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
                    gt = gt_episode[:, -1]  # (b, 1)
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
                    gt = episode.input[action_type][:, -1]
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
                    gt_prev = gt_episode[:, -2]  # (b, 1)
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
                    gt_prev = gt_episode[:, -2]  # (b, 1)
                    gt = gt_episode[:, -1]  # (b, 1)
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
                            gt_hist = gt_episode.mean(dim=1)  # (b, 1)
                            prediction = x[..., 0]  # (b, 1)
                            return prediction - gt_hist

                        case (Modality.DISCRETE, "turn_signal"):
                            gt_hist = torch.mode(gt_episode, dim=1).values  # (b, 1)
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
                    gt = gt_episode[:, -1]  # (b, 1)
                    match action_type:
                        case (Modality.CONTINUOUS, _):
                            gt_hist = gt_episode.mean(dim=1)  # (b, 1)
                            return gt - gt_hist

                        case (Modality.DISCRETE, "turn_signal"):
                            gt_hist = torch.mode(gt_episode, dim=1).values
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
                        lambda k: episode.get(k)[:, -1],
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
                        xy=xy,
                        heading_deg=heading,
                        history_steps=self.history_steps,
                    )  # (b, steps, 2)
                    predictions[key] = Prediction(
                        value=TensorDict({"xy": traj_gt}, batch_size=[b]),
                        timestep_indices=timestep_indices,
                    )

        return TensorDict(predictions).auto_batch_size_(2)  # ty:ignore[invalid-argument-type]
