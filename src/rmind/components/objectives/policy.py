from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from typing import Annotated, Any, Literal, Self, final, override

import torch
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    InstanceOf,
    model_validator,
    validate_call,
)
from structlog import get_logger
from tensordict import TensorDict
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F

from rmind.components.action_transform import ActionTransform
from rmind.components.base import Modality
from rmind.components.containers import ModuleDict
from rmind.components.episode import Episode
from rmind.components.lds import LDSWeights
from rmind.components.loss import FlowMatchingLoss
from rmind.components.objectives.base import (
    Metrics,
    Objective,
    ObjectivePredictionKey,
    Prediction,
)
from rmind.components.readout import Readout, SingleReadout
from rmind.components.transformer import FlowActionDecoder

logger = get_logger(__name__)

FlowTimeSampling = Literal["uniform", "logit-normal"]
_VALIDATION_SEED = 0


@final
class PolicyObjective(Objective):
    class _Wiring(BaseModel):
        """Cross-component compatibility of the injected modules."""

        model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

        decoder: FlowActionDecoder
        action_keys: tuple[str, ...]
        action_transform: ActionTransform | None
        lds_weights: LDSWeights | None

        @model_validator(mode="after")
        def _check(self) -> Self:
            transform = self.action_transform
            if transform is None:
                raw_dim = self.decoder.action_dim
            else:
                if transform.action_keys != self.action_keys:
                    msg = (
                        f"action_transform action_keys {transform.action_keys} != "
                        f"objective action_keys {self.action_keys}"
                    )
                    raise ValueError(msg)
                if transform.model_dim != self.decoder.action_dim:
                    msg = (
                        f"action_transform.model_dim {transform.model_dim} != "
                        f"decoder.action_dim {self.decoder.action_dim} "
                        "(set flow_action_dim to match)"
                    )
                    raise ValueError(msg)
                raw_dim = transform.raw_dim

            if len(self.action_keys) != raw_dim:
                msg = (
                    f"action_targets must have {raw_dim} entries (raw_dim), "
                    f"got {len(self.action_keys)}"
                )
                raise ValueError(msg)

            lds = self.lds_weights
            if lds is not None:
                if lds.model_dim != self.decoder.action_dim:
                    msg = (
                        f"lds model_dim {lds.model_dim} != "
                        f"decoder.action_dim {self.decoder.action_dim}"
                    )
                    raise ValueError(msg)
                # LDS lives in the model (loss) space: the transform's model keys
                # when merging, else the raw action keys.
                model_keys = (
                    transform.model_action_keys
                    if transform is not None
                    else self.action_keys
                )
                if lds.model_keys != tuple(model_keys):
                    msg = (
                        f"lds model_keys {lds.model_keys} != "
                        f"model-space keys {tuple(model_keys)}"
                    )
                    raise ValueError(msg)
            return self

    @validate_call
    def __init__(  # noqa: PLR0913
        self,
        *,
        decoder: InstanceOf[FlowActionDecoder],
        loss: InstanceOf[FlowMatchingLoss],
        targets: Mapping[str, tuple[str, ...]],
        condition_tokens: tuple[tuple[Modality, str], ...],
        history_steps: Annotated[int, Field(gt=0)],
        flow_time_sampling: FlowTimeSampling = "logit-normal",
        action_transform: InstanceOf[ActionTransform] | None = None,
        lds_weights: InstanceOf[LDSWeights] | None = None,
        readout: InstanceOf[Readout] | None = None,
    ) -> None:
        super().__init__()
        self.targets = dict(targets)
        self.action_keys = tuple(self.targets)
        self.condition_tokens = tuple(condition_tokens)
        self._Wiring(
            decoder=decoder,
            action_keys=self.action_keys,
            action_transform=action_transform,
            lds_weights=lds_weights,
        )

        self.decoder = decoder
        self.loss = loss
        self.readout = readout if readout is not None else SingleReadout()
        self.history_steps = history_steps
        self.flow_time_sampling = flow_time_sampling
        self.action_transform = action_transform
        self.lds_weights = lds_weights

    def exportable(self) -> "ExportablePolicy":
        return ExportablePolicy(self.decoder, self.action_transform)

    @override
    def compute_metrics(
        self, *, episode: Episode, embedding: Tensor, batch: Any = None, **_kwargs: Any
    ) -> Metrics:
        condition_tokens = self._condition_tokens(episode=episode, embedding=embedding)
        target_actions = self._target_actions(batch)
        condition_tokens, target_actions = self._drop_nonfinite(
            condition_tokens, target_actions, episode=episode, embedding=embedding
        )

        target_actions_model = self._to_model_space(target_actions)

        generator = self._validation_generator(device=target_actions.device)
        noise = torch.randn(
            target_actions_model.shape,
            dtype=target_actions_model.dtype,
            device=target_actions_model.device,
            generator=generator,
        )
        flow_time = self.sample_flow_time(
            self.flow_time_sampling,
            target_actions_model.shape[0],
            dtype=target_actions_model.dtype,
            device=target_actions_model.device,
            generator=generator,
        )
        flow_time_broadcast = flow_time.view(-1, 1, 1)
        noised_actions = (
            1.0 - flow_time_broadcast
        ) * noise + flow_time_broadcast * target_actions_model
        target_action_flow = target_actions_model - noise

        predicted_action_flow = self.decoder(
            condition_tokens=condition_tokens,
            noised_actions=noised_actions,
            flow_time=flow_time,
        )

        loss = self.loss(
            predicted_action_flow,
            target_action_flow,
            noised_actions=noised_actions,
            flow_time=flow_time_broadcast,
            target_actions=target_actions_model,
            weight=self._lds_weight(target_actions),
        )
        metrics: Metrics = {
            "loss": loss,
            "flow_mse": F.mse_loss(predicted_action_flow, target_action_flow).detach(),
        }
        if not self.training:
            honest = self._to_raw_space(
                self.decoder.sample(
                    condition_tokens=condition_tokens,
                    noise=self._noise(
                        condition_tokens=condition_tokens, generator=generator
                    ),
                )
            )  # (B, H, A) raw
            tgt = target_actions.to(dtype=honest.dtype)
            metrics["sample_l1"] = (
                (honest - tgt).abs().flatten(start_dim=1).mean(dim=1).mean()
            )
        return metrics

    def _drop_nonfinite(
        self,
        condition_tokens: Tensor,
        target_actions: Tensor,
        *,
        episode: Episode,
        embedding: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Drop rows with a non-finite condition/target, naming the culprit field.

        A NaN in a condition input (e.g. waypoints — not range-filtered by the
        dataset query, only speed/gas/brake/steering are) propagates through the
        frozen encoder to NaN condition_tokens, then to a NaN loss for the whole
        batch. The attribution (and a NaN swallowed downstream never reaches
        RMIND_NAN_DEBUG) only runs when a row is non-finite.
        """
        finite = condition_tokens.flatten(1).isfinite().all(
            dim=1
        ) & target_actions.flatten(1).isfinite().all(dim=1)
        if bool(finite.all()):
            return condition_tokens, target_actions

        culprits: dict[str, int] = {}
        parsed = (
            episode
            .index[self.history_steps - 1]
            .select(*self.condition_tokens)
            .parse(embedding)
        )
        for path in self.condition_tokens:
            bad = int((~parsed.get(path).flatten(1).isfinite().all(dim=1)).sum())
            if bad:
                culprits[f"condition:{path[0].value}/{path[1]}"] = bad
        tgt_bad = int((~target_actions.flatten(1).isfinite().all(dim=1)).sum())
        if tgt_bad:
            culprits["target_actions"] = tgt_bad
        logger.warning(
            "dropping non-finite samples from flow loss",
            dropped=int((~finite).sum()),
            batch=int(finite.numel()),
            culprits=culprits,
        )
        if not bool(finite.any()):
            return condition_tokens, target_actions
        return condition_tokens[finite], target_actions[finite]

    def _readout_actions(self, condition_tokens: Tensor) -> Tensor:
        """Raw-space action chunk via the configured readout.

        Draws `self.readout.num_samples` candidates and lets the readout collapse
        them. Uses the global RNG (generator=None): re-seeding per batch would tie
        the noise to within-batch position, imprinting a period-`batch_size`
        artifact on plotted predictions.
        """
        draws = self._sample_raw_draws(condition_tokens, self.readout.num_samples)
        return self.readout(draws)

    def _sample_raw_draws(self, condition_tokens: Tensor, k: int) -> Tensor:
        """K raw-space candidate chunks per frame: (B, K, H, A_raw)."""
        cond_rep = condition_tokens.repeat_interleave(k, dim=0)
        draws = self.decoder.sample(
            condition_tokens=cond_rep,
            noise=self._noise(condition_tokens=cond_rep, generator=None),
        ).reshape(
            condition_tokens.shape[0],
            k,
            self.decoder.action_horizon,
            self.decoder.action_dim,
        )
        return self._to_raw_space(draws)  # (B, K, H, A) raw

    @override
    def predict(
        self,
        *,
        episode: Episode,
        embedding: Tensor,
        keys: AbstractSet[ObjectivePredictionKey],
        tokenizers: ModuleDict | None = None,
        batch: Any = None,
        **_kwargs: Any,
    ) -> TensorDict:
        del tokenizers
        key = ObjectivePredictionKey
        scores = {key.SCORE_L1, key.SCORE_SIGNED_ERROR}

        target = (
            self._target_actions(batch)
            if keys & ({key.GROUND_TRUTH} | scores)
            else None
        )
        predicted = (
            self._readout_actions(
                self._condition_tokens(episode=episode, embedding=embedding)
            )
            if keys & ({key.PREDICTION_VALUE} | scores)
            else None
        )

        slot = self._target_slice()
        predictions: dict[ObjectivePredictionKey, Prediction] = {}
        if key.GROUND_TRUTH in keys and target is not None:
            predictions[key.GROUND_TRUTH] = Prediction(
                value=self._trajectory_to_tensordict(target), timestep_indices=slot
            )
        if key.PREDICTION_VALUE in keys and predicted is not None:
            predictions[key.PREDICTION_VALUE] = Prediction(
                value=self._trajectory_to_tensordict(predicted), timestep_indices=slot
            )
        if key.SCORE_L1 in keys and predicted is not None and target is not None:
            predictions[key.SCORE_L1] = Prediction(
                value=self._trajectory_to_tensordict(
                    F.l1_loss(predicted, target, reduction="none")
                ),
                timestep_indices=slot,
            )
        if (
            key.SCORE_SIGNED_ERROR in keys
            and predicted is not None
            and target is not None
        ):
            predictions[key.SCORE_SIGNED_ERROR] = Prediction(
                value=self._trajectory_to_tensordict(predicted - target),
                timestep_indices=slot,
            )

        return TensorDict(predictions).auto_batch_size_(2)  # ty:ignore[invalid-argument-type]

    def _noise(
        self, *, condition_tokens: Tensor, generator: torch.Generator | None
    ) -> Tensor:
        return torch.randn(
            condition_tokens.shape[0],
            self.decoder.action_horizon,
            self.decoder.action_dim,
            dtype=condition_tokens.dtype,
            device=condition_tokens.device,
            generator=generator,
        )

    def _condition_tokens(self, *, episode: Episode, embedding: Tensor) -> Tensor:
        embeddings = (
            episode
            .index[self.history_steps - 1]
            .select(*self.condition_tokens)
            .parse(embedding)
        )
        parts = [embeddings.get(path) for path in self.condition_tokens]
        return torch.cat(parts, dim=1)

    def _target_actions(self, batch: Any) -> Tensor:
        actions: list[Tensor] = []
        for path in self.targets.values():
            value: Any = batch
            for key in path:
                value = value[key]
            actions.append(value)
        return torch.stack(actions, dim=-1)

    def _to_model_space(self, raw: Tensor) -> Tensor:
        """Raw actions -> the space the decoder/flow operates in (Gaussianized)."""
        return raw if self.action_transform is None else self.action_transform(raw)

    def _to_raw_space(self, model: Tensor) -> Tensor:
        """Decoder/flow-space actions -> raw, for raw-space metrics and outputs."""
        if self.action_transform is None:
            return model
        return self.action_transform.inverse(model)

    def _lds_weight(self, target_actions: Tensor) -> Tensor | None:
        """Per-chunk LDS weight (B, 1, model_dim) broadcastable over slots, or
        None when LDS is off.

        The label is the peak |physical action| over the horizon per model
        channel (longitudinal, steering) — peak-over-slots so a chunk's lead-in
        is upweighted with its maneuver. Physical (pre-Gaussianize) space, since
        the bins are fit there.
        """
        if self.lds_weights is None:
            return None
        phys = (
            self.action_transform.physical_model(target_actions)
            if self.action_transform is not None
            else target_actions
        )
        label = phys.abs().amax(dim=1)  # (B, model_dim)
        return self.lds_weights(label).unsqueeze(1)  # (B, 1, model_dim)

    def _target_slice(self) -> slice:
        start = self.history_steps
        return slice(start, start + self.decoder.action_horizon)

    def _trajectory_to_tensordict(self, trajectory: Tensor) -> TensorDict:
        return TensorDict(
            {
                Modality.CONTINUOUS: {
                    key: trajectory[..., idx]
                    for idx, key in enumerate(self.action_keys)
                }
            },  # ty:ignore[invalid-argument-type]
            batch_size=list(trajectory.shape[:2]),
            device=trajectory.device,
        )

    @staticmethod
    def sample_flow_time(  # noqa: PLR0913
        flow_time_sampling: str,
        batch_size: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | None = None,
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
    ) -> Tensor:
        # Convention: t=0 is noise, t=1 is the clean action (interpolant
        # x_t = (1-t)*noise + t*target). "Skew toward 0" therefore means more
        # supervision at the noisy end.
        match flow_time_sampling:
            case "uniform":
                return torch.rand(
                    batch_size, dtype=dtype, device=device, generator=generator
                )
            case "logit-normal":
                # Logit-normal timestep sampling follows SiT:
                # https://arxiv.org/abs/2401.08740
                # Official code: https://github.com/willisma/SiT
                # t = sigmoid(z*std + mean); mean > 0 skews toward t=1 (data),
                # mean < 0 toward t=0 (noise); std controls concentration.
                z = torch.randn(
                    batch_size, dtype=dtype, device=device, generator=generator
                )
                return torch.sigmoid(z * logit_std + logit_mean)
            case _:
                msg_0 = f"Invalid flow_time_sampling method: {flow_time_sampling}"
                raise ValueError(msg_0)

    def _validation_generator(self, *, device: torch.device) -> torch.Generator | None:
        if self.training:
            return None

        generator = torch.Generator(device=device)
        generator.manual_seed(_VALIDATION_SEED)
        return generator


@final
class ExportablePolicy(Module):
    """Deterministic, ONNX-exportable inference graph for the flow policy.

    forward(condition_tokens (B,S,D), noise (B,K,H,A_model))
        -> raw action draws (B,K,H,A_raw)

    = K decoder rollouts (fixed NFE) + inverse action transform. It emits the K
    candidate trajectories and stops there: the readout (a single draw or the
    winner-take-all consensus in readout.py) runs host-side as postprocessing,
    deliberately OUTSIDE the graph — its data-dependent control flow (sort /
    nonzero / per-frame loops) doesn't belong in a static graph, and keeping it
    host-side lets it be retuned without re-exporting. Noise is an INPUT (not
    sampled internally) so the graph is a pure function of its inputs.

    Everything here lowers to ONNX: the sampler loop unrolls at the fixed step
    count, and the inverse transform is searchsorted-free (ndtr -> Erf, uniform-
    grid lookup). Build via PolicyObjective.exportable().
    """

    def __init__(
        self, decoder: FlowActionDecoder, action_transform: ActionTransform | None
    ) -> None:
        super().__init__()
        self.decoder = decoder
        self.action_transform = action_transform

    @override
    def forward(self, condition_tokens: Tensor, noise: Tensor) -> Tensor:
        b, k, h, a = noise.shape
        cond = condition_tokens.repeat_interleave(k, dim=0)
        traj = self.decoder.sample(
            condition_tokens=cond, noise=noise.reshape(b * k, h, a)
        )
        raw = (
            traj
            if self.action_transform is None
            else self.action_transform.inverse(traj)
        )
        return raw.reshape(b, k, h, raw.shape[-1])
