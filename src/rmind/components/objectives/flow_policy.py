from collections.abc import Set as AbstractSet
from typing import Any, final, override

import torch
from pydantic import InstanceOf, validate_call
from structlog import get_logger
from tensordict import TensorDict
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F

from rmind.components.base import Modality, SummaryToken
from rmind.components.containers import ModuleDict
from rmind.components.episode import Episode
from rmind.components.objectives.action_transform import GaussianizeActionTransform
from rmind.components.objectives.consensus import mode_aware_anchor
from rmind.components.objectives.maneuver_weights import ManeuverLossWeights
from rmind.components.objectives.base import (
    Metrics,
    Objective,
    ObjectivePredictionKey,
    Prediction,
)
from rmind.components.transformer import FlowActionDecoder

logger = get_logger(__name__)

POLICY_CONDITION_TOKENS: tuple[tuple[Modality, str], ...] = (
    (Modality.SUMMARY, SummaryToken.OBSERVATION_SUMMARY),
    (Modality.SUMMARY, SummaryToken.OBSERVATION_HISTORY),
    # Route intent: the baseline (PolicyObjective) conditions on waypoints; the
    # cross-attention decoder takes the waypoint tokens directly (no mean-pool).
    (Modality.CONTEXT, "waypoints"),
    # Raw scene: 256 encoded image patch tokens. Tested on the overfit and
    # found informationally redundant with the summaries (null result: same
    # floor, decoder substitutes one source for the other — see
    # flow_action_expert_improvements.md). Disabled for now; ckpts trained
    # with it (e.g. model-tfuv76yx) need `+fan.legacy_condition=false`-style
    # care: conditioning is baked into code, not the checkpoint.
    # (Modality.IMAGE, "cam_front_left"),
)

DEFAULT_ACTION_KEYS = ("gas_pedal", "brake_pedal", "steering_angle")
DEFAULT_BATCH_ACTION_PATHS: tuple[tuple[str, ...], ...] = (
    ("data", "meta/VehicleMotion/gas_pedal_normalized_target"),
    ("data", "meta/VehicleMotion/brake_pedal_normalized_target"),
    ("data", "meta/VehicleMotion/steering_angle_normalized_target"),
)
FLOW_TIME_LOGIT_NORMAL_MEAN = 0.0
FLOW_TIME_LOGIT_NORMAL_STD = 1.0
# pi0's time-step distribution defaults (arXiv:2410.24164): p(t) =
# Beta((s - t)/s; alpha, 1), which biases mass toward t -> 0 (the noisy end)
# for alpha > 1. See sample_flow_time for the (generator-safe) sampler.
FLOW_TIME_BETA_ALPHA = 1.5
FLOW_TIME_BETA_S = 0.999
# Number of equal-width flow-time buckets for the per-t flow-MSE diagnostic
# logged in validation (flow_mse_t<edge>): shows WHERE on the trajectory the
# field is least accurate, so p(t) can be shaped to match instead of guessed.
FLOW_TIME_LOSS_BUCKETS = 8
FLOW_TIME_SAMPLING_METHODS = {"uniform", "logit-normal", "beta"}
# predict()-time readout policies: "single" = one honest draw (legacy);
# "meank" = mean of predict_samples draws (variance reduction, mode-averaging);
# "mode" = winner-take-all consensus (see consensus.py — commits to the
# dominant mode; == meank on unimodal frames, the deployment recommendation).
PREDICT_READOUTS = {"single", "meank", "mode"}


@final
class FlowPolicyObjective(Objective):
    @validate_call
    def __init__(  # noqa: PLR0913
        self,
        *,
        decoder: InstanceOf[FlowActionDecoder],
        loss: InstanceOf[Module],
        history_steps: int = 1,
        action_keys: tuple[str, ...] = DEFAULT_ACTION_KEYS,
        batch_action_paths: tuple[tuple[str, ...], ...] = DEFAULT_BATCH_ACTION_PATHS,
        flow_time_sampling: str = "uniform",
        flow_time_logit_mean: float = FLOW_TIME_LOGIT_NORMAL_MEAN,
        flow_time_logit_std: float = FLOW_TIME_LOGIT_NORMAL_STD,
        flow_time_beta_alpha: float = FLOW_TIME_BETA_ALPHA,
        flow_time_beta_s: float = FLOW_TIME_BETA_S,
        validation_seed: int = 0,
        validation_sample_metrics: bool = True,
        prediction_samples: int = 1,
        chunk_delta_weight: float = 0.0,
        maneuver_thresholds: tuple[float, ...] | None = None,
        action_transform_stats: str | None = None,
        lds_stats: str | None = None,
        lds_alpha: float = 0.5,
        lds_cap: float = 15.0,
        waypoint_pe: bool = False,
        waypoint_count: int = 10,
        predict_readout: str = "single",
        predict_samples: int = 16,
    ) -> None:
        super().__init__()

        if predict_readout not in PREDICT_READOUTS:
            msg = f"predict_readout must be one of {PREDICT_READOUTS}, got {predict_readout!r}"
            raise ValueError(msg)
        if predict_samples <= 0:
            msg = f"predict_samples must be positive, got {predict_samples}"
            raise ValueError(msg)
        self.predict_readout = predict_readout
        self.predict_samples = predict_samples

        if history_steps <= 0:
            msg = f"history_steps must be positive, got {history_steps}"
            raise ValueError(msg)

        # Build the action transform first: it defines raw_dim (I/O channel
        # count, e.g. gas/brake/steering = 3) and model_dim (what the decoder /
        # flow operate on, e.g. 2 after the gas/brake merge). Identity (None) =>
        # raw_dim == model_dim == decoder.action_dim (fully backward compatible).
        if not action_transform_stats:  # None or "" (config default) => identity
            action_transform = None
            raw_dim = decoder.action_dim
        else:
            action_transform = GaussianizeActionTransform.from_stats_file(
                action_transform_stats
            )
            if action_transform.action_keys != action_keys:
                msg = (
                    "action_transform stats action_keys "
                    f"{action_transform.action_keys} != objective action_keys {action_keys}"
                )
                raise ValueError(msg)
            if action_transform.model_dim != decoder.action_dim:
                msg = (
                    f"action_transform.model_dim {action_transform.model_dim} != "
                    f"decoder.action_dim {decoder.action_dim} (set flow_action_dim to match)"
                )
                raise ValueError(msg)
            raw_dim = action_transform.raw_dim

        if len(action_keys) != raw_dim:
            msg = f"action_keys must have {raw_dim} entries (raw_dim), got {len(action_keys)}"
            raise ValueError(msg)

        if len(batch_action_paths) != raw_dim:
            msg = (
                f"batch_action_paths must have {raw_dim} entries (raw_dim), "
                f"got {len(batch_action_paths)}"
            )
            raise ValueError(msg)

        if flow_time_sampling not in FLOW_TIME_SAMPLING_METHODS:
            msg = f"Invalid flow_time_sampling method: {flow_time_sampling}"
            raise ValueError(msg)

        if flow_time_logit_std < 0:
            msg = f"flow_time_logit_std must be non-negative, got {flow_time_logit_std}"
            raise ValueError(msg)

        if flow_time_beta_alpha <= 0:
            msg = f"flow_time_beta_alpha must be positive, got {flow_time_beta_alpha}"
            raise ValueError(msg)

        if not 0.0 < flow_time_beta_s <= 1.0:
            msg = f"flow_time_beta_s must be in (0, 1], got {flow_time_beta_s}"
            raise ValueError(msg)

        if prediction_samples <= 0:
            msg = f"prediction_samples must be positive, got {prediction_samples}"
            raise ValueError(msg)

        if chunk_delta_weight < 0:
            msg = f"chunk_delta_weight must be non-negative, got {chunk_delta_weight}"
            raise ValueError(msg)

        if maneuver_thresholds is not None and len(maneuver_thresholds) != raw_dim:
            msg = (
                "maneuver_thresholds must match raw action channels, got "
                f"{len(maneuver_thresholds)} for raw_dim={raw_dim}"
            )
            raise ValueError(msg)

        self.decoder = decoder
        self.prediction_samples = prediction_samples
        self.history_steps = history_steps
        self.action_keys = action_keys
        self.batch_action_paths = batch_action_paths
        self.flow_time_sampling = flow_time_sampling
        self.flow_time_logit_mean = flow_time_logit_mean
        self.flow_time_logit_std = flow_time_logit_std
        self.flow_time_beta_alpha = flow_time_beta_alpha
        self.flow_time_beta_s = flow_time_beta_s
        self.validation_seed = validation_seed
        self.validation_sample_metrics = validation_sample_metrics
        # The within-chunk delta loss differences adjacent horizon slots, which
        # is undefined for a single-slot chunk: diff over a length-1 axis is
        # empty and mse_loss(empty, empty) = NaN. Disable it for
        # action_horizon < 2 (there is no within-chunk structure to shape).
        if chunk_delta_weight > 0 and decoder.action_horizon < 2:
            logger.warning(
                "disabling chunk_delta_weight: needs action_horizon >= 2",
                action_horizon=decoder.action_horizon,
                requested_weight=chunk_delta_weight,
            )
            chunk_delta_weight = 0.0
        self.chunk_delta_weight = chunk_delta_weight
        self.loss: Module = loss
        # Reference copy of the slot position embedding for the pe_drift
        # metric — a direct readout of whether the model is learning slot
        # identity (untrained PEs were the smoking gun of the constant-chunk
        # failure). Non-persistent: absent from checkpoints (old ckpts load
        # unchanged); for artifact-loaded models drift is measured from load.
        self.register_buffer(
            "position_embedding_reference",
            decoder.position_embedding.weight.detach().clone(),
            persistent=False,
        )
        # Per-channel maneuver-L1 thresholds (raw action space, abs). gas/brake/
        # steering have very different scales (e.g. brake ~always 0, gas bounded
        # well below steering's ±1), so one threshold for all flags almost no
        # gas/brake frames — derive per channel from the data (flow_action_
        # thresholds.py). None => the maneuver-L1 metric is off. Non-persistent.
        if maneuver_thresholds is None:
            self.maneuver_thresholds = None
        else:
            self.register_buffer(
                "maneuver_thresholds",
                torch.tensor(maneuver_thresholds, dtype=torch.float32),
                persistent=False,
            )
        # Invertible action transform (gas/brake merge + Gaussianize), built and
        # validated above. Train the flow in model space; invert samples back to
        # raw before any reported metric. None => identity. Registered as a
        # submodule so its (non-persistent) knot buffers move with .to(device).
        self.action_transform = action_transform

        # Per-chunk maneuver loss weighting (LDS): reweight the flow loss from
        # frequency toward importance (rare high-intensity maneuver chunks get
        # more gradient; cruise ~1). Off when lds_stats is empty/None. The
        # weight channels must match the model (loss) space, so validate against
        # the transform's model_dim (or raw decoder dim when no transform).
        if not lds_stats:  # None or "" (config default) => off
            self.maneuver_weights = None
        else:
            self.maneuver_weights = ManeuverLossWeights.from_stats_file(
                lds_stats, alpha=lds_alpha, cap=lds_cap
            )
            if self.maneuver_weights.model_dim != decoder.action_dim:
                msg = (
                    f"lds model_dim {self.maneuver_weights.model_dim} != "
                    f"decoder.action_dim {decoder.action_dim}"
                )
                raise ValueError(msg)
            model_keys = (
                action_transform.model_action_keys
                if action_transform is not None
                else action_keys
            )
            if self.maneuver_weights.model_keys != tuple(model_keys):
                msg = (
                    f"lds model_keys {self.maneuver_weights.model_keys} != "
                    f"model-space keys {tuple(model_keys)}"
                )
                raise ValueError(msg)

        # Waypoint position embedding: the encoder is FACTORIZED (RoPE on the
        # temporal axis only) and the within-timestep spatial attention has no
        # positional encoding, so the route's waypoint SEQUENCE is invisible —
        # the decoder cross-attends to the 10 waypoint tokens as an unordered
        # bag. Tag each encoder-output waypoint token with a learned per-index
        # embedding so the decoder can use route order. Added HERE (not in the
        # encoder) because the episode builder + encoder are frozen during flow
        # finetune; this lives in the trainable objective. Zero-init => identity
        # at start (no-op until the loss learns the offsets); nn.Embedding so
        # SelectiveAdamW weight-decay-excludes it, like the decoder's slot PE.
        if not waypoint_pe:
            self.waypoint_pos_embedding = None
        else:
            cond_dim = (
                decoder.condition_projection.in_features
                if isinstance(decoder.condition_projection, torch.nn.Linear)
                else decoder.action_projection.out_features
            )
            self.waypoint_pos_embedding = torch.nn.Embedding(waypoint_count, cond_dim)
            torch.nn.init.zeros_(self.waypoint_pos_embedding.weight)

    @override
    def forward(self, *, episode: Episode, embedding: Tensor) -> TensorDict:
        condition_tokens = self._condition_tokens(episode=episode, embedding=embedding)
        return self._trajectory_to_tensordict(
            self._to_raw_space(self.decoder.sample(condition_tokens=condition_tokens))
        )

    @override
    def compute_metrics(
        self, *, episode: Episode, embedding: Tensor, batch: Any, **_kwargs: Any
    ) -> Metrics:
        # Linear interpolant / action-flow target follows Flow Matching:
        # https://arxiv.org/abs/2210.02747
        condition_tokens = self._condition_tokens(episode=episode, embedding=embedding)
        target_actions = self._target_actions(batch)

        # Drop non-finite samples before any flow compute. A NaN in a condition
        # input (e.g. waypoints — not range-filtered by the dataset query, only
        # speed/gas/brake/steering are) propagates through the frozen encoder to
        # NaN condition_tokens, then to a NaN loss for the whole batch. Slicing
        # here is field-agnostic and safe: encoder/decoder run per-sample, so a
        # bad row never poisons its neighbours. On the single-drive overfit such
        # frames are no longer diluted, so this guard is load-bearing.
        finite = (
            condition_tokens.flatten(1).isfinite().all(dim=1)
            & target_actions.flatten(1).isfinite().all(dim=1)
        )
        if not bool(finite.all()):
            # Attribute the non-finiteness to a specific input so the culprit
            # field is named without disabling the guard (a NaN swallowed here
            # never reaches RMIND_NAN_DEBUG). Cheap: only runs when a row drops.
            culprits: dict[str, int] = {}
            parsed = (
                episode.index[self.history_steps - 1]
                .select(*POLICY_CONDITION_TOKENS)
                .parse(embedding)
            )
            for path in POLICY_CONDITION_TOKENS:
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
            if bool(finite.any()):
                condition_tokens = condition_tokens[finite]
                target_actions = target_actions[finite]

        # The flow operates in transformed (model) space; raw targets are kept
        # for raw-space metrics. With no transform, model space == raw space.
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
            logit_mean=self.flow_time_logit_mean,
            logit_std=self.flow_time_logit_std,
            beta_alpha=self.flow_time_beta_alpha,
            beta_s=self.flow_time_beta_s,
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

        # Per-chunk maneuver weight (LDS): (B, 1, model_dim) broadcast over
        # slots, or None (uniform). Depends only on the clean target, so it is
        # valid at every flow-time and weights both the flow and delta terms.
        weight = self._maneuver_weight(target_actions)

        # flow_mse is ALWAYS the unweighted mean (stable curve across runs, and
        # comparable to the unweighted baselines); `loss` is the weighted +
        # delta-augmented training objective. With LDS off and delta off,
        # loss == flow_mse.
        flow_mse = self.loss(predicted_action_flow, target_action_flow)
        metrics: Metrics = {"flow_mse": flow_mse.detach()}
        if weight is None:
            flow_term = flow_mse
        else:
            flow_se = (predicted_action_flow - target_action_flow).pow(2)
            flow_term = (weight * flow_se).mean()
            metrics["maneuver_weight_mean"] = weight.mean().detach()
        metrics["loss"] = flow_term
        if self.chunk_delta_weight > 0:
            # Within-chunk delta loss: match the chunk's internal shape, not
            # just its level. The plain flow MSE is dominated by the shared
            # (DC) component, whose optimum is a constant chunk at the
            # mid-horizon value — leaving the slot position embeddings
            # untrained (measured: PEs at init after 100 epochs, chunk slope
            # ratio vs GT = 0.00). Differencing adjacent slots removes the DC
            # entirely: this term is zero on flat segments and pure phase
            # signal at maneuvers, and no constant chunk can satisfy it.
            # Implied clean-action estimate from the linear interpolant:
            # x1_hat = x_t + (1 - t) * v_hat  (exact when v_hat = x1 - x0).
            implied_actions = (
                noised_actions + (1.0 - flow_time_broadcast) * predicted_action_flow
            )
            delta_pred = implied_actions.diff(dim=1)
            delta_target = target_actions_model.diff(dim=1)
            if weight is None:
                chunk_delta_mse = torch.nn.functional.mse_loss(delta_pred, delta_target)
            else:
                # Same per-chunk weight (broadcast over the H-1 slot-diffs) so
                # the two loss terms stay on a consistent importance scale.
                chunk_delta_mse = (weight * (delta_pred - delta_target).pow(2)).mean()
            metrics["chunk_delta_mse"] = chunk_delta_mse.detach()
            metrics["loss"] = flow_term + self.chunk_delta_weight * chunk_delta_mse
        if not self.training:
            # Relative drift of the slot position embeddings from their
            # reference (init for fresh runs): ~0 means the model is not
            # learning slot identity and the chunk stays a constant.
            pe = self.decoder.position_embedding.weight
            reference = self.position_embedding_reference
            metrics["pe_drift"] = (
                (pe - reference).norm() / reference.norm().clamp_min(1e-12)
            ).detach()
            # Per-t flow-MSE: bucket the per-sample flow loss by its flow_time so
            # the val curve shows WHERE the field is least accurate, rather than
            # one t-averaged scalar. Read against the t-sampling density: a bucket
            # that stays high is under-sampled (or intrinsically hard) and is a
            # candidate to up-weight in p(t). Raw squared error (not self.loss) so
            # the diagnostic is independent of the training reduction. Empty
            # buckets are skipped (the t-distribution may not cover all bins).
            per_sample_se = (
                (predicted_action_flow - target_action_flow)
                .pow(2)
                .flatten(start_dim=1)
                .mean(dim=1)
            )
            bucket = (flow_time * FLOW_TIME_LOSS_BUCKETS).long().clamp_(
                max=FLOW_TIME_LOSS_BUCKETS - 1
            )
            for i in range(FLOW_TIME_LOSS_BUCKETS):
                in_bucket = bucket == i
                if bool(in_bucket.any()):
                    hi = (i + 1) / FLOW_TIME_LOSS_BUCKETS
                    metrics[f"flow_mse_t{hi:.2f}"] = per_sample_se[in_bucket].mean().detach()
        if not self.training and self.validation_sample_metrics:
            # One honest draw per frame (deployment-realistic: at inference we
            # take a single sample). Draw in model space, then invert to raw —
            # all reported L1/maneuver metrics are raw-space (flow-space loss is
            # not comparable across transforms; raw-space numbers decide). The
            # best-of-N curve was removed: a single draw is the number that
            # matters, and bo metrics flattered the model with oracle selection.
            samples = self._to_raw_space(
                self._draw_samples(
                    condition_tokens=condition_tokens, generator=generator
                )
            )  # (1, B, H, A) raw
            target = target_actions.to(dtype=samples.dtype).unsqueeze(0)
            per_sample_l1 = (samples - target).abs().flatten(start_dim=2).mean(dim=2)
            metrics["sample_l1"] = per_sample_l1[0].mean()

            if self.maneuver_thresholds is not None:
                # Maneuver-L1: the honest single-draw L1 restricted to "active"
                # frames per channel — |GT_c| above a per-channel, data-derived
                # threshold. drive-wide L1 is dominated by flat/cruise frames and
                # hides exactly the maneuvers that matter; this is the scoreboard
                # that rewards committing to them vs the mean-reverting baseline.
                # maneuver_frac/<key> reports the flagged fraction (sanity-check
                # the threshold). Computed on draw 0, matching sample_l1.
                honest = samples[0]  # (B, H, A) single honest draw
                tgt = target[0]  # (B, H, A) GT in samples dtype
                abs_err = (honest - tgt).abs()
                thr = self.maneuver_thresholds.to(tgt.dtype)  # (A,)
                active = tgt.abs() > thr  # (B, H, A)
                for c, key in enumerate(self.action_keys):
                    frames = active[..., c]
                    metrics[f"maneuver_frac/{key}"] = frames.float().mean().detach()
                    if bool(frames.any()):
                        metrics[f"maneuver_l1/{key}"] = (
                            abs_err[..., c][frames].mean().detach()
                        )

        return metrics

    @override
    def predict(
        self,
        *,
        episode: Episode,
        embedding: Tensor,
        keys: AbstractSet[ObjectivePredictionKey],
        tokenizers: ModuleDict | None = None,
        batch: Any,
        **_kwargs: Any,
    ) -> TensorDict:
        del tokenizers

        predictions: dict[ObjectivePredictionKey, Prediction] = {}
        target_actions: Tensor | None = None
        prediction_actions: Tensor | None = None

        if (key := ObjectivePredictionKey.GROUND_TRUTH) in keys:
            target_actions = self._target_actions(batch)
            predictions[key] = Prediction(
                value=self._trajectory_to_tensordict(target_actions),
                timestep_indices=self._target_slice(),
            )

        if keys & {
            ObjectivePredictionKey.PREDICTION_VALUE,
            ObjectivePredictionKey.SCORE_L1,
        }:
            condition_tokens = self._condition_tokens(
                episode=episode, embedding=embedding
            )
            # Readout policy (predict_readout):
            #   single — one honest draw, matching the sample_l1 metric.
            #   meank  — mean of predict_samples draws (variance reduction;
            #            mode-AVERAGING: splits the difference on bimodal frames).
            #   mode   — winner-take-all consensus (consensus.py): commit to the
            #            dominant cluster; == meank on unimodal frames. Measured
            #            held-out spike steering L1: single 0.62 / meank 0.53 /
            #            mode 0.34 (77% of held-out spike frames are bimodal).
            # All use the global RNG (generator=None): re-seeding per batch makes
            # noise a function of within-batch position, which imprints a
            # period-`batch_size` artifact on plotted predictions.
            if self.predict_readout == "single":
                prediction_actions = self._to_raw_space(
                    self.decoder.sample(
                        condition_tokens=condition_tokens,
                        noise=self._noise(
                            condition_tokens=condition_tokens, generator=None
                        ),
                    )
                )
            else:
                # One BATCHED decoder pass over all K draws (B*K), like the fan/
                # meank scripts — the encoder ran once upstream; only the small
                # flow decoder sees the K-fold batch, so wall-clock barely grows.
                k = self.predict_samples
                cond_rep = condition_tokens.repeat_interleave(k, dim=0)
                draws = self._to_raw_space(
                    self.decoder.sample(
                        condition_tokens=cond_rep,
                        noise=self._noise(
                            condition_tokens=cond_rep, generator=None
                        ),
                    ).reshape(
                        condition_tokens.shape[0],
                        k,
                        self.decoder.action_horizon,
                        self.decoder.action_dim,
                    )
                )  # (B, K, H, A) raw
                if self.predict_readout == "meank":
                    prediction_actions = draws.mean(dim=1)
                else:
                    steer_idx = next(
                        (
                            i
                            for i, key_ in enumerate(self.action_keys)
                            if "steering" in key_
                        ),
                        len(self.action_keys) - 1,
                    )
                    prediction_actions, _, _ = mode_aware_anchor(draws, steer_idx)

        if (key := ObjectivePredictionKey.PREDICTION_VALUE) in keys:
            if prediction_actions is None:
                msg = "prediction_actions were not computed"
                raise RuntimeError(msg)

            predictions[key] = Prediction(
                value=self._trajectory_to_tensordict(prediction_actions),
                timestep_indices=self._target_slice(),
            )

        if (key := ObjectivePredictionKey.SCORE_L1) in keys:
            if target_actions is None:
                target_actions = self._target_actions(batch)

            if prediction_actions is None:
                msg = "prediction_actions were not computed"
                raise RuntimeError(msg)

            predictions[key] = Prediction(
                value=self._trajectory_to_tensordict(
                    F.l1_loss(prediction_actions, target_actions, reduction="none")
                ),
                timestep_indices=self._target_slice(),
            )

        return TensorDict(predictions).auto_batch_size_(2)  # ty:ignore[invalid-argument-type]

    def _draw_samples(
        self, *, condition_tokens: Tensor, generator: torch.Generator | None
    ) -> Tensor:
        # N seeded trajectories: (N, B, H, A).
        return torch.stack(
            [
                self.decoder.sample(
                    condition_tokens=condition_tokens,
                    noise=self._noise(
                        condition_tokens=condition_tokens, generator=generator
                    ),
                )
                for _ in range(self.prediction_samples)
            ],
            dim=0,
        )

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
            .select(*POLICY_CONDITION_TOKENS)
            .parse(embedding)
        )
        parts = [embeddings.get(path) for path in POLICY_CONDITION_TOKENS]
        if self.waypoint_pos_embedding is not None:
            # Add the per-index waypoint PE to the waypoint token block only.
            wp_i = POLICY_CONDITION_TOKENS.index((Modality.CONTEXT, "waypoints"))
            wp = parts[wp_i]  # (B, n_waypoints, dim)
            n = wp.shape[1]
            if n != self.waypoint_pos_embedding.num_embeddings:
                msg = (
                    f"waypoint_count {self.waypoint_pos_embedding.num_embeddings} != "
                    f"actual waypoint tokens {n}"
                )
                raise ValueError(msg)
            pos = torch.arange(n, device=wp.device)
            parts[wp_i] = wp + self.waypoint_pos_embedding(pos).to(wp.dtype)
        return torch.cat(parts, dim=1)

    def _target_actions(self, batch: Any) -> Tensor:
        actions: list[Tensor] = []
        for path in self.batch_action_paths:
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

    def _maneuver_weight(self, target_actions: Tensor) -> Tensor | None:
        """Per-chunk LDS weight (B, 1, model_dim) broadcastable over slots, or
        None when LDS is off.

        The label is the peak |physical action| over the horizon per model
        channel (longitudinal, steering) — peak-over-slots so a chunk's lead-in
        is upweighted with its maneuver. Physical (pre-Gaussianize) space, since
        the bins are fit there.
        """
        if self.maneuver_weights is None:
            return None
        phys = (
            self.action_transform.physical_model(target_actions)
            if self.action_transform is not None
            else target_actions
        )
        label = phys.abs().amax(dim=1)  # (B, model_dim)
        return self.maneuver_weights(label).unsqueeze(1)  # (B, 1, model_dim)

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
    def sample_flow_time(
        flow_time_sampling: str,
        batch_size: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | None = None,
        logit_mean: float = FLOW_TIME_LOGIT_NORMAL_MEAN,
        logit_std: float = FLOW_TIME_LOGIT_NORMAL_STD,
        beta_alpha: float = FLOW_TIME_BETA_ALPHA,
        beta_s: float = FLOW_TIME_BETA_S,
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
            case "beta":
                # pi0's time-step distribution (arXiv:2410.24164):
                #   p(t) = Beta((s - t)/s; alpha, 1)
                # i.e. w = (s - t)/s ~ Beta(alpha, 1), so t = s * (1 - w). The
                # rationale (pi0 sec. on flow matching): action prediction is
                # most constrained / hardest at high noise, so concentrate
                # supervision near t=0. alpha > 1 skews mass toward t=0; alpha=1
                # is uniform on [0, s]. The second Beta parameter is fixed to 1
                # because Beta(alpha, 1) has CDF w^alpha, making inverse-transform
                # sampling w = U^(1/alpha) exact AND generator-aware
                # (torch.distributions.Beta ignores `generator`, which would
                # break the fixed-seed validation determinism).
                u = torch.rand(
                    batch_size, dtype=dtype, device=device, generator=generator
                )
                w = u.pow(1.0 / beta_alpha)
                return beta_s * (1.0 - w)
            case _:
                msg_0 = f"Invalid flow_time_sampling method: {flow_time_sampling}"
                raise ValueError(msg_0)

    def _validation_generator(self, *, device: torch.device) -> torch.Generator | None:
        if self.training:
            return None

        generator = torch.Generator(device=device)
        generator.manual_seed(self.validation_seed)
        return generator
