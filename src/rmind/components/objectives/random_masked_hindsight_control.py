from collections.abc import Callable
from collections.abc import Set as AbstractSet
from functools import lru_cache
from typing import final, override

import numpy as np
import torch
from einops.layers.torch import Rearrange
from pydantic import InstanceOf, validate_call
from tensordict import TensorDict
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F
from torch.utils._pytree import tree_map  # noqa: PLC2701

from rmind.components.containers import ModuleDict
from rmind.components.episode import (
    Episode,
    Index,
    Modality,
    SummaryToken,
    Timestep,
    TokenType,
)
from rmind.components.mask import (
    AttentionMask,
    AttentionMaskLegend,
    TorchAttentionMaskLegend,
)
from rmind.components.objectives.base import (
    Metrics,
    Objective,
    Prediction,
    PredictionKey,
    Targets,
)


@final
class RandomMaskedHindsightControlObjective(Objective):
    MASKED_TOKEN_FILL_VALUE = -1.0

    @validate_call
    def __init__(
        self,
        *,
        encoder: InstanceOf[Module] | None = None,
        heads: InstanceOf[ModuleDict],
        losses: InstanceOf[ModuleDict] | None = None,
        targets: Targets | None = None,
    ) -> None:
        super().__init__()

        self.encoder: Module | None = encoder
        self.heads = heads
        self.losses: ModuleDict | None = losses
        self.targets: Targets | None = targets

        self._build_attention_mask: Callable[..., AttentionMask] = lru_cache(
            maxsize=2, typed=True
        )(self.build_attention_mask)

    @override
    def compute_metrics(
        self, episode: Episode, final_embedding_norm: InstanceOf[Module]
    ) -> Metrics:
        episode, mask_action_timestep = self._mask_episode(episode)
        mask = self._build_attention_mask(
            episode.index, episode.timestep, legend=TorchAttentionMaskLegend
        )

        embedding = self.encoder(
            src=final_embedding_norm(episode.embeddings_packed),
            mask=mask.mask.to(episode.device),
        )  # ty:ignore[call-non-callable]

        keys_action = episode.timestep.get(TokenType.ACTION).keys(
            include_nested=True, leaves_only=True
        )
        index_action = episode.index.select(*keys_action)
        embeddings = index_action[mask_action_timestep].parse(embedding)
        logits = self.heads(embeddings.to_dict())

        targets = tree_map(
            lambda k: episode.get(k)[:, mask_action_timestep],
            self.targets,
            is_leaf=lambda x: isinstance(x, tuple),
        )

        losses = self.losses(
            tree_map(Rearrange("b t 1 d -> (b t 1) d"), logits),
            tree_map(Rearrange("b t 1 -> (b t)"), targets),
        )  # ty:ignore[call-non-callable]

        return {"loss": losses}

    @override
    def predict(
        self,
        episode: Episode,
        *,
        keys: AbstractSet[PredictionKey],
        tokenizers: ModuleDict | None = None,
    ) -> TensorDict:
        predictions: dict[PredictionKey, Prediction] = {}
        b, _t = episode.input.batch_size

        if (key := PredictionKey.GROUND_TRUTH) in keys:
            predictions[key] = Prediction(
                value=episode.input.select(*self.heads.tree_paths()),
                timestep_indices=slice(None),
            )

        if keys & {
            PredictionKey.PREDICTION_VALUE,
            PredictionKey.PREDICTION_PROBS,
            PredictionKey.SCORE_LOGPROB,
            PredictionKey.SCORE_L1,
            PredictionKey.SUMMARY_EMBEDDINGS,
        }:
            episode, timestep_indices = self._mask_episode(episode)
            mask = self._build_attention_mask(
                episode.index, episode.timestep, legend=TorchAttentionMaskLegend
            )

            embedding = self.encoder(
                src=episode.embeddings_packed, mask=mask.mask.to(episode.device)
            )  # ty:ignore[call-non-callable]

            keys_action = episode.timestep.get(TokenType.ACTION).keys(
                include_nested=True, leaves_only=True
            )
            index_action = episode.index.select(*keys_action)
            embeddings = index_action[timestep_indices].parse(embedding)

            logits = TensorDict(
                self.heads(embeddings.to_dict()), batch_size=[b, len(timestep_indices)]
            )

            if (key := PredictionKey.PREDICTION_VALUE) in keys:
                predictions[key] = Prediction(
                    value=logits.apply(lambda x: x.argmax(dim=-1)).named_apply(  # ty:ignore[possibly-missing-attribute]
                        lambda k, v: tokenizers.get_deepest(k).invert(v),  # ty:ignore[call-non-callable, possibly-missing-attribute]
                        nested_keys=True,
                    ),
                    timestep_indices=timestep_indices,
                )

            if (key := PredictionKey.PREDICTION_PROBS) in keys:
                predictions[key] = Prediction(
                    value=logits.apply(lambda x: x.softmax(dim=-1)),
                    timestep_indices=timestep_indices,
                )

            if (key := PredictionKey.SCORE_LOGPROB) in keys:
                predictions[key] = Prediction(
                    value=(
                        logits
                        .apply(lambda x: x.softmax(dim=-1))
                        .apply(Rearrange("b t 1 d -> b t d"))  # ty:ignore[possibly-missing-attribute]
                        .apply(  # ty:ignore[possibly-missing-attribute]
                            lambda probs, tokens: probs.gather(dim=-1, index=tokens),
                            episode.input_tokens[:, timestep_indices],  # ty:ignore[invalid-argument-type]
                        )
                        .apply(lambda x: -torch.log(x))  # ty:ignore[possibly-missing-attribute]
                    ),
                    timestep_indices=timestep_indices,
                )

            if (key := PredictionKey.SCORE_L1) in keys:
                predictions[key] = Prediction(
                    value=(
                        logits
                        .apply(lambda x: x.argmax(dim=-1))
                        .named_apply(  # ty:ignore[possibly-missing-attribute]
                            lambda k, v: tokenizers.get_deepest(k).invert(v),  # ty:ignore[possibly-missing-attribute, call-non-callable]
                            nested_keys=True,
                        )
                        .apply(  # ty:ignore[possibly-missing-attribute]
                            lambda pred, gt: F.l1_loss(pred, gt, reduction="none"),
                            episode.input[:, timestep_indices],  # ty:ignore[invalid-argument-type]
                            nested_keys=True,
                        )
                    ),
                    timestep_indices=timestep_indices,
                )

            if (key := PredictionKey.SUMMARY_EMBEDDINGS) in keys:
                predictions[key] = episode.index.select(Modality.SUMMARY)[[-1]].parse(
                    embedding
                )

        return TensorDict(predictions).auto_batch_size_(2)  # ty:ignore[invalid-argument-type]

    @classmethod
    def _mask_episode(cls, episode: Episode) -> tuple[Episode, Tensor]:
        episode = episode.clone(recurse=True)
        b, t = episode.input.batch_size
        device = episode.device

        mask = torch.zeros((b, t), dtype=torch.bool, device=device)

        mask_action = mask.clone()
        mask_action_timestep = torch.from_numpy(
            np.random.choice(t, 2, replace=False)  # noqa: NPY002
        )
        mask_action[:, mask_action_timestep] = True
        keys_action = episode.timestep.get(TokenType.ACTION).keys(
            include_nested=True, leaves_only=True
        )
        episode.projected_embeddings.select(*keys_action).masked_fill_(
            mask_action, cls.MASKED_TOKEN_FILL_VALUE
        )

        mask_observation = mask.clone()
        mask_observation_timestep = torch.from_numpy(
            np.random.choice(t, 1, replace=False)  # noqa: NPY002
        )
        mask_observation[:, mask_observation_timestep] = True
        keys_observation = episode.timestep.get(TokenType.OBSERVATION).keys(
            include_nested=True, leaves_only=True
        )
        episode.projected_embeddings.select(*keys_observation).masked_fill_(
            mask_observation, cls.MASKED_TOKEN_FILL_VALUE
        )

        return episode, mask_action_timestep

    @classmethod
    def build_attention_mask(
        cls, index: Index, timestep: Timestep, *, legend: AttentionMaskLegend
    ) -> AttentionMask:
        """Build bidirectional attention mask with action isolation.

        Unlike other objectives, this uses BIDIRECTIONAL attention (can see future),
        which is appropriate for hindsight control where we mask some actions and
        predict them from surrounding context (including future observations).

        The only restriction is ACTION ISOLATION: actions at any timestep cannot
        attend to actions at other timesteps. This prevents actions from copying
        each other during the masked prediction task.

        Attention pattern:
        - Observations can attend to everything (past, current, future)
        - Actions can only attend to observations and summaries, NOT other actions
        - This enables learning action prediction from full trajectory context
          while preventing trivial action-to-action copying
        """
        length: int = index.max(reduce=True).item() + 1
        mask = AttentionMask(
            mask=torch.full((length, length), legend.DO_ATTEND.value),
            legend=legend,
            device="cpu",
        )

        (t,) = index.batch_size
        action_keys = timestep.get(TokenType.ACTION).keys(
            include_nested=True, leaves_only=True
        )

        for step in range(t):
            past, current, future = index[:step], index[step], index[step + 1 :]
            current_actions = current.select(*action_keys)
            current_action_summary = current.select((
                Modality.SUMMARY,
                SummaryToken.ACTION_SUMMARY,
            ))
            past_actions = past.select(*action_keys)
            past_action_summary = past.select((
                Modality.SUMMARY,
                SummaryToken.ACTION_SUMMARY,
            ))
            future_actions = future.select(*action_keys)
            future_action_summary = future.select((
                Modality.SUMMARY,
                SummaryToken.ACTION_SUMMARY,
            ))

            # Action isolation: actions cannot see other actions across timesteps
            mask = (
                mask
                .do_not_attend(current_actions, past_actions)
                .do_not_attend(current_actions, past_action_summary)
                .do_not_attend(current_actions, future_actions)
                .do_not_attend(current_actions, future_action_summary)
                .do_not_attend(current_action_summary, past_actions)
                .do_not_attend(current_action_summary, past_action_summary)
                .do_not_attend(current_action_summary, future_actions)
                .do_not_attend(current_action_summary, future_action_summary)
            )

        return mask
