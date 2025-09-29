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
    SpecialToken,
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
    PredictionResultKey,
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

        self.encoder = encoder
        self.heads = heads
        self.losses = losses
        self.targets = targets

        self._build_attention_mask = lru_cache(maxsize=2, typed=True)(
            self.build_attention_mask
        )

    @override
    def compute_metrics(self, episode: Episode) -> Metrics:
        episode, mask_action_timestep = self._mask_episode(episode)
        mask = self._build_attention_mask(
            episode.index, episode.timestep, legend=TorchAttentionMaskLegend
        )

        embedding = self.encoder(
            src=episode.embeddings_packed, mask=mask.mask.to(episode.device)
        )  # pyright: ignore[reportOptionalCall]

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

        losses = self.losses(  # pyright: ignore[reportOptionalCall]
            tree_map(Rearrange("b t 1 d -> (b t 1) d"), logits),
            tree_map(Rearrange("b t 1 -> (b t)"), targets),
        )

        return {"loss": losses}

    @override
    def predict(
        self,
        episode: Episode,
        *,
        result_keys: AbstractSet[PredictionResultKey],
        tokenizers: ModuleDict | None = None,
    ) -> TensorDict:
        b, t = episode.input.batch_size
        result = {}

        if (result_key := PredictionResultKey.GROUND_TRUTH) in result_keys:
            result[result_key] = episode.input.select(*self.heads.tree_paths())

        if result_keys & {
            PredictionResultKey.PREDICTION_VALUE,
            PredictionResultKey.PREDICTION_PROBS,
            PredictionResultKey.SCORE_LOGPROB,
            PredictionResultKey.SCORE_L1,
            PredictionResultKey.SUMMARY_EMBEDDINGS,
        }:
            episode, mask_action_timestep = self._mask_episode(episode)
            mask = self._build_attention_mask(
                episode.index, episode.timestep, legend=TorchAttentionMaskLegend
            )

            embedding = self.encoder(
                src=episode.embeddings_packed, mask=mask.mask.to(episode.device)
            )  # pyright: ignore[reportOptionalCall]

            keys_action = episode.timestep.get(TokenType.ACTION).keys(
                include_nested=True, leaves_only=True
            )
            index_action = episode.index.select(*keys_action)
            embeddings = index_action[mask_action_timestep].parse(embedding)

            logits = TensorDict(
                self.heads(embeddings.to_dict()),
                batch_size=[b, len(mask_action_timestep)],
            )

            def timestep_padder(x: Tensor) -> Tensor:
                """Insert NaN at all indices except `masked_action_timestep_idx`."""
                match x.shape:
                    case (b, _, *rest):
                        size = (b, t, *rest)

                    case _:
                        raise NotImplementedError

                out = torch.full(size=size, fill_value=torch.nan, device=x.device)
                out[:, mask_action_timestep] = x.to(out.dtype)
                return out

            if (result_key := PredictionResultKey.PREDICTION_VALUE) in result_keys:
                result[result_key] = (
                    logits.apply(lambda x: x.argmax(dim=-1))
                    .named_apply(  # pyright: ignore[reportAttributeAccessIssue]
                        lambda k, v: tokenizers.get_deepest(k).invert(v),  # pyright: ignore[reportOptionalMemberAccess, reportCallIssue]
                        nested_keys=True,
                    )
                    .apply(timestep_padder, batch_size=[b, t])
                )

            if (result_key := PredictionResultKey.PREDICTION_PROBS) in result_keys:
                result[result_key] = logits.apply(lambda x: x.softmax(dim=-1)).apply(  # pyright: ignore[reportAttributeAccessIssue]
                    timestep_padder, batch_size=[b, t]
                )

            if (result_key := PredictionResultKey.SCORE_LOGPROB) in result_keys:
                result[result_key] = (
                    logits.apply(lambda x: x.softmax(dim=-1))
                    .apply(Rearrange("b t 1 d -> b t d"))  # pyright: ignore[reportAttributeAccessIssue]
                    .apply(timestep_padder, batch_size=[b, t])
                    .apply(
                        lambda probs, tokens: probs.gather(dim=-1, index=tokens),
                        episode.input_tokens,
                    )
                    .apply(lambda x: -torch.log(x))
                )

            if (result_key := PredictionResultKey.SCORE_L1) in result_keys:
                result[result_key] = (
                    logits.apply(lambda x: x.argmax(dim=-1))
                    .named_apply(  # pyright: ignore[reportAttributeAccessIssue]
                        lambda k, v: tokenizers.get_deepest(k).invert(v),  # pyright: ignore[reportOptionalMemberAccess, reportCallIssue]
                        nested_keys=True,
                    )
                    .apply(timestep_padder, batch_size=[b, t])
                    .apply(
                        lambda pred, gt: F.l1_loss(pred, gt, reduction="none"),
                        episode.input,
                        nested_keys=True,
                    )
                )

            if (result_key := PredictionResultKey.SUMMARY_EMBEDDINGS) in result_keys:
                result[result_key] = episode.index.select(Modality.SPECIAL)[[-1]].parse(
                    embedding
                )

        return TensorDict(result).auto_batch_size_(2)

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
        episode.input_embeddings.select(*keys_action).masked_fill_(
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
        episode.input_embeddings.select(*keys_observation).masked_fill_(
            mask_observation, cls.MASKED_TOKEN_FILL_VALUE
        )

        return episode, mask_action_timestep

    @classmethod
    def build_attention_mask(  # noqa: PLR0914
        cls, index: Index, timestep: Timestep, *, legend: AttentionMaskLegend
    ) -> AttentionMask:
        length: int = index.max(reduce=True).item() + 1  # pyright: ignore[reportAttributeAccessIssue, reportAssignmentType]
        mask = AttentionMask(
            mask=torch.full((length, length), legend.DO_ATTEND.value),
            legend=legend,
            device=index.device,
        )

        (t,) = index.batch_size  # pyright: ignore[reportAssignmentType]
        action_keys = timestep.get(TokenType.ACTION).keys(
            include_nested=True, leaves_only=True
        )

        for step in range(t):
            past, current, future = index[:step], index[step], index[step + 1 :]
            current_actions = current.select(*action_keys)
            current_action_summary = current.select((
                Modality.SPECIAL,
                SpecialToken.ACTION_SUMMARY,
            ))
            past_actions = past.select(*action_keys)
            past_action_summary = past.select((
                Modality.SPECIAL,
                SpecialToken.ACTION_SUMMARY,
            ))
            future_actions = future.select(*action_keys)
            future_action_summary = future.select((
                Modality.SPECIAL,
                SpecialToken.ACTION_SUMMARY,
            ))

            mask = (
                mask.do_not_attend(current_actions, past_actions)
                .do_not_attend(current_actions, past_action_summary)
                .do_not_attend(current_actions, future_actions)
                .do_not_attend(current_actions, future_action_summary)
                .do_not_attend(current_action_summary, past_actions)
                .do_not_attend(current_action_summary, past_action_summary)
                .do_not_attend(current_action_summary, future_actions)
                .do_not_attend(current_action_summary, future_action_summary)
            )

        observation_summary = index.select((
            Modality.SPECIAL,
            SpecialToken.OBSERVATION_SUMMARY,
        ))
        observation_history = index.select((
            Modality.SPECIAL,
            SpecialToken.OBSERVATION_HISTORY,
        ))
        observation_keys = timestep.get(TokenType.OBSERVATION).keys(
            include_nested=True, leaves_only=True
        )
        for modality_i in observation_keys:
            mask = mask.do_not_attend(
                index.select(modality_i), observation_summary
            ).do_not_attend(index.select(modality_i), observation_history)
            for modality_j in observation_keys:
                if modality_i != modality_j:
                    mask = mask.do_not_attend(
                        index.select(modality_i), index.select(modality_j)
                    )

        return mask
