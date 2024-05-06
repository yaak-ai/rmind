from collections.abc import Sequence
from functools import lru_cache

import numpy as np
import torch
from einops.layers.torch import Rearrange
from tensordict import TensorDict
from torch.nn import Module, ModuleDict
from typing_extensions import override

from cargpt.components.episode import (
    EpisodeBuilder,
    Index,
    Modality,
    SpecialToken,
    Timestep,
    TokenType,
)
from cargpt.components.mask import (
    AttentionMask,
    AttentionMaskLegend,
    XFormersAttentionMaskLegend,
)
from cargpt.components.objectives.common import PredictionResultKey


class RandomMaskedHindsightControlObjective(Module):
    def __init__(self, heads: ModuleDict, losses: ModuleDict | None) -> None:
        super().__init__()
        self.heads = heads
        self.losses = losses

    @override
    def forward(
        self,
        inputs: TensorDict,
        episode_builder: EpisodeBuilder,
        encoder: Module,
    ) -> TensorDict:
        _, t = inputs.batch_size
        masked_action_timestep_idx = np.random.choice(t, 2, replace=False).tolist()
        masked_observation_timestep_idx = np.random.choice(t, 1, replace=False).tolist()
        episode = episode_builder.build_episode(
            inputs,
            masked_action_timestep_idx=masked_action_timestep_idx,
            masked_observation_timestep_idx=masked_observation_timestep_idx,
        )
        mask = self._build_attention_mask(episode.index, episode.timestep)
        embedding = encoder(src=episode.packed_embeddings, mask=mask.data)
        index = episode.index.select(*episode.timestep.keys(TokenType.ACTION))  # pyright: ignore[reportAttributeAccessIssue]
        embeddings = index[masked_action_timestep_idx].parse(embedding)
        logits = embeddings.named_apply(
            lambda k, v: self.heads.get(k)(v),
            nested_keys=True,
        )

        labels = episode.tokenized.select(*logits.keys(True, True))[
            :, masked_action_timestep_idx
        ]

        logits = logits.apply(Rearrange("b t 1 d -> (b t 1) d"), batch_size=[])
        labels = labels.apply(Rearrange("b t 1 -> (b t 1)"), batch_size=[])
        loss = logits.named_apply(
            lambda k, _logits, _labels: self.losses.get(k)(_logits, _labels),  # pyright: ignore
            labels,
            nested_keys=True,
        )

        return TensorDict.from_dict({"loss": loss}, batch_size=[])

    def predict(
        self,
        inputs: TensorDict,
        episode_builder: EpisodeBuilder,
        encoder: Module,
        *,
        result_keys: Sequence[PredictionResultKey] = tuple(PredictionResultKey),  # pyright: ignore[reportCallInDefaultInitializer]
    ) -> TensorDict:
        b, t = inputs.batch_size
        result = TensorDict({}, batch_size=[b, t])

        if not result_keys:
            return result

        masked_action_timestep_idx = np.random.choice(t, 2, replace=False).tolist()
        masked_observation_timestep_idx = np.random.choice(t, 1, replace=False).tolist()
        episode = episode_builder.build_episode(
            inputs,
            masked_action_timestep_idx=masked_action_timestep_idx,
            masked_observation_timestep_idx=masked_observation_timestep_idx,
        )

        if (result_key := PredictionResultKey.PREDICTION) in result_keys:
            mask = self._build_attention_mask(episode.index, episode.timestep)
            embedding = encoder(src=episode.packed_embeddings, mask=mask.data)
            index = episode.index.select(*episode.timestep.keys(TokenType.ACTION))  # pyright: ignore[reportAttributeAccessIssue]
            embeddings = index[masked_action_timestep_idx].parse(embedding)
            logits = embeddings.named_apply(
                lambda k, v: self.heads.get(k)(v),
                nested_keys=True,
            )

            prediction_tokens = logits.apply(lambda x: x.argmax(dim=-1))
            prediction = prediction_tokens.named_apply(
                lambda k, v: episode_builder.detokenizers.get(k)(v),  # pyright: ignore[reportOptionalMemberAccess]
                nested_keys=True,
            ).apply(Rearrange("b t 1 -> b t"))

            # insert NaN at all indices except `masked_action_timestep_idx`
            def padder(x):
                out = torch.full((b, t), fill_value=torch.nan, device=x.device)
                out[:, masked_action_timestep_idx] = x
                return out

            prediction = prediction.apply(padder, batch_size=[b, t])

            result[result_key] = prediction

        if (result_key := PredictionResultKey.GROUND_TRUTH) in result_keys:
            ground_truth = episode.inputs.select(*(k for k, _ in self.heads.flatten()))

            result[result_key] = ground_truth

        if (result_key := PredictionResultKey.ATTENTION) in result_keys:
            # TODO
            raise NotImplementedError

        return result

    @classmethod
    @lru_cache(maxsize=1, typed=True)
    def _build_attention_mask(
        cls,
        index: Index,
        timestep: Timestep,
        legend: AttentionMaskLegend = XFormersAttentionMaskLegend,
    ) -> AttentionMask:
        mask = AttentionMask(  # pyright: ignore
            data=torch.full((index.max + 1, index.max + 1), legend.DO_ATTEND),
            legend=legend,
            batch_size=[],
            device=index.device,  # pyright: ignore[reportAttributeAccessIssue]
        )

        (t,) = index.batch_size  # pyright: ignore[reportAttributeAccessIssue]
        for step in range(t):
            past, current, future = index[:step], index[step], index[step + 1 :]  # pyright: ignore
            current_actions = current.select(*timestep.keys(TokenType.ACTION))
            current_action_summary = current.select((
                Modality.SPECIAL,
                SpecialToken.ACTION_SUMMARY,
            ))
            past_actions = past.select(*timestep.keys(TokenType.ACTION))
            past_action_summary = past.select((
                Modality.SPECIAL,
                SpecialToken.ACTION_SUMMARY,
            ))
            future_actions = future.select(*timestep.keys(TokenType.ACTION))
            future_action_summary = future.select((
                Modality.SPECIAL,
                SpecialToken.ACTION_SUMMARY,
            ))

            mask = (
                mask._do_not_attend(
                    current_actions,
                    past_actions,
                )
                ._do_not_attend(
                    current_actions,
                    past_action_summary,
                )
                ._do_not_attend(
                    current_actions,
                    future_actions,
                )
                ._do_not_attend(
                    current_actions,
                    future_action_summary,
                )
                ._do_not_attend(
                    current_action_summary,
                    past_actions,
                )
                ._do_not_attend(
                    current_action_summary,
                    past_action_summary,
                )
                ._do_not_attend(
                    current_action_summary,
                    future_actions,
                )
                ._do_not_attend(
                    current_action_summary,
                    future_action_summary,
                )
            )

        return mask
