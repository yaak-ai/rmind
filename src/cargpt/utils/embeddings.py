from typing import Any, override

import pytorch_lightning as pl
from hydra.utils import instantiate
from jaxtyping import Float  # noqa: TC002
from omegaconf import DictConfig
from tensordict import TensorDict
from torch import Tensor  # noqa: TC002

from cargpt.components.episode import Modality, SpecialToken
from cargpt.components.objectives.base import PredictionResultKey


class Embeddings(pl.LightningModule):
    # TODO: merge into ControlTransformer?
    def __init__(self, base: DictConfig):
        super().__init__()
        self.base = instantiate(base)

    @override
    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> TensorDict:
        input = self.base._build_input(batch)
        episode = self.base.episode_builder.build_episode(input)
        predictions = TensorDict.from_dict({
            name: {
                PredictionResultKey.SUMMARY_EMBEDDINGS: {
                    Modality.SPECIAL: self._compute_summary_embeddings(
                        objective, episode
                    )
                }
            }
            for name, objective in self.base.objectives.items()
        })

        return TensorDict.from_dict({"input": input, "predictions": predictions})

    def _compute_summary_embeddings(self, objective, episode) -> TensorDict:
        mask = objective._build_attention_mask(episode.index, episode.timestep)
        embedding = self.base.encoder(src=episode.packed_embeddings, mask=mask.data)

        index = episode.index[[-1]]

        observation_summary: Float[Tensor, "b t 1 d"] = (
            index.select(k := (Modality.SPECIAL, SpecialToken.OBSERVATION_SUMMARY))
            .parse(embedding)
            .get(k)
        )

        observation_history: Float[Tensor, "b t 1 d"] = (
            index.select(k := (Modality.SPECIAL, SpecialToken.OBSERVATION_HISTORY))
            .parse(embedding)
            .get(k)
        )

        action_summary: Float[Tensor, "b t 1 d"] = (
            index.select(k := (Modality.SPECIAL, SpecialToken.ACTION_SUMMARY))
            .parse(embedding)
            .get(k)
        )

        return TensorDict(
            {
                SpecialToken.OBSERVATION_SUMMARY: observation_summary,
                SpecialToken.ACTION_SUMMARY: action_summary,
                SpecialToken.OBSERVATION_HISTORY: observation_history,
            },
            batch_size=[],
            device=embedding.device,
        )
