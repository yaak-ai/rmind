from typing import Any

import pytorch_lightning as pl
from hydra.utils import instantiate
from jaxtyping import Float  # noqa: TCH002
from omegaconf import DictConfig
from tensordict import TensorDict
from torch import Tensor  # noqa: TCH002
from typing_extensions import override
from yaak_datasets import Batch

from cargpt.components.episode import Modality, SpecialToken


class Embeddings(pl.LightningModule):
    def __init__(self, base: DictConfig):
        super().__init__()
        self.base = instantiate(base)

    @override
    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> TensorDict:
        return self.embeddings_step(batch)

    def embeddings_step(self, batch: Batch) -> TensorDict:  # pyright: ignore[reportGeneralTypeIssues]
        inputs = self.base._build_input(batch)
        episode = self.base.episode_builder.build_episode(inputs)

        embeddigs = TensorDict({}, batch_size=[], device=inputs.device)

        for name, objective in self.base.objectives.items():
            summaries = self.summary_embeddings(objective, episode)
            for token_name, embedding in summaries.items():
                embeddigs[f"{name}/{token_name}"] = embedding

        return embeddigs

    def summary_embeddings(self, objective, episode) -> TensorDict:
        mask = objective._build_attention_mask(episode.index, episode.timestep)
        embedding = self.base.encoder(src=episode.packed_embeddings, mask=mask.data)

        index = episode.index[[-1]]

        observation_summary: Float[Tensor, "b t 1 d"] = (
            index.select(k := (Modality.SPECIAL, SpecialToken.OBSERVATION_SUMMARY))
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
            },
            batch_size=[],
            device=embedding.device,
        )
