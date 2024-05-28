from typing import TYPE_CHECKING, Any

import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig
from tensordict import TensorDict
from typing_extensions import override
from yaak_datasets import Batch

from cargpt.components.episode import Modality, SpecialToken

if TYPE_CHECKING:
    from jaxtyping import Float
    from torch import Tensor


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

        return TensorDict(
            {
                name: self.summary_embeddings(objective, episode)
                for name, objective in self.base.objectives.items()
            },
            batch_size=[],
            device=inputs.device,
        )

    def summary_embeddings(self, objective, episode) -> TensorDict:
        mask = objective._build_attention_mask(episode.index, episode.timestep)
        embedding = self.base.encoder(src=episode.packed_embeddings, mask=mask.data)

        index = episode.index[:-1]

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
                "observation_summary": observation_summary,
                "action_summary": action_summary,
            },
            batch_size=[],
            device=embedding.device,
        )
