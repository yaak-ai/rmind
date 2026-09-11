from typing import final, override

from pydantic import InstanceOf, validate_call
from torch import Tensor
from torch.nn import Module

from rmind.components.base import Modality
from rmind.components.episode import Episode
from rmind.components.objectives.base import world_latent_context


@final
class WorldModelLatent(Module):
    """The model's latent stage: L = latent(Q=[MASK]+PE, K=V=context[OS;AS])."""

    @validate_call
    def __init__(
        self, *, latent: InstanceOf[Module]
    ) -> None:
        super().__init__()
        self.latent = latent

    @override
    def forward(self, *, episode: Episode, embedding: Tensor) -> Tensor:
        context = world_latent_context(episode, embedding)
        query = episode.embeddings.get((Modality.UTILITY, "latent"))
        return self.latent({"query": query, "key": context, "value": context})
