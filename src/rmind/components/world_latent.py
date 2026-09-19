from typing import final, override

from pydantic import InstanceOf, validate_call
from torch import Tensor
from torch.nn import Module

from rmind.components.base import Modality
from rmind.components.episode import Episode
from rmind.components.objectives.base import (
    world_latent_context,
    world_latent_os_as,
)


@final
class WorldModelLatent(Module):
    """The model's latent stage.

    With a ``conditioner`` (Conditional Self-Attention), the action summary AS
    *updates* the observation summary OS -> OS', and the latent reads only the
    updated observations:  L = latent(Q=[MASK]+PE, K=V=OS').  The action enters
    solely as a condition that reweights which OS tokens matter; it never becomes
    key/value content.

    Without a ``conditioner`` it falls back to the original K=V=[OS; AS].
    """

    @validate_call
    def __init__(
        self,
        *,
        latent: InstanceOf[Module],
        conditioner: InstanceOf[Module] | None = None,
    ) -> None:
        super().__init__()
        self.latent = latent
        self.conditioner: Module | None = conditioner

    @override
    def forward(self, *, episode: Episode, embedding: Tensor) -> Tensor:
        query = episode.embeddings.get((Modality.UTILITY, "latent"))

        if self.conditioner is not None:
            observation_summary, action_summary = world_latent_os_as(episode, embedding)
            # AS conditions OS -> updated observation summary OS'
            context = self.conditioner(observation_summary, action_summary)
        else:
            context = world_latent_context(episode, embedding)  # [OS; AS]

        return self.latent({"query": query, "key": context, "value": context})
