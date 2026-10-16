from typing import final, override

from pydantic import InstanceOf, validate_call
from torch import Tensor
from torch.nn import Module

from rmind.components.base import Modality
from rmind.components.episode import Episode
from rmind.components.objectives.base import world_latent_context, world_latent_os_as


@final
class WorldModelLatent(Module):
    """The model's latent stage.

    With a ``condition_bias``, the learnable latent queries read the observation
    summary, K = V = OS, and the condition summary CS (the ``action_summary`` slot:
    speed, waypoints, actions) enters only as an additive bias on their cross-attention
    logits:  L = latent(Q=[MASK]+PE, K=V=OS, bias=b(OS, CS)).  L is scene content
    gathered where the condition makes it relevant.

    Without it the latent reads K=V=[OS; AS].
    """

    @validate_call
    def __init__(
        self,
        *,
        latent: InstanceOf[Module],
        condition_bias: InstanceOf[Module] | None = None,
    ) -> None:
        super().__init__()
        self.latent = latent
        self.condition_bias: Module | None = condition_bias

    @override
    def forward(self, *, episode: Episode, embedding: Tensor) -> Tensor:
        query = episode.embeddings.get((Modality.UTILITY, "latent"))

        if self.condition_bias is not None:
            observation_summary, condition_summary = world_latent_os_as(
                episode, embedding
            )
            bias = self.condition_bias(
                observation_summary, condition_summary, num_queries=query.shape[-2]
            )
            return self.latent({
                "query": query,
                "key": observation_summary,
                "value": observation_summary,
                "attn_bias": bias,
            })

        context = world_latent_context(episode, embedding)  # [OS; AS]
        return self.latent({"query": query, "key": context, "value": context})
