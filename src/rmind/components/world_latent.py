from typing import final, override

from einops import repeat
from pydantic import InstanceOf, validate_call
from torch import Tensor
from torch.nn import Module

from rmind.components.base import Modality
from rmind.components.episode import Episode
from rmind.components.objectives.base import PATCHES, world_latent_context


@final
class WorldModelLatent(Module):
    """The model's latent stage: L = latent(Q=[MASK]+PE, K=V=context[OS;AS])."""

    @validate_call
    def __init__(
        self,
        *,
        latent: InstanceOf[Module],
        patch_pos_embed: InstanceOf[Module],
    ) -> None:
        super().__init__()
        self.latent = latent
        self.patch_pos_embed = patch_pos_embed

    @override
    def forward(self, *, episode: Episode, embedding: Tensor) -> Tensor:
        context = world_latent_context(episode, embedding)
        p = episode.get(PATCHES).shape[-2]
        query = repeat(
            episode.embeddings.get((Modality.UTILITY, "latent")),
            "b t 1 d -> b t p d",
            p=p,
        )
        query = self.patch_pos_embed(query)
        return self.latent({"query": query, "key": context, "value": context})
