"""LeJEPA SIGReg objective — anti-collapse regularizer on the encoder embedding.

Adds the Sketched Isotropic Gaussian Regularization term (arXiv:2511.08544) to
the control-transformer's objective set. rmind sums every objective's loss, so
attaching this alongside `forward_dynamics` realizes the LeJEPA objective

    L = L_pred  +  lambda * SIGReg(embedding)

with the recommended lambda = 0.05 folded into `weight`. SIGReg drives the
encoder embedding toward an isotropic standard Gaussian, keeping effective rank
high and preventing the representation collapse a naive unfrozen fine-tune
suffers from.
"""

from __future__ import annotations

from collections.abc import Set as AbstractSet
from typing import Any, final, override

from pydantic import validate_call
from tensordict import TensorDict
from torch import Tensor

from rmind.components.containers import ModuleDict
from rmind.components.episode import Episode
from rmind.components.objectives.base import (
    Metrics,
    Objective,
    ObjectivePredictionKey,
)
from rmind.components.sigreg import SIGReg


@final
class SIGRegObjective(Objective):
    @validate_call
    def __init__(
        self,
        *,
        weight: float = 0.05,
        num_slices: int = 1024,
        num_points: int = 17,
    ) -> None:
        super().__init__()
        self.weight = weight
        self.sigreg = SIGReg(num_slices=num_slices, num_points=num_points)

    @override
    def compute_metrics(self, *, episode: Episode, embedding: Tensor) -> Metrics:  # noqa: ARG002
        return {"loss": {"sigreg": self.weight * self.sigreg(embedding)}}

    @override
    def predict(
        self,
        *,
        episode: Episode,  # noqa: ARG002
        embedding: Tensor,  # noqa: ARG002
        keys: AbstractSet[ObjectivePredictionKey],  # noqa: ARG002
        tokenizers: ModuleDict | None = None,  # noqa: ARG002
        **_kwargs: Any,
    ) -> TensorDict:
        return TensorDict({}, batch_size=[])
