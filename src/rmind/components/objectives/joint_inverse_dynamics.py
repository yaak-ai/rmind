from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from typing import final, override

import torch
import torch.nn.functional as F
from pydantic import InstanceOf, validate_call
from tensordict import TensorDict
from torch import Tensor
from torch.nn import Module
from torch.utils._pytree import tree_leaves, tree_map  # noqa: PLC2701

from rmind.components.base import Modality, SummaryToken
from rmind.components.containers import ModuleDict
from rmind.components.episode import Episode
from rmind.components.objectives.base import (
    Metrics,
    Objective,
    ObjectivePredictionKey,
    Prediction,
)

type CodeTargets = Mapping[str, Mapping[str, tuple[str, ...]]]


@final
class JointInverseDynamicsObjective(Objective):
    """Inverse dynamics predicting a frozen tokenizer's joint residual-VQ codes.
    """

    @validate_call
    def __init__(  # noqa: PLR0913
        self,
        *,
        tokenizer: InstanceOf[ModuleDict],
        projection: InstanceOf[ModuleDict],
        heads: InstanceOf[ModuleDict],
        losses: InstanceOf[ModuleDict],
        targets: CodeTargets,
        norm: InstanceOf[Module] | None = None,
    ) -> None:
        super().__init__()

        self.tokenizer = tokenizer.requires_grad_(False).eval()  # noqa: FBT003
        self.projection = projection
        self.heads = heads
        self.losses = losses
        self.targets: CodeTargets = targets
        self.norm: Module | None = norm

    @override
    def train(self, mode: bool = True) -> "JointInverseDynamicsObjective":
        super().train(mode)
        self.tokenizer.eval()  # keep the frozen tokenizer's VQ EMA from updating
        return self

    def _observation_summary(self, episode: Episode, embedding: Tensor) -> Tensor:
        if self.norm is not None:
            embedding = self.norm(embedding)
        k = (Modality.SUMMARY, SummaryToken.OBSERVATION_SUMMARY)
        # drop the singleton summary-token dim: (b, t, 1, d) -> (b, t, d)
        return episode.index.select(k).parse(embedding).get(k).squeeze(-2)

    @override
    def compute_metrics(self, *, episode: Episode, embedding: Tensor) -> Metrics:
        features = self._observation_summary(episode, embedding)[:, :-1]
        (tokenizer,) = tree_leaves(self.tokenizer)
        (heads,) = tree_leaves(self.heads)  # one head per quantizer

        with torch.no_grad():
            codes = tree_map(
                lambda path: episode.get(path)[:, 1:],
                self.targets,
                is_leaf=lambda x: isinstance(x, tuple),
            )  # {joint: {actions: (B, T-1, num_quantizers)}}

        r = self.projection(features)  # {joint: {actions: (B, T-1, latent)}}
        loss = None
        for q in range(tokenizer.quantizer.num_quantizers):
            step = self.losses(
                tree_map(lambda rq, q=q: heads[q](rq).flatten(0, 1), r),
                tree_map(lambda c, q=q: c[..., q].flatten(), codes),
            )
            loss = step if loss is None else tree_map(torch.add, loss, step)
            codebook = tokenizer.quantizer.codebook(q)
            # out-of-place: heads[q](r) is saved for backward, so r must not be mutated
            r = tree_map(
                lambda rq, c, q=q, cb=codebook: rq - F.embedding(c[..., q], cb),
                r,
                codes,
            )

        return {"loss": loss}

    @override
    def predict(
        self,
        *,
        episode: Episode,
        embedding: Tensor,
        keys: AbstractSet[ObjectivePredictionKey],
        tokenizers: ModuleDict | None = None,
    ) -> TensorDict:
        predictions: dict[ObjectivePredictionKey, Prediction] = {}

        if (key := ObjectivePredictionKey.GROUND_TRUTH) in keys:
            gt = tree_map(
                episode.get,
                self.targets,
                is_leaf=lambda x: isinstance(x, tuple),
            )
            predictions[key] = Prediction(
                value=TensorDict(gt), timestep_indices=slice(None)
            )

        if ObjectivePredictionKey.PREDICTION_VALUE in keys:
            features = self._observation_summary(episode, embedding)[:, :-1]
            (tokenizer,) = tree_leaves(self.tokenizer)
            (heads,) = tree_leaves(self.heads)

            r = self.projection(features)
            codes: list[CodeTargets] = []
            for q in range(tokenizer.quantizer.num_quantizers):
                code = tree_map(lambda rq, q=q: heads[q](rq).argmax(dim=-1), r)
                codes.append(code)
                codebook = tokenizer.quantizer.codebook(q)
                r = tree_map(
                    lambda rq, c, cb=codebook: rq - F.embedding(c, cb), r, code
                )

            value = tree_map(lambda *cs: torch.stack(cs, dim=-1), *codes)
            predictions[ObjectivePredictionKey.PREDICTION_VALUE] = Prediction(
                value=TensorDict(value), timestep_indices=slice(1, None)
            )

        return TensorDict(predictions).auto_batch_size_(2)
