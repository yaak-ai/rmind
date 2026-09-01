from abc import ABC, abstractmethod
from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from enum import StrEnum, auto, unique
from typing import Any, Never, NotRequired, TypedDict

import torch
from tensordict import TensorClass, TensorDict
from torch import Tensor
from torch.nn import Module
from torch.utils._pytree import Context, register_pytree_node  # noqa: PLC2701

from rmind.components.base import Modality, SummaryToken, TensorTree
from rmind.components.containers import ModuleDict
from rmind.components.episode import Episode

type Targets = Mapping[Modality, Mapping[str, tuple[str, ...]]]
type CodeTargets = Mapping[str, Mapping[str, tuple[str, ...]]]

_K_OS = (Modality.SUMMARY, SummaryToken.OBSERVATION_SUMMARY)
_K_AS = (Modality.SUMMARY, SummaryToken.ACTION_SUMMARY)

PATCHES = ("input_embeddings", "image", "cam_front_left")


def world_latent_context(episode: "Episode", embedding: Tensor) -> Tensor:
    """[OS ; AS] summary tokens over all timesteps -> (b, t, 65, d)."""
    summary = episode.index.select(_K_OS, _K_AS).parse(embedding)
    return torch.cat([summary.get(_K_OS), summary.get(_K_AS)], dim=-2)


@unique
class ObjectivePredictionKey(StrEnum):
    PREDICTION_VALUE = auto()
    PREDICTION_STD = auto()
    PREDICTION_PROBS = auto()
    SCORE_LOGPROB = auto()
    SCORE_L1 = auto()
    GROUND_TRUTH = auto()
    SUMMARY_EMBEDDINGS = auto()
    SCORE_L1_REL = auto()
    PREDICTION_DIFF_PREV = auto()
    GROUND_TRUTH_DIFF_PREV = auto()
    PREDICTION_DIFF_HIST = auto()
    GROUND_TRUTH_DIFF_HIST = auto()
    SCORE_SIGNED_ERROR = auto()


class Prediction(TensorClass["autocast"]):  # ty:ignore[unsupported-base]
    value: TensorDict
    time_index: Tensor | None = None  # for timestep-wise sparse values


class Metrics(TypedDict):
    loss: TensorTree | None
    _artifacts: NotRequired[TensorTree]


def _not_implemented(*_args: Any, **_kwargs: Any) -> Never:
    raise NotImplementedError


def objective_flatten(objective: Module) -> tuple[list[Module], Context]:
    keys, values = zip(*sorted(objective.named_children()), strict=True)
    return list(values), keys


class Objective(Module, ABC):
    def __init_subclass__(cls) -> None:
        register_pytree_node(
            cls, flatten_fn=objective_flatten, unflatten_fn=_not_implemented
        )
        return super().__init_subclass__()

    def __getitem__(self, name: str) -> Any:
        return getattr(self, name)

    @abstractmethod
    def compute_metrics(self, *, episode: Episode, embedding: Tensor) -> Metrics: ...

    @abstractmethod
    def predict(
        self,
        *,
        episode: Episode,
        embedding: Tensor,
        keys: AbstractSet[ObjectivePredictionKey],
        tokenizers: ModuleDict | None = None,
    ) -> TensorDict: ...
