from abc import ABC, abstractmethod
from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from enum import StrEnum, auto, unique
from typing import Any, Never, TypedDict

from tensordict import TensorDict
from torch.nn import Module
from torch.utils._pytree import Context, register_pytree_node  # noqa: PLC2701

from rmind.components.base import TensorTree
from rmind.components.containers import ModuleDict
from rmind.components.episode import Episode, Modality

type Targets = Mapping[Modality, Mapping[str, tuple[str, ...]]]


@unique
class PredictionResultKey(StrEnum):
    PREDICTION_VALUE = auto()
    PREDICTION_STD = auto()
    PREDICTION_PROBS = auto()
    SCORE_LOGPROB = auto()
    SCORE_L1 = auto()
    GROUND_TRUTH = auto()
    SUMMARY_EMBEDDINGS = auto()


class Metrics(TypedDict):
    loss: TensorTree | None


def _not_implemented(*_args: Any, **_kwargs: Any) -> Never:
    raise NotImplementedError


def objective_flatten(objective: Module) -> tuple[list[Module], Context]:
    keys, values = zip(*sorted(objective.named_children()), strict=True)
    return values, keys


class Objective(Module, ABC):
    def __init_subclass__(cls) -> None:
        register_pytree_node(
            cls, flatten_fn=objective_flatten, unflatten_fn=_not_implemented
        )
        return super().__init_subclass__()

    def __getitem__(self, name: str) -> Any:
        return getattr(self, name)

    @abstractmethod
    def compute_metrics(self, episode: Episode) -> Metrics: ...

    @abstractmethod
    def predict(
        self,
        episode: Episode,
        *,
        result_keys: AbstractSet[PredictionResultKey],
        tokenizers: ModuleDict | None = None,
    ) -> TensorDict: ...
