from abc import ABC, abstractmethod
from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from enum import StrEnum, auto, unique
from typing import Any, Never

from optree import register_pytree_node
from optree.utils import unzip2
from tensordict import TensorDict
from torch.nn import Module

from rmind.components.episode import Episode, Modality
from rmind.utils.containers import OPTREE_NAMESPACE, ModuleDict

type Targets = Mapping[Modality, Mapping[str, tuple[str, ...]]]


@unique
class PredictionResultKey(StrEnum):
    PREDICTION_VALUE = auto()
    PREDICTION_STD = auto()
    PREDICTION_PROBS = auto()
    SCORE_LOGPROB = auto()
    SCORE_L1 = auto()
    GROUND_TRUTH = auto()
    ATTENTION = auto()
    SUMMARY_EMBEDDINGS = auto()


def _not_implemented(*_args: Any, **_kwargs: Any) -> Never:
    raise NotImplementedError


def objective_flatten(
    objective: Module,
) -> tuple[tuple[Module, ...], list[str], tuple[str, ...]]:
    keys, values = unzip2(sorted(objective.named_children()))
    return values, list(keys), keys


class Objective(Module, ABC):
    def __init_subclass__(cls) -> None:
        register_pytree_node(  # pyright: ignore[reportUnusedCallResult]
            cls,  # pyright: ignore[reportArgumentType]
            flatten_func=objective_flatten,  # pyright: ignore[reportArgumentType]
            unflatten_func=_not_implemented,
            namespace=OPTREE_NAMESPACE,
        )
        return super().__init_subclass__()

    def __getitem__(self, name: str) -> Any:
        return getattr(self, name)

    @abstractmethod
    def predict(
        self,
        *,
        episode: Episode,
        encoder: Module,
        result_keys: AbstractSet[PredictionResultKey],
        tokenizers: ModuleDict | None = None,
    ) -> TensorDict: ...
