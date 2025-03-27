from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from enum import auto, unique
from typing import Any

from backports.strenum import StrEnum
from optree import register_pytree_node
from optree.utils import unzip2
from tensordict import TensorDict
from torch.nn import Module
from typing_extensions import override

from cargpt.components.episode import Episode, Modality
from cargpt.utils.containers import OPTREE_NAMESPACE, ModuleDict

Targets = Mapping[Modality, Mapping[str, tuple[str, ...]]]

@unique
class ObjectiveName(StrEnum):
    FORWARD_DYNAMICS = auto()
    INVERSE_DYNAMICS = auto()
    RANDOM_MASKED_HINDSIGHT_CONTROL = auto()
    MEMORY_EXTRACTION = auto()
    POLICY = auto()


@unique
class PredictionResultKey(StrEnum):
    PREDICTION = auto()
    PREDICTION_STD = auto()
    PREDICTION_PROBS = auto()
    SCORE_LOGPROB = auto()
    SCORE_L1 = auto()
    GROUND_TRUTH = auto()
    ATTENTION = auto()
    SUMMARY_EMBEDDINGS = auto()


def _not_implemented(*_args, **_kwargs):
    raise NotImplementedError


def objective_flatten(
    objective: Module,
) -> tuple[tuple[Module, ...], list[str], tuple[str, ...]]:
    keys, values = unzip2(sorted(objective.named_children()))
    return values, list(keys), keys


class Objective(Module):
    def __init_subclass__(cls) -> None:
        register_pytree_node(
            cls,  # pyright: ignore[reportArgumentType]
            flatten_func=objective_flatten,  # pyright: ignore[reportArgumentType]
            unflatten_func=_not_implemented,
            namespace=OPTREE_NAMESPACE,
        )
        return super().__init_subclass__()

    def __getitem__(self, name: str) -> Any:
        return getattr(self, name)

    @override
    def forward(self, episode: Episode, encoder: Module) -> TensorDict:
        raise NotImplementedError

    def predict(
        self,
        *,
        episode: Episode,
        encoder: Module,
        result_keys: AbstractSet[PredictionResultKey],
        tokenizers: ModuleDict | None = None,
    ) -> TensorDict:
        raise NotImplementedError
