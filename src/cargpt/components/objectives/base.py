from collections.abc import Set as AbstractSet
from enum import StrEnum, auto
from typing import Any, override

from tensordict import TensorDict
from torch.nn import Module
from torch.utils._pytree import Context, KeyEntry, MappingKey, register_pytree_node

from cargpt.components.episode import EpisodeBuilder


class ObjectiveName(StrEnum):
    FORWARD_DYNAMICS = auto()
    INVERSE_DYNAMICS = auto()
    RANDOM_MASKED_HINDSIGHT_CONTROL = auto()
    MEMORY_EXTRACTION = auto()
    POLICY = auto()


class PredictionResultKey(StrEnum):
    PREDICTION = auto()
    PREDICTION_STD = auto()
    PREDICTION_PROBS = auto()
    SCORE_LOGPROB = auto()
    SCORE_L1 = auto()
    GROUND_TRUTH = auto()
    ATTENTION = auto()


def _not_implemented(*_args, **_kwargs):
    raise NotImplementedError


def objective_flatten(objective: Module) -> tuple[list[Module], tuple[str, ...]]:
    keys, values = zip(*sorted(objective.named_children()), strict=True)
    return values, keys


def objective_flatten_with_keys(
    objective: Module,
) -> tuple[list[tuple[KeyEntry, Any]], Context]:
    values, context = objective_flatten(objective)
    return [(MappingKey(k), v) for k, v in zip(context, values, strict=True)], context  # pyright: ignore[reportReturnType]


class Objective(Module):
    def __init_subclass__(cls) -> None:
        register_pytree_node(
            cls,
            flatten_fn=objective_flatten,
            unflatten_fn=_not_implemented,
            flatten_with_keys_fn=objective_flatten_with_keys,
        )
        return super().__init_subclass__()

    @override
    def forward(
        self, inputs: TensorDict, episode_builder: EpisodeBuilder, encoder: Module
    ) -> TensorDict:
        raise NotImplementedError

    def predict(
        self,
        inputs: TensorDict,
        episode_builder: EpisodeBuilder,
        encoder: Module,
        *,
        result_keys: AbstractSet[PredictionResultKey] | None,
    ) -> TensorDict:
        raise NotImplementedError
