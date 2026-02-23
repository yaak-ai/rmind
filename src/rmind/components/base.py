from enum import StrEnum, auto, unique
from typing import Protocol, runtime_checkable

from torch import Tensor


@runtime_checkable
class Invertible(Protocol):
    def invert(self, input: Tensor) -> Tensor: ...


type TensorTree = dict[str, Tensor | TensorTree]


@unique
class TokenType(StrEnum):
    OBSERVATION = auto()
    ACTION = auto()
    SPECIAL = auto()


@unique
class Modality(StrEnum):
    IMAGE = auto()
    CONTINUOUS = auto()
    DISCRETE = auto()
    SUMMARY = auto()
    CONTEXT = auto()
    FORESIGHT = auto()
    UTILITY = auto()


@unique
class SummaryToken(StrEnum):
    OBSERVATION_SUMMARY = auto()
    OBSERVATION_HISTORY = auto()
    ACTION_SUMMARY = auto()


@unique
class PositionEncoding(StrEnum):
    OBSERVATIONS = auto()
    ACTIONS = auto()
    SPECIAL = auto()
    TIMESTEP = auto()
    CONTEXT = auto()
