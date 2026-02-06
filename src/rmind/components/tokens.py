from enum import StrEnum, auto, unique


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
