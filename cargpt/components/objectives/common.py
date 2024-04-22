from enum import StrEnum, auto


class ObjectiveName(StrEnum):
    FORWARD_DYNAMICS = auto()
    INVERSE_DYNAMICS = auto()
    RANDOM_MASKED_HINDSIGHT_CONTROL = auto()
    COPYCAT = auto()
