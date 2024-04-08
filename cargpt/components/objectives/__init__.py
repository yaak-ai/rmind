from enum import StrEnum, auto

from .copycat import CopycatObjective
from .forward_dynamics import ForwardDynamicsPredictionObjective
from .inverse_dynamics import InverseDynamicsPredictionObjective
from .random_masked_hindsight_control import RandomMaskedHindsightControlObjective


class Objective(StrEnum):
    FORWARD_DYNAMICS = auto()
    INVERSE_DYNAMICS = auto()
    RANDOM_MASKED_HINDSIGHT_CONTROL = auto()
    COPYCAT = auto()


__all__ = [
    "CopycatObjective",
    "ForwardDynamicsPredictionObjective",
    "InverseDynamicsPredictionObjective",
    "RandomMaskedHindsightControlObjective",
]
