from .common import ObjectiveName
from .copycat import CopycatObjective
from .forward_dynamics import ForwardDynamicsPredictionObjective
from .inverse_dynamics import InverseDynamicsPredictionObjective
from .random_masked_hindsight_control import RandomMaskedHindsightControlObjective
from .scheduler import ObjectiveScheduler

__all__ = [
    "CopycatObjective",
    "ForwardDynamicsPredictionObjective",
    "InverseDynamicsPredictionObjective",
    "ObjectiveName",
    "ObjectiveScheduler",
    "RandomMaskedHindsightControlObjective",
]
