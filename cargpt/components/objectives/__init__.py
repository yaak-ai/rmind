from .copycat import CopycatObjective
from .forward_dynamics import ForwardDynamicsPredictionObjective
from .inverse_dynamics import InverseDynamicsPredictionObjective
from .random_masked_hindsight_control import RandomMaskedHindsightControlObjective

__all__ = [
    "CopycatObjective",
    "ForwardDynamicsPredictionObjective",
    "InverseDynamicsPredictionObjective",
    "RandomMaskedHindsightControlObjective",
]
