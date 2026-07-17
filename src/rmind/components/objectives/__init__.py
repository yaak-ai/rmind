from .base import Objective
from .forward_dynamics import ForwardDynamicsPredictionObjective
from .inverse_dynamics import InverseDynamicsPredictionObjective
from .memory_extraction import MemoryExtractionObjective
from .policy import PolicyObjective
from .sigreg_objective import SIGRegObjective

__all__ = [
    "ForwardDynamicsPredictionObjective",
    "SIGRegObjective",
    "InverseDynamicsPredictionObjective",
    "MemoryExtractionObjective",
    "Objective",
    "PolicyObjective",
]
