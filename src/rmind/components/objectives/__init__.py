from .base import Objective
from .forward_dynamics import ForwardDynamicsPredictionObjective
from .inverse_dynamics import InverseDynamicsPredictionObjective
from .memory_extraction import MemoryExtractionObjective
from .policy import PolicyObjective

__all__ = [
    "ForwardDynamicsPredictionObjective",
    "InverseDynamicsPredictionObjective",
    "MemoryExtractionObjective",
    "Objective",
    "PolicyObjective",
]
