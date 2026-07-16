from .base import Objective
from .forward_dynamics import ForwardDynamicsPredictionObjective
from .goal_foresight import GoalConditionedForesightObjective
from .inverse_dynamics import InverseDynamicsPredictionObjective
from .memory_extraction import MemoryExtractionObjective
from .policy import PolicyObjective

__all__ = [
    "ForwardDynamicsPredictionObjective",
    "GoalConditionedForesightObjective",
    "InverseDynamicsPredictionObjective",
    "MemoryExtractionObjective",
    "Objective",
    "PolicyObjective",
]
