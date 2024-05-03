from enum import StrEnum, auto


class ObjectiveName(StrEnum):
    FORWARD_DYNAMICS = auto()
    INVERSE_DYNAMICS = auto()
    RANDOM_MASKED_HINDSIGHT_CONTROL = auto()
    COPYCAT = auto()


class PredictionResultKey(StrEnum):
    PREDICTION = auto()
    PREDICTION_PROBS = auto()
    SCORE_LOGPROB = auto()
    SCORE_L1 = auto()
    GROUND_TRUTH = auto()
    ATTENTION = auto()
