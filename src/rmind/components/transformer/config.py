from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field


class AttentionRolloutPredictionConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    head_fusion: Literal["mean", "max", "min"] = "max"
    discard_ratio: Annotated[float, Field(ge=0.0, le=1.0)] | None = 0.9


class EncoderPredictionConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    attention_rollout: AttentionRolloutPredictionConfig | None = None
