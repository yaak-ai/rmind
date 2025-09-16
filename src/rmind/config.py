from collections.abc import Callable
from typing import Any, ClassVar, Literal

from hydra.utils import instantiate
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ImportString,
    SerializationInfo,
    field_serializer,
)


class HydraConfig[T](BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        extra="allow",
        validate_assignment=True,
        populate_by_name=True,
        serialize_by_alias=True,
    )

    target: ImportString[type[T] | Callable[..., T]] = Field(
        serialization_alias="_target_", validation_alias="_target_"
    )
    recursive: bool = Field(alias="_recursive_", default=True)
    convert: Literal["none", "partial", "object", "all"] = Field(
        alias="_convert_", default="all"
    )
    partial: bool = Field(alias="_partial_", default=False)

    def instantiate(self, **kwargs: object) -> T:
        return instantiate(self.model_dump(), **kwargs)

    @field_serializer("target", when_used="always")
    def serialize_target(self, target: Any, _info: SerializationInfo) -> str:  # noqa: PLR6301
        return ImportString._serialize(target)  # noqa: SLF001
