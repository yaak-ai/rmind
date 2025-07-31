from collections.abc import Callable
from typing import ClassVar, Literal

from hydra.utils import instantiate
from pydantic import BaseModel, ConfigDict, Field, ImportString


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
