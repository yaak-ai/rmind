from collections.abc import Callable
from inspect import ismethod
from typing import Any, ClassVar, Literal

from hydra.utils import get_object, instantiate
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ImportString,
    SerializationInfo,
    field_serializer,
    field_validator,
)


class HydraConfig[T](BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        extra="allow",
        validate_assignment=True,
        populate_by_name=True,
        serialize_by_alias=True,
    )

    target: type[T] | Callable[..., T] = Field(
        serialization_alias="_target_", validation_alias="_target_"
    )
    recursive: bool = Field(alias="_recursive_", default=True)
    convert: Literal["none", "partial", "object", "all"] = Field(
        alias="_convert_", default="all"
    )
    partial: bool = Field(alias="_partial_", default=False)

    def instantiate(self, **kwargs: object) -> T:
        return instantiate(self.model_dump(), **kwargs)

    @field_validator("target", mode="before")
    @classmethod
    def resolve_target(cls, target: Any) -> Any:
        if isinstance(target, str):
            try:
                target = get_object(target)
            except Exception as e:
                msg = f"unable to resolve Hydra target {target!r}"
                raise ValueError(msg) from e

        return target

    @field_serializer("target", when_used="always")
    def serialize_target(self, target: Any, _info: SerializationInfo) -> str:  # noqa: PLR6301
        if (
            ismethod(target)
            and isinstance(target.__self__, type)
            and isinstance(target.__name__, str)
        ):
            return f"{target.__self__.__module__}.{target.__self__.__qualname__}.{target.__name__}"

        if isinstance(target, type) or callable(target):
            module = getattr(target, "__module__", None)
            qualname = getattr(target, "__qualname__", None)
            if isinstance(module, str) and isinstance(qualname, str):
                return f"{module}.{qualname}"

        return ImportString._serialize(target)  # noqa: SLF001
