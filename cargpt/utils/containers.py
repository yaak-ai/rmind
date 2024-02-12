from functools import reduce
from typing import Any, Mapping

from torch.nn import Module
from torch.nn import ModuleDict as _ModuleDict


class ModuleDict(_ModuleDict):
    """A convenience wrapper around torch.nn.ModuleDict"""

    __unspecified = object()

    def __init__(self, **modules: Module | Mapping[str, Module]) -> None:
        """recursive instantiation for the sake of smaller YAMLs"""

        _modules = {
            k: v if isinstance(v, Module) else ModuleDict(**v)
            for k, v in modules.items()
        }

        super().__init__(modules=_modules)

    def get(self, key: str | tuple[str, ...], *, default: Any = __unspecified):
        """recursive access mimicking TensorDict.get"""

        key_tuple = (key,) if isinstance(key, str) else key

        try:
            return reduce(ModuleDict.__getitem__, key_tuple, self)  # pyright: ignore
        except KeyError:
            if default is self.__unspecified:
                raise

            return default
