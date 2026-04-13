from operator import attrgetter
from typing import override

import pytorch_lightning as pl
from hydra.utils import get_class
from pydantic import InstanceOf, validate_call
from pytorch_lightning.callbacks import Callback
from structlog import get_logger
from torch import nn

logger = get_logger(__name__)


class FreezeModules(Callback):
    @validate_call
    def __init__(
        self, paths: list[str] | None = None, types: list[str] | None = None
    ) -> None:
        self.paths = paths or []
        resolved_types: tuple[type, ...] = tuple(get_class(t) for t in types or [])
        if non_modules := tuple(
            t for t in resolved_types if not issubclass(t, nn.Module)
        ):
            msg = f"types must be nn.Module subclasses, got: {non_modules}"
            raise TypeError(msg)
        self.types = resolved_types

    def _resolve(self, pl_module: pl.LightningModule) -> list[tuple[str, nn.Module]]:
        resolved: list[tuple[str, nn.Module]] = []

        for path in self.paths:
            try:
                module = attrgetter(path)(pl_module)
            except AttributeError:
                logger.exception("freeze path not found on pl_module", path=path)
                continue
            resolved.append((path, module))

        if self.types:
            for name, module in pl_module.named_modules():
                if isinstance(module, self.types):
                    resolved.append((name, module))

        return resolved

    @override
    @validate_call
    def setup(
        self,
        trainer: InstanceOf[pl.Trainer],
        pl_module: InstanceOf[pl.LightningModule],
        stage: str,
    ) -> None:
        for path, module in self._resolve(pl_module):
            if frozen_params := tuple(
                k
                for k, v in module.named_parameters(recurse=True)
                if not v.requires_grad
            ):
                logger.warning(
                    "freezing module with already-frozen params",
                    path=path,
                    params=frozen_params,
                )
            module.requires_grad_(False).eval()  # noqa: FBT003
            logger.info("froze module", path=path, type=type(module).__name__)
