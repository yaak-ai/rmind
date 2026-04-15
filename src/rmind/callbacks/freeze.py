from operator import attrgetter
from typing import override

import pytorch_lightning as pl
from pydantic import ImportString, InstanceOf, validate_call
from pytorch_lightning.callbacks import Callback
from structlog import get_logger
from torch import nn

logger = get_logger(__name__)


class FreezeModules(Callback):
    @validate_call
    def __init__(
        self,
        paths: set[str] | None = None,
        types: set[ImportString[type[nn.Module]]] | None = None,
    ) -> None:
        self.paths = paths or set()
        self.types = tuple(types or set())

    def _resolve(self, pl_module: pl.LightningModule) -> list[tuple[str, nn.Module]]:
        resolved: list[tuple[str, nn.Module]] = [
            (path, attrgetter(path)(pl_module)) for path in self.paths
        ]

        if self.types:
            resolved.extend(
                (name, module)
                for name, module in pl_module.named_modules()
                if isinstance(module, self.types)
            )

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
