from collections.abc import Iterator
from operator import attrgetter
from typing import override

import pytorch_lightning as pl
from pydantic import InstanceOf, validate_call
from pytorch_lightning.callbacks import Callback
from structlog import get_logger
from torch import nn

from rmind.components.lora import LoRALinear, convert_multihead_attention
from rmind.components.transformer.attention import MaskedSelfAttention

logger = get_logger(__name__)


class LoraInjector(Callback):
    @validate_call
    def __init__(
        self, paths: set[str], r: int = 8, alpha: int = 16, dropout: float = 0.05
    ) -> None:
        self.paths = paths
        self.r = r
        self.alpha = alpha
        self.dropout = dropout

    @staticmethod
    def _convert_attention(root: nn.Module) -> None:
        for _, module in list(root.named_modules()):
            if isinstance(module, MaskedSelfAttention) and isinstance(
                module.attn, nn.MultiheadAttention
            ):
                module.attn = convert_multihead_attention(module.attn)

    def _wrap_linears(self, root: nn.Module) -> Iterator[str]:
        targets = [
            (name, module)
            for name, module in root.named_modules()
            if isinstance(module, nn.Linear)
        ]
        for name, linear in targets:
            *parent_path, attr = name.split(".")
            parent = attrgetter(".".join(parent_path))(root) if parent_path else root
            setattr(
                parent,
                attr,
                LoRALinear(linear, r=self.r, alpha=self.alpha, dropout=self.dropout),
            )
            yield name

    @override
    @validate_call
    def setup(
        self,
        trainer: InstanceOf[pl.Trainer],
        pl_module: InstanceOf[pl.LightningModule],
        stage: str,
    ) -> None:
        for path in self.paths:
            root = attrgetter(path)(pl_module)
            was_training = root.training
            root.requires_grad_(False)  # noqa: FBT003

            self._convert_attention(root)
            wrapped = list(self._wrap_linears(root))

            # freshly constructed submodules (converted attention, LoRA
            # wrappers) default to training=True regardless of root's actual
            # mode at injection time -- propagate it so dropout stays
            # consistent with whatever mode the root was already in.
            root.train(was_training)

            trainable_params = sum(
                p.numel() for p in root.parameters() if p.requires_grad
            )
            total_params = sum(p.numel() for p in root.parameters())
            logger.info(
                "injected lora",
                path=path,
                wrapped=wrapped,
                trainable_params=trainable_params,
                total_params=total_params,
                r=self.r,
                alpha=self.alpha,
            )
