from typing import override

import pytorch_lightning as pl
import torch
from pydantic import InstanceOf, validate_call
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.plugins.precision.amp import MixedPrecision
from structlog import get_logger

logger = get_logger(__name__)


class AutoPrecisionCallback(Callback):
    """Downgrades bf16-mixed to 16-mixed on GPUs without native bf16 support (e.g. Turing)."""

    @override
    @validate_call
    def setup(
        self,
        trainer: InstanceOf[pl.Trainer],
        pl_module: InstanceOf[pl.LightningModule],
        stage: str,
    ) -> None:
        if not torch.cuda.is_available():
            return
        plugin = trainer.precision_plugin
        if not isinstance(plugin, MixedPrecision):
            return
        if plugin.precision != "bf16-mixed":
            return
        if torch.cuda.is_bf16_supported():
            return

        plugin.precision = "16-mixed"
        if plugin.scaler is None:
            plugin.scaler = torch.cuda.amp.GradScaler()

        logger.info("bf16 not supported, downgraded precision", precision="16-mixed")
