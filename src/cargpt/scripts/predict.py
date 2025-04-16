#! /usr/bin/env python
from typing import TYPE_CHECKING

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning.utilities.types import (
    _PREDICT_OUTPUT,  # pyright: ignore[reportPrivateUsage]
)
from structlog import get_logger

logger = get_logger(__name__)


if TYPE_CHECKING:
    import pytorch_lightning as pl


@hydra.main(version_base=None)
def predict(cfg: DictConfig) -> _PREDICT_OUTPUT | None:
    logger.debug("instantiating model", target=cfg.model._target_)
    model: pl.LightningModule = instantiate(cfg.model)

    logger.debug("instantiating datamodule", target=cfg.datamodule._target_)
    datamodule: pl.LightningDataModule = instantiate(cfg.datamodule)

    logger.debug("instantiating trainer", target=cfg.trainer._target_)
    trainer: pl.Trainer = instantiate(cfg.trainer)

    logger.debug("starting prediction")

    return trainer.predict(model=model, datamodule=datamodule, return_predictions=False)


if __name__ == "__main__":
    import logging

    logging.getLogger("xformers").setLevel(logging.ERROR)

    predict()
