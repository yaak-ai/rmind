#! /usr/bin/env python
from __future__ import annotations

import multiprocessing as mp
from typing import TYPE_CHECKING

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from structlog import get_logger

logger = get_logger(__name__)


if TYPE_CHECKING:
    import pytorch_lightning as pl


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    torch.set_float32_matmul_precision(cfg.matmul_precision)

    logger.debug("instantiating model", target=cfg.model._target_)
    model: pl.LightningModule = instantiate(cfg.model)

    logger.debug("instantiating datamodule", target=cfg.datamodule._target_)
    datamodule: pl.LightningDataModule = instantiate(cfg.datamodule)

    logger.debug("instantiating trainer", target=cfg.trainer._target_)
    trainer: pl.Trainer = instantiate(cfg.trainer)

    logger.debug("starting prediction")

    trainer.predict(model=model, datamodule=datamodule, return_predictions=False)


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_forkserver_preload(["rbyte", "polars"])

    main()
