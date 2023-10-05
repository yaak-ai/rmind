#!/usr/bin/env python
import multiprocessing as mp
import sys

import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig

from cargpt.utils.logging import setup_logging

OmegaConf.register_new_resolver("eval", eval)


@hydra.main(version_base=None, config_path="config", config_name="predict.yaml")
def predict(cfg: DictConfig):
    logger.debug("instantiating model", target=cfg.model._target_)
    model: pl.LightningModule = instantiate(cfg.model)

    logger.debug("instantiating datamodule", target=cfg.datamodule._target_)
    datamodule: pl.LightningDataModule = instantiate(cfg.datamodule)

    logger.debug("instantiating trainer", target=cfg.trainer._target_)
    trainer: pl.Trainer = instantiate(cfg.trainer)

    logger.debug("starting prediction")
    trainer.predict(
        model=model,
        datamodule=datamodule,
        return_predictions=False,
    )


@logger.catch(onerror=lambda _: sys.exit(1))
def main():
    mp.set_start_method("spawn", force=True)
    setup_logging()

    predict()


if __name__ == "__main__":
    main()
