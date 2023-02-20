#!/usr/bin/env python
import multiprocessing as mp
import sys

import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from jaxtyping import install_import_hook
from loguru import logger
from omegaconf import DictConfig, OmegaConf

import wandb


def _train(cfg: DictConfig):
    pl.seed_everything(cfg.seed, workers=True)

    logger.debug("instantiating model", target=cfg.model._target_)
    model: pl.LightningModule = instantiate(cfg.model)

    logger.debug("instantiating datamodule", target=cfg.datamodule._target_)
    datamodule: pl.LightningDataModule = instantiate(cfg.datamodule)

    logger.debug("instantiating trainer", target=cfg.trainer._target_)
    trainer: pl.Trainer = instantiate(cfg.trainer)

    logger.debug("starting training")
    trainer.fit(model=model, datamodule=datamodule)


@hydra.main(version_base=None, config_path="config", config_name="train.yaml")
def train(cfg: DictConfig):
    _run = wandb.init(
        config=OmegaConf.to_container(  # type: ignore
            cfg,
            resolve=True,
            throw_on_missing=True,
        ),
        **cfg.wandb,
    )

    # TODO: code logging

    return _train(cfg)


@logger.catch(onerror=lambda _: sys.exit(1))
def main():
    mp.set_start_method("spawn", force=True)

    train()


if __name__ == "__main__":
    with install_import_hook("cargpt", ("beartype", "beartype")):
        main()
