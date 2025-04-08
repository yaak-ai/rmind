import sys
from pathlib import Path
from subprocess import check_output  # noqa: S404

import hydra
import pytorch_lightning as pl
import wandb
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only
from torch import multiprocessing as mp

from cargpt.utils.logging import setup_logging


def _train(cfg: DictConfig):
    pl.seed_everything(cfg.seed, workers=True)

    logger.debug("instantiating model", target=cfg.model._target_)
    model: pl.LightningModule = instantiate(cfg.model)

    logger.debug("instantiating datamodule", target=cfg.datamodule._target_)
    datamodule: pl.LightningDataModule = instantiate(cfg.datamodule)

    logger.debug("instantiating trainer", target=cfg.trainer._target_)
    trainer: pl.Trainer = instantiate(cfg.trainer)

    logger.debug("starting training")

    return trainer.fit(model=model, datamodule=datamodule)


@hydra.main(version_base=None)
def train(cfg: DictConfig):
    if (
        run := rank_zero_only(wandb.init)(
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),  # pyright: ignore[reportArgumentType]
            **cfg.wandb,
        )
    ) is not None:
        paths = {
            Path(path).resolve()
            for path in check_output(
                ["git", "ls-files"],  # noqa: S607
                universal_newlines=True,
            ).splitlines()
        }

        run.log_code(root=".", include_fn=lambda path: Path(path).resolve() in paths)

    return _train(cfg)


@logger.catch(onerror=lambda _: sys.exit(1))
def main():
    mp.set_start_method("forkserver", force=True)
    mp.set_forkserver_preload(["rbyte"])
    setup_logging()

    train()


if __name__ == "__main__":
    main()
