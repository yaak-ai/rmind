import multiprocessing as mp
import sys
from pathlib import Path
from subprocess import check_output  # noqa: S404

import hydra
import pytorch_lightning as pl
import wandb
from hydra.utils import instantiate
from lightning_utilities.core.rank_zero import rank_zero_only
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from cargpt.utils.logging import setup_logging

OmegaConf.register_new_resolver("eval", eval)


def _train(cfg: DictConfig):
    pl.seed_everything(
        cfg.seed,
        workers=True,
    )  # pyright: ignore[reportUnusedCallResult]

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
    run = None
    if rank_zero_only.rank == 0:
        run = wandb.init(
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),  # type: ignore
            **cfg.wandb,
        )

    if run is not None:
        paths = {
            Path(path).resolve()
            for path in check_output(
                ["git", "ls-files"],  # noqa: S603, S607
                universal_newlines=True,
            ).splitlines()
        }
        run.log_code(  # pyright: ignore[reportUnusedCallResult]
            root=".",
            include_fn=lambda path: Path(path).resolve() in paths,
        )

    return _train(cfg)


@logger.catch(onerror=lambda _: sys.exit(1))
def main():
    mp.set_start_method("spawn", force=True)
    mp.set_forkserver_preload(["torch"])
    setup_logging()

    train()


if __name__ == "__main__":
    main()
