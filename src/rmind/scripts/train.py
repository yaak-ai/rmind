from pathlib import Path
from subprocess import check_output  # noqa: S404

import hydra
import pytorch_lightning as pl
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only
from structlog import get_logger

logger = get_logger(__name__)


def _train(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed, workers=True)  # pyright: ignore[reportUnusedCallResult]
    torch.set_float32_matmul_precision(cfg.matmul_precision)

    logger.debug("instantiating model", target=cfg.model._target_)
    model: pl.LightningModule = instantiate(cfg.model)

    logger.debug("instantiating datamodule", target=cfg.datamodule._target_)
    datamodule: pl.LightningDataModule = instantiate(cfg.datamodule)

    logger.debug("instantiating trainer", target=cfg.trainer._target_)
    trainer: pl.Trainer = instantiate(cfg.trainer)

    logger.debug("starting training")

    return trainer.fit(model=model, datamodule=datamodule)


@hydra.main(version_base=None)
def train(cfg: DictConfig) -> None:
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

        _ = run.log_code(
            root=".", include_fn=lambda path: Path(path).resolve() in paths
        )

    return _train(cfg)


if __name__ == "__main__":
    import logging

    logging.getLogger("xformers").setLevel(logging.ERROR)

    train()
