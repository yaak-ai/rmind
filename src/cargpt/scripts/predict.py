import multiprocessing as mp
import sys

import hydra
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig

from cargpt.utils.logging import setup_logging


@hydra.main(version_base=None)
def predict(cfg: DictConfig):
    logger.debug("instantiating model", target=cfg.model._target_)
    model = instantiate(cfg.model)

    logger.debug("instantiating datamodule", target=cfg.datamodule._target_)
    datamodule = instantiate(cfg.datamodule)

    logger.debug("instantiating trainer", target=cfg.trainer._target_)
    trainer = instantiate(cfg.trainer)

    logger.debug("starting prediction")
    return trainer.predict(model=model, datamodule=datamodule, return_predictions=False)


@logger.catch(onerror=lambda _: sys.exit(1))
def main():
    mp.set_start_method("spawn", force=True)
    mp.set_forkserver_preload(["torch"])
    setup_logging()

    predict()


if __name__ == "__main__":
    main()
