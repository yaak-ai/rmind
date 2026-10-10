import hashlib
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

from rmind.utils.precision import auto_precision

logger = get_logger(__name__)


def check_pins(cfg: DictConfig) -> None:
    """Refuse a pinned input whose bytes changed (`tokenizer_ckpt_sha256`).

    Raises:
        ValueError: if `tokenizer_ckpt` does not hash to `tokenizer_ckpt_sha256`.
    """
    if (expected := cfg.get("tokenizer_ckpt_sha256")) is None:
        return
    path = Path(cfg.tokenizer_ckpt)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        msg = f"tokenizer_ckpt {path} has sha256 {digest}, the config pins {expected}"
        raise ValueError(msg)


def _train(cfg: DictConfig) -> None:
    check_pins(cfg)
    pl.seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision(cfg.matmul_precision)

    logger.debug("instantiating model", target=cfg.model._target_)
    model: pl.LightningModule = instantiate(cfg.model)

    logger.debug("instantiating datamodule", target=cfg.datamodule._target_)
    datamodule: pl.LightningDataModule = instantiate(cfg.datamodule)

    cfg.trainer.precision = auto_precision(cfg.trainer.precision)
    logger.debug("instantiating trainer", target=cfg.trainer._target_)
    trainer: pl.Trainer = instantiate(cfg.trainer)

    logger.debug("starting training")

    return trainer.fit(
        model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path")
    )


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    if (
        run := rank_zero_only(wandb.init)(
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),  # ty:ignore[invalid-argument-type]
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
    import multiprocessing as mp

    mp.set_forkserver_preload(["rbyte", "polars"])

    main()
