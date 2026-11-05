from pathlib import Path
from subprocess import check_output  # noqa: S404

import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only
from structlog import get_logger

import wandb
from rmind.utils.precision import auto_precision

logger = get_logger(__name__)


def _train(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision(cfg.matmul_precision)

    logger.debug("instantiating model", target=cfg.model._target_)
    model: pl.LightningModule = instantiate(cfg.model)

    logger.debug("instantiating datamodule", target=cfg.datamodule._target_)
    datamodule: pl.LightningDataModule = instantiate(cfg.datamodule)

    cfg.trainer.precision = auto_precision(cfg.trainer.precision)
    logger.debug("instantiating trainer", target=cfg.trainer._target_)
    trainer: pl.Trainer = instantiate(cfg.trainer)

    ckpt_path = cfg.get("ckpt_path")
    if (artifact := cfg.get("ckpt_artifact")) is not None:
        # resume from a checkpoint artifact (weights, optimizer, lr scheduler, epoch / step)
        run = wandb.run
        artifact_obj = (
            run.use_artifact(artifact)
            if run is not None and not run.disabled
            else wandb.Api().artifact(artifact, type="model")
        )
        ckpt_path = str(Path(artifact_obj.download()) / "model.ckpt")

    if ckpt_path is not None and not cfg.get("ckpt_resume_optimizer_state", True):
        # keep weights, lr scheduler and step; drop Adam moments (re-estimated within a few
        # steps). Needed for checkpoints whose optimizer parameter order cannot be matched.
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        for state in ckpt["optimizer_states"]:
            state["state"] = {}
        ckpt_path = str(Path(ckpt_path).with_name("model_reset_optimizer.ckpt"))
        torch.save(ckpt, ckpt_path)

    logger.debug("starting training", ckpt_path=ckpt_path)

    return trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)


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
