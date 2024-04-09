from pathlib import Path

from pytorch_lightning.loggers.wandb import WandbLogger


class LoadableFromArtifact:
    @classmethod
    def load_from_wandb_artifact(
        cls,
        artifact: str,
        filename: str = "model.ckpt",
        **kwargs,
    ):
        artifact_dir = WandbLogger.download_artifact(artifact, artifact_type="model")
        ckpt_path = Path(artifact_dir) / filename  # pyright: ignore

        return cls.load_from_checkpoint(ckpt_path, **kwargs)
