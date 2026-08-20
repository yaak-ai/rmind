from pathlib import Path
from typing import Any, Self


class LoadableFromArtifact:
    @staticmethod
    def download_wandb_artifact(artifact: str, filename: str = "model.ckpt") -> Path:
        import wandb  # noqa: PLC0415

        run = wandb.run
        artifact_obj = (
            run.use_artifact(artifact)
            if run is not None and not run.disabled
            else wandb.Api().artifact(artifact, type="model")
        )

        artifact_dir = artifact_obj.download()
        return Path(artifact_dir) / filename

    @classmethod
    def load_from_wandb_artifact(
        cls, artifact: str, filename: str = "model.ckpt", **kwargs: Any
    ) -> Self:
        ckpt_path = cls.download_wandb_artifact(artifact, filename)
        return cls.load_from_checkpoint(ckpt_path, **kwargs)  # ty:ignore[unresolved-attribute]
