from __future__ import annotations

from pathlib import Path
from typing import Any

from typing_extensions import Self


class LoadableFromArtifact:
    @classmethod
    def load_from_wandb_artifact(
        cls, artifact: str, filename: str = "model.ckpt", **kwargs: Any
    ) -> Self:
        import wandb  # noqa: PLC0415

        run = wandb.run
        artifact_obj = (
            run.use_artifact(artifact)
            if run is not None and not run.disabled
            else wandb.Api().artifact(artifact, type="model")
        )

        artifact_dir = artifact_obj.download()
        ckpt_path = Path(artifact_dir) / filename

        return cls.load_from_checkpoint(ckpt_path, **kwargs)  # ty:ignore[unresolved-attribute]
