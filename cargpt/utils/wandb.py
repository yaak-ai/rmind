from pathlib import Path
from typing import Callable

import more_itertools as mit

import wandb


class LoadableFromArtifact:
    @classmethod
    def load_from_wandb_artifact(cls, name: str, **kwargs):
        get_artifact: Callable[..., wandb.Artifact] = (
            wandb.run.use_artifact  # type: ignore
            if wandb.run is not None
            and not isinstance(wandb.run, wandb.sdk.lib.RunDisabled)
            else wandb.Api().artifact
        )

        artifact = get_artifact(name, type="model")
        artifact_dir = artifact.download()
        ckpt_path = mit.one(Path(artifact_dir).glob("*.ckpt"))

        return cls.load_from_checkpoint(ckpt_path.as_posix(), **kwargs)  # type: ignore[attr-defined]
