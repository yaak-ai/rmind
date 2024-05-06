from pathlib import Path

import wandb
from wandb.sdk.lib import RunDisabled


class LoadableFromArtifact:
    @classmethod
    def load_from_wandb_artifact(
        cls,
        artifact: str,
        filename: str = "model.ckpt",
        **kwargs,
    ):
        match wandb.run:
            case RunDisabled() | None:  # pyright: ignore[reportUnnecessaryComparison]
                artifact_obj = wandb.Api().artifact(artifact, type="model")

            case _:
                artifact_obj = wandb.run.use_artifact(artifact)

        artifact_dir = artifact_obj.download()
        ckpt_path = Path(artifact_dir) / filename

        return cls.load_from_checkpoint(ckpt_path, **kwargs)  # pyright: ignore[reportAttributeAccessIssue]
