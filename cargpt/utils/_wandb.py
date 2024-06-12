from pathlib import Path


class LoadableFromArtifact:
    @classmethod
    def load_from_wandb_artifact(
        cls, artifact: str, filename: str = "model.ckpt", **kwargs
    ):
        import wandb  # noqa: PLC0415

        match wandb.run:
            case wandb.sdk.lib.RunDisabled() | None:  # pyright: ignore[reportAttributeAccessIssue]
                artifact_obj = wandb.Api().artifact(artifact, type="model")

            case _:
                artifact_obj = wandb.run.use_artifact(artifact)

        artifact_dir = artifact_obj.download()
        ckpt_path = Path(artifact_dir) / filename

        return cls.load_from_checkpoint(ckpt_path, **kwargs)  # pyright: ignore[reportAttributeAccessIssue]
