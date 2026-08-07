from pathlib import Path
from typing import override

import pytorch_lightning as pl
import torch
from pydantic import InstanceOf, validate_call
from pytorch_lightning.callbacks import Callback
from structlog import get_logger
from torch import Tensor

logger = get_logger(__name__)


class CheckpointWeightLoader(Callback):
    """Copy overlapping weights from a checkpoint into the current model.

    Unlike `ControlTransformer.load_from_checkpoint`, the model is built from the
    *current* config and only parameters present in both — with matching shapes —
    are copied. Use it to carry a pretrained backbone/encoder onto a different
    embodiment, whose token set and action heads necessarily differ.

    Place it before `ModuleFreezer` in the callback list: both act in `setup`,
    and weights must land before anything is frozen.
    """

    @validate_call
    def __init__(
        self,
        *,
        checkpoint_path: Path | None = None,
        artifact: str | None = None,
        filename: str = "model.ckpt",
        include: set[str] | None = None,
        exclude: set[str] | None = None,
    ) -> None:
        if checkpoint_path is not None and artifact is not None:
            msg = "specify at most one of `checkpoint_path`, `artifact`"
            raise ValueError(msg)

        self.checkpoint_path: Path | None = checkpoint_path
        self.artifact: str | None = artifact
        self.filename: str = filename
        self.include: set[str] = include or set()
        self.exclude: set[str] = exclude or set()
        self._loaded: bool = False

    def _resolve_path(self) -> Path | None:
        if self.checkpoint_path is not None:
            return self.checkpoint_path

        if self.artifact is None:
            return None

        import wandb  # ruff: ignore[import-outside-top-level]

        run = wandb.run
        artifact = (
            run.use_artifact(self.artifact)
            if run is not None and not run.disabled
            else wandb.Api().artifact(self.artifact, type="model")
        )

        return Path(artifact.download()) / self.filename

    def _select(
        self, source: dict[str, Tensor], target: dict[str, Tensor]
    ) -> tuple[dict[str, Tensor], list[str]]:
        selected: dict[str, Tensor] = {}
        mismatched: list[str] = []

        for key, value in source.items():
            if self.include and not any(key.startswith(p) for p in self.include):
                continue

            if any(key.startswith(p) for p in self.exclude):
                continue

            match target.get(key):
                case None:
                    continue

                case existing if existing.shape != value.shape:
                    mismatched.append(key)

                case _:
                    selected[key] = value

        return selected, mismatched

    @override
    @validate_call
    def setup(
        self,
        trainer: InstanceOf[pl.Trainer],
        pl_module: InstanceOf[pl.LightningModule],
        stage: str,
    ) -> None:
        if stage != "fit" or self._loaded:
            return

        if (path := self._resolve_path()) is None:
            logger.warning(
                "no checkpoint specified, training from scratch",
                hint="set `checkpoint_path` or `artifact`",
            )

            return

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        selected, mismatched = self._select(
            checkpoint["state_dict"], pl_module.state_dict()
        )
        missing, _ = pl_module.load_state_dict(selected, strict=False)

        logger.info(
            "loaded checkpoint weights",
            checkpoint=path.as_posix(),
            loaded=len(selected),
            shape_mismatched=len(mismatched),
            randomly_initialized=len(missing),
        )
        for key in mismatched:
            logger.debug("shape mismatch, skipped", key=key)

        self._loaded = True
