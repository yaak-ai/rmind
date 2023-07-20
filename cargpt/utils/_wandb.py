from pathlib import Path
from typing import Callable, Dict, List

import more_itertools as mit
import wandb
from einops import rearrange
from jaxtyping import Float
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities.parsing import AttributeDict
from torch import Tensor
from wandb.sdk.interface.artifacts import Artifact
from wandb.sdk.lib import RunDisabled
from wandb.wandb_run import Run


class LoadableFromArtifact:
    @classmethod
    def load_from_wandb_artifact(cls, name: str, **kwargs):
        get_artifact: Callable[..., Artifact] = (
            wandb.run.use_artifact
            if wandb.run is not None and not isinstance(wandb.run, RunDisabled)
            else wandb.Api().artifact
        )

        artifact = get_artifact(name, type="model")
        artifact_dir = artifact.download()
        ckpt_path = mit.one(Path(artifact_dir).glob("*.ckpt"))

        return cls.load_from_checkpoint(ckpt_path.as_posix(), **kwargs)  # type: ignore


class ValOutputsLoggingTableMixin:
    trainer: Trainer
    logger: WandbLogger
    hparams: AttributeDict

    @property
    def val_table_main_columns(self):
        return getattr(self, "_val_table_main_columns")

    @val_table_main_columns.setter
    def val_table_main_columns(self, columns: List[str]):
        setattr(self, "_val_table_main_columns", list(columns))

    @property
    def val_table(self):
        return getattr(self, "_val_table", None)

    @val_table.setter
    def val_table(self, wandb_table: wandb.Table):
        setattr(self, "_val_table", wandb_table)

    @val_table.deleter
    def val_table(self):
        delattr(self, "_val_table")

    def is_outputs_logging_active(self):
        return (
            isinstance(logger := self.logger, WandbLogger)
            and isinstance(logger.experiment, Run)
            and self.hparams.get("log", {}).get("validation", {}).get("outputs")
            and self.trainer.state.stage != "sanity_check"
        )

    def _init_val_outputs_logging(self, outputs_dict: Dict[str, Tensor]):
        if not self.is_outputs_logging_active():
            return

        if self.val_table is not None:
            return

        if set(outputs_dict.keys()) != set(self.val_table_main_columns):
            raise ValueError(
                f"different keys provided {list(outputs_dict)} than "
                f"declared in val_table_main_columns"
            )

        columns = []
        for key in self.val_table_main_columns:
            _, TS = outputs_dict[key].shape
            for ts in range(TS):
                columns.append(f"{key}_{ts}")

        self.val_table = wandb.Table(columns=columns)

    def _finish_val_outputs_logging(self):
        if not self.is_outputs_logging_active():
            return

        run: Run = self.logger.experiment

        assert self.val_table is not None
        self.val_table.add_column("_step", list(map(int, self.val_table.get_index())))
        artifact = wandb.Artifact(f"run-{run.id}-val_outputs", "run_table")
        artifact.add(self.val_table, "outputs")
        run.log_artifact(artifact)
        # Cleanup after epoch
        del self.val_table

    def _log_val_outputs_dict(self, outputs_dict: Dict[str, Float[Tensor, "b col ts"]]):
        if not self.is_outputs_logging_active():
            return

        self._init_val_outputs_logging(outputs_dict=outputs_dict)

        data: Float[Tensor, "b C"] = rearrange(
            [outputs_dict[column] for column in self.val_table_main_columns],
            "col b ts -> b (col ts)",
        )

        assert self.val_table is not None
        for row in data.tolist():
            self.val_table.add_data(*row)
