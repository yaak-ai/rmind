"""ARM-SELECTION metric on val for the bimanual nero patch policy (callback).

Every val batch (sanity check excluded), the policy's newest code argmax is
decoded for every (window, frame, side) row -- argmax, never sampled, so the
number does not depend on `sample_codes` -- mapped to ABSOLUTE robot units with
the model's own `_absolute` (the tokenizer standardizer + relative anchor), and
compared with the demonstration's chunk through `side_excursion` over the
executed horizon (`EXEC_STEPS` real steps). At the end of the val epoch the frames
are grouped by their take's active-arm class (the shared split JSON) and
`rmind.data.nero_bimanual.arm_selection` is logged as `val/arm_select/...`, the
same keys the ACT family logs.

Needs the take id per sample (`input_id`), which the bimanual datamodules' val
collate (`flat_rbyte_batch_with_ids`) adds; a batch without it is skipped. Costs
one extra trunk forward per val batch (no model change: it reuses the policy's own
row/decode helpers, read-only), i.e. roughly doubles val compute on the epochs it
runs; `every_n_epochs` (yaml: `nero_arm_select_every_n_epochs`) thins it out --
it always runs on the last epoch.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, override

import numpy as np
import pytorch_lightning as pl
import torch

from rmind.data.nero_bimanual import EXEC_STEPS, INPUT_ID, arm_selection, side_excursion
from rmind.data.nero_robot import ARM_AXES, SIDES

__all__ = ["NeroArmSelectionLogger"]

_ARM = slice(ARM_AXES[0], ARM_AXES[-1] + 1)


class NeroArmSelectionLogger(pl.Callback):
    def __init__(
        self,
        *,
        split_file: str | Path | None = None,
        exec_steps: int = EXEC_STEPS,
        every_n_epochs: int = 1,
    ) -> None:
        from rmind.scripts.nero_split_lib import (  # noqa: PLC0415
            SPLIT_JSON,
            take_classes,
        )

        self.classes = take_classes(Path(split_file) if split_file else SPLIT_JSON)
        self.exec_steps = int(exec_steps)
        if int(every_n_epochs) < 1:
            msg = f"every_n_epochs must be >= 1, got {every_n_epochs}"
            raise ValueError(msg)
        self.every_n_epochs = int(every_n_epochs)
        self._reset()

    def active(self, trainer: pl.Trainer) -> bool:
        """Whether this val epoch computes the metric (every n-th, and the last)."""
        if trainer.sanity_checking:
            return False
        epoch = int(trainer.current_epoch)
        max_epochs = trainer.max_epochs
        # max_epochs -1 / None: unbounded, no last epoch
        last = max_epochs is not None and max_epochs > 0 and epoch + 1 >= max_epochs
        return last or (epoch + 1) % self.every_n_epochs == 0

    def _reset(self) -> None:
        self._pred: list[np.ndarray] = []
        self._demo: list[np.ndarray] = []
        self._take: list[np.ndarray] = []

    @override
    def on_validation_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self._reset()

    @torch.no_grad()
    def excursions(self, pl_module: Any, batch: Any) -> tuple[np.ndarray, np.ndarray]:
        """(b*T, S) predicted and demonstrated arm excursions (nan for an invalid side)."""
        features = pl_module._features(batch)  # noqa: SLF001
        rows = pl_module._robot_rows(batch, features)  # noqa: SLF001
        decoded = pl_module._decode(rows["offsets"], rows["code_logits"].argmax(dim=-1))  # noqa: SLF001
        anchor = pl_module._anchor_rows(batch, rows["row_valid"])  # noqa: SLF001
        pred = pl_module._absolute(decoded, anchor, rows["side"]).float()  # noqa: SLF001
        demo = pl_module._absolute(rows["target"], anchor, rows["side"]).float()  # noqa: SLF001
        steps = rows["real"].clone()
        steps[:, self.exec_steps :] = False
        state = anchor.float().cpu().numpy()
        valid = steps.cpu().numpy()
        row_valid = rows["row_valid"].cpu().numpy()
        out = []
        for chunk in (pred, demo):
            exc = np.full(row_valid.shape, np.nan)
            exc[row_valid] = side_excursion(chunk.cpu().numpy(), state, valid, _ARM)
            out.append(exc.reshape(-1, len(SIDES)))
        return out[0], out[1]

    @override
    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if not self.active(trainer) or INPUT_ID not in batch:
            return
        if not getattr(pl_module, "robot", False):
            return
        with trainer.precision_plugin.forward_context():
            pred, demo = self.excursions(pl_module, batch)
        frames = pred.shape[0] // len(batch[INPUT_ID])
        self._pred.append(pred)
        self._demo.append(demo)
        self._take.append(np.repeat(np.asarray(batch[INPUT_ID], dtype=object), frames))

    def compute(self) -> dict[str, float]:
        if not self._pred:
            return {}
        take = np.concatenate(self._take)
        unknown = sorted({t for t in take.tolist() if t not in self.classes})
        if unknown:
            msg = f"val takes missing from the split file's class table: {unknown}"
            raise KeyError(msg)
        cls = np.asarray([self.classes[t] for t in take.tolist()], dtype=object)
        return arm_selection(
            "val/",
            tuple(SIDES),
            np.concatenate(self._pred),
            np.concatenate(self._demo),
            take,
            cls,
        )

    @override
    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if trainer.sanity_checking:
            return
        metrics = {k: v for k, v in self.compute().items() if math.isfinite(v)}
        if metrics:
            pl_module.log_dict(metrics, sync_dist=False, rank_zero_only=True)
        self._reset()
