from typing import override

import pytorch_lightning as pl
import torch
import torchvision
from optree import tree_flatten_with_path
from pytorch_lightning.callbacks import Callback
from rbyte import Dataset
from structlog import get_logger
from torch.utils.data import DataLoader

from rmind.components.loss import LogitBiasMixin
from rmind.utils.containers import OPTREE_NAMESPACE

logger = get_logger(__name__)


class LogitBiasSetter(Callback):
    @override
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.train_dataloader is None:
            trainer.fit_loop.setup_data()

        match dataloader := trainer.train_dataloader:
            case DataLoader():
                match dataset := dataloader.dataset:
                    case Dataset():
                        pass

                    case _:
                        raise NotImplementedError
            case _:
                raise NotImplementedError

        objectives = pl_module.objectives
        targets: list[tuple[str, tuple[str, ...], LogitBiasMixin]] = []

        for objective_key, objective in objectives.items():
            loss_keys, losses, _ = tree_flatten_with_path(
                objective.losses, namespace=OPTREE_NAMESPACE
            )
            for loss_key, loss in zip(loss_keys, losses, strict=True):
                match loss:
                    case LogitBiasMixin(logit_bias=None):
                        targets.append((objective_key, loss_key, loss))

                    case _:
                        pass

        if not targets:
            return

        input_keys, batch_keys, _ = tree_flatten_with_path(
            pl_module.input_builder.keys, is_leaf=lambda x: isinstance(x, tuple)
        )
        loss_keys = {k_loss for (_, k_loss, _) in targets}
        batch_keys = {
            batch_key
            for batch_key, input_key in zip(batch_keys, input_keys, strict=True)
            if input_key in loss_keys
        }

        batch = dataset.get_batch(slice(-1), keys=batch_keys)  # pyright: ignore[reportArgumentType]

        input = (
            pl_module.input_builder.forward(batch.to(device=pl_module.device))
            .apply(torch.flatten, batch_size=[])
            # `*_diff`s contain NaNs for last timestep
            .apply(lambda x: x[~x.isnan()])
        )
        labels = pl_module.episode_builder.tokenizers.forward(input)

        for objective_key, loss_key, loss in targets:
            logger.debug("setting logit bias", objective=objective_key, loss=loss_key)
            head = objectives[objective_key].heads.get(loss_key)
            freq = torch.bincount(
                labels[*loss_key], weights=None, minlength=self._get_out_features(head)
            )
            loss.logit_bias = ((freq + 1) / freq.sum()).log()

    @staticmethod
    def _get_out_features(head: torch.nn.Module) -> int:
        match head:
            case torch.nn.Linear():
                return head.out_features
            case torch.nn.Sequential():
                try:
                    last_linear_layer = next(
                        layer
                        for layer in reversed(head)
                        if isinstance(layer, torch.nn.Linear)
                    )
                except StopIteration:
                    raise ValueError("No Linear layer found")
                else:
                    return last_linear_layer.out_features
            case _:
                msg = f"Unsupported head type: {type(head)}"
                raise NotImplementedError(msg)
