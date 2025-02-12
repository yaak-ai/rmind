from typing import override

import pytorch_lightning as pl
import torch
from loguru import logger
from optree import tree_flatten_with_path
from pytorch_lightning.callbacks import Callback

from cargpt.components.loss import LogitBiasMixin
from cargpt.utils.containers import OPTREE_NAMESPACE


class LogitBiasSetter(Callback):
    @override
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        objectives = pl_module.objectives
        targets = []

        for objective_key, objective in pl_module.objectives.items():
            loss_keys, losses, _ = tree_flatten_with_path(
                objective.losses, namespace=OPTREE_NAMESPACE
            )
            for loss_key, loss in zip(loss_keys, losses, strict=True):
                match loss:
                    case LogitBiasMixin(logit_bias=None):
                        targets.append((objective_key, loss_key, loss))

        input_keys, batch_keys, _ = tree_flatten_with_path(
            pl_module.input_builder.keys, is_leaf=lambda x: isinstance(x, tuple)
        )
        loss_keys = {k_loss for (_, k_loss, _) in targets}
        batch_keys = {
            batch_key
            for batch_key, input_key in zip(batch_keys, input_keys, strict=True)
            if input_key in loss_keys
        }
        dataset = trainer.datamodule.train_dataloader().dataset  # pyright: ignore[reportAttributeAccessIssue]
        batch = dataset.get_batch(slice(-1), keys=batch_keys).to(pl_module.device)

        input = (
            pl_module.input_builder.forward(batch)
            .apply(torch.flatten, batch_size=[])
            .apply(lambda x: x[~x.isnan()])  # `*_diff`s contain NaNs for last timestep
        )
        labels = pl_module.episode_builder.tokenizers.forward(input)

        for objective_key, loss_key, loss in targets:
            logger.debug("setting logit bias", objective=objective_key, loss=loss_key)
            head = objectives[objective_key].heads.get(loss_key)
            freq = torch.bincount(
                labels[*loss_key], weights=None, minlength=head.out_features
            )
            loss.logit_bias = ((freq + 1) / freq.sum()).log()
