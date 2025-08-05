from typing import override

import pytorch_lightning as pl
import rbyte
import torch
from pydantic import InstanceOf, validate_call
from pytorch_lightning.callbacks import Callback
from structlog import get_logger
from tensordict import TensorClass, TensorDict
from torch.utils._pytree import (
    KeyPath,
    key_get,  # noqa: PLC2701
    tree_flatten_with_path,  # noqa: PLC2701
    tree_map,  # noqa: PLC2701
)

from rmind.components.loss import LogitBiasMixin
from rmind.models.control_transformer import ControlTransformer
from rmind.utils.pytree import path_to_key

logger = get_logger(__name__)


class LogitBiasSetter(Callback):
    @override
    @validate_call
    def on_fit_start(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, trainer: InstanceOf[pl.Trainer], pl_module: InstanceOf[ControlTransformer]
    ) -> None:
        objectives = pl_module.objectives
        losses: list[tuple[str, KeyPath, LogitBiasMixin]] = []

        for objective_key, objective in objectives.items():
            for loss_keypath, loss in tree_flatten_with_path(objective.losses)[0]:
                match loss:
                    case LogitBiasMixin(logit_bias=None):
                        losses.append((objective_key, loss_keypath, loss))

                    case _:
                        pass

        if not losses:
            return

        keypaths, _ = tree_flatten_with_path(
            pl_module.episode_builder.input_transform[0].paths,  # pyright: ignore[reportIndexIssue, reportAttributeAccessIssue]
            is_leaf=lambda x: isinstance(x, tuple),
        )
        loss_keypaths = {keypath for (_, keypath, _) in losses}
        batch_keys = {
            path_to_key(batch_keypath)
            for input_keypath, batch_keypath in keypaths
            if input_keypath in loss_keypaths
        }

        if trainer.train_dataloader is None:
            trainer.fit_loop.setup_data()

        match dataset := trainer.train_dataloader.dataset:  # pyright: ignore[reportOptionalMemberAccess]
            case rbyte.Dataset():
                batch = dataset.get_batch(slice(-1), keys=batch_keys).to_tensordict()  # pyright: ignore[reportArgumentType]

            case TensorDict() | TensorClass():  # used in tests
                batch = dataset.select(*batch_keys).to_tensordict()

            case _:
                raise NotImplementedError

        with torch.inference_mode():
            input = pl_module.episode_builder.input_transform(  # pyright: ignore[reportCallIssue]
                batch.to(device=pl_module.device).to_dict()
            )
            input = tree_map(
                lambda x: torch.flatten(
                    x[~x.isnan()]  # `*_diff`s contain NaNs for last timestep
                )
                if x is not None
                else None,
                input,
            )
            labels = pl_module.episode_builder.tokenizers(input)  # pyright: ignore[reportCallIssue]

        for objective_key, loss_keypath, loss in losses:
            logger.debug(
                "setting logit bias", objective=objective_key, loss=loss_keypath
            )
            loss_head = key_get(objectives[objective_key].heads, loss_keypath)
            loss_labels = key_get(labels, loss_keypath)
            minlength = self._get_out_features(loss_head)
            freq = torch.bincount(input=loss_labels, weights=None, minlength=minlength)
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
                    msg = "No Linear layer found"
                    raise ValueError(msg) from None
                else:
                    return last_linear_layer.out_features
            case _:
                msg = f"Unsupported head type: {type(head)}"
                raise NotImplementedError(msg)
