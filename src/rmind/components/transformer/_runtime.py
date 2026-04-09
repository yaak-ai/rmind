from collections.abc import Iterable

from torch import Tensor
from torch.nn.modules.module import Module
from torch.utils.checkpoint import checkpoint


def freeze_module(module: Module, *, freeze: bool | None) -> None:
    if freeze is not None:
        module.requires_grad_(not freeze).train(not freeze)


def run_layer_stack(
    layers: Iterable[Module], x: Tensor, *extra_args: Tensor, training: bool
) -> Tensor:
    for layer in layers:
        if training:
            x = checkpoint(layer, x, *extra_args, use_reentrant=False)
        else:
            x = layer(x, *extra_args)

    return x
