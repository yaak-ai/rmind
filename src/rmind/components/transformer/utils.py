from collections.abc import Iterable

from torch import Tensor
from torch.nn.modules.module import Module
from torch.utils.checkpoint import checkpoint


def run_layer_stack(
    layers: Iterable[Module], x: Tensor, *extra_args: Tensor, training: bool
) -> Tensor:
    for layer in layers:
        if training:
            x = checkpoint(layer, x, *extra_args, use_reentrant=False)
        else:
            x = layer(x, *extra_args)

    return x
