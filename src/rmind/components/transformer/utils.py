from collections.abc import Iterable

from torch import Tensor
from torch.nn.modules.module import Module
from torch.utils.checkpoint import checkpoint


def run_layer_stack(
    layers: Iterable[Module], x: Tensor, *extra_args: object, training: bool
) -> Tensor:
    """Run `layers` in sequence, checkpointing each one while training.

    `extra_args` are forwarded unchanged and are not required to be tensors: the
    FlexAttention path passes a `BlockMask` here, which `checkpoint` treats as an
    ordinary non-differentiable argument (it is stored by reference and re-passed
    on recompute, exactly like the bool mask it replaces).
    """
    for layer in layers:
        if training:
            x = checkpoint(layer, x, *extra_args, use_reentrant=False)
        else:
            x = layer(x, *extra_args)

    return x
