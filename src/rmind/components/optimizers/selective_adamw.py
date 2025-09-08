from typing import Any

from pydantic import InstanceOf, validate_call
from torch.nn import Module
from torch.optim.adamw import AdamW


class SelectiveAdamW(AdamW):
    """AdamW with selective weight decay.

    https://stats.stackexchange.com/questions/576463/why-not-perform-weight-decay-on-layernorm-embedding
    """

    @validate_call
    def __init__(
        self,
        module: InstanceOf[Module],
        *,
        weight_decay: float = 1e-2,
        weight_decay_module_blacklist: tuple[type[Module], ...],
        **kwargs: Any,
    ) -> None:
        if "params" in kwargs or weight_decay == 0.0:
            raise ValueError

        weight_decay_param_blacklist = set()
        submodules = dict(module.named_modules())
        params = dict(module.named_parameters())
        for param_name in params:
            submodule_name, param_type = param_name.rsplit(sep=".", maxsplit=1)
            match param_type:
                case "weight":
                    if isinstance(
                        submodules[submodule_name], weight_decay_module_blacklist
                    ):
                        weight_decay_param_blacklist.add(param_name)

                case "bias" | "in_proj_bias":
                    weight_decay_param_blacklist.add(param_name)

                case "mask_embedding":
                    weight_decay_param_blacklist.add(param_name)

                # https://github.com/pytorch/pytorch/blob/v2.7.0/torch/nn/modules/activation.py#L1091
                case "in_proj_weight":
                    pass

                case _:
                    msg = f"Optimizer does not have a rule for parameter type: {param_type}"
                    raise NotImplementedError(msg)
        weight_decay_param_whitelist = params.keys() - weight_decay_param_blacklist

        param_groups = [
            {
                "weight_decay": 0.0,
                "params": [params[k] for k in weight_decay_param_blacklist],
            },
            {
                "weight_decay": weight_decay,
                "params": [params[k] for k in weight_decay_param_whitelist],
            },
        ]

        super().__init__(params=param_groups, **kwargs)
