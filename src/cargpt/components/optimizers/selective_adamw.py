from collections.abc import Iterable
from typing import Any

from loguru import logger
from torch.nn import Module
from torch.optim.adamw import AdamW


class SelectiveAdamW(AdamW):
    """
    AdamW with selective weight decay

    https://stats.stackexchange.com/questions/576463/why-not-perform-weight-decay-on-layernorm-embedding
    """

    def __init__(
        self,
        module: Module,
        *,
        weight_decay: float = 1e-2,
        weight_decay_module_blacklist: Iterable[type[Module]],
        **kwargs: dict[str, Any],
    ):
        if "params" in kwargs or weight_decay == 0.0:
            raise ValueError

        weight_decay_module_blacklist = tuple(weight_decay_module_blacklist)
        weight_decay_param_blacklist = set()

        for submodule_name, submodule in module.named_modules():
            for k, _ in submodule.named_parameters():
                param_name = f"{submodule_name}.{k}" if submodule_name else k
                _, param_type = param_name.rsplit(".", maxsplit=1)

                match param_type:
                    case "weight":
                        if (
                            isinstance(submodule, weight_decay_module_blacklist)
                            | ("depth_decoder" in param_name)
                            | ("pose_decoder" in param_name)
                        ):
                            weight_decay_param_blacklist.add(param_name)

                    case "bias":
                        weight_decay_param_blacklist.add(param_name)

                    case _:
                        raise NotImplementedError

        logger.trace("overriding weight_decay=0", params=weight_decay_param_blacklist)

        params = dict(module.named_parameters())
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

        super().__init__(params=param_groups, **kwargs)  # pyright: ignore[reportArgumentType]
