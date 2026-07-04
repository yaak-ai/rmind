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
        if "params" in kwargs or weight_decay == 0.0:  # noqa: RUF069
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

                # https://github.com/pytorch/pytorch/blob/v2.7.0/torch/nn/modules/activation.py#L1091
                case (
                    "in_proj_weight"
                    | "cls_token"
                    | "reg_token"
                    | "query"
                    | "gamma_1"
                    | "gamma_2"
                ):
                    pass

                # GRU/LSTM weight matrices (weight_ih_l0, weight_hh_l0, …) — apply weight decay
                case _ if param_type.startswith("weight_") and "_l" in param_type:
                    pass

                # GRU/LSTM bias vectors (bias_ih_l0, bias_hh_l0, …) — no weight decay
                case _ if param_type.startswith("bias_") and "_l" in param_type:
                    weight_decay_param_blacklist.add(param_name)

                case _:
                    msg = f"Handling of param_type '{param_type}' is not implemented"
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
