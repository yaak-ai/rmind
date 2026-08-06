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
            # top-level parameters (e.g. PatchPolicy's fusion gains) have no
            # module prefix; their "submodule" is the root module itself
            submodule_name, _, param_type = param_name.rpartition(".")
            match param_type:
                # fusion_norm scale gains: scalar calibration parameters,
                # no weight decay (decay would pull the goal gain toward 0
                # and re-open the patch/goal scale gap it exists to close)
                case "fusion_patch_gain" | "fusion_goal_gain":
                    weight_decay_param_blacklist.add(param_name)
                case "weight":
                    if isinstance(
                        submodules[submodule_name], weight_decay_module_blacklist
                    ):
                        weight_decay_param_blacklist.add(param_name)

                case "bias" | "in_proj_bias":
                    weight_decay_param_blacklist.add(param_name)

                # https://github.com/pytorch/pytorch/blob/v2.7.0/torch/nn/modules/activation.py#L1091
                # `pos_embed`/`gamma` (timm ViT positional embedding / LayerScale,
                # e.g. DINOv2): keep weight decay off, matching Embedding/LayerNorm
                case "pos_embed" | "gamma":
                    weight_decay_param_blacklist.add(param_name)

                case (
                    "in_proj_weight" | "cls_token" | "reg_token" | "gamma_1" | "gamma_2"
                ):
                    pass

                case _:
                    msg = f"Handling of param_type '{param_type}' is not implemented"
                    raise NotImplementedError(msg)

        weight_decay_param_whitelist = params.keys() - weight_decay_param_blacklist

        # sorted: set iteration order is salted per process, and torch's
        # Optimizer.load_state_dict maps saved state onto params POSITIONALLY --
        # unsorted groups corrupt Adam moments on any cross-process resume
        param_groups = [
            {
                "weight_decay": 0.0,
                "params": [params[k] for k in sorted(weight_decay_param_blacklist)],
            },
            {
                "weight_decay": weight_decay,
                "params": [params[k] for k in sorted(weight_decay_param_whitelist)],
            },
        ]

        super().__init__(params=param_groups, **kwargs)
