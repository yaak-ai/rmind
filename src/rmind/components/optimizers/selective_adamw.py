from typing import Any

from pydantic import InstanceOf, validate_call
from torch.nn import Module
from torch.optim.adamw import AdamW


def _partition_overrides(
    overrides: dict[str, float], decayed: set[str]
) -> dict[str, set[str]]:
    """Assign decayed param names to weight-decay override prefixes.

    Params under an override prefix (and not blacklisted -- biases/norms stay
    decay-free) get their own group with that decay instead of the global one.

    Raises:
        ValueError: on overlapping prefixes or a prefix matching nothing.
    """
    groups: dict[str, set[str]] = {prefix: set() for prefix in overrides}
    for param_name in sorted(decayed):
        matches = [
            prefix
            for prefix in overrides
            if param_name == prefix or param_name.startswith(prefix + ".")
        ]
        if len(matches) > 1:
            msg = (
                f"weight_decay_overrides prefixes overlap on '{param_name}': {matches}"
            )
            raise ValueError(msg)
        if matches:
            groups[matches[0]].add(param_name)
    for prefix, names in groups.items():
        if not names:
            msg = f"weight_decay_overrides prefix '{prefix}' matched no decayed params"
            raise ValueError(msg)
    return groups


class SelectiveAdamW(AdamW):
    """AdamW with selective weight decay.

    https://stats.stackexchange.com/questions/576463/why-not-perform-weight-decay-on-layernorm-embedding
    """

    @validate_call
    def __init__(  # noqa: C901
        self,
        module: InstanceOf[Module],
        *,
        weight_decay: float = 1e-2,
        weight_decay_module_blacklist: tuple[type[Module], ...],
        weight_decay_overrides: dict[str, float] | None = None,
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
                # and re-open the patch/goal scale gap it exists to close).
                # intra_position_gain is the same pattern for the intra-frame
                # position embedding (causal_frame.py) -- decay would pull it
                # back toward the content/position scale gap it closes.
                case "fusion_patch_gain" | "fusion_goal_gain" | "intra_position_gain":
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
                    "in_proj_weight"
                    | "cls_token"
                    | "reg_token"
                    | "readout_token"
                    | "register_tokens"
                    | "gamma_1"
                    | "gamma_2"
                ):
                    pass

                # rmind.components.lora.LoRALinear: weight-like factors adapting
                # a frozen nn.Linear, decayed same as the base weight would be
                case "lora_A" | "lora_B":
                    pass

                # rmind.components.backbone_registers.RegisterViTBackbone's
                # per-camera compression registers -- NOT the `reg_token` case
                # above (unrelated param, deliberately renamed to avoid this
                # exact collision, see that class's docstring). Decay-free:
                # initialized at N(0, 1e-6), and weight decay would fight the
                # gradient that has to grow them away from ~0.
                case "camera_reg_token":
                    weight_decay_param_blacklist.add(param_name)

                case _:
                    msg = f"Handling of param_type '{param_type}' is not implemented"
                    raise NotImplementedError(msg)

        weight_decay_param_whitelist = params.keys() - weight_decay_param_blacklist

        override_groups = _partition_overrides(
            weight_decay_overrides or {}, weight_decay_param_whitelist
        )
        for names in override_groups.values():
            weight_decay_param_whitelist -= names
        overrides = weight_decay_overrides or {}

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
            *(
                {
                    "weight_decay": overrides[prefix],
                    "params": [params[k] for k in sorted(override_groups[prefix])],
                }
                for prefix in sorted(overrides)
            ),
        ]

        super().__init__(params=param_groups, **kwargs)
