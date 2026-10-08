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
    def __init__(  # ruff: ignore[complex-structure]
        self,
        module: InstanceOf[Module],
        *,
        weight_decay: float = 1e-2,
        weight_decay_module_blacklist: tuple[type[Module], ...],
        weight_decay_overrides: dict[str, float] | None = None,
        lr_overrides: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        """`lr_overrides` `{module prefix: lr}`: those params get their own lr
        (each weight-decay group is split by prefix; e.g. a fine-tuned image
        encoder at 1e-5 under a 1e-4 trunk). Params that never get a gradient
        (frozen) are left out of every group, so they cannot be decayed.

        Raises:
            ValueError: on `params` in kwargs, zero weight decay, or an
                `lr_overrides` prefix matching no trainable parameter.
        """
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

                # nero: learned substitution tokens (no_goal / no_depth / no_hand
                # stand in for a missing input and must not be pulled toward 0),
                # the token-scale gain of NormedTokenEmbedding (same reason as the
                # fusion gains) and AxisShrinkage's threshold (playbook: never
                # weight-decay tau)
                case (
                    "no_goal" | "no_depth" | "no_hand" | "token_gain" | "raw_threshold"
                ):
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

        if lr_overrides:
            param_groups = _split_lr_groups(
                param_groups, lr_overrides, {id(p): n for n, p in params.items()}
            )

        super().__init__(params=param_groups, **kwargs)


def _split_lr_groups(
    groups: list[dict[str, Any]], lr_overrides: dict[str, float], names: dict[int, str]
) -> list[dict[str, Any]]:
    """Split every group by `lr_overrides` prefix; trainable params only.

    Raises:
        ValueError: on a prefix that matches no trainable parameter.
    """
    used: set[str] = set()
    out: list[dict[str, Any]] = []
    for group in groups:
        rest = []
        by_prefix: dict[str, list[Any]] = {
            prefix: [] for prefix in sorted(lr_overrides)
        }
        for param in group["params"]:
            if not param.requires_grad:
                continue
            name = names[id(param)]
            match = next(
                (
                    prefix
                    for prefix in sorted(lr_overrides)
                    if name == prefix or name.startswith(prefix + ".")
                ),
                None,
            )
            if match is None:
                rest.append(param)
            else:
                by_prefix[match].append(param)
                used.add(match)
        if rest:
            out.append({**group, "params": rest})
        out.extend(
            {**group, "params": ps, "lr": lr_overrides[prefix]}
            for prefix, ps in by_prefix.items()
            if ps
        )
    missing = set(lr_overrides) - used
    if missing:
        msg = f"lr_overrides prefixes {sorted(missing)} match no trainable parameter"
        raise ValueError(msg)
    return out
