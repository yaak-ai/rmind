from typing import Any, ClassVar

import torch
from tensordict import TensorDict
from torch import Tensor


class RuleBasedCluster:
    """Assigns named cluster labels by applying priority-ordered rules to batch fields.

    Each field is reduced to a per-episode scalar [B].
    Rules are evaluated in order; the first matching rule wins.
    Episodes matching no rule get the `default` label.

    Supported condition ops: ge, lt, abs_ge, abs_lt.
    """

    _OPS: ClassVar = {
        "ge": lambda v, t: v >= t,
        "lt": lambda v, t: v < t,
        "abs_ge": lambda v, t: v.abs() >= t,
        "abs_lt": lambda v, t: v.abs() < t,
    }

    def __init__(
        self, fields: dict[str, dict], rules: list[dict], default: str = "other"
    ) -> None:
        self._fields = fields
        self._rules = rules
        self._default = default

    @staticmethod
    def _extract(data: dict[str, Any], spec: dict) -> Tensor:
        values: Tensor = data["data"][spec["key"]]
        match reduce := spec["reduce"]:
            case "last":
                result = values[:, -1]  # (b,)
            case "last_diff":
                result = values[:, -1] - values[:, -2]  # (b,)
            case _:
                msg = f"unsupported reduce: {reduce!r}"
                raise ValueError(msg)
        return result

    def __call__(
        self, batch: dict[str, TensorDict], _predictions: TensorDict
    ) -> list[str]:
        scalars = {
            name: self._extract(batch, spec).cpu()
            for name, spec in self._fields.items()
        }
        b = next(iter(scalars.values())).shape[0]

        labels = [self._default] * b
        assigned = torch.zeros(b, dtype=torch.bool)

        for rule in self._rules:
            mask = torch.ones(b, dtype=torch.bool)
            for field, cond in rule["when"].items():
                for op, threshold in cond.items():
                    mask &= self._OPS[op](scalars[field], threshold)

            newly_matched = mask & ~assigned
            for i in newly_matched.nonzero(as_tuple=True)[0].tolist():
                labels[i] = rule["name"]
            assigned |= newly_matched

        return labels
