from typing import override

import pytest
import torch
from torch import Tensor
from torch.testing import assert_close, make_tensor
from torch.utils._pytree import KeyPath, keystr, tree_map_with_path  # noqa: PLC2701

from rmind.components.containers import ModuleDict


class Multiplier(torch.nn.Module):
    def __init__(self, alpha: float) -> None:
        super().__init__()

        self.alpha: float = alpha

    @override
    def forward(self, input: Tensor) -> Tensor:
        return input * self.alpha


def assert_leaves_equal(kp: KeyPath, actual: Tensor, expected: Tensor) -> None:
    assert_close(
        actual,
        expected,
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
        check_device=True,
        check_dtype=True,
        msg=lambda msg, kp=kp: f"{msg}\nkeypath: {keystr(kp)}",
    )


def test_moduledict(device: torch.device) -> None:
    transform = Multiplier(alpha=2.0).to(device)
    transforms = ModuleDict(modules={"image": transform}).to(device)

    assert set(transforms.tree_paths()) == {("image",)}

    assert transforms.get("image") is transform
    with pytest.raises(KeyError):
        _ = transforms.get(("image", "cam_front_left"))

    assert transforms.get_deepest(("image", "cam_front_left")) is transform

    with pytest.raises(KeyError):
        _ = transforms.get("does_not_exist")

    assert transforms.get("does_not_exist", default=None) is None

    x = make_tensor(8, dtype=torch.float, device=device)
    tree_map_with_path(assert_leaves_equal, transforms(x), {"image": transform(x)})

    x = {"image": {"cam_front_left": make_tensor(10, dtype=torch.float, device=device)}}
    tree_map_with_path(
        assert_leaves_equal,
        transforms(x),
        {"image": {"cam_front_left": transform(x["image"]["cam_front_left"])}},
    )


class OnLeafMultiplier(torch.nn.Module):
    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha: float = alpha

    @override
    def forward(self, input: dict[str, Tensor]) -> Tensor:
        return input["query"] * input["context"] * self.alpha


def test_moduledict_is_leaf(device: torch.device) -> None:
    on_is_leaf_transform = OnLeafMultiplier(alpha=3.0).to(device)
    regular_transform = Multiplier(alpha=2.0).to(device)

    modules = ModuleDict(
        modules={"foresight": on_is_leaf_transform, "state": regular_transform}
    ).to(device)

    x = {
        "foresight": {
            "query": make_tensor(4, 8, dtype=torch.float, device=device),
            "context": make_tensor(4, 8, dtype=torch.float, device=device),
        },
        "state": make_tensor(4, 8, dtype=torch.float, device=device),
    }

    tree_map_with_path(
        assert_leaves_equal,
        modules(
            x,
            is_leaf=lambda obj: (
                isinstance(obj, dict) and "query" in obj and "context" in obj
            ),
        ),
        {
            "foresight": on_is_leaf_transform(x["foresight"]),
            "state": regular_transform(x["state"]),
        },
    )
