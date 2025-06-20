from typing import override

import pytest
import torch
from torch import Tensor
from torch.testing import make_tensor
from torch.utils._pytree import (
    tree_leaves,  # noqa: PLC2701
    tree_map,  # noqa: PLC2701
)

from rmind.components.containers import ModuleDict


class Multiplier(torch.nn.Module):
    def __init__(self, alpha: float) -> None:
        super().__init__()

        self.alpha: float = alpha

    @override
    def forward(self, input: Tensor) -> Tensor:
        return input * self.alpha


def test_moduledict() -> None:
    transform = Multiplier(alpha=2.0)
    transforms = ModuleDict(modules={"image": transform})

    assert set(transforms.tree_paths()) == {("image",)}

    assert transforms.get("image") is transform
    with pytest.raises(KeyError):
        _ = transforms.get(("image", "cam_front_left"))

    assert transforms.get_deepest(("image", "cam_front_left")) is transform

    with pytest.raises(KeyError):
        _ = transforms.get("does_not_exist")

    assert transforms.get("does_not_exist", default=None) is None

    x = make_tensor(8, dtype=torch.float, device="cpu")
    assert all(
        tree_leaves(tree_map(torch.equal, transforms(x), {"image": transform(x)}))
    )

    x = {"image": {"cam_front_left": make_tensor(10, dtype=torch.float, device="cpu")}}

    assert all(
        tree_leaves(
            tree_map(
                torch.equal,
                transforms(x),
                {"image": {"cam_front_left": transform(x["image"]["cam_front_left"])}},
            )
        )
    )
