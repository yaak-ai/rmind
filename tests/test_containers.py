from typing import override

import pytest
import torch
from tensordict import TensorDict
from torch import Tensor
from torch.testing import make_tensor

from rmind.utils.containers import ModuleDict


class Multiplier(torch.nn.Module):
    def __init__(self, alpha: float) -> None:
        super().__init__()

        self.alpha: float = alpha

    @override
    def forward(self, x: Tensor) -> Tensor:
        return x * self.alpha


def test_moduledict() -> None:
    image_transform = Multiplier(alpha=2.0)
    transforms = ModuleDict(image=image_transform)

    assert set(transforms.tree_paths()) == {("image",)}

    assert transforms.get("image") is image_transform
    with pytest.raises(KeyError):
        _ = transforms.get(("image", "cam_front_left"))

    assert transforms.get_deepest(("image", "cam_front_left")) is image_transform

    with pytest.raises(KeyError):
        _ = transforms.get("does_not_exist")

    assert transforms.get("does_not_exist", default=None) is None

    x = make_tensor(8, dtype=torch.float, device="cpu")
    assert (transforms.forward(x) == TensorDict({"image": image_transform(x)})).all()  # pyright: ignore[reportAttributeAccessIssue]

    x = TensorDict({
        "image": {"cam_front_left": make_tensor(10, dtype=torch.float, device="cpu")}  # pyright: ignore[reportArgumentType]
    })

    assert (
        transforms.forward(x)
        == TensorDict({
            "image": {"cam_front_left": image_transform(x["image"]["cam_front_left"])}  # pyright: ignore[reportArgumentType]
        })
    ).all()  # pyright: ignore[reportAttributeAccessIssue]
