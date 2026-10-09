"""patch/sweep: SelectiveAdamW lr_overrides and the ResNet patch encoder."""

import pytest
import torch
from torch import nn

from rmind.components.optimizers.selective_adamw import SelectiveAdamW
from rmind.components.resnet_backbone import ResNetBackbone


class _Toy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.image_encoder = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
        self.trunk = nn.Linear(4, 4)
        self.frozen = nn.Linear(4, 4).requires_grad_(False)  # noqa: FBT003


def test_lr_overrides_split_groups_and_skip_frozen() -> None:
    model = _Toy()
    opt = SelectiveAdamW(
        model,
        lr=1e-4,
        weight_decay=0.1,
        weight_decay_module_blacklist=(nn.LayerNorm,),
        lr_overrides={"image_encoder": 1e-5},
    )
    names = {id(p): n for n, p in model.named_parameters()}
    seen = {}
    for group in opt.param_groups:
        for p in group["params"]:
            seen[names[id(p)]] = (group["lr"], group["weight_decay"])
    assert seen["image_encoder.0.weight"] == (1e-5, 0.1)
    assert seen["image_encoder.1.weight"] == (1e-5, 0.0)  # LayerNorm: no decay
    assert seen["trunk.weight"] == (1e-4, 0.1)
    assert seen["trunk.bias"] == (1e-4, 0.0)
    assert not any(n.startswith("frozen") for n in seen)


def test_lr_overrides_unknown_prefix_refused() -> None:
    with pytest.raises(ValueError, match="match no trainable"):
        SelectiveAdamW(
            _Toy(),
            lr=1e-4,
            weight_decay=0.1,
            weight_decay_module_blacklist=(nn.LayerNorm,),
            lr_overrides={"nope": 1e-5},
        )


def test_resnet_backbone_grid_and_frozen_bn() -> None:
    m = ResNetBackbone("resnet18", weights=None)
    out = m(torch.rand(2, 1, 3, 416, 640))
    assert out.shape == (2, 1, 512, 13, 20)
    # FrozenBatchNorm2d: statistics/affine are buffers, never parameters
    assert not any("bn" in n or "downsample.1" in n for n, _ in m.named_parameters())
