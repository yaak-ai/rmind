"""torchvision ResNet trunk as a patch-token image encoder (the ACT backbone).

`ResNetBackbone("resnet18", weights="IMAGENET1K_V1")` is the trunk lerobot's ACT
builds: torchvision ImageNet weights, `FrozenBatchNorm2d` (batch statistics AND
affine fixed at the ImageNet values -- the BN layers never update, so train and
eval forwards agree and the export needs no BN folding decision), avgpool/fc
dropped. The output is the final (stride-32) feature map `(..., C, H/32, W/32)`,
`C` = 512 for resnet18/34; flatten it to tokens with `einops Rearrange` exactly
like the timm backbone. Expects ImageNet-normalised float input (the model's
`image_transform`).
"""

from __future__ import annotations

from math import prod
from typing import override

import torchvision
from torch import Tensor, nn
from torchvision.ops.misc import FrozenBatchNorm2d


class ResNetBackbone(nn.Module):
    def __init__(
        self,
        model_name: str = "resnet18",
        *,
        weights: str | None = "IMAGENET1K_V1",
        frozen_bn: bool = True,
    ) -> None:
        super().__init__()
        builder = getattr(torchvision.models, model_name)
        resnet = builder(
            weights=weights,
            norm_layer=FrozenBatchNorm2d if frozen_bn else nn.BatchNorm2d,
        )
        # everything up to layer4: conv1 bn1 relu maxpool layer1..layer4
        self.body = nn.Sequential(*list(resnet.children())[:-2])
        self.out_channels = int(resnet.fc.in_features)

    @override
    def forward(self, input: Tensor) -> Tensor:
        *b, c, h, w = input.shape
        x = self.body(input.reshape(prod(b), c, h, w))
        return x.reshape(*b, *x.shape[1:])
