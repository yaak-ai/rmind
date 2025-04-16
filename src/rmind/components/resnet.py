from math import prod
from typing import override

from torch import Tensor, nn
from torchvision.models import ResNet


class ResnetBackbone(nn.Module):
    def __init__(self, resnet: ResNet, *, freeze: bool | None = None) -> None:
        super().__init__()

        self.resnet: ResNet = resnet

        if freeze is not None:
            self.requires_grad_(not freeze).train(not freeze)  # pyright: ignore[reportUnusedCallResult]

    @override
    def forward(self, x: Tensor) -> Tensor:
        *b, c, h, w = x.shape
        x = x.view(prod(b), c, h, w)

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        *_, c, h, w = x.shape
        return x.view(*b, c, h, w)
