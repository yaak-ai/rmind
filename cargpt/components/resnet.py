from math import prod

from jaxtyping import Float
from torch import Tensor, nn
from torchvision.models import ResNet


class ResnetBackbone(nn.Module):
    def __init__(self, resnet: ResNet, *, freeze: bool = True) -> None:
        super().__init__()

        self.resnet = resnet.requires_grad_(not freeze).train(not freeze)

    def forward(self, x: Float[Tensor, "*b c1 h1 w1"]) -> Float[Tensor, "*b c2 h2 w2"]:
        *B, C, H, W = x.shape
        x = x.view(prod(B), C, H, W)

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        *_, C, H, W = x.shape
        return x.view(*B, C, H, W)
