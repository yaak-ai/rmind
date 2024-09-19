from math import prod
from typing import override

from jaxtyping import Float
from tensordict import TensorDict
from torch import Tensor, nn
from torchvision.models import ResNet


class ResnetBackbone(nn.Module):
    def __init__(self, resnet: ResNet, *, freeze: bool | None = None) -> None:
        super().__init__()

        self.resnet = resnet

        if freeze is not None:
            self.requires_grad_(not freeze).train(not freeze)  # pyright: ignore[reportUnusedCallResult]

    @override
    def forward(
        self,
        x: Float[Tensor, "*b c1 h1 w1"],
        # ) -> list[Float[Tensor, "*b c2 h2 w2"]]
    ) -> TensorDict:
        *b, c, h, w = x.shape
        x = x.view(prod(b), c, h, w)

        out_t = TensorDict({})

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        out_t["0"] = x
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        out_t["1"] = x
        x = self.resnet.layer2(x)
        out_t["2"] = x
        x = self.resnet.layer3(x)
        out_t["3"] = x
        x = self.resnet.layer4(x)
        out_t["4"] = x

        return out_t.apply(lambda x: x.view(*b, *x.shape[-3:]), batch_size=b)
