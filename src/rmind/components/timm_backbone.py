from math import prod

from typing_extensions import override

from timm import create_model
from torch import Tensor, nn


class TimmBackbone(nn.Module):
    def __init__(
        self,
        model_name: str = "vit_small_patch16_dinov3.lvd1689m",
        *,
        freeze: bool | None = None,
        out_indices: list[int] | None = None,
        img_size: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.model: nn.Module = create_model(
            model_name,
            pretrained=True,
            features_only=True,
            out_indices=out_indices,
            img_size=img_size,
        )

        if freeze is not None:
            self.requires_grad_(not freeze).train(not freeze)

    @override
    def forward(self, input: Tensor) -> Tensor:
        *b, c, h, w = input.shape
        x = input.view(prod(b), c, h, w)
        x = self.model(x)[-1]
        *_, c, h, w = x.shape
        return x.view(*b, c, h, w)
