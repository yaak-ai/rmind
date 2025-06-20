from typing import override

from torch import Tensor, nn


class IdentityEncoder(nn.Module):
    @override
    def __init__(self) -> None:
        super().__init__()

    @override
    def forward(self, *, src: Tensor, mask: Tensor) -> Tensor:
        return src
