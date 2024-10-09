from collections.abc import Sequence
from math import prod

import torch
from torch import Tensor, nn
from typing_extensions import override

from .types import Pose


class PoseDecoder(nn.Module):
    def __init__(self, in_channels, out_mask: Sequence[int], *, freeze=False):
        super().__init__()

        self._out_mask = torch.tensor(out_mask).to(torch.bool)

        out_channels = self._out_mask.count_nonzero().item()

        self.net = nn.Sequential(
            nn.Linear(in_channels, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, int(out_channels), 1)
        )

        if freeze:
            _ = self.requires_grad_(False)

    @override
    def forward(self, x: Tensor) -> Pose:
        *b, c = x.shape
        x = self.net.forward(x)

        out = torch.full_like(x, torch.nan).to(x.device)
        out[..., self._out_mask] = x

        return out.view(*b, 6)
