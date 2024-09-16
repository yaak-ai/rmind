from collections.abc import Sequence

import torch
from jaxtyping import Float
from torch import Tensor, nn
from typing_extensions import override

# 6D: x (right), y (down), z (fwd), theta_x, theta_y, theta_z
Pose = Float[Tensor, "... 6"]


class PoseDecoder(nn.Module):
    def __init__(self, in_channels, out_mask: Sequence[int], *, freeze=False):
        super().__init__()

        self._out_mask = torch.tensor(out_mask).to(torch.bool)

        out_channels = self._out_mask.count_nonzero().item()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 256, 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, int(out_channels), 1),
        )

        if freeze:
            _ = self.requires_grad_(False)

    @override
    def forward(self, x) -> Pose:
        x = self.net.forward(x)
        x = x.mean(3).mean(2)

        out = torch.full((x.shape[0], 6), torch.nan).to(x.device)
        out[:, self._out_mask] = x

        return out
