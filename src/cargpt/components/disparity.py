from collections import OrderedDict
from math import prod

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from jaxtyping import Float
from tensordict import TensorDict
from torch import Tensor, nn
from typing_extensions import override


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    @override
    def forward(self, x):
        return self.nonlin(self.conv(x))


class Conv3x3(nn.Module):
    """Layer to pad and convolve input"""

    def __init__(self, in_channels, out_channels, *, use_refl=True):
        super().__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        return self.conv(self.pad(x))


def upsample(x):
    """Upsample input tensor by a factor of 2"""
    return F.interpolate(x, scale_factor=2, mode="nearest")


class AlignmentLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class AlignmentLayerLinear(nn.Module):
    def __init__(self, c, h, w):
        super().__init__()
        self.conv = nn.AvgPool2d
        self.linear = nn.Linear(c * h * w, c * h * w)

    def forward(self, x):
        b, c, h, w = x.shape

        x_flat = rearrange(x, "b c h w -> b (c h w)")
        x_aligned = self.linear(x_flat)  # Now shape will be (b, out_features
        return rearrange(x_aligned, "b (c h w) -> b c h w", c=c, h=h, w=w)


class DepthDecoder(nn.Module):
    def __init__(
        self, num_ch_enc, scales=(0, 1, 2, 3), num_output_channels=1, use_skips=True
    ):
        super().__init__()

        # The scalar values alpha and beta are used to constrain the estimated depth Z(x) to the range [0.01, 88]
        # units where, x is the output from the sigmoid function:
        #
        #   Disparity:  D(x) = (alpha * x + beta)
        #   Depth:      Z(x) = 1. / D(x)
        #
        # The max depth for float32 is
        # torch.log(torch.Tensor([torch.finfo(torch.float32).max - 1])).item() = 88.72283935546875
        # otherwise torch.exp(depth) calls produce infinity values and a training collapses

        self.alpha = 10
        self.beta = 1 / 88

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = "nearest"
        self.scales = scales
        if self.scales != [0]:
            raise NotImplementedError("Only `scales=[0]` is implemented now")

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs["upconv", i, 0] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs["upconv", i, 1] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs["dispconv", s] = Conv3x3(
                self.num_ch_dec[s], self.num_output_channels
            )

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()
        # self.alignment_layer = AlignmentLayer(
        #     in_channels=num_ch_enc[-1], out_channels=num_ch_enc[-1]
        # )
        # self.alignment_layer = AlignmentLayerLinear(c=num_ch_enc[-1], h=10, w=18)

        self.init_weights()

    def init_weights(self):
        for m in self.decoder.modules():
            if isinstance(m, nn.Conv2d):
                _ = torch.nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                _ = nn.init.constant_(m.weight, 1)
                _ = nn.init.constant_(m.bias, 0)

    def forward(self, input_features: TensorDict) -> Tensor:
        bs = input_features.batch_size

        input_features = input_features.apply(
            Rearrange("b t ... -> (b t) ..."), batch_size=[prod(bs)]
        )

        x = input_features["4"]
        # x = self.alignment_layer(x)

        for i in range(4, -1, -1):
            x = self.convs["upconv", i, 0](x)
            x = [upsample(x)]

            if self.use_skips and i > 0:
                x += [input_features[str(i - 1)]]

            x = torch.cat(x, 1)
            x = self.convs["upconv", i, 1](x)

        # NOTE: hardfix scales = 0 and return only last one
        res = self.alpha * self.sigmoid(self.convs["dispconv", 0](x)) + self.beta
        return Rearrange("(b t) ... -> b t ...", b=bs[0])(res)
