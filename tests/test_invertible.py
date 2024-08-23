from itertools import pairwise

import torch
from cargpt.components.nn import Sequential
from cargpt.components.norm import Scaler, UniformBinner
from torch.testing import assert_close, make_tensor


def test_scaler():
    module = Scaler(in_range=[0.0, 100.0], out_range=[-1.0, 1.0])

    x = make_tensor(
        1024,
        dtype=torch.float,
        device="cpu",
        low=module.in_range[0],
        high=module.in_range[1],
    )

    x_rt = module.invert(module.forward(x))

    assert_close(x_rt, x)


def test_uniform_binner():
    module = UniformBinner(range=[5.0, 130.0], bins=1024)
    x = make_tensor(
        1024, dtype=torch.float, device="cpu", low=module.range[0], high=module.range[1]
    )
    x_rt = module.invert(module.forward(x))

    bin_width = (module.range[1] - module.range[0]) / module.bins
    assert_close(x_rt, x, rtol=0.0, atol=bin_width / 2.0)


def test_sequential():
    module = Sequential(
        *(
            Scaler(in_range=in_range, out_range=out_range)
            for in_range, out_range in pairwise([0.0, 10.0**x] for x in range(1, 6))
        )
    )

    x = make_tensor(
        1024,
        dtype=torch.float,
        device="cpu",
        low=module[0].in_range[0].item(),
        high=module[0].in_range[1].item(),
    )

    x_rt = module.invert(module.forward(x))

    assert_close(x_rt, x)
