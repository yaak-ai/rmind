from itertools import pairwise

import torch
from torch.testing import assert_close, make_tensor

from rmind.components.nn import Sequential
from rmind.components.norm import Scaler, UniformBinner


def test_scaler(device: torch.device) -> None:
    module = Scaler(in_range=(0.0, 100.0), out_range=(-1.0, 1.0)).to(device)

    x = make_tensor(
        1024,
        dtype=torch.float,
        device=device,
        low=module.in_range[0].item(),
        high=module.in_range[1].item(),
    )

    x_rt = module.invert(module(x))

    assert_close(x_rt, x)


def test_uniform_binner(device: torch.device) -> None:
    module = UniformBinner(range=(5.0, 130.0), bins=1024).to(device)
    x = make_tensor(
        1024,
        dtype=torch.float,
        device=device,
        low=module.range[0].item(),
        high=module.range[1].item(),
    )
    x_rt = module.invert(module(x))

    bin_width = (module.range[1] - module.range[0]) / module.bins
    assert_close(x_rt, x, rtol=0.0, atol=(bin_width / 2.0).item())


def test_sequential(device: torch.device) -> None:
    module = Sequential(
        *(
            Scaler(in_range=in_range, out_range=out_range)
            for in_range, out_range in pairwise([0.0, 10.0**x] for x in range(1, 6))
        )
    ).to(device)

    x = make_tensor(
        1024,
        dtype=torch.float,
        device=device,
        low=module[0].in_range[0].item(),  # ty:ignore[not-subscriptable]
        high=module[0].in_range[1].item(),  # ty:ignore[not-subscriptable]
    )

    x_rt = module.invert(module(x))

    assert_close(x_rt, x)
