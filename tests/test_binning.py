import torch
from torch.testing import assert_close, make_tensor

from cargpt.components.norm import UniformBinner, UniformUnbinner


def test_binning():
    value_range = [5.0, 130.0]
    num_bins = 1024
    bin_width = (value_range[1] - value_range[0]) / num_bins

    binner = UniformBinner(in_range=value_range, num_bins=num_bins)
    unbinner = UniformUnbinner(out_range=value_range, num_bins=num_bins)

    values = make_tensor(
        1024,
        dtype=torch.float,
        device="cpu",
        low=value_range[0],
        high=value_range[1],
    )
    binned = binner(values)
    unbinned = unbinner(binned)

    assert_close(actual=unbinned, expected=values, rtol=0.0, atol=bin_width / 2.0)
