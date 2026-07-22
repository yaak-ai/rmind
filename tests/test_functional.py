import math

import torch
from torch.testing import assert_close

from rmind.utils.functional import diff_last


def test_diff_last_append() -> None:
    x = torch.tensor([[1.0, 2.0, 4.0]])
    out = diff_last(x, append=math.nan)
    assert_close(out[:, :2], torch.tensor([[1.0, 2.0]]))
    assert torch.isnan(out[:, 2]).all()


def test_diff_last_prepend() -> None:
    x = torch.tensor([[1.0, 2.0, 4.0]])
    out = diff_last(x, prepend=math.nan)
    assert torch.isnan(out[:, 0]).all()
    assert_close(out[:, 1:], torch.tensor([[1.0, 2.0]]))


def test_diff_last_neither() -> None:
    x = torch.tensor([[1.0, 2.0, 4.0]])
    out = diff_last(x)
    assert_close(out, torch.tensor([[1.0, 2.0]]))
