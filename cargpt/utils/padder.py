from functools import partial

import torch
from torch.nn import functional as F


def nan_padder(pad: tuple[int, ...]):
    return partial(
        F.pad,
        pad=pad,
        mode="constant",
        value=torch.nan,
    )
