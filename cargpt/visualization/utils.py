from typing import Any, Callable, Optional, Sequence

import more_itertools as mit
import torch
from jaxtyping import Float
from torch import Tensor
from torchvision.transforms import Normalize


class Unnormalize(Normalize):
    def __init__(
        self,
        mean: Sequence[float],
        std: Sequence[float],
        **kwargs: Any,
    ) -> None:
        _mean: Float[Tensor, "c"] = torch.tensor(mean)
        _std: Float[Tensor, "c"] = torch.tensor(std)

        super().__init__(
            mean=(-_mean / _std).tolist(),
            std=(1.0 / _std).tolist(),
            **kwargs,
        )


def get_images(
    batch, transform: Callable, clip_idx: Optional[int] = None
) -> Float[Tensor, "..."]:
    clips = mit.one(batch["clips"].values())
    frames = clips["frames"]
    if clip_idx is not None:
        frames = frames[:, clip_idx, ...]
    imgs = transform(frames)
    return imgs.clamp(min=0.0, max=1.0)
