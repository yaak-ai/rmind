"""Concurrent drop-in replacement for `rbyte.io.PathTensorSource`.

Upstream's `PathTensorSource.__getitem__` reads every index in a batch with a
fully serial Python loop (`torch.stack([self._getitem(i) for i in indexes])`),
so a whole batch of per-frame NFS opens+reads+decodes happens one at a time
regardless of how many dataloader worker threads are configured. `py-spy`
profiling of a live training run showed dataloader threads spending most of
their time blocked in `Path.open()`/`read_bytes()` for exactly this reason.

`Path.read_bytes()` and `simplejpeg.decode_jpeg` are both native calls that
release the GIL, so this class instead fans the per-index work in a batch out
over a small thread pool: N NFS round-trips (and N JPEG decodes) actually
overlap instead of serializing.

`rbyte.types.TensorSource` is a `Protocol` (structurally typed), so this
doesn't need to inherit from it — matching `__init__`'s keyword arguments
(plus one extra `max_workers` knob) is enough to be a drop-in replacement via
Hydra's `_target_:`.

See: https://github.com/yaak-ai/rbyte/blob/v0.38.1/rbyte/io/path/tensor_source.py
"""

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from os import PathLike
from pathlib import Path

import numpy.typing as npt
import torch
from pydantic import validate_call
from torch import Tensor


class ConcurrentPathTensorSource:
    @validate_call
    def __init__(
        self,
        *,
        path: PathLike[str],
        decoder: Callable[[bytes], npt.ArrayLike],
        index_transform: Callable[..., object] | None = None,
        max_workers: int = 8,
    ) -> None:
        self._path = Path(path)
        self._decoder = decoder
        self._index_transform = index_transform
        self._max_workers = max_workers

    @cached_property
    def _path_posix(self) -> str:
        return self._path.resolve().as_posix()

    def _decode(self, path: str) -> npt.ArrayLike:
        return self._decoder(Path(path).read_bytes())

    def _getitem(self, index: object) -> Tensor:
        if self._index_transform is not None:
            index = self._index_transform(index)

        path = self._path_posix.format(index)
        array = self._decode(path)

        return torch.from_numpy(array)

    def __getitem__(self, indexes: object | Sequence[object]) -> Tensor:
        match indexes:
            case Sequence():
                arrays = self._getitems_concurrent(indexes)
                return torch.stack(arrays)
            case _:
                return self._getitem(indexes)

    def _getitems_concurrent(self, indexes: Sequence[object]) -> list[Tensor]:
        # skip the thread pool for trivially small batches, where its
        # setup/teardown overhead isn't worth it
        max_workers = min(self._max_workers, len(indexes))
        if max_workers <= 1:
            return [self._getitem(i) for i in indexes]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(self._getitem, indexes))

    def __len__(self) -> int:
        raise NotImplementedError
