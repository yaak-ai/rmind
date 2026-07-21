from collections.abc import Callable, Iterable, Sized
from typing import Any, Literal

import torch
import torchdata.nodes as tn
from pydantic import InstanceOf, PositiveInt, validate_call
from torch import Generator
from torch.utils.data import BatchSampler, WeightedRandomSampler, default_collate
from torchdata.nodes.loader import LoaderIterator

from rbyte.dataloader import BatchIndexableDataset, MapAndCollate


class WeightedDataNodeDataLoader[T](Iterable[T], Sized):
    """Like ``rbyte.dataloader.TorchDataNodeDataLoader`` but draws samples WITH
    REPLACEMENT proportional to a per-sample weight column (``dataset.data[weight_key]``).

    Each epoch draws ``num_samples`` indices (default ``len(dataset)``), so the epoch
    size matches uniform sampling while rare samples (higher weight) are drawn more often
    and dominant ones less — nothing is dropped from disk. Use with a dataset that emits a
    ``meta/weight`` column (e.g. inverse maneuver-bin frequency) to oversample rare
    driving behaviour without shrinking the dataset.
    """

    @validate_call
    def __init__(  # noqa: PLR0913
        self,
        *,
        dataset: InstanceOf[BatchIndexableDataset],
        weight_key: str = "meta/weight",
        batch_size: int = 1,
        num_samples: int | None = None,
        num_workers: PositiveInt = 1,
        collate_fn: Callable[[...], Any] | None = None,
        pin_memory: bool = False,
        pin_memory_device: str = "",
        drop_last: bool = False,
        in_order: bool = True,
        method: Literal["thread", "process"] = "thread",
        multiprocessing_context: Literal["spawn", "forkserver", "fork"] | None = None,
        generator: InstanceOf[Generator] | None = None,
        prefetch_factor: int = 2,
        max_concurrent: int | None = None,
        snapshot_frequency: int = 1,
        prebatch: int | None = None,
    ) -> None:
        self._dataset = dataset

        weights = dataset.data[weight_key].detach().to(torch.double).flatten().cpu()
        num_samples = num_samples if num_samples is not None else len(dataset)
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=num_samples,
            replacement=True,
            generator=generator,
        )
        self._sampler = BatchSampler(
            sampler, batch_size=batch_size, drop_last=drop_last
        )

        node = tn.SamplerWrapper(self._sampler)
        node = tn.ParallelMapper(
            source=node,
            map_fn=MapAndCollate(dataset, collate_fn or default_collate),
            num_workers=num_workers,
            in_order=in_order,
            method=method,
            multiprocessing_context=multiprocessing_context,
            max_concurrent=max_concurrent,
            snapshot_frequency=snapshot_frequency,
            prebatch=prebatch,
        )

        if pin_memory:
            node = tn.PinMemory(node, pin_memory_device=pin_memory_device)

        node = tn.Prefetcher(node, prefetch_factor=num_workers * prefetch_factor)

        self._loader = tn.Loader(node)

    def __iter__(self) -> LoaderIterator[T]:
        return iter(self._loader)

    def __len__(self) -> int:
        return len(self._sampler)

    @property
    def dataset(self) -> BatchIndexableDataset:
        return self._dataset
