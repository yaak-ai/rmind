"""DDP-aware drop-in for ``rbyte.dataloader.TorchDataNodeDataLoader``.

rbyte's loader hard-codes ``RandomSampler`` and is not a ``torch.utils.data
.DataLoader``, so Lightning cannot inject a ``DistributedSampler``. Under
``trainer.devices>1`` every rank would draw an identical batch sequence
(Lightning seeds all ranks alike) and the all-reduced gradient degenerates to
the single-rank one — 2x the compute for an unchanged effective batch.

This loader swaps in a ``DistributedSampler`` (disjoint per-rank shards, so
world_size x per-rank batch = the effective batch) whenever
``torch.distributed`` is initialized, and falls back to rbyte's exact
behavior otherwise. ``DistributedSamplerEpochSetter`` forwards the epoch to
the sampler for per-epoch reshuffling (Lightning only does this for plain
DataLoaders).

Only ``method: thread`` workers are supported under DDP: ``method: process``
forks after CUDA initialization and deadlocks (fork + CUDA + locked
shared-memory TensorDict).
"""

from collections.abc import Callable, Iterable, Sized
from typing import Any, Literal, override

import pytorch_lightning as pl
import torch.distributed
import torchdata.nodes as tn
from pydantic import InstanceOf, PositiveInt, validate_call
from rbyte.dataloader import BatchIndexableDataset, MapAndCollate
from structlog import get_logger
from torch import Generator
from torch.utils.data import (
    BatchSampler,
    DistributedSampler,
    RandomSampler,
    SequentialSampler,
    default_collate,
)
from torchdata.nodes.loader import LoaderIterator

logger = get_logger(__name__)


class DistributedTorchDataNodeDataLoader[T](Iterable[T], Sized):
    """rbyte TorchDataNodeDataLoader with DDP-sharded sampling."""

    @validate_call
    def __init__(  # noqa: PLR0913
        self,
        *,
        dataset: InstanceOf[BatchIndexableDataset],
        batch_size: int = 1,
        shuffle: bool | None = None,
        num_workers: PositiveInt = 1,
        collate_fn: Callable[..., Any] | None = None,
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
        seed: int = 0,
    ) -> None:
        self._dataset = dataset

        distributed = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )
        if distributed:
            if method != "thread":
                msg = (
                    "method='process' deadlocks under DDP (fork after CUDA init); "
                    "use method='thread'"
                )
                raise ValueError(msg)
            self._distributed_sampler = DistributedSampler(
                dataset,  # type: ignore[arg-type]
                shuffle=bool(shuffle),
                drop_last=drop_last,
                seed=seed,
            )
            sampler = self._distributed_sampler
            logger.info(
                "distributed sampler",
                rank=self._distributed_sampler.rank,
                world_size=self._distributed_sampler.num_replicas,
                shard_len=self._distributed_sampler.num_samples,
            )
        else:
            self._distributed_sampler = None
            sampler = (
                RandomSampler(dataset, generator=generator)  # type: ignore[assignment]
                if shuffle
                else SequentialSampler(dataset)  # type: ignore[assignment]
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

    def set_epoch(self, epoch: int) -> None:
        if self._distributed_sampler is not None:
            self._distributed_sampler.set_epoch(epoch)

    def __iter__(self) -> LoaderIterator[T]:
        return iter(self._loader)

    def __len__(self) -> int:
        return len(self._sampler)

    @property
    def dataset(self) -> BatchIndexableDataset:
        return self._dataset


class DistributedSamplerEpochSetter(pl.Callback):
    """Forward the epoch to DDP-sharded loaders for per-epoch reshuffling."""

    @override
    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        loader = trainer.train_dataloader
        set_epoch = getattr(loader, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(trainer.current_epoch)
