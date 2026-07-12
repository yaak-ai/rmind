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
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._drop_last = drop_last
        self._generator = generator
        self._seed = seed
        self._method: Literal["thread", "process"] = method
        self._node_kwargs = {
            "num_workers": num_workers,
            "in_order": in_order,
            "multiprocessing_context": multiprocessing_context,
            "max_concurrent": max_concurrent,
            "snapshot_frequency": snapshot_frequency,
            "prebatch": prebatch,
        }
        self._collate_fn = collate_fn
        self._pin_memory = pin_memory
        self._pin_memory_device = pin_memory_device
        self._prefetch_factor = prefetch_factor

        self._distributed_sampler: DistributedSampler[Any] | None = None
        self._sampler: BatchSampler | None = None
        self._loader: tn.Loader[T] | None = None
        self._built_distributed: bool | None = None

    @staticmethod
    def _distributed_now() -> bool:
        return torch.distributed.is_available() and torch.distributed.is_initialized()

    def _ensure_built(self) -> None:
        # Rebuild if the process group appeared after a pre-DDP len()/iter()
        # call — an unsharded graph under DDP silently degenerates the
        # effective batch to a single rank's.
        if self._loader is not None and (
            self._built_distributed is False and self._distributed_now()
        ):
            logger.warning("rebuilding dataloader: process group appeared after build")
            self._loader = None
        if self._loader is None:
            self._build()

    def _build(self) -> None:
        """Build the sampler + node graph on first use.

        Deferred past __init__ because hydra instantiates the loader before
        ``trainer.fit`` initializes the DDP process group; the sharding
        decision must be taken at iteration time.

        Raises:
            ValueError: if ``method='process'`` is configured under DDP.
        """
        distributed = self._distributed_now()
        self._built_distributed = distributed
        if distributed:
            if self._method != "thread":
                msg = (
                    "method='process' deadlocks under DDP (fork after CUDA init); "
                    "use method='thread'"
                )
                raise ValueError(msg)
            self._distributed_sampler = DistributedSampler(
                self._dataset,  # type: ignore[arg-type]
                shuffle=bool(self._shuffle),
                drop_last=self._drop_last,
                seed=self._seed,
            )
            sampler = self._distributed_sampler
            logger.info(
                "distributed sampler",
                rank=self._distributed_sampler.rank,
                world_size=self._distributed_sampler.num_replicas,
                shard_len=self._distributed_sampler.num_samples,
            )
        else:
            sampler = (  # type: ignore[assignment]
                RandomSampler(self._dataset, generator=self._generator)  # type: ignore[arg-type]
                if self._shuffle
                else SequentialSampler(self._dataset)  # type: ignore[arg-type]
            )

        self._sampler = BatchSampler(
            sampler, batch_size=self._batch_size, drop_last=self._drop_last
        )

        node = tn.SamplerWrapper(self._sampler)
        node = tn.ParallelMapper(
            source=node,
            map_fn=MapAndCollate(self._dataset, self._collate_fn or default_collate),
            method=self._method,
            **self._node_kwargs,
        )

        if self._pin_memory:
            node = tn.PinMemory(node, pin_memory_device=self._pin_memory_device)

        node = tn.Prefetcher(
            node,
            prefetch_factor=self._node_kwargs["num_workers"] * self._prefetch_factor,
        )

        self._loader = tn.Loader(node)

    def set_epoch(self, epoch: int) -> None:
        if self._distributed_sampler is not None:
            self._distributed_sampler.set_epoch(epoch)

    @override
    def __iter__(self) -> LoaderIterator[T]:
        self._ensure_built()
        assert self._loader is not None  # noqa: S101
        return iter(self._loader)

    @override
    def __len__(self) -> int:
        self._ensure_built()
        assert self._sampler is not None  # noqa: S101
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
