"""Populate + activate the frozen-encoder output cache before fine-tuning.

At ``on_fit_start`` (after ``ModuleFreezer`` has frozen episode_builder+encoder),
this callback either reuses a valid on-disk cache or runs a single no-grad pass
over the train/val dataloaders to populate it, then attaches the
:class:`EncoderCache` to the model so the step methods take the fast path.
"""

from typing import Any, override

import pytorch_lightning as pl
import torch
from pydantic import InstanceOf, validate_call
from pytorch_lightning.callbacks import Callback
from structlog import get_logger

from rmind.components.encoder_cache import DEFAULT_SELECT, EncoderCache
from rmind.models.control_transformer import ControlTransformer

logger = get_logger(__name__)


class EncoderCachePopulator(Callback):
    """Build (or reuse) and activate an :class:`EncoderCache` for FT.

    ```yaml
    _target_: rmind.callbacks.EncoderCachePopulator
    root: ${hydra:run.dir}/encoder_cache
    checkpoint: ${model.artifact}
    dataset_fingerprint: ${datamodule_fingerprint}
    ```
    """

    @validate_call
    def __init__(
        self,
        *,
        root: str,
        checkpoint: str,
        dataset_fingerprint: str,
        select: tuple[tuple[str, ...], ...] = DEFAULT_SELECT,
        sample_id_key: tuple[str, ...] = ("data", "meta/sample_id"),
    ) -> None:
        self._root = root
        self._checkpoint = checkpoint
        self._dataset_fingerprint = dataset_fingerprint
        self._select = select
        self._sample_id_key = sample_id_key

    @staticmethod
    def _num_samples(dataloader: Any) -> int:
        return len(dataloader.dataset)

    @staticmethod
    def _sequential_loader(dataloader: Any) -> Any:
        """Clone a dataloader as a full, order-stable pass.

        Population MUST cover every ``sample_id`` exactly once, so we force
        ``shuffle=False`` and ``drop_last=False`` (the training loaders use
        ``shuffle=True``/``drop_last=True``, which would leave a random tail
        unwritten and crash retrieval mid-epoch).
        """
        from rbyte.types import Batch  # noqa: PLC0415

        return type(dataloader)(
            dataset=dataloader.dataset,
            batch_size=getattr(dataloader, "batch_size", 1) or 1,
            shuffle=False,
            drop_last=False,
            num_workers=2,
            collate_fn=Batch.to_dict,
            method="thread",
        )

    @torch.no_grad()
    def _populate_stage(
        self,
        *,
        cache: EncoderCache,
        stage: str,
        dataloader: Any,
        pl_module: ControlTransformer,
    ) -> None:
        if cache.is_populated(stage):
            logger.info("encoder cache already populated", stage=stage)
            return

        logger.info("populating encoder cache", stage=stage)
        device = pl_module.device
        loader = self._sequential_loader(dataloader)
        # Populate under the trainer's precision (e.g. bf16-mixed) autocast, so
        # the stored embedding matches the numerical path the live training_step
        # would have produced for the frozen encoder. forward_context() yields a
        # single-use context manager, so re-acquire it per batch.
        precision_plugin = pl_module.trainer.precision_plugin
        n_batches = 0
        for raw_batch in loader:
            batch = pl_module.transfer_batch_to_device(raw_batch, device, 0)
            with precision_plugin.forward_context():
                episode = pl_module.episode_builder(batch)
                embedding = pl_module.encoder(
                    src=episode.embeddings_flattened, mask=episode.attention_mask
                )
            cache.store_batch(stage, batch=batch, episode=episode, embedding=embedding)
            n_batches += 1

        cache.mark_complete(stage)
        logger.info("encoder cache populated", stage=stage, batches=n_batches)

    @override
    @validate_call
    def on_fit_start(
        self, trainer: InstanceOf[pl.Trainer], pl_module: InstanceOf[ControlTransformer]
    ) -> None:  # ty:ignore[invalid-method-override]
        datamodule = trainer.datamodule  # ty:ignore[unresolved-attribute]
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()

        num_samples = {"train": self._num_samples(train_loader)}
        if val_loader is not None:
            num_samples["val"] = self._num_samples(val_loader)

        cache = EncoderCache(
            root=self._root,
            checkpoint=self._checkpoint,
            dataset_fingerprint=self._dataset_fingerprint,
            num_samples=num_samples,
            precision=str(trainer.precision),
            select=self._select,
            sample_id_key=self._sample_id_key,
        )

        was_training = pl_module.training
        pl_module.eval()
        self._populate_stage(
            cache=cache, stage="train", dataloader=train_loader, pl_module=pl_module
        )
        if val_loader is not None:
            self._populate_stage(
                cache=cache, stage="val", dataloader=val_loader, pl_module=pl_module
            )
        if was_training:
            pl_module.train()

        pl_module.encoder_cache = cache
        logger.info("encoder cache activated", root=self._root)
