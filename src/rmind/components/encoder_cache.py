"""Encoder-output caching for frozen-backbone fine-tuning.

During fine-tuning the ``ModuleFreezer`` callback freezes both ``episode_builder``
(which contains the DINOv3 image backbone -- the dominant cost) and ``encoder``.
They run in eval with ``requires_grad=False``, so their output is deterministic
per input across epochs. Only the (cheap) objective heads train.

This module precomputes, per sample, the small slice of the post-encoder
``embedding`` that the objective actually reads -- the last-timestep token groups
returned by ``episode.index[-1].select(*select).parse(embedding)`` -- together
with ``episode.input`` (the targets). At train time the cached slice is packed
into a compact embedding and a synthetic :class:`Episode` is handed to the
objective UNCHANGED, skipping DINOv3 and the encoder entirely.

Correctness rests on three facts (verified to bit-identical losses):

1. ``episode_builder`` + ``encoder`` are deterministic in eval (no augmentation;
   the only dropout lives in the frozen encoder).
2. ``policy.norm`` is a per-token ``LayerNorm`` (no cross-token mixing), so
   ``norm(full)[selected] == norm(selected)`` and we may cache the groups
   pre-norm and apply the trainable norm at train time.
3. The objective only ever ``select``s the configured groups before ``parse``,
   so a compact embedding holding just those groups -- with a rebased index --
   produces identical reads.

CACHE INVALIDATION: the cache is valid ONLY while ``episode_builder`` + ``encoder``
are frozen and unchanged, and the dataset definition (drives / frame selection)
is unchanged. The cache is stamped with ``(checkpoint_artifact, dataset_fingerprint)``
and refuses to load on mismatch. ``meta/sample_id`` is per-dataset-split, so the
cache is namespaced by stage (``train`` / ``val``).
"""

from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Any, Self, final

import orjson
import torch
from pydantic import BaseModel, ConfigDict, validate_call
from structlog import get_logger
from tensordict import TensorDict
from torch import Tensor

from rmind.components.episode import Episode, Index
from rmind.utils.patch import monkeypatched

logger = get_logger(__name__)

STAMP_FILE = "stamp.json"
DEFAULT_SELECT: tuple[tuple[str, ...], ...] = (
    ("summary", "observation_summary"),
    ("summary", "observation_history"),
    ("context", "waypoints"),
)
# top-level groups of episode.input the objective reads as targets. The image
# branch of episode.input (~4.7 MB/sample) is intentionally NOT cached.
DEFAULT_INPUT_SELECT: tuple[str, ...] = ("continuous", "discrete")


class CacheStamp(BaseModel):
    """Identifies the (frozen weights, dataset) the cache was built against."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    checkpoint: str
    dataset_fingerprint: str
    select: tuple[tuple[str, ...], ...]
    input_select: tuple[str, ...]
    # the embedding's numerics depend on the autocast precision it was computed
    # under; reusing a bf16 cache under a 16-mixed run would silently mismatch.
    precision: str


def _patched_orjson_dumps():  # noqa: ANN202
    # TensorDict.memmap serializes metadata via orjson.dumps, which chokes on
    # StrEnum keys without OPT_NON_STR_KEYS (see prediction/_tensordict.py).
    return monkeypatched(
        orjson, "dumps", partial(orjson.dumps, option=orjson.OPT_NON_STR_KEYS)
    )


@final
class EncoderCache:
    """On-disk cache of per-sample frozen encoder-output slices.

    Storage layout under ``root``::

        <root>/<stage>/stamp.json        # CacheStamp
        <root>/<stage>/groups/           # memmapped TensorDict, row == sample_id
        <root>/<stage>/input/            # memmapped TensorDict (targets)
        <root>/<stage>/index/            # memmapped Index (sample-invariant)
        <root>/<stage>/written           # 1D bool memmap: which rows are populated
    """

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(  # noqa: PLR0913
        self,
        *,
        root: str,
        checkpoint: str,
        dataset_fingerprint: str,
        num_samples: dict[str, int],
        precision: str = "32-true",
        select: Sequence[Sequence[str]] = DEFAULT_SELECT,
        input_select: Sequence[str] = DEFAULT_INPUT_SELECT,
        sample_id_key: tuple[str, ...] = ("data", "meta/sample_id"),
        device: str | torch.device = "cpu",
    ) -> None:
        self.root: Path = Path(root)
        self.num_samples: dict[str, int] = dict(num_samples)
        self.select: tuple[tuple[str, ...], ...] = tuple(tuple(s) for s in select)
        self.input_select: tuple[str, ...] = tuple(input_select)
        self.sample_id_key: tuple[str, ...] = tuple(sample_id_key)
        self.device: torch.device = torch.device(device)
        self.stamp: CacheStamp = CacheStamp(
            checkpoint=checkpoint,
            dataset_fingerprint=dataset_fingerprint,
            select=self.select,
            input_select=self.input_select,
            precision=str(precision),
        )

        self._stores: dict[str, _StageStore] = {}

    def _stage_dir(self, stage: str) -> Path:
        return self.root / stage

    def _sample_ids(self, batch: Any) -> Tensor:
        node: Any = batch
        for key in self.sample_id_key:
            node = node[key]
        return node.flatten().long()

    # -- population -------------------------------------------------------

    def _complete_marker(self, stage: str) -> Path:
        return self._stage_dir(stage) / "COMPLETE"

    def is_populated(self, stage: str) -> bool:
        """True only if a *complete*, stamp-matching pass exists on disk.

        Gated on an explicit COMPLETE marker (written by mark_complete) AND a
        full written-mask, so a partial/aborted populate never silently
        activates the fast path with garbage rows.
        """
        if not self._complete_marker(stage).exists():
            return False
        store = self._stores.get(stage)
        if store is None:
            try:
                store = _StageStore.open(self._stage_dir(stage), self.stamp)
            except (FileNotFoundError, ValueError) as exc:
                logger.debug("cache not openable", stage=stage, reason=str(exc))
                return False
            self._stores[stage] = store
        return bool(store.written.all())

    def mark_complete(self, stage: str) -> None:
        """Record that every sample for ``stage`` has been written.

        Raises:
            RuntimeError: if any row is missing (an incomplete pass cannot be
                marked complete).
        """
        store = self._stores.get(stage)
        if store is None:
            msg = f"refusing to mark stage {stage!r} complete: nothing written"
            raise RuntimeError(msg)
        if not bool(store.written.all()):
            missing = int((~store.written).sum())
            msg = f"refusing to mark stage {stage!r} complete: {missing} rows unwritten"
            raise RuntimeError(msg)
        self._complete_marker(stage).touch()

    def store_batch(
        self, stage: str, *, batch: Any, episode: Episode, embedding: Tensor
    ) -> None:
        """Capture the configured slice for one batch (call under no_grad)."""
        sample_ids = self._sample_ids(batch).cpu()

        select_keys = [tuple(s) for s in self.select]
        parsed = episode.index[-1].select(*select_keys).parse(embedding)
        groups = TensorDict(
            {k: parsed.get(k) for k in select_keys}, batch_size=parsed.batch_size
        ).cpu()
        # Only cache the (tiny) target fields the objective reads; the image
        # branch of episode.input (~4.7 MB/sample) is dropped.
        input_td = episode.input.select(*self.input_select).cpu()

        store = self._stores.get(stage)
        if store is None:
            index_replay = _build_index_replay(episode.index, self.select)
            store = _StageStore.create(
                self._stage_dir(stage),
                self.stamp,
                num_samples=self.num_samples[stage],
                group_template=groups[0],  # ty:ignore[invalid-argument-type]
                input_template=input_td[0],  # ty:ignore[invalid-argument-type]
                index_replay=index_replay,
            )
            self._stores[stage] = store

        store.write_rows(sample_ids, groups=groups, input_td=input_td)

    # -- retrieval --------------------------------------------------------

    def build_cached(
        self, stage: str, batch: Any, *, device: torch.device
    ) -> tuple[Episode, Tensor]:
        """Return ``(synthetic_episode, compact_embedding)`` for ``batch``.

        Skips episode_builder + encoder entirely; uses only ``meta/sample_id``.
        """
        store = self._stores.get(stage)
        if store is None:
            store = _StageStore.open(self._stage_dir(stage), self.stamp)
            self._stores[stage] = store

        sample_ids = self._sample_ids(batch).cpu()
        groups, input_td = store.read_rows(sample_ids)
        groups = groups.to(device)
        input_td = input_td.to(device)
        index_replay = store.index_replay.to(device)

        ordered = [groups.get(tuple(k)) for k in self.select]
        embedding_compact = torch.cat(ordered, dim=1)

        b, t = input_td.batch_size
        episode = Episode(
            input=input_td,
            input_tokens=TensorDict(batch_size=[b, t]),
            input_embeddings=TensorDict(batch_size=[b, t]),
            embeddings=TensorDict(batch_size=[b, t]),
            index=index_replay,
            embeddings_flattened=torch.empty(b, 0, device=device),
            attention_mask=None,
            batch_size=[],
            device=device,
        )
        return episode, embedding_compact


def _build_index_replay(index: Index, select: tuple[tuple[str, ...], ...]) -> Index:
    """Rebase the selected groups (last timestep) into a contiguous [0..N) buffer.

    Keeps a leading timestep dim of size 1 so the objective's ``index[-1]`` works.
    Pack order matches ``torch.cat(select, dim=1)`` at retrieval time.
    """
    last = index[-1:].select(*select)
    rebased = last.to_tensordict(retain_none=False).clone()
    off = 0
    for key in select:
        leaf = last.get(tuple(key))
        n = leaf.numel()
        rebased.set(
            tuple(key), torch.arange(off, off + n, device=leaf.device).view_as(leaf)
        )
        off += n
    return Index.from_tensordict(rebased)


@final
class _StageStore:
    """Per-stage memmapped storage. Row index == ``sample_id``."""

    def __init__(
        self,
        *,
        groups: TensorDict,
        input_td: TensorDict,
        index_replay: Index,
        written: Tensor,
    ) -> None:
        self.groups: TensorDict = groups
        self.input: TensorDict = input_td
        self.index_replay: Index = index_replay
        self.written: Tensor = written

    @classmethod
    def create(  # noqa: PLR0913
        cls,
        path: Path,
        stamp: CacheStamp,
        *,
        num_samples: int,
        group_template: TensorDict,
        input_template: TensorDict,
        index_replay: Index,
    ) -> Self:
        path.mkdir(parents=True, exist_ok=True)
        (path / STAMP_FILE).write_bytes(
            orjson.dumps(stamp.model_dump(), option=orjson.OPT_NON_STR_KEYS)
        )

        with _patched_orjson_dumps():
            groups = group_template.expand(num_samples, *group_template.batch_size)
            groups = groups.memmap(str(path / "groups"), copy_existing=True)
            input_mm = input_template.expand(
                num_samples, *input_template.batch_size
            ).memmap(str(path / "input"), copy_existing=True)
            index_replay.to_tensordict(retain_none=False).memmap(
                str(path / "index"), copy_existing=True
            )

        written = torch.zeros(num_samples, dtype=torch.bool)
        TensorDict({"written": written}, batch_size=[num_samples]).memmap(
            str(path / "written"), copy_existing=True
        )
        written_mm = TensorDict.load_memmap(str(path / "written")).get("written")

        return cls(
            groups=groups,
            input_td=input_mm,
            index_replay=index_replay,
            written=written_mm,
        )

    @classmethod
    def open(cls, path: Path, stamp: CacheStamp) -> Self:
        stamp_path = path / STAMP_FILE
        if not stamp_path.exists():
            msg = f"no cache stamp at {stamp_path}"
            raise FileNotFoundError(msg)

        on_disk = CacheStamp.model_validate(orjson.loads(stamp_path.read_bytes()))
        if on_disk != stamp:
            msg = (
                f"cache stamp mismatch at {path}: on-disk={on_disk!r} "
                f"requested={stamp!r}; refusing to load a stale cache"
            )
            raise ValueError(msg)

        groups = TensorDict.load_memmap(str(path / "groups"))
        input_mm = TensorDict.load_memmap(str(path / "input"))
        index_replay = Index.from_tensordict(
            TensorDict.load_memmap(str(path / "index"))
        )
        written = TensorDict.load_memmap(str(path / "written")).get("written")
        return cls(
            groups=groups, input_td=input_mm, index_replay=index_replay, written=written
        )

    def write_rows(
        self, sample_ids: Tensor, *, groups: TensorDict, input_td: TensorDict
    ) -> None:
        for local, sid in enumerate(sample_ids.tolist()):
            self.groups[sid] = groups[local]
            self.input[sid] = input_td[local]
            self.written[sid] = True

    def read_rows(self, sample_ids: Tensor) -> tuple[TensorDict, TensorDict]:
        if not self.written[sample_ids].all():
            missing = sample_ids[~self.written[sample_ids]].tolist()
            msg = f"cache miss for sample_ids={missing}"
            raise KeyError(msg)
        return self.groups[sample_ids].clone(), self.input[sample_ids].clone()  # ty:ignore[invalid-return-type]
