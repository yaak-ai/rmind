"""Opt-in acceptance of operator-relabelled nero takes (nutron-cli convert.py --accept-relabel).

rbyte's `NeroRobotReader` (like convert.py) trains a take only if BOTH copies of
its outcome say success: the MCAP `episode_outcome` record (cannot be edited
after the fact) and `outcome.json` (what a relabel rewrites). A take discarded by
accident and relabelled success by the operator has no (or a non-success) MCAP
outcome, so it is refused.

`NeroRobotRelabelReader` is that reader plus convert.py's opt-in rule: when
`outcome.json` says success AND carries relabel provenance (`relabeled_from`
present, `source` starting with `relabel:`), it overrides the MCAP outcome. Every
take accepted this way is logged (warning, with the relabel source). Any other
take goes to rbyte unchanged, so every other check (and refusal) still applies.
The override only swaps the MCAP `episode_outcome` record seen by rbyte's
`read_episode` for the duration of that one call (the samples pipeline runs the
readers one call at a time per worker process).
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from rbyte.samples.nero import NeroRobotReader, robot
from structlog import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from os import PathLike

    import polars as pl

logger = get_logger(__name__)

RELABEL_SOURCE_PREFIX = "relabel:"


def relabel_provenance(outcome: dict[str, Any]) -> str | None:
    """outcome.json's relabel source when it carries relabel provenance, else None."""
    source = outcome.get("source")
    if (
        "relabeled_from" in outcome
        and isinstance(source, str)
        and source.startswith(RELABEL_SOURCE_PREFIX)
    ):
        return source
    return None


def mcap_outcome(path: str | PathLike[str]) -> str | None:
    """The MCAP `episode_outcome.outcome` of a take (None when absent)."""
    with Path(path).open("rb") as f:
        for record in robot.make_reader(f).iter_metadata():
            if record.name == "episode_outcome":
                return dict(record.metadata).get("outcome")
    return None


def relabel_override(path: str | PathLike[str]) -> dict[str, Any] | None:
    """The relabel record when convert.py's --accept-relabel rule applies to the
    take at `path` (a data.mcap), else None."""
    outcome_path = Path(path).parent / "outcome.json"
    if not outcome_path.exists():
        return None
    outcome = json.loads(outcome_path.read_text())
    source = relabel_provenance(outcome)
    if outcome.get("outcome") != "success" or source is None:
        return None
    recorded = mcap_outcome(path)
    if recorded == "success":
        return None
    return {
        "source": source,
        "relabeled_from": outcome.get("relabeled_from"),
        "mcap_outcome": recorded,
    }


class _OutcomeOverride:
    """An mcap reader whose metadata says `episode_outcome.outcome = success`."""

    def __init__(self, inner: Any, record: dict[str, Any]) -> None:
        self._inner = inner
        self._record = record

    def iter_metadata(self) -> Iterator[Any]:
        for r in self._inner.iter_metadata():
            if r.name != "episode_outcome":
                yield r
        yield SimpleNamespace(
            name="episode_outcome",
            metadata={"outcome": "success", "source": self._record["source"]},
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


@contextmanager
def _patched_outcome(record: dict[str, Any]) -> Iterator[None]:
    original = robot.make_reader

    def make(*args: Any, **kwargs: Any) -> Any:
        return _OutcomeOverride(original(*args, **kwargs), record)

    vars(robot)["make_reader"] = make
    try:
        yield
    finally:
        vars(robot)["make_reader"] = original


class NeroRobotRelabelReader:
    """`NeroRobotReader` + the opt-in relabel rule (see module docstring)."""

    def __init__(
        self,
        *,
        chunk_size: int = 100,
        max_pad_steps: int | None = None,
        camera_cond: Sequence[Sequence[float]] | None = None,
        accept_relabel: bool = True,
    ) -> None:
        kwargs: dict[str, Any] = {
            "chunk_size": chunk_size,
            "max_pad_steps": max_pad_steps,
        }
        if camera_cond is not None:
            kwargs["camera_cond"] = camera_cond
        self._reader = NeroRobotReader(**kwargs)
        self._accept_relabel = accept_relabel

    def __call__(self, path: str | PathLike[str]) -> pl.DataFrame:
        record = relabel_override(path) if self._accept_relabel else None
        if record is None:
            return self._reader(path)
        logger.warning("accepted relabelled take", path=str(path), **record)
        with _patched_outcome(record):
            return self._reader(path)
