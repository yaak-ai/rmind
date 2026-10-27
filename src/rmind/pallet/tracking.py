"""wandb tracking for the pallet detector: runs, label datasets and weights as artifacts.

Everything goes to `yaak/pallet-detector`. Runs keep wandb's generated names; what a
run is lives in its tags and notes. Label files are logged as dataset artifacts
(`entry-labels`, `multi-labels`, ...) and *used* by the runs that train or
evaluate on them, so every checkpoint records the label version it came from;
wandb deduplicates by content, so re-logging an unchanged file adds no version.
"""

from pathlib import Path
from typing import Any

ENTITY = "yaak"
PROJECT = "pallet-detector"


def start(
    job_type: str,
    config: dict[str, Any],
    tags: list[str],
    notes: str | None = None,
    *,
    enabled: bool = True,
) -> Any:
    """A wandb run, or None when tracking is off (all helpers accept None)."""
    if not enabled:
        return None
    import wandb  # ruff: ignore[import-outside-top-level]

    return wandb.init(
        entity=ENTITY,
        project=PROJECT,
        job_type=job_type,
        config=_plain(config),
        tags=tags,
        notes=notes,
    )


def use_labels(
    run: Any, path: Path, name: str, metadata: dict[str, Any] | None = None
) -> None:
    """Log `path` as a dataset artifact `name` (new version only if the content changed) and mark it used."""
    if run is None or run.offline:  # artifacts cannot be used offline
        return
    import wandb  # ruff: ignore[import-outside-top-level]

    art = wandb.Artifact(
        name, type="dataset", metadata={"file": str(path), **(metadata or {})}
    )
    art.add_file(str(path))
    run.use_artifact(art)


def log_model(
    run: Any,
    path: Path,
    name: str,
    metadata: dict[str, Any],
    aliases: list[str] | None = None,
) -> None:
    if run is None:
        return
    import wandb  # ruff: ignore[import-outside-top-level]

    art = wandb.Artifact(name, type="model", metadata=_plain(metadata))
    art.add_file(str(path))
    run.log_artifact(art, aliases=aliases or ["latest"])


def _plain(d: dict[str, Any]) -> dict[str, Any]:
    return {k: str(v) if isinstance(v, Path) else v for k, v in d.items()}
