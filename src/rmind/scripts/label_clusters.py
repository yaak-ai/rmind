from pathlib import Path
from typing import Any

import hydra
import polars as pl
from hydra.utils import instantiate
from omegaconf import DictConfig
from structlog import get_logger
from tqdm import tqdm

logger = get_logger(__name__)


def _label_split(loader: Any, cluster_fn: Any, split: str, output_dir: Path) -> None:
    records: list[dict[str, Any]] = []

    for batch in tqdm(loader, desc=split):
        labels = cluster_fn(batch, {})
        sample_ids = batch["data"]["meta/sample_id"].tolist()
        records.extend(
            {"meta/sample_id": sid, "cluster": label}
            for sid, label in zip(sample_ids, labels, strict=False)
        )

    df = pl.DataFrame(records)
    path = output_dir / f"{split}.parquet"
    df.write_parquet(path)

    dist = df.group_by("cluster").len().sort("cluster")
    logger.info(
        "saved cluster labels",
        split=split,
        path=str(path),
        n=len(df),
        distribution=dict(
            zip(dist["cluster"].to_list(), dist["len"].to_list(), strict=False)
        ),
    )


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    output_dir = Path(cfg.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    datamodule = instantiate(cfg.datamodule)
    cluster_fn = instantiate(cfg.cluster_fn)

    splits = []
    if (loader := datamodule.train_dataloader()) is not None:
        splits.append(("train", loader))
    if (loader := datamodule.val_dataloader()) is not None:
        splits.append(("val", loader))

    for split, loader in splits:
        _label_split(loader, cluster_fn, split, output_dir)


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_forkserver_preload(["rbyte", "polars"])
    main()
