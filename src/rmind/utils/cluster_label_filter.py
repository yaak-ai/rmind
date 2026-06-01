from pathlib import Path

import polars as pl


class ClusterLabelFilter:
    """PipeFunc-compatible stage that joins cluster labels into the sample DataFrame.

    Reads a parquet file with columns (meta/sample_id, cluster) produced by
    ``rmind-label-clusters`` and left-joins it onto the sample DataFrame so that
    every row gets a ``cluster`` column.  Optionally filters to keep only rows
    whose cluster is in ``clusters``.

    When ``labels_path`` is None or the file does not exist the DataFrame is
    returned unchanged so the stage is a safe no-op in unconfigured runs.
    """

    def __init__(
        self, labels_path: str | None = None, clusters: list[str] | None = None
    ) -> None:
        self._labels_path = (
            None if labels_path is None else Path(labels_path).expanduser()
        )
        self._clusters = clusters

    def __call__(self, df: pl.DataFrame) -> pl.DataFrame:
        if self._labels_path is None or not self._labels_path.exists():
            return df

        labels = pl.read_parquet(self._labels_path).select("meta/sample_id", "cluster")
        df = df.join(labels, on="meta/sample_id", how="left")

        if self._clusters is not None:
            df = df.filter(pl.col("cluster").is_in(self._clusters))

        return df.drop("cluster")
