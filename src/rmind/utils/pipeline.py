import polars as plr  # noqa: ICN001


def left_join_parquet(
    df: plr.DataFrame,
    *,
    path: str,
    on: str | list[str],
    right_on: str | list[str] | None = None,
    select: list[str] | None = None,
) -> plr.DataFrame:
    """Left-join df with a parquet file on the given key(s). No-op if path is empty.

    right_on: column name(s) in the parquet when they differ from `on`.
    select: columns to bring in from the parquet (besides the join key). If None, all columns are included.
    """
    if not path:
        return df
    key = (
        [right_on]
        if isinstance(right_on, str)
        else (
            list(right_on)
            if right_on is not None
            else ([on] if isinstance(on, str) else list(on))
        )
    )
    right = plr.read_parquet(path, columns=key + select if select is not None else None)
    if right_on is not None:
        on_list = [on] if isinstance(on, str) else list(on)
        right_on_list = [right_on] if isinstance(right_on, str) else list(right_on)
        right = right.rename(dict(zip(right_on_list, on_list, strict=False)))
    return df.join(right, on=on, how="left")


def drop_overrepresented_by_loss(
    df: plr.DataFrame,
    *,
    columns: list[str] | None,
    n_bins: int = 100,
    max_bin_freq: float,
    seed: int = 0,
) -> plr.DataFrame:
    """Cap each loss bin at max_bin_freq * len(df) to flatten the loss distribution.

    Bins are equal-width over the [p001, p999] range of the mean loss across columns.
    No-op if any of the specified columns are missing from df.
    """
    if not columns or any(c not in df.columns for c in columns) or max_bin_freq == 1:
        return df

    max_bin_size = max(1, int(len(df) * max_bin_freq))

    score_col, bin_col, rank_col = "_score_", "_bin_", "_rank_"

    def _scalar_expr(c: str) -> plr.Expr:
        return (
            plr.col(c).arr.mean() if isinstance(df.schema[c], plr.Array) else plr.col(c)
        )

    score = plr.mean_horizontal([_scalar_expr(c) for c in columns])

    scores = df.select(score.alias(score_col))[score_col]
    finite = scores.filter(scores.is_finite())
    q_lo = finite.quantile(0.001)
    q_hi = finite.quantile(0.999)
    if q_lo is None or q_hi is None:
        return df
    lo, hi = float(q_lo), float(q_hi)
    breaks = [lo + (hi - lo) * i / n_bins for i in range(1, n_bins)]

    return (
        df
        .with_columns(score.alias(score_col))
        .with_columns(plr.col(score_col).cut(breaks, left_closed=True).alias(bin_col))
        .with_columns(
            plr.int_range(plr.len()).shuffle(seed=seed).over(bin_col).alias(rank_col)
        )
        .filter(plr.col(rank_col) < max_bin_size)
        .drop([score_col, bin_col, rank_col])
    )
