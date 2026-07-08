import numpy as np
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
    key = [right_on] if isinstance(right_on, str) else (list(right_on) if right_on is not None else ([on] if isinstance(on, str) else list(on)))
    right = plr.read_parquet(path, columns=key + select if select is not None else None)
    if right_on is not None:
        on_list = [on] if isinstance(on, str) else list(on)
        right_on_list = [right_on] if isinstance(right_on, str) else list(right_on)
        right = right.rename(dict(zip(right_on_list, on_list)))
    return df.join(right, on=on, how="left")


def drop_overrepresented_still_zero(
    df: plr.DataFrame,
    *,
    gas_column: str,
    speed_column: str,
    still_zero_keep_frac: float,
    speed_max_kmh: float = 15.0,
    speed_tick: int = 4,
    gas_eps: float = 1e-6,
    still_zero_seed: int = 0,
) -> plr.DataFrame:
    """Downsample windows where gas stays exactly zero for the whole window at low speed.

    These windows outnumber launch onsets ~8:1 at launch-relevant states and pull the
    Gaussian mean into the between-modes dead zone (mean-splitting). Capping them
    rebalances press vs no-press at launch without touching the architecture.
    Windows with any gas activity, or at speed >= speed_max_kmh (tick `speed_tick`,
    the last history tick), are always kept. No-op if the columns are missing
    (e.g. no scores_parquet joined) or keep_frac == 1.
    """
    if (
        gas_column not in df.columns
        or speed_column not in df.columns
        or still_zero_keep_frac == 1
    ):
        return df

    gas = df[gas_column].to_numpy().reshape(len(df), -1)
    speed = df[speed_column].to_numpy().reshape(len(df), -1)
    # null-joined rows become NaN -> not boring -> kept
    boring = (np.nan_to_num(gas, nan=1.0) <= gas_eps).all(axis=1) & (
        np.nan_to_num(speed[:, speed_tick], nan=np.inf) < speed_max_kmh
    )
    rng = np.random.default_rng(still_zero_seed)
    keep = ~boring | (rng.random(len(df)) < still_zero_keep_frac)
    return df.filter(plr.Series(keep))


def drop_overrepresented_brake_hold(
    df: plr.DataFrame,
    *,
    brake_column: str,
    speed_column: str,
    brake_hold_keep_frac: float,
    speed_max_kmh: float = 15.0,
    speed_tick: int = 4,
    press_eps: float = 0.01,
    brake_hold_seed: int = 1,
) -> plr.DataFrame:
    """Downsample windows where the brake is held for the entire window at low speed.

    Launches in the data begin from a brake hold; held-brake stopped windows vastly
    outnumber release transitions and teach the policy to ride the brake at launch
    states (closed loop: predicted brake ~0.06 sits above rsim's 0.05 gas mutex).
    Windows containing a release (brake drops below press_eps at any tick) are
    always kept. No-op if the columns are missing or keep_frac == 1.
    """
    if (
        brake_column not in df.columns
        or speed_column not in df.columns
        or brake_hold_keep_frac == 1
    ):
        return df

    brake = df[brake_column].to_numpy().reshape(len(df), -1)
    speed = df[speed_column].to_numpy().reshape(len(df), -1)
    # null-joined rows become NaN -> comparison False -> not boring -> kept
    boring = (np.nan_to_num(brake, nan=0.0) > press_eps).all(axis=1) & (
        np.nan_to_num(speed[:, speed_tick], nan=np.inf) < speed_max_kmh
    )
    rng = np.random.default_rng(brake_hold_seed)
    keep = ~boring | (rng.random(len(df)) < brake_hold_keep_frac)
    return df.filter(plr.Series(keep))


def drop_overrepresented_low_loss(
    df: plr.DataFrame,
    *,
    loss_columns: list[str],
    low_loss_keep_frac: float,
    low_loss_quantile: float = 0.5,
    low_loss_seed: int = 2,
) -> plr.DataFrame:
    """Downsample windows whose mean loss (over columns) is below the given quantile.

    Hard-example mining — the flip side of drop_overrepresented_by_loss: instead of
    flattening the loss histogram, concentrate training on what the reference
    checkpoint gets wrong (for gas, that is exactly the launch-onset tail).
    Rows with null/non-finite scores are kept. No-op if columns are missing or
    keep_frac == 1.
    """
    if any(c not in df.columns for c in loss_columns) or low_loss_keep_frac == 1:
        return df

    score = df.select(
        plr.mean_horizontal([plr.col(c) for c in loss_columns]).alias("_score_")
    )["_score_"].to_numpy()
    finite = score[np.isfinite(score)]
    thr = float(np.quantile(finite, low_loss_quantile))
    boring = np.nan_to_num(score, nan=np.inf) < thr
    rng = np.random.default_rng(low_loss_seed)
    keep = ~boring | (rng.random(len(df)) < low_loss_keep_frac)
    return df.filter(plr.Series(keep))


def drop_overrepresented_by_loss(
    df: plr.DataFrame,
    *,
    columns: list[str],
    n_bins: int = 100,
    max_bin_freq: float,
    seed: int = 0,
) -> plr.DataFrame:
    """Cap each loss bin at max_bin_freq * len(df) to flatten the loss distribution.

    Bins are equal-width over the [p001, p999] range of the mean loss across columns.
    No-op if any of the specified columns are missing from df.
    """
    if any(c not in df.columns for c in columns) or max_bin_freq == 1:
        return df

    max_bin_size = max(1, int(len(df) * max_bin_freq))

    score_col, bin_col, rank_col = "_score_", "_bin_", "_rank_"
    score = plr.mean_horizontal([plr.col(c) for c in columns])

    scores = df.select(score.alias(score_col))[score_col]
    finite = scores.filter(scores.is_finite())
    lo = float(finite.quantile(0.001))
    hi = float(finite.quantile(0.999))
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
