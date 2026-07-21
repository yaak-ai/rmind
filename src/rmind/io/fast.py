from typing import final

import numpy as np
import polars as pl
from pydantic import validate_call
from structlog import get_logger

logger = get_logger(__name__)


@final
class FASTBuilder:
    """FAST-style action features (https://arxiv.org/abs/2501.09747).

    Per-channel Discrete Cosine Transform (DCT-II) over the time axis, keeping the first
    `k` low-frequency coefficients. Smooth action trajectories compact their energy into
    the low frequencies, so a handful of coefficients captures the maneuver.

    Runs as the last step before binning/sampling: for each configured column (a
    fixed-length ``Array``/``List`` of length ``T`` per row) it emits an ``Array`` column
    ``"<column>/dct"`` of the first ``k`` coefficients, computed as a polars
    ``map_batches`` expression (lazy-compatible; the DCT is a dense matmul run in numpy).
    With ``gamma`` set, coefficients are FAST scale-and-round quantized
    (``round(gamma * C)`` as ``Int32``) so downstream binning is an exact-integer
    group-by; otherwise they are kept as ``Float32``.
    """

    __name__ = __qualname__

    @validate_call
    def __init__(
        self,
        *,
        columns: list[str],
        k: int,
        gamma: float | None = None,
        norm: str = "ortho",
        suffix: str = "/dct",
    ) -> None:
        self._columns = columns
        self._k = k
        self._gamma = gamma
        self._norm = norm
        self._suffix = suffix
        self._basis_cache: dict[int, np.ndarray] = {}

    def __call__(self, input: pl.DataFrame) -> pl.DataFrame:
        return self._build(input)

    def _basis_t(self, length: int) -> np.ndarray:
        """Transposed first-`k` DCT-II matrix; shape (length, k) for X @ basis_t."""
        basis = self._basis_cache.get(length)
        if basis is None:
            if self._k > length:
                msg = f"k={self._k} exceeds signal length {length}"
                raise ValueError(msg)
            n = np.arange(length)
            k = np.arange(self._k)[:, None]
            m = np.cos(np.pi * (2 * n + 1) * k / (2 * length))  # (k, length)
            if self._norm == "ortho":
                alpha = np.full((self._k, 1), np.sqrt(2.0 / length))
                alpha[0] = np.sqrt(1.0 / length)
                m = m * alpha
            basis = np.ascontiguousarray(m.T)  # (length, k)
            self._basis_cache[length] = basis
        return basis

    def _expr(self, column: str, length: int) -> pl.Expr:
        basis_t = self._basis_t(length)
        dtype = pl.Int32 if self._gamma is not None else pl.Float32
        out_dtype = pl.Array(dtype, self._k)
        gamma = self._gamma

        def transform(s: pl.Series) -> pl.Series:
            x = s.to_numpy()
            if x.dtype == object:  # List column -> ragged object array
                x = np.asarray(s.to_list())
            x = np.ascontiguousarray(x, dtype=np.float64).reshape(s.len(), length)
            coeffs = x @ basis_t  # (rows, k)
            if gamma is not None:
                coeffs = np.round(gamma * coeffs)
            return pl.Series(coeffs).cast(out_dtype)

        return (
            pl.col(column)
            .map_batches(transform, return_dtype=out_dtype)
            .alias(f"{column}{self._suffix}")
        )

    @staticmethod
    def _length(series: pl.Series) -> int:
        first = series.to_numpy()[0]
        return int(np.asarray(first).shape[0])

    def _build(self, input: pl.DataFrame) -> pl.DataFrame:
        exprs = [
            self._expr(column, self._length(input.get_column(column)))
            for column in self._columns
        ]
        logger.debug(
            "building FAST coefficients",
            columns=self._columns,
            k=self._k,
            gamma=self._gamma,
        )
        return input.with_columns(exprs)
