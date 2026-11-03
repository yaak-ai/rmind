"""FAST tokenizer for ego trajectories (https://arxiv.org/abs/2501.09747), applied to poses.

encode: poses (P, A) -> DCT-II over the P poses, per axis -> round(gamma_a * c) -> integers
        -> flatten low to high frequency with the axes interleaved (c0_x c0_y c0_th c1_x ...)
        -> one character per integer -> BPE token ids
decode: the inverse; rounding is the only lossy step, BPE is lossless.

Unlike a learned VQ tokenizer, the DCT and the rounding are fixed, so the reconstruction error
is set by `gammas` alone and is the same for every kind of trajectory. Only the BPE merge
table is fitted (`fit`), and it only changes how many tokens a trajectory takes.
"""

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Self, final

import numpy as np
from tokenizers import Tokenizer, models, trainers

# integers are written as code points BASE + (value - value_min); BASE skips control/ASCII
# characters, and the whole alphabet stays far below the surrogate range (0xD800)
_BASE = 0x100
_MAX_ALPHABET = 0xD800 - _BASE


def dct_basis(n: int) -> np.ndarray:
    """Orthonormal DCT-II matrix (n, n): coefficients = basis @ signal, signal = basis.T @ coefficients."""
    k = np.arange(n)[:, None]
    m = np.cos(np.pi * (2 * np.arange(n) + 1) * k / (2 * n))
    scale = np.full((n, 1), np.sqrt(2.0 / n))
    scale[0] = np.sqrt(1.0 / n)
    return m * scale


@final
class FASTTrajectoryTokenizer:
    def __init__(
        self,
        *,
        num_poses: int = 10,
        gammas: Sequence[float] = (10.0, 10.0, 100.0),
        value_min: int | None = None,
        value_max: int | None = None,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        self.num_poses = num_poses
        self.gammas = np.asarray(gammas, dtype=np.float64)
        self.num_axes = len(self.gammas)
        self.value_min = value_min
        self.value_max = value_max
        self.tokenizer = tokenizer
        self._basis = dct_basis(num_poses)

    # --- fixed part: DCT + rounding ----------------------------------------------------

    def coefficients(self, poses: np.ndarray) -> np.ndarray:
        """(..., P, A) poses -> (..., P, A) rounded DCT coefficients (int64)."""
        c = np.einsum("kn,...na->...ka", self._basis, np.asarray(poses, dtype=np.float64))
        return np.round(c * self.gammas).astype(np.int64)

    def poses(self, coefficients: np.ndarray) -> np.ndarray:
        """(..., P, A) integer coefficients -> (..., P, A) poses."""
        c = np.asarray(coefficients, dtype=np.float64) / self.gammas
        return np.einsum("kn,...ka->...na", self._basis, c)

    # --- integers <-> characters ---------------------------------------------------------

    def _fitted(self) -> tuple[Tokenizer, int, int]:
        """Return (tokenizer, value_min, value_max).

        Raises:
            RuntimeError: before fit() or load().
        """
        if self.tokenizer is None or self.value_min is None or self.value_max is None:
            msg = "tokenizer is not fitted; call fit() or load()"
            raise RuntimeError(msg)
        return self.tokenizer, self.value_min, self.value_max

    @staticmethod
    def _to_string(coefficients: np.ndarray, value_min: int, value_max: int) -> str:
        flat = np.clip(coefficients.reshape(-1), value_min, value_max)  # (P*A,) frequency-major
        return "".join(chr(_BASE + int(v) - value_min) for v in flat)

    def _from_string(self, text: str, value_min: int) -> np.ndarray:
        values = [ord(ch) - _BASE + value_min for ch in text]
        n = self.num_poses * self.num_axes
        values = (values + [0] * n)[:n]   # malformed lengths: pad with zeros (high frequencies) or truncate
        return np.asarray(values, dtype=np.int64).reshape(self.num_poses, self.num_axes)

    # --- BPE ------------------------------------------------------------------------------

    def fit(self, poses: np.ndarray, *, vocab_size: int = 4096, margin: int = 16) -> Self:
        """Fit the alphabet range and the BPE merges on (N, P, A) training poses.

        Raises:
            ValueError: if the alphabet is too large, or `vocab_size` leaves no room for merges.
        """
        coefficients = self.coefficients(poses)
        self.value_min = int(coefficients.min()) - margin
        self.value_max = int(coefficients.max()) + margin
        alphabet_size = self.value_max - self.value_min + 1
        if alphabet_size > _MAX_ALPHABET:
            msg = f"alphabet of {alphabet_size} values is too large; lower gammas"
            raise ValueError(msg)
        if vocab_size <= alphabet_size:
            msg = f"vocab_size {vocab_size} must exceed the alphabet size {alphabet_size} to leave room for merges"
            raise ValueError(msg)
        alphabet = [chr(_BASE + i) for i in range(alphabet_size)]
        tokenizer = Tokenizer(models.BPE())
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, initial_alphabet=alphabet, show_progress=False)
        value_min, value_max = self.value_min, self.value_max
        tokenizer.train_from_iterator(
            (self._to_string(c, value_min, value_max) for c in coefficients), trainer=trainer, length=len(coefficients)
        )
        self.tokenizer = tokenizer
        return self

    def encode(self, poses: np.ndarray) -> list[list[int]]:
        """(N, P, A) poses -> one list of token ids per trajectory."""
        tokenizer, value_min, value_max = self._fitted()
        strings = [self._to_string(c, value_min, value_max) for c in self.coefficients(poses)]
        return [e.ids for e in tokenizer.encode_batch(strings)]

    def decode(self, token_ids: Iterable[Sequence[int]]) -> np.ndarray:
        """Token id lists -> (N, P, A) poses."""
        tokenizer, value_min, _ = self._fitted()
        vocab = {i: tok for tok, i in tokenizer.get_vocab().items()}
        coefficients = [self._from_string("".join(vocab.get(i, "") for i in ids), value_min) for ids in token_ids]
        return self.poses(np.stack(coefficients))

    @property
    def vocab_size(self) -> int:
        return self._fitted()[0].get_vocab_size()

    # --- persistence ----------------------------------------------------------------------

    def save(self, directory: str | Path) -> None:
        tokenizer, _, _ = self._fitted()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(directory / "tokenizer.json"))
        meta = {"num_poses": self.num_poses, "gammas": self.gammas.tolist(), "value_min": self.value_min, "value_max": self.value_max}
        (directory / "fast_trajectory.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, directory: str | Path) -> "FASTTrajectoryTokenizer":
        directory = Path(directory)
        meta = json.loads((directory / "fast_trajectory.json").read_text())
        return cls(**meta, tokenizer=Tokenizer.from_file(str(directory / "tokenizer.json")))
