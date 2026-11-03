from pathlib import Path

import numpy as np
import pytest

from rmind.components.fast_trajectory import FASTTrajectoryTokenizer, dct_basis


def _arcs(n: int = 400, seed: int = 0) -> np.ndarray:
    """(n, 10, 3) circular-arc trajectories with random speed and yaw rate (x right, y forward, theta)."""
    rng = np.random.default_rng(seed)
    t = np.arange(1, 11) / 3.0
    v = rng.uniform(0.0, 20.0, (n, 1))
    w = rng.uniform(-0.4, 0.4, (n, 1))
    th = w * t
    eps = 1e-6
    safe_w = np.where(np.abs(w) < eps, eps, w)
    x = v / safe_w * (1 - np.cos(th))
    y = v / safe_w * np.sin(th)
    return np.stack([x, y, th], -1)


def test_dct_basis_is_orthonormal() -> None:
    m = dct_basis(10)
    assert np.allclose(m @ m.T, np.eye(10), atol=1e-12)


def test_rounding_is_the_only_loss() -> None:
    poses = _arcs()
    tok = FASTTrajectoryTokenizer(gammas=(10.0, 10.0, 100.0))
    rec = tok.poses(tok.coefficients(poses))
    # orthonormal basis: per-axis error norm <= half a rounding step per coefficient
    bound = np.sqrt(10) * 0.5 / tok.gammas
    assert (np.linalg.norm(rec - poses, axis=1) <= bound + 1e-9).all()


def test_bpe_round_trip_is_lossless_and_saves(tmp_path: Path) -> None:
    poses = _arcs()
    tok = FASTTrajectoryTokenizer(gammas=(2.0, 2.0, 20.0)).fit(poses, vocab_size=1024)
    ids = tok.encode(poses)
    n_chars = 30  # 10 coefficients x 3 axes, one character each
    assert max(len(i) for i in ids) < n_chars  # merges shorten the strings
    assert np.array_equal(tok.coefficients(tok.decode(ids)), tok.coefficients(poses))
    tok.save(tmp_path)
    loaded = FASTTrajectoryTokenizer.load(tmp_path)
    assert loaded.encode(poses) == ids
    assert np.allclose(loaded.decode(ids), tok.decode(ids))


def test_malformed_sequences_decode_to_the_right_shape() -> None:
    poses = _arcs(50)
    tok = FASTTrajectoryTokenizer(gammas=(2.0, 2.0, 20.0)).fit(poses, vocab_size=1024)
    ids = tok.encode(poses[:2])
    out = tok.decode([ids[0][:1], ids[1] + ids[1]])  # too short, too long
    assert out.shape == (2, 10, 3)


def test_vocab_must_exceed_alphabet() -> None:
    with pytest.raises(ValueError, match="vocab_size"):
        FASTTrajectoryTokenizer(gammas=(10.0, 10.0, 100.0)).fit(_arcs(), vocab_size=64)
