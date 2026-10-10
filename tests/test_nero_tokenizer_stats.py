"""Self-contained NeroChunkTokenizer checkpoints (embedded standardizer + event ref)."""

import hashlib
import shutil
import warnings
from pathlib import Path
from typing import Any

import pytest
import torch

from rmind.data.nero_robot import AxisStandardizer, EventReference
from rmind.models.nero_chunk_tokenizer import STATS_KEY, NeroChunkTokenizer
from rmind.scripts import nero_tokenizer_embed_stats
from tests.test_nero_split_tokenizer import A, H, _config


def _stats(root: Path, seed: int = 0) -> tuple[Path, Path]:
    g = torch.Generator().manual_seed(seed)
    std = AxisStandardizer(
        mean=torch.randn(2, A, generator=g), std=torch.rand(2, A, generator=g) + 0.5
    )
    ref = EventReference(
        reference=torch.randn(A, generator=g).tolist(),
        relative_mode="none",
        standardizer_sha256=std.digest,
    )
    root.mkdir(parents=True, exist_ok=True)
    std.save(root / "action_standardizer_none.json")
    ref.save(root / "event_reference_none.json")
    return root / "action_standardizer_none.json", root / "event_reference_none.json"


def _tokenizer(stats: Path) -> NeroChunkTokenizer:
    """Hparams hold absolute stats PATHS, like every real checkpoint."""
    std, ref = _stats(stats)
    torch.manual_seed(0)
    config: dict = _config() | {
        "standardizer": {
            "_target_": "rmind.data.nero_robot.AxisStandardizer.load",
            "path": str(std),
        },
        "event_reference": str(ref),
    }
    return NeroChunkTokenizer(**config).eval()


def _save(tok: NeroChunkTokenizer, path: Path, *, embed: bool) -> Path:
    ckpt: dict[str, Any] = {
        "state_dict": tok.state_dict(),
        "hyper_parameters": dict(tok.hparams),
        "pytorch-lightning_version": "2.5.0",
    }
    if embed:
        tok.on_save_checkpoint(ckpt)  # what Trainer.save_checkpoint calls
    torch.save(ckpt, path)
    return path


def _same(a: NeroChunkTokenizer, b: NeroChunkTokenizer) -> None:
    x = torch.randn(8, H, A, generator=torch.Generator().manual_seed(3))
    codes = a(x)
    assert torch.equal(b(x), codes)
    assert torch.equal(b.invert(codes), a.invert(codes))
    assert b.standardizer_digest == a.standardizer_digest
    assert b.standardizer.sides == a.standardizer.sides
    assert torch.equal(b.event_reference, a.event_reference)
    for key, value in a.state_dict().items():
        assert torch.equal(b.state_dict()[key], value), key


def test_checkpoint_embeds_and_loads_without_the_stats_files(tmp_path: Path) -> None:
    tok = _tokenizer(tmp_path / "stats")
    path = _save(tok, tmp_path / "tok.ckpt", embed=True)
    stats = torch.load(path, weights_only=False)[STATS_KEY]
    assert stats["action_standardizer"]["sha256"] == tok.standardizer_digest
    assert stats["event_reference"]["source"] == str(
        tmp_path / "stats" / "event_reference_none.json"
    )

    shutil.rmtree(tmp_path / "stats")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no "differs" warning without the files
        loaded = NeroChunkTokenizer.load_from_checkpoint(path, map_location="cpu")
    _same(tok, loaded.eval())

    # re-saving the loaded model embeds the same stats again
    again = _save(loaded, tmp_path / "again.ckpt", embed=True)
    assert torch.load(again, weights_only=False)[STATS_KEY] == stats
    _same(tok, NeroChunkTokenizer.load_from_checkpoint(again).eval())


def test_legacy_checkpoint_still_reads_the_hparams_paths(tmp_path: Path) -> None:
    tok = _tokenizer(tmp_path / "stats")
    path = _save(tok, tmp_path / "tok.ckpt", embed=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        loaded = NeroChunkTokenizer.load_from_checkpoint(path, map_location="cpu")
    _same(tok, loaded.eval())
    assert loaded.hparams["standardizer"] == tok.hparams["standardizer"]
    assert loaded.hparams["event_reference"] == tok.hparams["event_reference"]
    assert loaded.standardizer.source == tok.standardizer.source

    shutil.rmtree(tmp_path / "stats")  # the old host-bound behaviour, unchanged
    with pytest.raises(Exception, match="FileNotFoundError"):
        NeroChunkTokenizer.load_from_checkpoint(path, map_location="cpu")


def test_embedded_stats_win_and_a_different_file_warns(tmp_path: Path) -> None:
    tok = _tokenizer(tmp_path / "stats")
    path = _save(tok, tmp_path / "tok.ckpt", embed=True)
    _stats(tmp_path / "stats", seed=1)  # the paths now hold OTHER stats
    with pytest.warns(UserWarning, match="differs from the one embedded") as record:
        loaded = NeroChunkTokenizer.load_from_checkpoint(path, map_location="cpu")
    assert len(record) == 2  # standardizer and event reference  # noqa: PLR2004
    _same(tok, loaded.eval())


def test_embedded_payload_must_match_its_sha(tmp_path: Path) -> None:
    tok = _tokenizer(tmp_path / "stats")
    path = _save(tok, tmp_path / "tok.ckpt", embed=True)
    ckpt = torch.load(path, weights_only=False)
    ckpt[STATS_KEY]["action_standardizer"]["payload"]["mean"][0] += 1.0
    torch.save(ckpt, path)
    with pytest.raises(ValueError, match="embedded action standardizer hashes"):
        NeroChunkTokenizer.load_from_checkpoint(path, map_location="cpu")


def test_embed_script_upgrades_a_legacy_checkpoint(tmp_path: Path) -> None:
    tok = _tokenizer(tmp_path / "stats")
    legacy = _save(tok, tmp_path / "tok.ckpt", embed=False)
    sha = hashlib.sha256(legacy.read_bytes()).hexdigest()
    # on a host without the stored path: the files from another directory
    shutil.move(tmp_path / "stats", tmp_path / "nas")
    with pytest.raises(FileNotFoundError, match="--stats-dir"):
        nero_tokenizer_embed_stats.main([str(legacy)])

    nero_tokenizer_embed_stats.main([str(legacy), "--stats-dir", str(tmp_path / "nas")])
    out = tmp_path / "tok.selfcontained.ckpt"
    assert hashlib.sha256(legacy.read_bytes()).hexdigest() == sha  # untouched
    with pytest.raises(FileExistsError):
        nero_tokenizer_embed_stats.embed(legacy, out, tmp_path / "nas")
    with pytest.raises(ValueError, match="already embeds"):
        nero_tokenizer_embed_stats.embed(out, tmp_path / "x.ckpt", tmp_path / "nas")

    shutil.rmtree(tmp_path / "nas")
    loaded = NeroChunkTokenizer.load_from_checkpoint(out, map_location="cpu").eval()
    _same(tok, loaded)
    embedded = torch.load(out, weights_only=False)[STATS_KEY]
    assert embedded["action_standardizer"]["source"] == str(
        tmp_path / "stats" / "action_standardizer_none.json"
    )


def test_embed_script_refuses_stats_that_are_not_the_checkpoints(
    tmp_path: Path,
) -> None:
    tok = _tokenizer(tmp_path / "stats")
    legacy = _save(tok, tmp_path / "tok.ckpt", embed=False)
    _stats(tmp_path / "other", seed=1)
    with pytest.raises(ValueError, match="buffers hash to"):
        nero_tokenizer_embed_stats.embed(
            legacy, tmp_path / "out.ckpt", tmp_path / "other"
        )
    assert not (tmp_path / "out.ckpt").exists()
