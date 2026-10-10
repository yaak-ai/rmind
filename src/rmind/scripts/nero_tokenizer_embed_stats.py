"""Embed a NeroChunkTokenizer checkpoint's stats so it loads on any host.

A tokenizer checkpoint saved before `nero_tokenizer_stats` existed builds its
action standardizer and event reference from the absolute JSON paths in its
hparams, so it only loads where those paths exist. This writes a NEW checkpoint,
`<name>.selfcontained.ckpt` next to the input by default, with both payloads
embedded under `checkpoint["nero_tokenizer_stats"]`. The original is never
written: tokenizer pins (`tokenizer_ckpt_sha256`) hash it, and the new file has
its own sha256 (printed), so re-pin deliberately if you switch a run to it.

The embedded standardizer is rebuilt from the checkpoint's own
`standardizer.mean/std` buffers (what the model computes with) and must hash to
the stats file's sha256; the event reference must equal the `event_reference`
buffer. Either mismatch refuses. The JSON files are read from the hparams paths
or, with `--stats-dir`, by file name from that directory (e.g. the
byte-identical NAS copy on a host without the original path)::

    python -m rmind.scripts.nero_tokenizer_embed_stats \\
        /nasa/max/nero-sweep-ckpts/paper2_tok_q4/checkpoints/epoch=999-step=150000.ckpt \\
        --stats-dir /nasa/max/nero-cache/cube-bimanual/stats_c10_v3 --out /tmp/q4.ckpt
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import torch

from rmind.data.nero_robot import AxisStandardizer, EventReference
from rmind.models.nero_chunk_tokenizer import (
    STATS_KEY,
    NeroChunkTokenizer,
    embedded_stats,
)

__all__ = ["embed", "main"]

SUFFIX = ".selfcontained.ckpt"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _locate(stored: str, stats_dir: Path | None) -> Path:
    path = Path(stored) if stats_dir is None else stats_dir / Path(stored).name
    if not path.is_file():
        msg = f"{path} not found (pass --stats-dir with a copy of {stored})"
        raise FileNotFoundError(msg)
    return path


def _standardizer(
    hparams: dict[str, Any], state: dict[str, Any], stats_dir: Path | None
) -> AxisStandardizer:
    """From the checkpoint's buffers; must hash to the stats file's sha256.

    Raises:
        ValueError: if the buffers are not the file's numbers.
    """
    stored = (hparams.get("standardizer") or {}).get("path")
    if stored is None:  # the identity default: nothing to read
        return AxisStandardizer(
            mean=state["standardizer.mean"], std=state["standardizer.std"]
        )
    path = _locate(stored, stats_dir)
    std = AxisStandardizer(
        mean=state["standardizer.mean"],
        std=state["standardizer.std"],
        sides=AxisStandardizer.load(path).sides,
        source=stored,
    )
    if std.digest != _sha256(path):
        msg = (
            f"the checkpoint's standardizer buffers hash to {std.digest}, "
            f"{path} to {_sha256(path)}"
        )
        raise ValueError(msg)
    return std


def _event_reference(
    stored: str | None,
    state: dict[str, Any],
    std: AxisStandardizer,
    stats_dir: Path | None,
) -> EventReference | None:
    """The stats file; must equal the checkpoint's `event_reference` buffer.

    Raises:
        ValueError: if it does not, or it is pinned to another standardizer.
    """
    if stored is None:
        return None
    ref = EventReference.load(_locate(stored, stats_dir))
    buffer = state["event_reference"]
    if not torch.equal(torch.tensor(ref.reference, dtype=buffer.dtype), buffer):
        msg = f"{stored} differs from the checkpoint's event_reference buffer"
        raise ValueError(msg)
    if ref.standardizer_sha256 != std.digest:
        msg = f"{stored} is pinned to standardizer {ref.standardizer_sha256}"
        raise ValueError(msg)
    return ref


def embed(ckpt: Path, out: Path, stats_dir: Path | None = None) -> dict[str, Any]:
    """Write `out` = `ckpt` + embedded stats; returns a small report.

    Raises:
        FileExistsError: if `out` exists.
        ValueError: if `ckpt` already embeds its stats, or the files do not match
            the checkpoint's buffers.
        RuntimeError: if the result does not reload to the same standardizer.
    """
    if out.exists():
        msg = f"{out} exists; refusing to overwrite"
        raise FileExistsError(msg)
    before = _sha256(ckpt)
    checkpoint = torch.load(ckpt, map_location="cpu", weights_only=False)
    if STATS_KEY in checkpoint:
        msg = f"{ckpt} already embeds its stats ({STATS_KEY})"
        raise ValueError(msg)
    hparams, state = checkpoint["hyper_parameters"], checkpoint["state_dict"]
    std = _standardizer(hparams, state, stats_dir)
    ref_stored = hparams.get("event_reference")
    ref = _event_reference(ref_stored, state, std, stats_dir)

    checkpoint[STATS_KEY] = embedded_stats(std, ref, ref_stored)
    with out.open("xb") as f:  # never the input: it exists, so "x" refuses it
        torch.save(checkpoint, f)
    loaded = NeroChunkTokenizer.load_from_checkpoint(out, map_location="cpu")
    if loaded.standardizer_digest != std.digest:  # pragma: no cover
        msg = "the self-contained checkpoint does not reload to the same standardizer"
        raise RuntimeError(msg)
    return {
        "in": str(ckpt),
        "in_sha256": before,
        "out": str(out),
        "out_sha256": _sha256(out),
        "standardizer_sha256": std.digest,
        "event_reference_sha256": None if ref is None else ref.digest,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("ckpt", type=Path, nargs="+")
    parser.add_argument(
        "--stats-dir",
        type=Path,
        help="read the stats JSONs by file name from here, not the hparams paths",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help=f"output file (one ckpt) or directory; default <name>{SUFFIX} beside it",
    )
    args = parser.parse_args(argv)
    if args.out is not None and len(args.ckpt) > 1 and not args.out.is_dir():
        parser.error("--out must be a directory with several checkpoints")
    for ckpt in args.ckpt:
        name = ckpt.name.removesuffix(".ckpt") + SUFFIX
        if args.out is None:
            out = ckpt.with_name(name)
        elif args.out.is_dir():
            out = args.out / name
        else:
            out = args.out
        report = embed(ckpt, out, args.stats_dir)
        for key, value in report.items():
            print(f"{key}: {value}")  # noqa: T201
        print()  # noqa: T201


if __name__ == "__main__":
    main()
