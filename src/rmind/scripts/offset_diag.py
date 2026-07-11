"""Shared diagnostics harness for the JointPolicyObjective offset-head probes.

Provides three building blocks for the Phase-1 diagnostics of the
offset-supervision bug (offset head trained at sampled instead of
ground-truth codes):

- `load_policy`: load the finetuned `ControlTransformer` from a local
  checkpoint file, in eval mode, with `sample_codes=False` on the policy
  objective so all probes match export-time (argmax) decoding.
- `build_dataloader`: build the train/val/train_debug dataloader exactly as
  the finetune experiment does (hydra compose of
  `experiment=yaak/control_transformer/finetune`); first instantiation of a
  split builds the rbyte sample cache over NFS.
- `iter_policy_tensors`: yield per-batch policy tensors (features, code
  logits, the full offset table, GT chunk, GT codes, normalized GT target)
  under `torch.inference_mode`.

Usage (smoke test / cache build):
    uv run python -m rmind.scripts.offset_diag --split val --max-batches 2
    uv run python -m rmind.scripts.offset_diag --split train --skip-model
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils._pytree import tree_map  # noqa: PLC2701

from rmind.components.objectives.joint_policy import JointPolicyObjective
from rmind.models.control_transformer import ControlTransformer

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from rbyte.dataloader import TorchDataNodeDataLoader
    from torch import Tensor

    from rmind.models.action_tokenizer import ActionTokenizer

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "config"
DEFAULT_CKPT = REPO_ROOT / "artifacts" / "model-2gqxhjod:v9" / "model.ckpt"

EXPECTED_NUM_QUANTIZERS = 4
EXPECTED_CODEBOOK_SIZE = 16


def load_policy(
    ckpt_path: str | Path = DEFAULT_CKPT, device: str | torch.device = "cuda"
) -> ControlTransformer:
    """Load the finetuned model from a local checkpoint, ready for probing.

    The model is set to eval, gradients are disabled, and the policy
    objective decodes with argmax (`sample_codes=False`) to match the export
    path (`config/export/yaak/control_transformer/finetuned.yaml`).

    Note: the checkpoint hparams reference the frozen action tokenizer as a
    wandb artifact (`yaak/rmind/model-gkzgn6gk:v9`); resolving it hits the
    wandb API but reuses the already-downloaded files under `./artifacts`.

    Raises:
        TypeError: if `objectives['policy']` is not a `JointPolicyObjective`.
        ValueError: if the tokenizer quantizer geometry is not G=4, C=16.
    """
    model = ControlTransformer.load_from_checkpoint(
        Path(ckpt_path), map_location="cpu", weights_only=False
    )
    model = model.to(torch.device(device)).eval().requires_grad_(requires_grad=False)

    policy = model.objectives["policy"]
    if not isinstance(policy, JointPolicyObjective):
        msg = f"objectives['policy'] is {type(policy).__name__}, expected JointPolicyObjective"
        raise TypeError(msg)

    quantizer = cast("ActionTokenizer", policy.tokenizer).quantizer
    if (quantizer.num_quantizers, quantizer.codebook_size) != (
        EXPECTED_NUM_QUANTIZERS,
        EXPECTED_CODEBOOK_SIZE,
    ):
        msg = (
            f"unexpected quantizer geometry: G={quantizer.num_quantizers} "
            f"C={quantizer.codebook_size}, expected G={EXPECTED_NUM_QUANTIZERS} "
            f"C={EXPECTED_CODEBOOK_SIZE}"
        )
        raise ValueError(msg)

    policy.sample_codes = False  # match export-time (argmax) decoding
    return model


def build_dataloader(
    split: str,
    batch_size: int = 32,
    num_workers: int = 2,
    *,
    shuffle: bool | None = None,
) -> TorchDataNodeDataLoader[dict[str, Any]]:
    """Instantiate a dataloader of the finetune experiment for `split`.

    `split` is one of 'train', 'val', 'train_debug'. The 3hz episode
    geometry (episode_length=6, episode_step=10, clip_length=11, ...) comes
    from the composed experiment config and is left untouched. First call
    per split builds the rbyte sample cache under `.rbyte_cache` (slow, NFS).
    `shuffle=None` keeps the experiment default (train: True, val: True).

    Raises:
        ValueError: if `split` is not a known split name.
    """
    if split not in {"train", "val", "train_debug"}:
        msg = f"unknown split: {split!r}"
        raise ValueError(msg)

    overrides = [
        "experiment=yaak/control_transformer/finetune",
        # only cfg.datamodule is consumed; these just satisfy mandatory keys
        "model.artifact=placeholder/placeholder:v0",
        "action_tokenizer_artifact=placeholder/placeholder:v0",
    ]

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="train", overrides=overrides)

    if split == "train_debug":
        # graft the debug dataset into the composed tree; its interpolations
        # (${paths.*}, ${clip_length}, ...) resolve against the root config
        cfg.datamodule.train.dataset = OmegaConf.load(
            CONFIG_DIR / "dataset" / "yaak" / "train_debug.yaml"
        )

    # make the cache path independent of the caller's working directory
    cfg.paths.rbyte.cache = str(REPO_ROOT / ".rbyte_cache")

    node = cfg.datamodule.val if split == "val" else cfg.datamodule.train
    node.batch_size = batch_size
    node.num_workers = num_workers
    if shuffle is not None:
        node.shuffle = shuffle

    return instantiate(node)


def shutdown_dataloader(dataloader: Any) -> None:
    """Best-effort shutdown of `TorchDataNodeDataLoader` worker threads.

    torchdata.nodes has no public shutdown API; abandoning an iterator
    mid-epoch can abort the process at interpreter exit ("terminate called
    without an active exception"). Walking the node graph and calling the
    private `_shutdown` hooks joins the threads; a later `iter()` on the
    same loader resets and restarts them, so the loader stays reusable.
    """
    root = getattr(getattr(dataloader, "_loader", None), "root", None)
    stack: list[Any] = [root]
    seen: set[int] = set()
    while stack:
        node = stack.pop()
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        shutdown = getattr(node, "_shutdown", None)
        if callable(shutdown):
            shutdown()
        stack.extend(getattr(node, attr, None) for attr in ("_it", "source", "root"))


def _to_device(batch: Any, device: torch.device) -> Any:
    return tree_map(
        lambda x: x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x,
        batch,
    )


def iter_policy_tensors(
    model: ControlTransformer,
    dataloader: Iterable[dict[str, Any]],
    device: str | torch.device = "cuda",
    max_batches: int | None = None,
) -> Iterator[dict[str, Tensor]]:
    """Yield per-batch policy tensors for offset diagnostics.

    Yields:
    - `features`     (b, 1152): policy features, built exactly like
                     `ControlTransformer.validation_step` (episode_builder +
                     encoder) followed by `JointPolicyObjective._features`
    - `code_logits`  (b, 4, 16): per-quantizer code logits
    - `offsets`      (b, 4, 16, 24): the full per-(quantizer, code) offset table
    - `chunk`        (b, 6, 4): raw GT action chunk of the last timestep
    - `target_codes` (b, 4): GT residual-VQ codes, `tokenizer(chunk)`
    - `target`       (b, 24): normalized GT action vector
    """
    device = torch.device(device)
    policy = cast("JointPolicyObjective", model.objectives["policy"])
    tokenizer = cast("ActionTokenizer", policy.tokenizer)

    try:
        with torch.inference_mode():
            for batch_idx, batch_cpu in enumerate(dataloader):
                if max_batches is not None and batch_idx >= max_batches:
                    break

                batch = _to_device(batch_cpu, device)
                episode = model.episode_builder(batch)
                embedding = model.encoder(
                    src=episode.embeddings_flattened, mask=episode.attention_mask
                )

                features = policy._features(episode, embedding)  # noqa: SLF001
                code_logits, offsets = policy._heads(features)  # noqa: SLF001

                chunk = episode.get(policy.chunk)[:, -1]  # (b, horizon, fields)
                target_codes = tokenizer(chunk)
                target = tokenizer._normalize(chunk.flatten(-2, -1))  # noqa: SLF001

                yield {
                    "features": features,
                    "code_logits": code_logits,
                    "offsets": offsets,
                    "chunk": chunk,
                    "target_codes": target_codes,
                    "target": target,
                }
    finally:
        # join worker threads on early exit (incl. caller break -> GeneratorExit)
        # so abandoned iterators cannot abort the process at interpreter exit
        shutdown_dataloader(dataloader)


def _smoke(args: argparse.Namespace) -> None:
    t0 = time.perf_counter()
    loader = build_dataloader(
        args.split, batch_size=args.batch_size, num_workers=args.num_workers
    )
    t_build = time.perf_counter() - t0
    n_samples = len(loader.dataset)
    print(  # noqa: T201
        f"[offset_diag] split={args.split} dataset built in {t_build:.1f}s: "
        f"{n_samples} samples, {len(loader)} batches of {args.batch_size}"
    )

    if args.skip_model:
        return

    model = load_policy(args.ckpt, args.device)
    print(f"[offset_diag] model loaded from {args.ckpt} on {args.device}")  # noqa: T201

    t1 = time.perf_counter()
    n_batches = 0
    for tensors in iter_policy_tensors(
        model, loader, device=args.device, max_batches=args.max_batches
    ):
        n_batches += 1
        shapes = {k: tuple(v.shape) for k, v in tensors.items()}
        checksums = {k: float(v.float().abs().sum()) for k, v in tensors.items()}
        print(f"[offset_diag] batch {n_batches} shapes: {shapes}")  # noqa: T201
        print(f"[offset_diag] batch {n_batches} |sum|: {checksums}")  # noqa: T201
    t_iter = time.perf_counter() - t1
    print(  # noqa: T201
        f"[offset_diag] {n_batches} batches in {t_iter:.2f}s "
        f"({n_batches / t_iter:.2f} batches/s)"
    )


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_forkserver_preload(["rbyte", "polars"])

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split", default="val", choices=["train", "val", "train_debug"]
    )
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=2)
    parser.add_argument(
        "--skip-model", action="store_true", help="only build the dataset cache"
    )
    _smoke(parser.parse_args())
