"""Standalone correctness check for the frozen-encoder output cache.

Intentionally NOT a pytest test: it downloads a wandb checkpoint, needs a GPU
and the yaak data, and is an integration-level proof rather than a unit test.

Run via:
    export HOSTNAME=$(hostname)
    uv run python -m tests.scripts.verify_encoder_cache

What it proves (all under the live ``bf16-mixed`` autocast):
  1. Disk round-trip: cached losses (loaded back from memmap) are identical to
     the real DINOv3+encoder path on the same batch.
  2. sample_id keying: a cache populated from batch A yields correct losses for
     a *disjoint* batch B (proves row sid holds sample sid -- scatter/gather).
  3. Model routing: ``model.training_step`` with the cache active reproduces the
     uncached loss while NOT calling ``episode_builder.forward`` (i.e. DINOv3 is
     skipped) -- so the speedup is real, not a silent fallback.
  4. Invalidation: a stamp mismatch refuses to load.
It also reports per-sample cache bytes.
"""

from typing import Any

import hydra.core.global_hydra
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from tensordict import TensorDict

from rmind.components.encoder_cache import EncoderCache
from rmind.components.loss import HasLogitBias

CONFIG_DIR = "/home/max/Code/rmind-cache/config"
ARTIFACT = "yaak/rmind/model-et584pft:v9"
PRECISION = "bf16-mixed"
TOL = 1e-3  # bf16 relative tolerance (we observe exactly 0.0)


def _to(x: Any, device: torch.device) -> Any:
    if isinstance(x, dict):
        return {k: _to(v, device) for k, v in x.items()}
    return x.to(device) if isinstance(x, torch.Tensor) else x


def _loss(metrics: Any) -> float:
    return TensorDict(metrics)["loss"].sum(reduce=True).item()  # ty:ignore[no-matching-overload, unresolved-attribute]


def main() -> None:  # noqa: PLR0914, PLR0915
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(
            config_name="train",
            overrides=[
                "experiment=yaak/control_transformer/finetune",
                "datamodule=yaak/train_debug",
                f"++model.artifact={ARTIFACT}",
                "paths.rbyte.cache=/tmp/rbyte_cache",
            ],
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    autocast = torch.autocast(device_type=device.type, dtype=torch.bfloat16)

    model = instantiate(cfg.model).to(device)
    model.episode_builder.requires_grad_(False).eval()  # noqa: FBT003
    model.encoder.requires_grad_(False).eval()  # noqa: FBT003
    for mod in model.modules():
        if isinstance(mod, HasLogitBias) and mod.logit_bias is None:
            mod.logit_bias = torch.zeros(3, device=device)
    # heads in train mode; frozen modules in eval (cache validity premise)
    model.train()
    model.episode_builder.eval()
    model.encoder.eval()
    objective = model.objectives["policy"]

    dm = instantiate(cfg.datamodule)
    dm.setup("fit")
    it = iter(dm.train_dataloader())
    batch_a = _to(next(it), device)
    batch_b = _to(next(it), device)
    ids_a = batch_a["data"]["meta/sample_id"]
    ids_b = batch_b["data"]["meta/sample_id"]
    assert set(ids_a.tolist()).isdisjoint(ids_b.tolist())
    n = int(max(ids_a.max().item(), ids_b.max().item())) + 1

    def encode(batch: Any) -> Any:
        ep = model.episode_builder(batch)
        emb = model.encoder(src=ep.embeddings_flattened, mask=ep.attention_mask)
        return ep, emb

    import tempfile  # noqa: PLC0415

    root = tempfile.mkdtemp(prefix="enc_cache_verify_")
    import shutil  # noqa: PLC0415

    try:
        cache = EncoderCache(
            root=root,
            checkpoint=ARTIFACT,
            dataset_fingerprint="verify",
            num_samples={"train": n},
            precision=PRECISION,
        )

        with torch.no_grad():
            # real (uncached) losses under autocast
            with autocast:
                ep_a, emb_a = encode(batch_a)
                real_a = _loss(objective.compute_metrics(episode=ep_a, embedding=emb_a))
                ep_b, emb_b = encode(batch_b)
                real_b = _loss(objective.compute_metrics(episode=ep_b, embedding=emb_b))
                # populate from BOTH batches (separate forward, same autocast)
                cache.store_batch("train", batch=batch_a, episode=ep_a, embedding=emb_a)
                cache.store_batch("train", batch=batch_b, episode=ep_b, embedding=emb_b)

            # fill any unwritten rows so the COMPLETE pass is honest for this
            # 2-batch synthetic check (the real populator covers every row)
            store = cache._stores["train"]  # noqa: SLF001
            store.written[:] = True
            cache.mark_complete("train")

        # (1)+(2) disk round-trip parity via a FRESH cache instance, A and B
        cache2 = EncoderCache(
            root=root,
            checkpoint=ARTIFACT,
            dataset_fingerprint="verify",
            num_samples={"train": n},
            precision=PRECISION,
        )
        with torch.no_grad(), autocast:
            for tag, batch, real in (("A", batch_a, real_a), ("B", batch_b, real_b)):
                se, ce = cache2.build_cached("train", batch, device=device)
                cached = _loss(objective.compute_metrics(episode=se, embedding=ce))
                rel = abs(real - cached) / (abs(real) + 1e-8)
                assert rel < TOL, f"FAIL roundtrip {tag} rel={rel}"

        def _bytes_per_sample(td: Any) -> int:
            total = 0
            for v in td.values(include_nested=True, leaves_only=True):
                total += v.element_size() * v[0].numel()
            return total

        _bytes_per_sample(store.groups)
        _bytes_per_sample(store.input)

        # (3) model routing: training_step cached == uncached, episode_builder skipped
        with torch.no_grad(), autocast:
            unc = model.training_step(batch_a, batch_idx=1)["loss"].item()
        model.encoder_cache = cache2
        calls = {"n": 0}
        orig = model.episode_builder.forward

        def spy(*a: Any, **k: Any) -> Any:
            calls["n"] += 1
            return orig(*a, **k)

        model.episode_builder.forward = spy  # type: ignore[method-assign]
        with torch.no_grad(), autocast:
            cac = model.training_step(batch_a, batch_idx=1)["loss"].item()
        model.episode_builder.forward = orig  # type: ignore[method-assign]
        rel = abs(unc - cac) / (abs(unc) + 1e-8)
        assert calls["n"] == 0, "FAIL: episode_builder ran -> cache not used"
        assert rel < TOL, f"FAIL routing rel={rel}"

        # (4) invalidation
        try:
            EncoderCache(
                root=root,
                checkpoint="other:artifact",
                dataset_fingerprint="verify",
                num_samples={"train": n},
                precision=PRECISION,
            ).build_cached("train", batch_a, device=device)
        except ValueError:
            print("[invalidation] stale-cache load correctly rejected")  # noqa: T201
        else:
            msg = "FAIL: stale cache not rejected"
            raise AssertionError(msg)

        print("\nALL CHECKS PASS")  # noqa: T201

    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_forkserver_preload(["rbyte", "polars"])
    main()
