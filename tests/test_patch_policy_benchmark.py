"""Benchmark: `BlockCausalTransformer` trunk under each checkpointing policy.

The trunk is ~80% of `PatchPolicy`'s FLOPs, so isolating it at production shape
lets a checkpointing or `torch.compile` change be A/B'd in seconds instead of a
multi-hour NAS-backed training run.

Run with:
    uv run pytest tests/test_patch_policy_benchmark.py -s -v

Defaults to a batch a small GPU can hold. For the production number:
    RMIND_BENCH_BATCH=128 uv run pytest tests/test_patch_policy_benchmark.py -s -v

Times a forward+backward -- not a bare forward -- because checkpointing only
costs anything when there is a backward to recompute for. The assertion is a
loose regression guard; the printed table (wall-time AND peak memory per policy)
is the real output.
"""

import os
import time

import pytest
import torch
from torch import Tensor

from rmind.models.patch_policy import BlockCausalTransformer

# config/experiment/yaak/patch_policy/dinov3.yaml
DIM_MODEL = 512
NUM_LAYERS = 8
NUM_HEADS = 8
NUM_FRAMES = 6
TOKENS_PER_FRAME = 257  # num_patches (256) + 1 speed token
SEQ_LEN = NUM_FRAMES * TOKENS_PER_FRAME  # 1542

N_WARMUP = 3
N_BENCH = 10

# `True` = checkpoint every block (the historical default), `2` = every other,
# `False` = none
CHECKPOINT_POLICIES: tuple[bool | int, ...] = (True, 2, False)

_BYTES_PER_GB = 1024**3


def _batch_size() -> int:
    return int(os.getenv("RMIND_BENCH_BATCH", "16"))


def _make_trunk(
    *, checkpoint: bool | int, device: torch.device
) -> BlockCausalTransformer:
    torch.manual_seed(0)
    trunk = BlockCausalTransformer(
        dim_model=DIM_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        max_sequence_length=SEQ_LEN,
        checkpoint=checkpoint,
    )
    return trunk.to(device).train()


def _time_train_steps(
    trunk: BlockCausalTransformer, src: Tensor, *, n_warmup: int, n_bench: int
) -> tuple[list[float], float]:
    """Return per-step forward+backward times and peak allocated memory (GB)."""
    device = src.device

    def step() -> None:
        trunk.zero_grad(set_to_none=True)
        # same autocast context Lightning's bf16-mixed plugin establishes
        with torch.autocast(device.type, dtype=torch.bfloat16):
            out = trunk(src, num_frames=NUM_FRAMES)
        out.float().pow(2).mean().backward()

    for _ in range(n_warmup):
        step()
    torch.cuda.synchronize(device)

    torch.cuda.reset_peak_memory_stats(device)
    times: list[float] = []
    for _ in range(n_bench):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        step()
        torch.cuda.synchronize(device)
        times.append(time.perf_counter() - t0)

    return times, torch.cuda.max_memory_allocated(device) / _BYTES_PER_GB


def _stats(ts: list[float]) -> tuple[float, float]:
    mean = sum(ts) / len(ts)
    std = (sum((t - mean) ** 2 for t in ts) / len(ts)) ** 0.5
    return mean, std


@pytest.mark.skipif(not torch.cuda.is_available(), reason="benchmark requires a GPU")
def test_trunk_checkpointing_benchmark() -> None:
    device = torch.device("cuda")
    batch_size = _batch_size()

    # fp32 to match production: the encoder's input is a `torch.cat` of the bf16
    # `patch_projection` output with the fp32 `speed_embedding` output, and
    # `cat` type-promotes, so the residual stream really is fp32 here
    src = torch.randn(batch_size, SEQ_LEN, DIM_MODEL, device=device)

    results: dict[bool | int, tuple[float, float, float]] = {}
    for policy in CHECKPOINT_POLICIES:
        trunk = _make_trunk(checkpoint=policy, device=device)
        times, peak_gb = _time_train_steps(
            trunk, src, n_warmup=N_WARMUP, n_bench=N_BENCH
        )
        mean, std = _stats(times)
        results[policy] = (mean, std, peak_gb)
        del trunk
        torch.cuda.empty_cache()

    baseline_mean = results[True][0]
    rows = "\n".join(
        f"  checkpoint={policy!s:<5}  {mean * 1e3:7.1f} ms  +/-{std * 1e3:4.1f}  "
        f"{peak_gb:6.2f} GB   {baseline_mean / mean:.2f}x"
        for policy, (mean, std, peak_gb) in results.items()
    )
    print(  # noqa: T201
        f"\n{'':=<64}\n"
        f"  Trunk fwd+bwd  (batch={batch_size}, seq={SEQ_LEN}, "
        f"dim={DIM_MODEL}, layers={NUM_LAYERS})\n"
        f"{'':=<64}\n"
        f"{rows}\n"
        f"  speedup is vs checkpoint=True (the historical default)\n"
        f"{'':=<64}"
    )

    # not checkpointing must not be SLOWER -- it strictly removes a recompute
    assert results[False][0] <= baseline_mean, (
        f"checkpoint=False ({results[False][0] * 1e3:.1f} ms) is slower than "
        f"checkpoint=True ({baseline_mean * 1e3:.1f} ms), which should be impossible"
    )
