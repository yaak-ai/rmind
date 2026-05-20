"""Benchmark: torch.compile(encoder) vs eager — wall-clock time per forward pass.

Run with:
    uv run pytest tests/test_compile_benchmark.py -s -v

By default runs with the best available backend (inductor → cudagraphs → aot_eager).
To compare all backends explicitly:
    uv run pytest tests/test_compile_benchmark.py -s -v --backends inductor,cudagraphs,aot_eager

Warms up JIT / compiled kernels for N_WARMUP steps, then times N_BENCH steps
and prints a summary table. The assertion only guards against severe regression
(compiled > 1.5x slower than eager); the printed speedup is the real output.
"""

import copy
import time
from typing import TYPE_CHECKING, Any

import pytest
import torch
from torch.nn import Module

from rmind.components.base import TensorTree
from rmind.components.containers import ModuleDict
from rmind.models.control_transformer import ControlTransformer, PredictionConfig

if TYPE_CHECKING:
    from rmind.components.episode import Episode

N_WARMUP = 3
N_BENCH = 10

COMPILE_BACKENDS = ("inductor", "cudagraphs", "aot_eager")


def _probe(backend: str) -> bool:
    """Return True if torch.compile with this backend works on this machine."""
    try:
        compiled = torch.compile(lambda x: x + 1, backend=backend)
        compiled(torch.zeros(1))
    except Exception:  # noqa: BLE001
        return False
    return True


def _best_compile_backend() -> str:
    """Return the best torch.compile backend available on this machine."""
    for candidate in COMPILE_BACKENDS:
        if _probe(candidate):
            return candidate
    return "eager"


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "backend" in metafunc.fixturenames:
        opt = metafunc.config.getoption("--backends", default=None)
        backends = (
            [b.strip() for b in opt.split(",")] if opt else [_best_compile_backend()]
        )
        metafunc.parametrize("backend", backends)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _time_encoder_steps(  # noqa: PLR0913
    episode_builder: Module,
    encoder: Any,
    batch: dict[str, Any],
    device: torch.device,
    *,
    n_warmup: int,
    n_bench: int,
) -> list[float]:
    """Time only the encoder forward pass (the compiled portion)."""
    with torch.inference_mode():
        episode: Episode = episode_builder(batch)
        src = episode.embeddings_flattened
        mask = episode.attention_mask

    for _ in range(n_warmup):
        with torch.inference_mode():
            _ = encoder(src=src, mask=mask)
        _sync(device)

    times = []
    for _ in range(n_bench):
        _sync(device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            _ = encoder(src=src, mask=mask)
        _sync(device)
        times.append(time.perf_counter() - t0)

    return times


@pytest.fixture(scope="module")
def objectives(policy_objective: Any, device: torch.device) -> ModuleDict:
    return ModuleDict({"policy": policy_objective}).to(device)


@pytest.fixture(scope="module")
def control_transformer_eager(
    episode_builder: Module, objectives: ModuleDict, encoder: Any, device: torch.device
) -> ControlTransformer:
    return ControlTransformer(
        episode_builder=episode_builder,
        encoder=encoder,
        objectives=objectives,
        prediction_config=PredictionConfig(),
    ).to(device)


def test_compile_vs_eager_encoder(  # noqa: PLR0913, PLR0917
    control_transformer_eager: ControlTransformer,
    episode_builder: Module,
    objectives: ModuleDict,
    encoder: Any,
    batch_dict: TensorTree,
    device: torch.device,
    backend: str,
) -> None:
    """Compare encoder forward time: eager vs torch.compile for each backend."""
    if device.type == "mps" and backend == "inductor":
        pytest.skip("inductor has no MPS kernel support")
    if not _probe(backend):
        pytest.skip(f"backend '{backend}' not available on this machine")

    compiled_encoder = copy.deepcopy(encoder)
    compiled_encoder.compile(backend=backend)
    compiled_ct = ControlTransformer(
        episode_builder=episode_builder,
        encoder=compiled_encoder,
        objectives=objectives,
        prediction_config=PredictionConfig(),
    ).to(device)

    eager_times = _time_encoder_steps(
        control_transformer_eager.episode_builder,
        control_transformer_eager.encoder,
        batch_dict,
        device,
        n_warmup=N_WARMUP,
        n_bench=N_BENCH,
    )
    compiled_times = _time_encoder_steps(
        compiled_ct.episode_builder,
        compiled_ct.encoder,
        batch_dict,
        device,
        n_warmup=N_WARMUP,
        n_bench=N_BENCH,
    )

    def stats(ts: list[float]) -> tuple[float, float]:
        mean = sum(ts) / len(ts)
        std = (sum((t - mean) ** 2 for t in ts) / len(ts)) ** 0.5
        return mean, std

    eager_mean, eager_std = stats(eager_times)
    compiled_mean, compiled_std = stats(compiled_times)
    speedup = eager_mean / compiled_mean

    print(  # noqa: T201
        f"\n{'':=<60}\n"
        f"  Encoder benchmark  (backend={backend!r})\n"
        f"{'':=<60}\n"
        f"  eager:    {eager_mean * 1e3:7.2f} ms  +/-{eager_std * 1e3:.2f} ms\n"
        f"  compiled: {compiled_mean * 1e3:7.2f} ms  +/-{compiled_std * 1e3:.2f} ms\n"
        f"  speedup:  {speedup:.2f}x\n"
        f"{'':=<60}"
    )

    assert compiled_mean <= eager_mean * 1.5, (
        f"Compiled encoder ({compiled_mean * 1e3:.1f} ms) is more than 1.5x slower "
        f"than eager ({eager_mean * 1e3:.1f} ms) on backend '{backend}'"
    )
