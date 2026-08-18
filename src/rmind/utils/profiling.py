import os
import time
from collections.abc import Iterator
from contextlib import contextmanager

import torch.profiler
from structlog import get_logger

logger = get_logger(__name__)


@contextmanager
def maybe_profile(tag: str) -> Iterator[None]:
    """Wrap a block in `torch.profiler` iff the `TORCH_PROFILER` env var is set.

    Writes a chrome trace to `{TORCH_PROFILER_DIR}/torch_profiler_{tag}_{pid}_{ts}.json`
    (`TORCH_PROFILER_DIR` defaults to the cwd). A no-op (runs the block unprofiled) if
    `TORCH_PROFILER` is unset or profiler setup fails -- the wrapped block always runs
    exactly once either way.
    """
    if not os.getenv("TORCH_PROFILER"):
        yield
        return

    try:
        prof_ctx = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
    except Exception:  # noqa: BLE001 -- profiling must never break training
        logger.warning("torch profiler setup failed, running unprofiled", tag=tag)
        yield
        return

    trace_dir = os.getenv("TORCH_PROFILER_DIR", ".")
    fname = f"{trace_dir}/torch_profiler_{tag}_{os.getpid()}_{int(time.time())}.json"
    with prof_ctx as prof:
        yield
    try:
        prof.export_chrome_trace(fname)
    except Exception:  # noqa: BLE001 -- best-effort trace export
        logger.warning("failed to export torch profiler trace", tag=tag, fname=fname)
