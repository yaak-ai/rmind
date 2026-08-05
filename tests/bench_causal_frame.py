"""Benchmark: dense-mask SDPA vs block-sparse FlexAttention for the causal trunk.

Not a pytest module (pytest only collects `test_*.py`) -- run it directly on a
machine with a free GPU:

    nix develop /path/to/rmind-rqv --command bash -c \\
      'PYTHONPATH=$PWD/src .venv/bin/python tests/bench_causal_frame.py trunk'

Sections:

* `attention` -- one attention layer, fwd+bwd. Isolates the kernel: this is the
  term that grows quadratically with context length under a dense mask.
* `trunk` -- the whole `CausalFrameTransformer` in `.train()`, i.e. WITH gradient
  checkpointing, which is how it actually trains. This is the number the recipe is
  sized from, because the MLP (linear in tokens) dilutes the attention saving.
* `vit` -- the frozen image encoder's forward, for context: it is linear in
  `batch * num_frames` and is unaffected by any of this, so it sets the floor on
  what a longer context can cost.

Everything runs in bf16 (`trainer.precision: bf16-mixed`); the correctness work in
`tests/test_causal_frame.py` is fp32 on purpose.

`(num_frames, window)` pairs are deliberately off-diagonal as well as on it:
`num_frames == window` is plain block-causal and still grows as `F^2`, while
`num_frames > window` is the regime where block-sparsity makes the cost LINEAR in
context length. Results as of 2026-08-05 are in §11 of docs/decoder_only_kv_cache.md.

Output goes through `emit` rather than `print` because the repo's ruff config
auto-removes `print` calls (`select = ["ALL"]`, `fix = true`).
"""

import argparse
import gc
import sys
import time
from collections.abc import Callable, Iterable

import torch
from torch import Tensor
from torch.nn import functional as F

from rmind.components.transformer.causal_frame import (
    AttentionImpl,
    CausalFrameTransformer,
    apply_rope,
    flex_frame_attention,
    frame_block_causal_block_mask,
    frame_block_causal_mask,
    frame_rope_cos_sin,
)

TOKENS_PER_FRAME = 257

# (num_frames, window). The first row is today's 6-frame arm: the denominator for
# every ratio in the report.
GEOMETRIES: tuple[tuple[int, int], ...] = (
    (6, 6),
    (16, 6),
    (16, 16),
    (32, 8),
    (32, 16),
    (32, 32),
    (64, 16),
    (64, 64),
)
WIDTHS: tuple[tuple[int, int, int], ...] = (  # dim_model, num_heads, num_layers
    (512, 8, 8),
    (768, 12, 12),
)


def emit(line: str) -> None:
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def timed(
    fn: Callable[[], object], *, iters: int = 10, warmup: int = 3
) -> tuple[float, float]:
    """`(ms per call, peak MiB)`."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters, torch.cuda.max_memory_allocated() / 2**20


def _guard(fn: Callable[[], tuple[float, float]]) -> str:
    """Run a case, reporting OOM as a result rather than losing the sweep."""
    try:
        ms, peak = fn()
    except torch.OutOfMemoryError:
        return f"{'OOM':>9} {'-':>9}"
    except Exception as e:  # noqa: BLE001
        return f"ERR {type(e).__name__}: {str(e)[:60]}"
    finally:
        gc.collect()
        torch.cuda.empty_cache()
    return f"{ms:>9.2f} {peak:>9.0f}"


def bench_attention(batches: Iterable[int], dtype: torch.dtype) -> None:
    emit(
        f"{'F':>4} {'win':>4} {'d':>5} {'b':>3} {'seq':>6} {'impl':>5} "
        f"{'ms':>9} {'peakMiB':>9}"
    )
    for dim, heads, _ in WIDTHS:
        head_dim = dim // heads
        for num_frames, window in GEOMETRIES:
            seq = num_frames * TOKENS_PER_FRAME
            for batch in batches:
                for impl in ("sdpa", "flex"):

                    def run(  # noqa: ANN202, PLR0913
                        *,
                        impl: AttentionImpl = impl,  # ty:ignore[invalid-parameter-default]
                        batch: int = batch,
                        num_frames: int = num_frames,
                        window: int = window,
                        seq: int = seq,
                        heads: int = heads,
                        head_dim: int = head_dim,
                    ):
                        g = torch.Generator(device="cuda").manual_seed(0)
                        qkv = [
                            torch.randn(
                                batch,
                                heads,
                                seq,
                                head_dim,
                                generator=g,
                                device="cuda",
                                dtype=dtype,
                            ).requires_grad_()
                            for _ in range(3)
                        ]
                        grad = torch.randn(qkv[0].shape, device="cuda", dtype=dtype)
                        cos, sin = frame_rope_cos_sin(
                            torch.arange(seq, device="cuda") // TOKENS_PER_FRAME,
                            head_dim=head_dim,
                        )
                        cos, sin = cos.to(dtype), sin.to(dtype)
                        mask = (
                            frame_block_causal_block_mask(
                                num_frames,
                                TOKENS_PER_FRAME,
                                window=window,
                                device=torch.device("cuda"),
                            )
                            if impl == "flex"
                            else frame_block_causal_mask(
                                num_frames,
                                TOKENS_PER_FRAME,
                                window=window,
                                device=torch.device("cuda"),
                            )
                        )

                        def step() -> None:
                            q, k, v = qkv
                            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
                            out = (
                                flex_frame_attention(q, k, v, mask)  # ty:ignore[invalid-argument-type]
                                if impl == "flex"
                                else F.scaled_dot_product_attention(
                                    q,
                                    k,
                                    v,
                                    attn_mask=~mask,  # ty:ignore[unsupported-operator]
                                )
                            )
                            out.backward(grad)

                        return timed(step)

                    emit(
                        f"{num_frames:>4} {window:>4} {dim:>5} {batch:>3} {seq:>6} "
                        f"{impl:>5} {_guard(run)}"
                    )


def bench_trunk(batches: Iterable[int], dtype: torch.dtype) -> None:
    emit(
        f"{'F':>4} {'win':>4} {'d':>5} {'L':>3} {'b':>3} {'seq':>6} {'impl':>5} "
        f"{'ms':>9} {'peakMiB':>9}"
    )
    for dim, heads, layers in WIDTHS:
        for num_frames, window in GEOMETRIES:
            seq = num_frames * TOKENS_PER_FRAME
            for batch in batches:
                for impl in ("sdpa", "flex"):

                    def run(  # noqa: ANN202, PLR0913
                        *,
                        impl: AttentionImpl = impl,  # ty:ignore[invalid-parameter-default]
                        batch: int = batch,
                        num_frames: int = num_frames,
                        window: int = window,
                        seq: int = seq,
                        dim: int = dim,
                        heads: int = heads,
                        layers: int = layers,
                    ):
                        torch.manual_seed(0)
                        trunk = CausalFrameTransformer(
                            dim_model=dim,
                            num_layers=layers,
                            num_heads=heads,
                            tokens_per_frame=TOKENS_PER_FRAME,
                            window=window,
                            attn_dropout=0.0,
                            attention_impl=impl,
                        ).cuda()
                        trunk.train()  # gradient checkpointing, as in training
                        x = torch.randn(batch, seq, dim, device="cuda", dtype=dtype)
                        grad = torch.randn(x.shape, device="cuda", dtype=dtype)

                        def step() -> None:
                            trunk.zero_grad(set_to_none=True)
                            inp = x.detach().clone().requires_grad_()
                            with torch.autocast("cuda", dtype=dtype):
                                out = trunk(inp, num_frames=num_frames)
                            out.backward(grad)

                        return timed(step, iters=5, warmup=2)

                    emit(
                        f"{num_frames:>4} {window:>4} {dim:>5} {layers:>3} {batch:>3} "
                        f"{seq:>6} {impl:>5} {_guard(run)}"
                    )


def bench_vit(batches: Iterable[int], dtype: torch.dtype) -> None:
    """Frozen encoder forward, for the per-step context the trunk numbers live in."""
    import timm  # noqa: PLC0415

    model = (
        timm
        .create_model(
            "vit_small_patch14_dinov2.lvd142m", pretrained=False, img_size=224
        )
        .eval()
        .cuda()
        .to(dtype)
    )
    emit(f"{'images':>7} {'ms':>9} {'peakMiB':>9}")
    for n in batches:
        img = torch.randn(n, 3, 224, 224, device="cuda", dtype=dtype)

        def step(img: Tensor = img) -> None:
            with torch.no_grad():
                model.forward_features(img)

        emit(f"{n:>7} {_guard(lambda step=step: timed(step, iters=5, warmup=2))}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("section", choices=["attention", "trunk", "vit"])
    parser.add_argument("--batches", default="4,16")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    args = parser.parse_args()
    if not torch.cuda.is_available():
        msg = "no CUDA device"
        raise SystemExit(msg)
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    batches = [int(b) for b in args.batches.split(",")]
    emit(
        f"{torch.__version__} {torch.cuda.get_device_name(0)} dtype={args.dtype} "
        f"free/total GiB {[round(x / 2**30, 1) for x in torch.cuda.mem_get_info()]}"
    )
    started = time.perf_counter()
    {"attention": bench_attention, "trunk": bench_trunk, "vit": bench_vit}[
        args.section
    ](batches, dtype)
    emit(f"# {args.section} done in {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
