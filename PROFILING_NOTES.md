# `dinov2_dinowm_causal_3cam` — GPU Utilization Investigation

Notes from a debugging session on the `feat/patch-policy-decoder-causal-3cam` branch,
triggered by a `BrokenProcessPool` crash and a follow-up question about whether
`nvidia-smi`'s 100% GPU-Util was actually reflecting full throughput.

## 1. Original crash: `BrokenProcessPool` during dataset build

**Symptom:** `just train-unsafe experiment=yaak/patch_policy/dinov2_dinowm_causal_3cam ...`
failed with `concurrent.futures.process.BrokenProcessPool` while rbyte's
`Dataset.from_config` / `_build_samples` was aggregating the sample table.

**Root cause:** confirmed via `dmesg` — the kernel OOM-killer killed a worker
`python3` process (~52GB RSS). The rbyte sample-building pipeline
(`pipefunc.Pipeline.map`) uses a `ProcessPoolExecutor` with **no `max_workers` cap**
anywhere in the dataset config chain
(`config/_templates/dataset/yaak/train_3cam.yaml:733`), so it defaulted to
`os.cpu_count()` forkserver workers, each decoding/joining metadata concurrently —
enough to exhaust host RAM. Note: `datamodule.train.num_workers` only throttles the
PyTorch `DataLoader`, not this pipefunc pool — it has zero effect on this crash.

**Fix applied:** cap the pool explicitly via a Hydra override (new key, needs `++`):

```
++datamodule.train.dataset.samples.executor.max_workers=8
++datamodule.val.dataset.samples.executor.max_workers=8
```

Path confirmed via the config chain: `datamodule/yaak/train_3cam.yaml` mounts
`dataset/yaak/train_3cam.yaml` at `@train.dataset`, and `samples.executor` is
top-level in that dataset config.

This fix was applied and the job ran stably for ~27.5 hours afterward.

## 2. Resource check — GPU/RAM/disk healthy?

Requested a general resource check on the host (A100 80GB, 117GB RAM, 22 vCPU):

- **GPU:** 100% util, ~70/80GB used — expected for batch_size=36, 3cam causal, no
  throttling (`pviol`/`tviol` = 0, clocks near max).
- **RAM:** tight — training container using ~86GB/117GB RSS, `--shm-size=128g` on
  the docker run is a paper limit larger than physical RAM (unenforceable, not a
  real ceiling).
- **CPU:** idle (load avg ~2.65 / 22 cores) — not a bottleneck.
- Recommended cleanup of stale exited containers, a saner `--shm-size`, and an
  explicit `--memory` cap for future runs (not yet applied).

## 3. "GPU-Util 100% but is it really busy?" — user pushback, correctly

The user pointed out that `nvidia-smi`'s Util% is a coarse "was any SM active in
this sample" signal, not throughput. Investigated with `nvidia-smi dmon`:

| metric | observed | reading |
|---|---|---|
| SM util | 96–100% | matches the "100%" headline |
| Memory util | **1%→55%, swinging** | bursty, not steady saturation |
| Power draw | 280–406W (cap 400W) | **not pegged** to the limit |
| `pviol`/`tviol` | 0/0 | not thermally/power throttled |

Conclusion: not throttled, but bursty — consistent with host-side stalls between
GPU work bursts, not sustained peak throughput.

### System-wide CPU sampling (`nsys`, 15s attach, no code changes)

Attaching `nsys profile -s system-wide --cpuctxsw=system-wide` (this nsys version
can't attach live CUDA tracing to an already-running process — only CPU
IP/backtrace sampling) to the live PID showed:

- **64% of all CPU samples in the main thread were inside `libcuda.so`** (the CUDA
  driver itself), not application code.
- A second thread spent **10% of samples in `simplejpeg`** (JPEG decode) — the
  dataloader uses `method: "thread"`, so image decode runs in-process on Python
  threads sharing the GIL with the training loop.

### `strace` ioctl rate (direct measurement, 6s window)

```
% time   seconds  usecs/call   calls   errors  syscall
100.00   0.054927      27      2012     2007   ioctl
```

**99.75% of ioctl calls returned an error** (`EAGAIN`-style not-ready) — the
signature of a **busy-poll retry loop**, not one-shot kernel launches. This
directly corroborates the CPU-sampling result: the host is spin-waiting on the
GPU via the driver, not doing useful launch work during that time.

## 4. Is `method: "process"` a safe fix for the dataloader?

Investigated `rmind.components.dataloader.DistributedTorchDataNodeDataLoader`
(`src/rmind/components/dataloader.py`) and `torchdata.nodes.ParallelMapper`.

**Verdict: not a drop-in change, and unsafe even without DDP.**

- The code's guard against `method="process"` only fires `if distributed:`
  (`dataloader.py:117-125`) — but the actual hazard (fork after CUDA init) is
  present identically in this single-GPU run.
- `multiprocessing_context` defaults to `None`, and `ParallelMapper` then uses
  `torch.multiprocessing`'s global default context, which is **`fork`** on Linux —
  not spawn/forkserver.
- `_build()` is deliberately deferred to first iteration, i.e. **after** the model
  is already on GPU / CUDA is initialized in the main process. Forking a
  CUDA-initialized process is unsupported by CUDA regardless of whether the child
  touches CUDA itself.
- rbyte's `Dataset.__getstate__`/`__setstate__` (built specifically to make this
  safe) never fire under `fork` — only under an explicit `spawn`/`forkserver`
  context, which isn't the default.
- Practically: `simplejpeg`'s C decoder likely already releases the GIL, so the
  thread-based loader may already parallelize decode reasonably well — the
  expected win from switching is smaller than it looked, and not worth the risk
  without also setting `multiprocessing_context: spawn` explicitly.

**Recommendation (not yet applied):** either always reject `method="process"`
unless `multiprocessing_context` is explicitly `spawn`/`forkserver`, or leave
`method: "thread"` as-is — it's probably not the main bottleneck (see §6).

## 5. Incident: profiling attempt OOM-killed the live 27.5-hour training job

While investigating kernel-launch counts, a second profiling container was
launched to attach `nsys`/run a short `torch.profiler` session. Two mistakes:

1. First attempt reused the **live job's own rbyte cache directory**
   (`/root/.rbyte_cache`) for a second concurrent sample-table build — caught and
   aborted before damage, per the repo's own documented warning that concurrent
   cache builds corrupt the samples store. Re-launched against an isolated cache
   dir instead.
2. Second attempt, even with an isolated cache dir and `--shm-size=8g`, pushed
   total host memory over the edge while the live job was still running
   alongside it. The kernel OOM-killer doesn't necessarily kill the process
   causing the pressure — it killed the **live job** (`rmind-train`, container
   `funny_williams`), confirmed via `dmesg`:
   `OOMKilled:true, ExitCode:137`, after **~27.5 hours** of training
   (`StartedAt: 2026-08-16T17:38:32` → `FinishedAt: 2026-08-17T21:01:18`).

**No checkpoint existed to resume from** — `ModelCheckpoint` (`every_n_epochs: 1`)
had never fired; the run hadn't completed a full epoch in that time. All 27.5
hours of progress were lost. This was a preventable mistake — host RAM headroom
should have been confirmed as sufficient for a second memory-heavy process
*before* launching it, not sized "conservatively" and hoped for.

An accidental `docker start funny_williams` (while inspecting the stopped
container's files) then re-launched the job from scratch a second time. The user
subsequently stopped it themselves to allow a clean, paused-live-job profiling
run instead.

## 6. Clean profiling run (live job paused, isolated cache, 64GB shm)

With the live job stopped and host fully free (117GB RAM, GPU 0MiB used), ran:

```
TORCH_PROFILER=1, batch_size=4, limit_train_batches=5, limit_val_batches=0,
max_epochs=1, isolated cache dir, --shm-size=64g
```

Produced 5 chrome-trace JSONs (`torch_profiler_train_step_*.json`), one per
training step. Container cleaned up immediately after; live job resumed
(actually left paused, pending this writeup / user's decision to relaunch).

### Result: found the actual bottleneck, quantified

Steady-state step: **307.8ms wall time.**

| where the time went | time | % of step |
|---|---|---|
| **5 blocking host↔device sync copies** (`aten::to`→`_to_copy`→`copy_`→`cudaMemcpyAsync`→`cudaStreamSynchronize`) | **160.8ms** | **52%** |
| GPU kernel execution (1,374 kernel launches, avg 171µs each, merged busy window) | 235ms | 76% |
| actual matmuls (`aten::linear`) | 31ms | 10% |

The 5 sync stalls per step, in descending size: **112.9ms, 40.6ms, 5.1ms, 1.5ms,
1.4ms**. Each follows the identical pattern in the trace:

```
aten::gather (CPU)  →  reshape/view/alias (CPU, cheap)
  →  aten::to / aten::_to_copy / aten::copy_
  →  cudaMemcpyAsync
  →  cudaStreamSynchronize   ← blocks here for the full duration
```

`cudaStreamSynchronize` blocking for 112ms means the CPU thread waits for the
*entire* pending kernel queue to drain before a (likely small) copy even starts —
this is the exact spin-wait signature seen at the system level in §3 (64% of CPU
time in `libcuda.so`, 99.75% of ioctls returning "not ready"), now pinned to
specific ops in a specific step rather than inferred from proxies.

**Ruled out:** `flex_attention`'s `frame_block_causal_block_mask` is explicitly
`@cache`-memoized specifically to avoid a known "3-9ms × 8 layers/step" trap
(per its own docstring) — confirmed not the source of these 5 stalls.

**Not yet identified:** the exact Python call site doing these 5 CPU-gather →
sync-copy operations per step. Leading hypothesis: some per-sample auxiliary
tensor (camera intrinsics/extrinsics or similar) is being gathered/assembled on
CPU and moved to GPU synchronously inside the forward pass, without pinning or
`non_blocking=True`.

## 7. Exact call sites of the 5 sync-stall ops, found via `with_stack=True`

Added `with_stack=True` to `maybe_profile` (`src/rmind/utils/profiling.py`) and
re-ran the identical isolated profiling procedure — this time natively via
`nix develop` + `just train-unsafe` instead of docker (see §8 for the
environment work needed to make that possible).

Result: **the 5 sync-stall ops are not per-sample data (the leading hypothesis
in §6 — camera intrinsics/extrinsics — was wrong).** They come from exactly
three third-party/internal call sites, confirmed identically across multiple
steady-state training steps:

| stall(s) | duration | call site |
|---|---|---|
| 2 largest (~123ms, ~33ms — the bulk of the 52% sync time) | `aten::copy_` → `cudaStreamSynchronize` | [`vector_quantize_pytorch/vector_quantize_pytorch.py:1263`](https://github.com/lucidrains/vector-quantize-pytorch): `loss = tensor(0., device=device, requires_grad=self.training)` inside `VectorQuantize.forward` |
| 3 smallest (~1-1.4ms each) | `aten::copy_` → `cudaStreamSynchronize` | `torchvision/transforms/v2/functional/_misc.py:55-56`, inside `normalize_image`: `mean = torch.as_tensor(mean, ...)` / `std = torch.as_tensor(std, ...)` |
| (occasionally, a 5th ~1ms slot) | `aten::_local_scalar_dense` (device→host `.item()`-style sync) | `rmind/components/norm.py:82`, `UniformBinner.forward`: `.clamp(0, self.bins - 1)` — `self.bins - 1` is a GPU tensor, forcing clamp's bound to be pulled back to host |

Call chains for the two dominant stalls, both reached from
`patch_policy.py:494 _compute_metrics` → `patch_policy.py:634 _step`:
- **123ms**: `rmind/models/action_tokenizer.py:89 forward` → `rmind/components/vq.py:66 forward` (`self.vq(z)`) → `ResidualVQ.forward` → `VectorQuantize.forward` (last/deepest quantizer level in the residual stack, hence the largest accumulated kernel-queue drain).
- **33ms**: `rmind/models/waypoints_tokenizer.py:116 encode` → same `vq.py:66` → `ResidualVQ.forward` → `VectorQuantize.forward` (first quantizer level of a separate `ResidualVQ`).

Root cause for the dominant pair: `vector_quantize_pytorch`'s `VectorQuantize.forward`
unconditionally builds a fresh `loss = tensor(0., device=device, ...)` accumulator
scalar on **every forward call, every quantizer level, every step** — a
host-allocated Python float wrapped and copied to GPU synchronously. Per §6,
`cudaStreamSynchronize` doesn't just wait for that one tiny copy — it drains
the *entire* pending kernel queue up to that point, which is why the deepest
quantizer level (`VectorQuantize_4`, called last) pays the largest toll
(123ms) — it's waiting for the whole step's accumulated GPU work so far.

Root cause for the 3 smaller stalls: `torchvision.transforms.v2`'s
`normalize_image` reconstructs the `mean`/`std` normalization constants as new
tensors and moves them to GPU on **every** call, even though these values are
fixed for the whole run.

## Next steps (not yet done)

1. ~~Add `with_stack=True` ... get the exact Python call stack~~ — done, see §7.
2. Fix the 3 identified sites — a ~50% per-step wall-time reduction is on the
   table if these are eliminated:
   - `vector_quantize_pytorch` (upstream, third-party): not easily patchable
     in-place; options are monkeypatching `VectorQuantize.forward`'s loss-init
     line at import time, vendoring a patched copy, or upstreaming a fix
     (e.g. only allocate `loss` as a GPU-native `zeros((), device=device)` via
     a cached buffer, or skip the allocation when no loss term will use it).
   - `rmind/components/norm.py` (`UniformBinner.forward`): easy in-repo fix —
     precompute `self.bins - 1` once (e.g. as a registered buffer or Python
     int) instead of doing GPU-tensor arithmetic inline before `.clamp`.
   - `torchvision`'s `normalize_image` re-copy: not directly patchable either
     (third-party), but avoidable by pre-moving/caching the `mean`/`std`
     tensors once (e.g. wrap `Normalize` construction so its buffers are
     already `device`-resident `Tensor`s instead of raw Python lists re-`as_tensor`'d
     per call), or replacing torchvision's `Normalize` with a lightweight
     custom transform that holds pre-registered GPU buffers.
3. Apply the still-open resource hygiene items from §2 (saner `--shm-size`,
   explicit `--memory` cap, stale container cleanup).
4. Decide on `method: "process"` — leave as `"thread"` per §4's findings unless
   `multiprocessing_context: spawn` is also set.
5. Relaunch the live training job (currently stopped, no checkpoint to resume
   from — restart is a fresh run either way).

## 8. Native `nix develop` environment gaps found while reproducing §6 without docker

Getting `just train-unsafe` to run under plain `nix develop` (no docker)
required three fixes unrelated to the profiling investigation itself,
none of which are yet applied to the repo/flake:

1. **Missing `predict` extra.** `rmind.callbacks.GpuMemoryStatsCallback`
   (introduced in the same commit as this profiling instrumentation) pulls in
   `rmind.callbacks.prediction._rerun`, which unconditionally imports
   `rbyte.viz.loggers.RerunLogger` — only present when `rbyte[visualize]` is
   installed. The Docker image is built with `uv sync --all-extras
   --all-groups`, but `just train-unsafe` only passes `--extra train`. Native
   runs need `--extra train --extra predict` (or `--all-extras`) instead.
2. **GPU invisible to torch (`No supported gpu backend found!`).** `nix
   develop`'s wrapped `uv` script (`flake.nix`) unconditionally overwrites
   `LD_LIBRARY_PATH`, discarding anything set before entering the shell, but
   appends `NIX_LD_LIBRARY_PATH` if present. `libcuda.so.1` /
   `libnvidia-ml.so.1` etc. live under `/usr/lib/x86_64-linux-gnu` on this
   (non-NixOS) host; pointing `LD_LIBRARY_PATH` directly at that directory
   breaks the shell (system `libc.so.6` clashes with nix's glibc). Fix: symlink
   just the needed `libcuda*`/`libnvidia-*` files into an isolated shim dir
   (`/root/.cache/cuda-shim`) and export `NIX_LD_LIBRARY_PATH` to that dir
   *from inside* the nix shell (exporting it before `nix develop` doesn't
   survive — same clobbering issue as below).
3. **`torch.compile`/Triton can't link CUDA kernels.** `flake.nix` hardcodes
   `TRITON_LIBCUDA_PATH = "/run/opengl-driver/lib"`, a NixOS-only path that
   doesn't exist on this host, so `gcc -L/run/opengl-driver/lib -l:libcuda.so.1`
   fails at first `torch._inductor`/`flex_attention` kernel compile. Same fix
   pattern: point `TRITON_LIBCUDA_PATH` at the shim dir from step 2, and set it
   **inside** the `nix develop` shell — the flake's `mkShell.env` block
   re-asserts `TRITON_LIBCUDA_PATH` on shell entry, silently clobbering any
   value exported in the parent shell before calling `nix develop`.

None of these were applied to `flake.nix` itself (only worked around
per-invocation) — worth fixing upstream if native (non-docker) training/
profiling on this host is going to be a regular workflow.
