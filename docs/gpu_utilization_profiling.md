# GPU utilization / batch-size-OOM investigation (`patch_policy/dinov3_dropout`)

Notes from investigating run [`g4otplbp`](https://wandb.ai/yaak/alex-tmp/runs/g4otplbp)
(A100-SXM4-80GB, `datamodule.train.batch_size=128`, image `d0093d71f5027e354f33946411482006c1f46783`).
Question: GPU isn't fully utilized, and raising `batch_size` OOMs — where's the actual
bottleneck, and what's safe to try next?

## WandB evidence

Pulled via `wandb.Api().run("yaak/alex-tmp/g4otplbp").history(stream="system", ...)`
(full-resolution 15s system-metrics stream, ~6700 samples over ~28h):

- **GPU compute util** (`system.gpu.0.gpu`): post-startup (excludes a 975s cold-start
  stall) mean **84.5%**, median 100%, but **10.4%** of samples are exactly 0%. ~400
  discrete idle events, almost all 15–135s. This residual idle time — not memory — is
  the main "GPU busy%" gap.
- **GPU memory allocated%** (`system.gpu.0.memoryAllocated`, nvidia-smi "used" as % of
  80GB): clearly **bimodal** — ~78% of samples at ~57% (~45GB), ~22% at ~14% (~11GB).
  Matches train-step (forward+backward+optimizer state) vs. eval-step (forward only, no
  backward graph) memory — expected, not a bug.
- This metric is a **15s-cadence nvidia-smi snapshot**. It cannot see a memory spike
  that lasts a fraction of one training step, so "57% avg → looks like 35GB of
  headroom" is **not trustworthy evidence that batch_size can grow safely**. This is
  the crux of the OOM mystery: there's currently no visibility into real per-step peak
  memory, only a coarse time-average that misses transient spikes (e.g. checkpoint
  recompute during backward, allocator block churn on a new tensor shape).

## What's already applied (don't re-suggest these)

Traced through the code before concluding anything — several of the "obvious" batch-size
levers turn out to already be in place:

- **Activation/gradient checkpointing** is already on for the trainable trunk during
  training: `run_layer_stack` (`src/rmind/components/transformer/utils.py:8-15`) wraps
  each of the 8 `TransformerBlock`s in `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)` whenever `training=True`.
- **bf16-mixed precision** is already the default for both train and val
  (`config/trainer/default.yaml:8`) — PL applies the same autocast context to both, so
  there's no "val runs fp32" discrepancy to fix.
- The frozen `image_encoder`/`goal_encoder`/`tokenizer` already run under
  `torch.no_grad()` (`src/rmind/models/patch_policy.py:320-322`) — no autograd graph for
  the frozen dinov3 backbone.
- `goal_dropout`/`speed_dropout` substitute a learned "null" embedding per-batch-element
  (`patch_policy.py:291-309`); they do **not** change any tensor shape. Sequence length
  is always exactly `episode_length(6) × (num_patches(256)+1) = 1542` tokens — batches
  are not variable-size, so this dropout is not a source of variable/spiky memory.
- The attention op in `TransformerBlock.forward` (`patch_policy.py:74-81`) calls
  `nn.MultiheadAttention(..., need_weights=False)`. Traced into
  `torch/nn/functional.py:multi_head_attention_forward` — `need_weights=False` takes the
  branch that calls `scaled_dot_product_attention` directly, i.e. it **already** uses
  the fused/memory-efficient SDPA kernel and never materializes the full 1542×1542
  attention-score matrix. (The separate C++ "fast path" inside `nn.MultiheadAttention` is
  skipped because `training=True` and autocast is on, but that only changes which code
  path dispatches to SDPA, not whether SDPA is used.) Swapping to a hand-rolled
  `F.scaled_dot_product_attention` call (the pattern already used elsewhere in
  `src/rmind/components/transformer/attention.py`) would not materially change memory
  here.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is already exported globally in
  `justfile:10` — allocator fragmentation mitigation is already active.
- `torch.backends.cudnn.benchmark=True` already set (`config/trainer/default.yaml:5`).

## What's genuinely not yet done

1. **No per-step/peak memory instrumentation exists anywhere** — the only visibility is
   the coarse 15s wandb system metric, which is why the OOM cause is currently a
   mystery. → Added `GpuMemoryStatsCallback`
   (`src/rmind/callbacks/memory_stats.py`, wired in
   `config/trainer/callbacks/patch_policy.yaml`): resets
   `torch.cuda.reset_peak_memory_stats()` at the start of every train/val batch and logs
   `max_memory_allocated`/`max_memory_reserved` (GB) every `trainer.log_every_n_steps`
   steps to wandb as `system/train/max_memory_allocated_gb` etc. This makes the
   invisible per-step peak a real time series next to the existing 15s nvidia-smi one.
1. No use of `torch.profiler`/memory-timeline in this repo to see which op actually
   drives peak allocation, or whether checkpoint-recompute (which transiently redoes a
   layer's forward during backward) pushes the real peak well above the steady-state
   57%. There's also a `PredictMetricsCallback`
   (`config/trainer/callbacks/patch_policy.yaml`) producing the `predict/...` metrics
   seen in the wandb summary — worth confirming its memory footprint isn't a separate,
   larger spike than plain train/val forward passes.
1. `trainer.accumulate_grad_batches` is never set anywhere (PL default = 1) — a free,
   zero-code-change lever (`+trainer.accumulate_grad_batches=N` via Hydra CLI override,
   same `+` pattern already used for `max_concurrent`/`prefetch_factor` in
   `commands.sh`) to grow the *effective* batch size without raising per-step peak
   memory, by keeping the micro-batch at the currently-safe 128 (or smaller) and
   accumulating.
1. The residual ~10% GPU-idle time is a separate axis from OOM. Commit `d0093d7`
   already fixed the big serial-NFS-read bug (`ConcurrentPathTensorSource`,
   `src/rmind/io/path_tensor_source.py`, per-batch `ThreadPoolExecutor(max_workers=8)`).
   What's left is dataloader-level concurrency tuning on top of that fix:
   - `datamodule.train.num_workers=24` > the container's 22 CPUs. Since `method: thread` this is fine for I/O-wait, but each in-flight batch also spins up its own
     inner `ThreadPoolExecutor(max_workers=8)` for frame reads/decodes — with
     `max_concurrent=8` batches in flight, that's up to **64** concurrent JPEG-decode
     threads (real CPU work, not just I/O-wait) contending for 22 cores.
   - `max_concurrent=8` is a hard in-flight-batch cap
     (`torchdata.nodes.map.ParallelMapper`, enforced `≤ num_workers`); NAS latency
     variance with only 8 batches in flight can starve the prefetch queue when a couple
     of those 8 hit a slow NFS read together.
   - `prefetch_factor=1` → total lookahead queue depth = `num_workers × prefetch_factor = 24` batches (rbyte's default is `prefetch_factor=2` → 48). Lower than default,
     traded for host RAM.

## Recommended next steps, in order

1. **Ship the memory instrumentation** (done, see above) and let the next run report
   real per-step peak memory alongside the existing nvidia-smi snapshot.
1. Before touching the full multi-day job, run a short local profiler session
   (`just train-debug`, or `+trainer.limit_train_batches=`) wrapping a handful of steps
   in `torch.profiler.profile(profile_memory=True, record_shapes=True)` at (a)
   `batch_size=128` (known-good) and (b) the batch size you want to try. This gives an
   exact "which op/module owns the peak" answer instead of trial-and-error OOM crashes.
1. Try **gradient accumulation** (`+trainer.accumulate_grad_batches=N`) to reach a
   larger *effective* batch size without raising per-step peak memory — zero code
   change, purely a launch-command override.
1. Run one short dataloader-concurrency trial at a time (don't change `num_workers` and
   `prefetch_factor` together — the CPU-oversubscription and NAS-latency-starvation
   hypotheses point in opposite directions):
   - `num_workers` reduced to ≤22 (matching CPU count), `max_concurrent` at or below it.
   - `prefetch_factor` restored toward rbyte's default of 2.
     Compare the resulting `system.gpu.0.gpu==0` fraction with the same wandb-API query
     used above.
1. (Optional, later) `torch.compile` — not used for `patch_policy` today (only wired for
   `control_transformer` via `rmind.utils.functional.compiled`). Lower priority since
   checkpointing + SDPA are already active; would need validating against the existing
   `use_reentrant=False` checkpointing.

## Aside

`commands.sh` (repo root, untracked in git) has a plaintext W&B API key. It's live and
was used for this investigation — worth rotating and keeping out of the repo (`.env` +
gitignore) regardless of the above.

## Update: `torch.profiler` trace analysis (`/tmp/torch_trace_v1`, 303 chrome-trace

JSON dumps, `profile_memory=True`, `record_shapes=True`, A100-SXM4-80GB, single GPU)

This is a direct trace capture (not the coarse 15s wandb snapshot), so it resolves the
"what exactly happens during an idle stretch" question the WandB section above could
only speculate about.

**Aggregate, across all 303 windows:** GPU kernel-track span 170.1s, busy 130.7s →
**76.8% utilization**, 39.4s idle — same order of magnitude as the wandb-derived 84.5%,
confirming this capture is a representative slice of steady-state training (most
individual windows show a clean ~482ms/step at ~95.6% util; a handful are pathological
outliers spanning up to 4.8s for the *same* ~480–500ms of actual GPU work).

**Idle-time attribution** (gaps >20ms, dominant overlapping CPU/CUDA-runtime op by time
overlap):

| cause                                           | idle time | share |
| ----------------------------------------------- | --------- | ----- |
| `cudaHostAlloc` (allocating pinned host memory) | 6.48s     | 16.4% |
| unattributed (no op covers >50% of the gap)     | 21.18s    | 53.8% |
| device copy / sync                              | 0.06s     | 0.2%  |

Drilling into the "unattributed" gaps (dumping *every* event, any category/thread,
overlapping the largest ones) shows they aren't actually mysterious — each one starts
with the exact same signature as the `cudaHostAlloc` gaps: `aten::is_nonzero` →
`aten::item` → `aten::_local_scalar_dense` → `cudaStreamSynchronize` on a **bool**
scalar tensor, immediately followed by a long dead stretch (hundreds of ms to 3.5s)
where ordinary ops (`aten::detach`, `aten::pow`, `aten::to`, `aten::reshape`) show
absurdly inflated durations for what should be microsecond metadata ops. That's the
signature of the *calling thread itself* not getting scheduled — not GPU work, not
literal I/O wait, but the CPU thread sitting off-core.

**Correction (see below):** an earlier version of this note attributed the bool-scalar
`.item()` sync to Lightning's `loops/utilities.py:46` (`if loss is not None and not torch.isfinite(loss).all():`) / `training_epoch_loop.py:269` (`sigterm_tensor.item()`).
That's wrong — the `torch.profiler` block in `patch_policy.py:_step` only ever wrapped
`self._compute_metrics(batch)`; both of those Lightning-level checks run *after* `_step`
returns, outside the profiled scope, so they can't be what's showing up in these traces.
The real call site of the bool-tensor sync inside `_compute_metrics` is still
unconfirmed from op names alone — the fix applied below (a shared `maybe_profile`
helper with `with_stack`-ready call sites for `train_step`/`val_step`/`predict_step`)
sets up the next capture to pin this down properly instead of guessing from op names.

**Working hypothesis (still standing, now with much stronger evidence — see the v3
section below):** on a cache *miss*, `cudaHostAlloc` has to actually lock fresh physical
pages, which is normally fast but can take 100ms–800ms+ under host (CPU-side, not GPU)
memory pressure/fragmentation — and the wandb system metrics for this same run already
showed `system.memory_percent` sitting at 83–90% (see WandB evidence above). This reads
as a real, if partial, contributor to the residual idle time (more so than the
NAS-read/CPU-oversubscription theory from the original dataloader section, though both
point the same direction: too much host-side memory/thread pressure from the dataloader
stack: `num_workers=24`, prefetch queue depth 24, plus `ConcurrentPathTensorSource`'s
inner 8-way thread pool per in-flight batch) — see the v3 update below for the bigger
piece of this puzzle (validation/`predict_step`, not the dataloader).

**Next steps to confirm/fix, cheapest first:**

1. On the training host, during a run: `free -h` / `vmstat 1` — look for swap activity
   (`si`/`so`) or near-zero available RAM at the moments idle stalls happen in wandb.
1. Try lowering `datamodule.train.num_workers` (24 → e.g. 8–12) and/or `max_concurrent`
   on the next run — reduces the host-memory/thread pressure that's the suspected root
   cause, same experiment as item D above but now higher-priority given this evidence.
1. If (1) confirms swap/page pressure, reducing `prefetch_factor`/`num_workers` (fewer
   buffered batches resident in host RAM) should directly reduce `cudaHostAlloc` stall
   frequency — re-capture a short `torch.profiler` trace the same way to check.

## Update 2: `num_workers` trial (v2, wandb `nf39jicx`) — no clear improvement

Ran the exact experiment proposed above: `datamodule.train.num_workers=24→8`,
`max_concurrent=8→4`, everything else the same, no validation
(`+trainer.limit_val_batches=0`), captured to `/tmp/torch_trace_v2`.

Comparing v1 (train-debug dataset, `num_workers=24`) against v2 (`num_workers=8`),
**excluding each capture's first file** (both have a one-time warmup artifact — v1's
is the `cudaHostAlloc` cold-start burst above, v2's is `cudnn.benchmark` autotuning a
conv2d shape for the first time, ~2s, in `torch_profiler_3086614_1786636030.json`):

|                               | v1 (`num_workers=24`) | v2 (`num_workers=8`) |
| ----------------------------- | --------------------- | -------------------- |
| steady-state utilization      | 78.7%                 | 78.7% (identical)    |
| avg idle time/file            | ~0.117s               | ~0.124s (same)       |
| `cudaHostAlloc` share of idle | 9.6%                  | 3.0%                 |
| "unattributed" share of idle  | 57.5%                 | 16.3%                |

The `cudaHostAlloc`-specific share shrank, but **total idle time per step didn't
improve** — steady-state utilization is identical. `num_workers` was not the dominant
lever after all; something else (still present in both) accounts for most of the
residual idle time. (Caveat: v1 used the `train-debug` 3-drive dataset, not the full
production dataset, so this isn't a perfectly clean A/B — but the architecture-level
findings above aren't affected by that.)

## Update 3: memory instrumentation (`GpuMemoryStatsCallback`) finds the real culprit

Added a callback (`src/rmind/callbacks/memory_stats.py`) logging
`torch.cuda.max_memory_allocated()`/`max_memory_reserved()` per step, reset every
batch — real per-step peak memory, not a 15s nvidia-smi snapshot.

**v2 (`nf39jicx`, no validation, `batch_size=128`):** `system/train/max_memory_allocated_gb ≈ 9.4`, `max_memory_reserved_gb ≈ 10.4` — confirmed independently by nvidia-smi
(`system.gpu.0.memoryAllocatedBytes` maxed at 11.77GB for this run). Both agree: **the
true training-step memory footprint at batch_size=128 is only ~10-12GB**, not the ~45GB
(57% of 80GB) the original long production run showed at the same batch size.

**v3 (wandb `7qpdt0zc`, WITH validation this time — `+trainer.limit_val_batches=10`,
`+trainer.limit_train_batches=50`, otherwise identical to v2):**

| step (`trainer/global_step`)                         | `train/max_memory_reserved_gb` | `val/max_memory_reserved_gb` |
| ---------------------------------------------------- | ------------------------------ | ---------------------------- |
| 0 (pre-training sanity check, 2 val batches)         | —                              | **5.57**                     |
| 10 (first real train step, right after sanity check) | **44.80**                      | —                            |
| 50, 100 (steady state for the rest of the run)       | 45.45                          | 45.45                        |

The jump from ~10GB to ~45GB happens **between step 0 and step 10** — i.e. right after
Lightning's 2-batch pre-training sanity-check validation runs — and never comes back
down; every subsequent train and val step reports the same ~45GB reserved (`memory_reserved()`
is a process-wide cached-block high-water mark, shared between train and val, that only
grows). `max_memory_allocated` (live tensors actually in use right now) stays flat and
small throughout: ~9.4GB train, ~5.5GB val. So nothing in *steady-state* train or val
holds 45GB live — one thing, during that one sanity-check validation pass, transiently
needed ~40GB+, and the allocator has kept that much reserved ever since.

**Conclusion: the OOM risk when raising `batch_size` is not from training-step memory**
(~70GB of headroom) — **it's from whatever runs during validation**, almost certainly
`PredictMetricsCallback.on_validation_batch_end`'s extra `predict_step()` call (the only
code path materially different from a plain training step; already wrapped in
`@torch.no_grad()`, so it's not a backward-graph issue — something in
`PatchPolicy.forward()`/`_predict_chunk` itself is memory-heavy at batch_size=128).

## Update 4: v3's full run confirms it's also a *time* cost, not just memory

v3 ran to completion (10 epochs × 50 steps = 500 steps, `state: finished`). Splitting
its 596 trace files by step wall-time (a clean bimodal split, no ambiguity):

| bucket                  | n files | avg wall-time/step      | avg actual GPU work/step | utilization |
| ----------------------- | ------- | ----------------------- | ------------------------ | ----------- |
| fast (train steps)      | 431     | 536ms                   | 448ms                    | 83.5%       |
| slow (validation steps) | 165     | **1651ms** (~3x longer) | 465ms (same!)            | 28.2%       |

The "slow" bucket does essentially the **same amount of real GPU work** as a training
step (465ms vs 448ms busy) but takes over 3x as long in wall-clock — the extra
~1.1-1.2s per validation batch is pure stall, not compute. Those 165 files alone
account for ~84% of all idle time in the run (196s of 234s total idle), dragging
overall run utilization down to 53.9% versus the ~78.7% steady-state measured for
train-only runs (v1/v2).

Combined with Update 3, validation — specifically `PredictMetricsCallback`'s extra
`predict_step()` call — is expensive on **both** axes: it permanently inflates reserved
GPU memory from ~10GB to ~45GB, and it stalls the GPU for ~1.1-1.2s per batch without
extra GPU work to show for it. Both point at the same code path.

## Update 5: profiling `predict_step` directly

Added `src/rmind/utils/profiling.py` (`maybe_profile(tag)`, a shared context manager
extracted from the inline `TORCH_PROFILER`-gated block that used to live only in
`patch_policy.py:_step`) and wrapped `PredictMetricsCallback.on_validation_batch_end`'s
`pl_module.predict_step(batch)` call in `maybe_profile("predict_step")` too. Trace
files are now tagged by phase (`train_step_*`, `val_step_*`, `predict_step_*`), so the
next `TORCH_PROFILER=1` capture will isolate `predict_step`'s own trace directly instead
of needing the wall-time-based bucketing trick used in Update 4.

## Update 6: root cause found — `predict_step` runs in fp32, outside autocast (v4, wandb `ehso2306`)

Re-ran the identical v3 experiment with the new phase-tagged profiling
(`/tmp/torch_trace_v4`: 495 `train_step`, 100 `val_step`, 102 `predict_step` files).

**The direct proof, straight from the profiler's own memory-event log inside a single
mid-run `predict_step` file:** peak `Total Allocated` = **46.51 GB**, peak
`Total Reserved` = **48.80 GB** — this single call is directly responsible for the
~45GB permanent reservation found in Update 3. No more inference from timing
correlation needed.

**And the top ops inside that same file explain both the memory and the time cost:**

```
785.9ms  n=64  aten::copy_
785.1ms  n=98  aten::to
783.9ms  n=37  aten::_to_copy
```

plus kernels named `ampere_sgemm_128x128_tn` and
`fmha_cutlassF_f32_aligned_64x64...` — **`sgemm`/`f32`, i.e. fp32, not the `bf16`
GEMMs/flash-attention kernels seen everywhere in `train_step`/`val_step` traces.**
`predict_step` is running in full fp32.

**Why:** `PredictMetricsCallback.on_validation_batch_end` calls
`pl_module.predict_step(batch)` **directly** — a plain Python method call on the
`LightningModule` instance, not routed through `trainer.strategy.predict_step(...)`.
Lightning only enters its bf16-mixed autocast context
(`precision.py:171-189`, `forward_context()`) when it invokes `training_step`/
`validation_step`/`test_step`/`predict_step` through its own strategy/loop machinery —
a direct method call bypasses that entirely, so this call runs under Python's ambient
(fp32) default. `grep -n autocast` across `patch_policy.py`/`predict_metrics.py` finds
nothing that re-establishes it. fp32 activations are exactly 2x the size of bf16, which
lines up with the memory jump, and fp32 GEMM/attention kernels plus the huge
`copy_`/`to`/`_to_copy` overhead (contiguous bf16↔fp32 conversions all over the forward
pass) account for the time cost too.

**This also explains Update 4's "train_step gets slow too" puzzle:** splitting
`train_step` files by position within their ~50-step epoch, idle time concentrates
hard in the first ~15 steps of each epoch (right after a validation pass):

|                               | n   | utilization |
| ----------------------------- | --- | ----------- |
| first ~15 steps of each epoch | 150 | 30.7%       |
| rest of epoch                 | 345 | 60.3%       |

The first-few-steps-of-next-epoch tax (158s of the 262s total `train_step` idle time —
60% of it — concentrated in just 30% of the files) plus the exact same
`cudaHostAlloc` cold-start signature from Update 1 recurs at the very first `train_step`
of the run (`idx=0`, right after the pre-training sanity check) — i.e. `predict_step`'s
huge fp32 allocation disrupts the pinned-host-memory allocator every time it runs, and
that disruption "leaks" into the next several training steps, not just the validation
batch itself.

**Fix (applied):** `PredictMetricsCallback.on_validation_batch_end` now wraps
`pl_module.predict_step(batch)` in `trainer.precision_plugin.predict_step_context()` —
Lightning's own precision-context method (`plugins/precision/precision.py:183-186`,
`with self.forward_context(): yield`), the same one Lightning would enter itself if this
hook were invoked through its normal strategy/loop machinery. This automatically
respects whatever precision is actually configured (bf16-mixed here, but also correct if
`auto_precision` ever downgrades to `16-mixed` on non-bf16-capable hardware) rather than
hardcoding a dtype.

**Expected effect, to verify with another `TORCH_PROFILER=1` capture:**
`predict_step`'s memory footprint should roughly halve, its kernels should switch to the
bf16 tensor-core paths seen in `train_step`/`val_step` traces, the permanent ~45GB
reservation should drop close to train/val's own ~10GB/~5.5GB range, and the
epoch-boundary `train_step` slowdown (Update 4/6) should largely disappear since the
pinned-allocator disruption goes away.

## Update 7: IMPORTANT CORRECTION — every utilization number above is forward-pass-only

**All the `train_step` timing and utilization figures in this document measure the forward
pass, not a training step.** `maybe_profile` wraps only `self._compute_metrics(batch)`
inside `PatchPolicy._step`; the backward pass runs after `_step` returns, in Lightning's
optimizer loop, i.e. outside the profiled window.

Verified directly on the existing `/tmp/torch_trace_v5` capture (491 `train_step` traces):
the only attention kernel present is `fmha_cutlassF` — **F for forward** — and there are
**zero** kernels with `backward` in the name. Grep any `train_step` trace for
`fmha_cutlassB` to confirm.

What this invalidates:

- "~448ms of GPU work per step", "76.8% / 78.7% / 84.5% utilization", "~88ms idle per
  step" (Updates 1, 2, 4) all have the wrong denominator. A full training step does
  roughly 2.5–3x the GPU work reported here. Direct measurement of the trunk alone at
  batch 128 gives **1129ms forward+backward** with checkpointing on, against the ~460ms
  this document attributes to a whole step.
- Therefore the *share* of wall-clock lost to host-side stalls is materially smaller than
  concluded, and **compute optimizations rank above dataloader ones** — the reverse of the
  "Recommended next steps" ordering above.

The idle gaps themselves are real and still worth fixing. Only their relative weight
changes.

**Also resolved here: the unexplained bool-scalar sync.** Update 1 found an
`aten::is_nonzero → aten::item → aten::_local_scalar_dense → cudaStreamSynchronize`
signature and Update 6's correction ruled out the two Lightning-level checks without
identifying the real call site. It is `torch.multinomial` in `PatchPolicy._sample_codes`,
reached every training step from `_compute_metrics` (the `sampled_recon` diagnostic;
`sample_codes` defaults to `True` and no config overrides it). ATen's `multinomial`
validates the distribution with an `.item<bool>()` on a bool scalar. In the v5 traces this
accounts for **~140ms of CPU time per forward window**. A Gumbel-max rewrite removes it.

**And one conclusion above is now known to be incomplete.** The "what's already applied"
section reasons that swapping `nn.MultiheadAttention` for hand-rolled SDPA "would not
materially change memory here." That is correct about memory, but it never evaluated
throughput — and on throughput the block-causal mask is the single largest cost in the
model. Because SDPA receives an arbitrary (dense float) mask, the flash backend is
excluded outright (`SDPBackend.FLASH_ATTENTION` → "No available kernel"), leaving the
cutlass mem-efficient kernel at ~26 TF/s, about 10% of A100 bf16 peak. Attention is ~32%
of forward kernel time and ~61% of a full trunk step.

Full measured breakdown, the options and their costs, and the execution recipe now live in
**`docs/gpu_throughput_plan.md`**.
