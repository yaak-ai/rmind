# Reader-side (dataloader → GPU) speedup — instructions for the next agent

Companion to `docs/gpu_utilization_profiling.md` (GPU-idle-time investigation) and
`docs/gpu_throughput_plan.md` (GPU compute-side speedups, measured). This plan is scoped
to the **reader side only**: everything from `ConcurrentPathTensorSource` reading JPEGs
off NAS through to the tensor landing on GPU, ready for `forward()`. Do not re-scope into
attention/`torch.compile`/checkpointing work — that's `gpu_throughput_plan.md` §1/§3/§4
and is already measured and gated on a human decision (§4 there). This plan executes and
extends `gpu_throughput_plan.md` §2 ("cut the 45% wasted host data path") and §5
(precomputed features) — read those two sections before starting, they already contain
real numbers and exact file/line pointers. Don't re-derive what's already measured there.

**Read `gpu_throughput_plan.md` §0 first and use its environment recipe verbatim** (nix
devshell requirement, `libstdc++.so.6` failure otherwise). Every command below assumes it.

## Why this is a separate investigation from the GPU-idle-time one

`torch.profiler` (as wired via `maybe_profile`, `src/rmind/utils/profiling.py`) is
**structurally blind to the reader side**. Confirmed directly on `/tmp/torch_trace_v4`/`v5`:
every trace file shows exactly one CPU thread (the main training thread) plus the GPU
stream track — dataloader worker threads never appear, because (a) they run on separate
Python threads that mostly call non-instrumented code (`Path.read_bytes()`,
`simplejpeg.decode_jpeg`) rather than `aten::*` ops, and (b) the profiler window only
opens *after* Lightning has already handed `_step()` a fully-formed, on-device batch —
whatever the reader did to produce it happened before profiling even started. So: no
amount of re-reading the existing `torch_trace_*` directories will answer "is the reader
fast enough" — new instrumentation is required. That's Step 1 below.

**Known ground truth so far (don't re-test these):**
- `num_workers` 24→8 A/B'd with no effect on steady-state utilization
  (`gpu_utilization_profiling.md` Update 2). Don't retry this axis alone.
- Production validation is ~1% of a real epoch's steps (`gpu_throughput_plan.md` §5,
  "not worth doing" list) — do not spend time optimizing the val/`predict_step` reader
  path specifically; everything here targets the **train** dataloader.
- The dataset over-fetches: `clip_length = episode_length(6) + clip_horizon(6) - 1 = 11`
  frames are read and decoded per sample, but `ChunkFields.forward`
  (`src/rmind/components/nn.py:245`, `value.narrow(self.dim, 0, self.episode_length)`)
  narrows to 6 **on GPU, after transfer** — confirmed still true in the current tree.
  `ConcurrentPathTensorSource.__getitem__` (`src/rmind/io/path_tensor_source.py`) also
  confirmed to build a fresh `ThreadPoolExecutor` on every call (not hoisted).

## Step 0: instrument the reader in isolation (no model, no GPU)

Before touching any code, get a model-independent number for "how fast can the reader
alone produce batches". Use `build_dataloader` from `src/rmind/scripts/offset_diag.py`
(`split="train"`, real production dataset, not `train_debug`) as the starting point — it
already handles the Hydra compose + `.rbyte_cache` path plumbing:

```python
from rmind.scripts.offset_diag import build_dataloader, shutdown_dataloader
import time

loader = build_dataloader("train", batch_size=128, num_workers=8)  # match production
it = iter(loader)
next(it)  # warm up: first batch pays cache/connection setup, exclude it
t0 = time.perf_counter()
n = 50
for _ in range(n):
    batch = next(it)
dt = time.perf_counter() - t0
print(f"{n / dt:.2f} batches/s  ({dt / n * 1000:.1f} ms/batch)")
shutdown_dataloader(loader)
```

Compare `ms/batch` against the trunk's own fwd+bwd cost from `gpu_throughput_plan.md` §2
(1129ms today, ~500ms if the §4 attention decision lands, 874ms with checkpointing off).
**If the reader is already faster than the trunk, there is no reader-side bottleneck to
fix** — report that finding and stop; don't implement speculative optimizations against a
pipeline stage that isn't binding. If it's slower or comparable, proceed.

Also instrument `ConcurrentPathTensorSource._getitem_concurrent`
(`src/rmind/io/path_tensor_source.py`) directly — wrap the NFS-read+decode call with a
`time.perf_counter()` delta and log via `structlog`, gated behind an env var the same way
`TORCH_PROFILER` gates `maybe_profile` (follow that pattern in
`src/rmind/utils/profiling.py` for consistency — a `maybe_time(tag)` sibling, or extend
the existing helper). This tells you the NFS-vs-decode split per frame, which determines
whether items 3/4 below (thread pool, `method=process`) or item 5 (stop over-fetching,
or precompute features) matter more.

## Step 1: narrow images before transfer, not after (`gpu_throughput_plan.md` §2a)

Currently the model narrows 11→6 frames on GPU (`ChunkFields.forward`, confirmed above),
so 11/6 ≈ 1.83x more image data crosses PCIe than needed (per batch of 128: 752 MiB vs
410 MiB at the current `image_resize`). This is a transfer-size fix only — it does **not**
reduce JPEG decode work (all 11 frames' worth of JPEGs are still read/decoded upstream by
`ConcurrentPathTensorSource`; that's item 4).

- Override `on_before_batch_transfer` on `PatchPolicy` (`src/rmind/models/patch_policy.py`)
  to narrow the image tensor to `episode_length` frames while it's still on CPU. Address
  the **pre-`Remapper`** layout — the raw batch key is `data/cam_front_left` (see
  `config/model/yaak/patch_policy/raw.yaml:12`), not `self.image` (that name only exists
  after `Remapper` runs, which is part of `input_transform`, which runs on GPU after
  transfer — don't try to reuse it here).
- No numerics change (same 6 frames end up selected either way) — this is pure
  data-volume reduction. `tests/test_patch_policy.py` should still pass unchanged.
- **Measure the transfer-specific win**, not just overall step time: in a `torch.profiler`
  capture (`TORCH_PROFILER=1`, per `docs/gpu_utilization_profiling.md`), sum
  `aten::to`/`aten::_to_copy`/`aten::copy_` durations before vs after — v5's baseline was
  **~765ms of CPU time per forward window**, the single largest CPU cost in that trace
  (`gpu_throughput_plan.md` §2). Expect this to drop toward ~6/11 of that figure.

## Step 2: hoist the thread pool (`gpu_throughput_plan.md` §2c)

`ConcurrentPathTensorSource.__getitem__` (`src/rmind/io/path_tensor_source.py`) builds and
tears down a `ThreadPoolExecutor(max_workers=8)` in `_getitems_concurrent` on **every
call** — once per drive-group per batch. Make it an instance-level pool (built once in
`__init__` or lazily via `cached_property`, matching the existing `_path_posix` pattern in
the same class), reused across calls. Pure overhead removal, no behavior change — cheap,
do this regardless of what Step 0's numbers say.

Verify: the class has no `__del__`/context-manager protocol today, so a persistent pool
needs an explicit shutdown path if the dataloader is ever torn down mid-run (check
`rmind.scripts.offset_diag.shutdown_dataloader`'s node-walking pattern for how this
codebase handles that elsewhere — same idea likely applies here, or simply document that
the pool lives for the process lifetime, matching how the dataloader itself behaves).

## Step 3: dataloader-config A/B, one variable at a time (`gpu_throughput_plan.md` §2b)

Both of these are Hydra launch-flag changes, no code. Test **one at a time**, using
Step 0's isolated reader-throughput harness so results aren't confounded by GPU-side
changes from Steps 1/2:

- `datamodule.train.method=process` — today `method: thread` runs all dataloading inside
  the training process, where the tensordict/polars glue in `rbyte.Dataset.get_batch`
  holds the GIL and contends with the thread launching CUDA kernels (this matches the
  "calling thread itself not getting scheduled" signature from
  `gpu_utilization_profiling.md`'s Update 1). `--shm-size=32g` is already set in the
  production docker launch; after Step 1 the in-flight data volume is ~10GB, should fit.
  **Risk to check first:** process-based workers need the dataset/collate objects to be
  picklable — `rbyte.Dataset`'s polars/tensordict internals may not be; if construction or
  the first batch throws a pickling error, that's the answer (stay on `thread`), don't
  fight it.
- `+datamodule.train.in_order=false` — `torchdata.nodes.map.ParallelMapper`
  (confirmed at `torchdata/nodes/map.py`) defaults to `in_order=True`, so one slow NFS
  batch head-of-line-blocks the output queue even though later batches are ready.
  Training is shuffled anyway, so batch order doesn't matter for correctness.

Record batches/s from Step 0's harness for baseline vs each flag, independently.

## Step 4 (conditional): stop reading the 5 unused frames at all (`gpu_throughput_plan.md` §2d)

**Only do this if Step 0's per-frame timing (or a repeat after Steps 1-3) shows JPEG
decode/NFS read — not transfer volume, not thread scheduling — is still the binding
cost.** This is the most invasive item: it needs a 6-wide index column for the
`cam_front_left` stream, since `get_batch` derives what to fetch from
`batch_data[stream_config.index]` (currently indexed for all 11 frames). Two ways to get
there, prefer the second:
- A narrowed index column in `config/_templates/dataset/yaak/train.yaml` — invalidates
  `.rbyte_cache`, needs a full cache rebuild (slow, NFS-bound, see
  `gpu_utilization_profiling.md`'s dataloader section on what `.rbyte_cache` actually
  caches before doing this).
- A thin `rbyte.Dataset` subclass in `rmind.io` — same extension pattern
  `ConcurrentPathTensorSource` already used to replace the upstream `PathTensorSource`.
  Preferred: no cache invalidation, and it's the established pattern in this repo for
  "drop in a `rbyte`-shaped replacement without forking the dependency".

## Step 5 (deferred, only after 1-4 land and are measured): precomputed frozen DINOv3 features

`gpu_throughput_plan.md` §5 has the numbers already (**~390GB bf16 storage**, ~195GB at
int8/fp8; removes the ViT forward, all JPEG decode, and cuts PCIe 752→151 MiB/batch since
the transform is fully deterministic — no augmentation). This is a large one-time
investment, only worth it if Steps 1-4 collectively don't close the gap found in Step 0.
Prior art for the shard format: `src/rmind/scripts/offset_head_retrain.py`
(`_flush_shard`, `CACHE_DTYPES`) — **but its `_load_cache` pulls a whole split into RAM and
won't scale**, don't copy that part. Extension point: a new `TensorSource`-shaped class
(`__getitem__` + `__len__`, structural `Protocol`, no inheritance needed — same pattern as
`ConcurrentPathTensorSource`) dropped into `streams.cam_front_left.sources.*._target_`.

## Verification protocol

1. `nix develop --command uv run --all-extras --group test pytest tests/test_patch_policy.py -q`
   after any code change (Steps 1, 2, 4) — correctness gate, ~1s, synthetic/CPU.
2. Step 0's isolated-reader harness, before and after each change — this is the number
   that actually answers "did the reader get faster", independent of anything GPU-side.
3. Short end-to-end A/B for the cumulative effect, same recipe as
   `gpu_throughput_plan.md` §6:
   ```bash
   WANDB_MODE=disabled just train-unsafe \
     experiment=yaak/patch_policy/dinov3_dropout \
     datamodule=yaak/train_debug \
     +trainer.limit_train_batches=50 +trainer.limit_val_batches=10 \
     paths.data=/mnt/verda-nas \
     paths.rbyte.cache=/mnt/verda-nas/alex/.rbyte_cache_patch_policy_dropout
   ```
   Record steps/s. Don't use this alone to judge reader-side changes, though — it's a
   `train_debug` 3-drive dataset, small enough that `.rbyte_cache`/OS page cache may mask
   real-NAS effects; Step 0's harness against the **real** `train` split (not
   `train_debug`) is the one that matters for the reader-vs-GPU question.
4. Append results to `docs/gpu_utilization_profiling.md` as further "Update N" entries
   (matching its evidence-first style — real numbers, repro commands, no estimates), or to
   `gpu_throughput_plan.md` §2 directly since that's where this work was originally scoped
   from. Don't create a fourth doc; extend one of the existing two.

## Success criterion

Step 0's isolated reader-throughput number should exceed the trunk's own fwd+bwd rate
(from `gpu_throughput_plan.md` §2) by a comfortable margin — i.e. the reader is
provably *not* the pipeline's bottleneck, backed by a measurement, not an assumption.
If Steps 1-4 get there, stop; Step 5 is not needed. If they don't, that's the trigger for
Step 5, and it should be brought back to the human before starting (large storage/compute
investment, same as the attention decision in `gpu_throughput_plan.md` §4).
