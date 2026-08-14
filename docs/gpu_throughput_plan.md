# `patch_policy` GPU throughput: measured plan / handoff recipe

Companion to `docs/gpu_utilization_profiling.md`. That document diagnosed and fixed a
memory bug (fp32 `predict_step`). This one is about **making training faster** now that
the run sits at ~10GB of an 80GB A100.

Everything below was **measured** on the A100-SXM4-80GB in this container at production
shape, not estimated. Reproduction commands are given for every number so the next agent
can re-derive rather than trust.

______________________________________________________________________

## 0. Environment recipe (read this first — it will cost you an hour otherwise)

`python` is not on `PATH`, and plain `uv run` fails with
`ImportError: libstdc++.so.6`. Torch only works inside the nix devshell:

```bash
nix develop --command uv run --all-extras --group test <cmd>
```

**For anything that invokes TorchInductor** (`torch.compile`, `flex_attention`) you also
need the CUDA driver stub on the *compile-time* linker path. Without it Inductor dies with
`ld.bfd: cannot find -l:libcuda.so.1`:

```bash
nix develop --command env LIBRARY_PATH=/root/.cache/rmind/nix-nvidia-driver-libs \
  uv run --all-extras --group test <cmd>
```

(The flake's `shellHook` populates that symlink dir for `LD_LIBRARY_PATH` at runtime, but
Inductor shells out to `gcc`, which reads `LIBRARY_PATH`. Worth fixing in `flake.nix`.)

Real training runs in Docker on `nvidia/cuda:12.8.0`, where neither issue exists.

______________________________________________________________________

## 1. Correction: the profiling doc measures the forward pass, not a training step

`maybe_profile` wraps `self._compute_metrics(batch)` inside `PatchPolicy._step`
(`src/rmind/models/patch_policy.py`). Backward runs *after* `_step` returns, in
Lightning's optimizer loop — **outside the profiled window**.

Verified directly on `/tmp/torch_trace_v5` (491 `train_step` traces): the only attention
kernel present is `fmha_cutlassF` (**F = forward**), and there are **zero** kernels with
`backward` in the name.

Consequences for anyone reading the older doc:

- "448ms of GPU work per step", "76.8% / 78.7% utilization", "88ms idle per step" all
  describe the **forward-pass window**. A full training step does roughly 2.5–3x that
  much GPU work.
- So the *fraction* of wall-clock lost to host-side stalls is smaller than the doc
  concluded, and **compute optimizations matter more than dataloader ones** — the reverse
  of the doc's Recommended-next-steps ordering.
- The idle gaps the doc found are still real. Only the denominator was wrong.

Reproduce: grep any `train_step` trace for `fmha_cutlassB` (absent) vs `fmha_cutlassF`
(present). For the full picture, load a trace's `traceEvents`, keep `ph == "X"` events with
`cat == "kernel"`, and take the union of their `[ts, ts+dur]` intervals for real GPU-busy
time — summing `dur` gives the same answer here, which itself confirms there is only one
stream with no kernel overlap.

______________________________________________________________________

## 2. Measured baseline

Trunk = `BlockCausalTransformer`, batch 128, seq 1542, dim 512, 8 layers, 8 heads,
bf16 autocast, forward **+ backward**:

| checkpointing policy                          | fwd+bwd   | peak allocated | speedup   |
| --------------------------------------------- | --------- | -------------- | --------- |
| `True` (every block — the historical default) | 1129.1 ms | 8.80 GB        | 1.00x     |
| `2` (every other block)                       | 1004.1 ms | 19.41 GB       | 1.12x     |
| `False` (none)                                | 874.4 ms  | 33.05 GB       | **1.29x** |

Reproduce:

```bash
nix develop --command env RMIND_BENCH_BATCH=128 \
  uv run --all-extras --group test pytest tests/test_patch_policy_benchmark.py -s -q
```

**Where the 874ms goes** (`torch.profiler`, checkpointing off):

| kernel                                                      | time      | share |
| ----------------------------------------------------------- | --------- | ----- |
| `fmha_cutlassB_..._dropout_sm80` (attention **backward**)   | 384 ms    | 44%   |
| `fmha_cutlassF_...` (attention **forward**)                 | 148 ms    | 17%   |
| all `ampere_bf16_s16816gemm_*` (the actual projections/MLP) | ~105 ms   | 12%   |
| LayerNorm / GELU / elementwise / dropout                    | remainder | ~27%  |

**Attention is ~61% of trunk time while being ~33% of its FLOPs.** It runs at ~26 TF/s
— about **10% of A100 bf16 peak**. Everything else is roughly fine.

Cross-check against production: local attention-forward is 8 layers x 19.3ms = 154ms;
the real v5 trace shows `fmha_cutlassF` at **147.7ms = 32% of all forward kernel time**.
Agreement within 5%, so the microbenchmark is representative — use it, it is 40s per A/B
instead of hours.

______________________________________________________________________

## 3. Why attention is slow, measured

Per layer, B=128 H=8 N=1542 head_dim=64, bf16:

| variant                                                    | fwd     | fwd+bwd     | TF/s  |
| ---------------------------------------------------------- | ------- | ----------- | ----- |
| float mask + `dropout=0.1` (**what the model does today**) | 19.3 ms | 70.6 ms     | 26.5  |
| **bool** mask + `dropout=0.1`                              | 18.4 ms | 70.7 ms     | 26.5  |
| float mask + `dropout=0`                                   | 11.4 ms | 46.5 ms     | 40.2  |
| no mask + `dropout=0.1`                                    | 4.8 ms  | 23.1 ms     | 80.8  |
| **FlexAttention + `BlockMask`**                            | 4.5 ms  | **23.3 ms** | 80.2  |
| `is_causal` (token-causal — *wrong mask*, reference only)  | 2.8 ms  | 15.3 ms     | 122.4 |

Backend probe, run explicitly:

```
SDPBackend.FLASH_ATTENTION: REJECTS the float mask -> "No available kernel. Aborting execution."
SDPBackend.EFFICIENT_ATTENTION: works with the float mask
```

So:

- **Any arbitrary mask locks out flash**, forcing the cutlass mem-efficient kernel. This is
  the dominant cost: 47.5 ms/layer.
- The mask *form* is irrelevant. bool vs float is 70.7 vs 70.6 ms and the outputs are
  **bit-identical** (max abs diff exactly 0.0). Converting `block_causal_mask` to bool-keep
  semantics buys nothing — do not bother.
- `attn_dropout=0.1` costs a further 24.1 ms/layer *when the mask is present* (only 2.1 ms
  without it). It is never set in any config — it is the `TransformerBlock` class default.

This also corrects `docs/gpu_utilization_profiling.md`'s "already applied" section, which
concluded that hand-rolling SDPA "would not materially change memory here." True about
memory, but it never evaluated throughput, and on throughput the mask is the single
biggest cost in the model.

______________________________________________________________________

## 4. Open decision — needs the human, do not pick unilaterally

**FlexAttention has no `dropout_p`.** Adopting it forces `attn_dropout` 0.1 -> 0, which is
a regularization change, not a pure kernel swap. The three options, with measured value:

| option                                            | trunk fwd+bwd                                      | model change?                                     |
| ------------------------------------------------- | -------------------------------------------------- | ------------------------------------------------- |
| A. FlexAttention + `BlockMask`, `attn_dropout=0`  | 1129 -> **~496 ms** (2.3x, with checkpointing off) | yes — drops attention dropout                     |
| B. `attn_dropout=0` only, keep the cutlass kernel | 1129 -> ~682 ms (1.65x)                            | yes — same regularization change, half the payoff |
| C. Leave attention alone                          | 1129 -> 874 ms (1.29x)                             | no                                                |

Option B is dominated: it takes the same accuracy risk as A for less than half the gain.
The real choice is A or C.

If A: `resid_dropout=0.1`, `mlp_dropout=0.1` and the new `speed_dropout`/`goal_dropout`
conditioning dropout all remain, so the model is not left unregularized — but it still
needs an accuracy A/B against the current arm before it goes into a long run.

______________________________________________________________________

## 5. Work items, in measured priority order

### Already applied in the working tree (tests pass: `tests/test_patch_policy.py`, 12 passed)

**Configurable trunk checkpointing.** 1129 -> 874 ms (1.29x), 8.8 -> 33.1 GB.

- `src/rmind/models/patch_policy.py`: `BlockCausalTransformer.__init__` gained
  `checkpoint: bool | int = True` (`True` = every block, `False` = none, `k` = every k-th),
  normalized to `self._checkpoint_every`; `forward` now runs its own layer loop with
  `_should_checkpoint(i)`.
  **It deliberately no longer calls `run_layer_stack`** — that helper is shared with
  `ControlTransformer`'s encoder (`components/transformer/encoder.py:47`) and decoder
  (`decoder.py:95`), so changing it would silently change that model too. Leave it alone.
- `config/model/yaak/patch_policy/raw.yaml`: `encoder.checkpoint: ${trunk_checkpointing}`
- `config/experiment/yaak/patch_policy/dinov3.yaml`: `trunk_checkpointing: false`
  (every patch_policy arm inherits from this file, so one place covers all of them)
- `tests/test_patch_policy_benchmark.py`: new A/B harness (see §2)

Peak allocated goes ~9.4 -> ~36 GB on a real step. Still less than half the card. If a
larger arm (e.g. `dinov3_vitb`) gets tight, set `trunk_checkpointing: 2` rather than
shrinking the batch.

### 1. Attention — pending the §4 decision. Worth ~379 ms/step, the largest single item.

If option A is chosen: replace `nn.MultiheadAttention` in `TransformerBlock`
(`src/rmind/models/patch_policy.py`) with a packed-qkv projection + `flex_attention`.

- Prior art for the packed-qkv layout is `RotaryMultiheadAttention` in
  `src/rmind/components/transformer/attention.py` — follow it, and **keep
  `in_proj_weight`/`out_proj` parameter names and shapes** so existing checkpoints still
  load.
- Build the `BlockMask` **once** and cache it (see item 4), not per forward.
  `create_block_mask(block_causal, B=None, H=None, Q_LEN=N, KV_LEN=N, device=...)` with
  `block_causal = lambda b, h, q, kv: (q // tokens_per_frame) >= (kv // tokens_per_frame)`.
  Note the inverted convention vs `block_causal_mask`, which returns True = *blocked*.
- `flex_attention` must be `torch.compile`d to be fast — an uncompiled call is far slower.
- Verify equivalence against the current path with `attn_dropout=0` on both sides: outputs
  should match to bf16 tolerance (~1e-2 relative), not exactly — different reduction order.

### 2. Remove the per-step `cudaStreamSynchronize`. Confirmed in production traces.

`_sample_codes` (`src/rmind/models/patch_policy.py`) calls `torch.multinomial`, whose ATen
validity check does `.item<bool>()` on a bool scalar. In the real v5 `train_step` trace
this shows up as **~140 ms of CPU time** in `aten::is_nonzero` / `aten::item` /
`aten::_local_scalar_dense` per forward window.

This is the signature Update 1 of the profiling doc flagged and Update 6 left unresolved.
Update 6 correctly ruled out the two Lightning-level checks (they run outside the profiled
scope) but did not find the real call site: it is `torch.multinomial`, reached every train
step from `_compute_metrics` (the `sampled_recon` diagnostic), and `sample_codes` defaults
to `True` with no config override.

Beyond its own cost, a per-step sync drains the CUDA queue, so the GPU has nothing queued
to hide host hiccups behind — it converts dataloader jitter directly into GPU idle.

**Fix:** Gumbel-max — `(logits + gumbel_noise).argmax(dim=-1)`. Same sampling distribution,
no sync, and cheaper (drops both the softmax and the multinomial kernel). Applies to
`predict_step` too.

**Verify:** distribution equality over many draws on a fixed batch (compare code
histograms), *not* per-draw equality. Then re-check with the sync detector below.

### 3. Cut the 45% wasted host data path.

The dataset is built at `clip_length = episode_length(6) + clip_horizon(6) - 1 = 11` so it
can be shared with the control_transformer configs. `rbyte.Dataset.get_batch` fetches
**all 11** image frames per sample; `ChunkFields.forward`
(`src/rmind/components/nn.py`, the `value.narrow(self.dim, 0, self.episode_length)` branch)
then narrows images to 6 — **on the GPU, after the transfer**.

Per batch of 128: `324x576x3 x 11 x 128` = **752 MiB** crosses PCIe where 410 MiB is
needed; 1408 JPEGs are decoded where 768 are needed; the prefetch queue holds ~18 GB of
host RAM where ~10 GB would do. In the v5 trace `aten::to` / `_to_copy` / `copy_` total
**~765 ms of CPU time** per forward window — the largest CPU cost in the trace.

- **3a.** Override `on_before_batch_transfer` on `PatchPolicy` to narrow the image tensor
  to `episode_length` frames on the CPU side. Note it must address the **pre-`Remapper`**
  layout (`data/cam_front_left`), not `self.image`. ~15 lines, no numerics change.
- **3b.** Launch-flag experiments, one at a time:
  - `datamodule.train.method=process` — today `method: thread` runs all dataloading *inside
    the training process*, where the tensordict/polars glue in `get_batch` holds the GIL and
    contends with the thread that launches CUDA kernels. This matches the doc's unexplained
    "the calling thread itself not getting scheduled" signature. `--shm-size=32g` is already
    set; after 3a the in-flight set is ~10 GB.
  - `+datamodule.train.in_order=false` — `torchdata`'s `ParallelMapper` defaults to
    `in_order=True`, so one slow NFS batch head-of-line-blocks the queue. Training is
    shuffled anyway.
- **3c.** `ConcurrentPathTensorSource.__getitem__`
  (`src/rmind/io/path_tensor_source.py`) builds and tears down a
  `ThreadPoolExecutor(max_workers=8)` on *every call* — once per drive-group per batch.
  Hoist it to an instance-level pool.
- **3d.** (only if 3a shows decode, not transfer, is binding) Stop reading the 5 unused
  frames at all. `get_batch` derives what to fetch from `batch_data[stream_config.index]`,
  so this needs a 6-wide index column for the `cam_front_left` stream: either a narrowed
  column in `config/_templates/dataset/yaak/train.yaml` (invalidates `.rbyte_cache`, needs
  an index rebuild) or a thin `rbyte.Dataset` subclass in `rmind.io`. Prefer the latter.

### 4. `torch.compile` the trunk + cache the mask.

Reuse the existing wrapper — `rmind.utils.functional.compiled` is already wired for
`ControlTransformer` at `config/model/yaak/control_transformer/raw.yaml`. It mutates in
place, so `state_dict` keys keep no `_orig_mod.` prefix and `PatchPolicy`'s
`InstanceOf[BlockCausalTransformer]` narrowing still holds. Wrap `encoder:` in
`config/model/yaak/patch_policy/raw.yaml` the same way, keeping the `disable` escape hatch
that `just train-debug` uses.

Do this **after** the attention decision — compiling a plain layer stack is far more
tractable than compiling through `torch.utils.checkpoint`, and FlexAttention needs compile
anyway.

Also: `block_causal_mask` is rebuilt on device every forward, and `nn.MultiheadAttention`
re-materializes it as a dense 1542x1542 float mask per layer. Cache it in a non-persistent
buffer keyed by `num_frames` — prior art at `src/rmind/components/episode.py:163-196`.

Shape note: train has `drop_last: true` so batch is a constant 128; val does not, so expect
one recompile on the final val batch.

### 5. Not worth doing (measured/derived, recorded so nobody re-litigates)

- **bool instead of float attention mask** — 70.7 vs 70.6 ms, bit-identical output.
- **Optimizing the tokenizers or heads** — `ActionTokenizer`/`WaypointsTokenizer` are ~5
  orders of magnitude below the trunk (VQ lookups + tiny MLPs on 768 rows).
- **`accumulate_grad_batches` for speed** — changes effective batch at identical FLOPs.
- **Chasing the duplicate `predict_step` forward in validation** — correct that it exists,
  but production val is 5 drives against 655 (~117 of ~15.4k steps/epoch), so it is well
  under 1% end-to-end. The doc's "3x slower val steps" came from artificial
  `limit_*_batches` runs.
- **Raising `num_workers`** — already A/B'd in the doc (Update 2), no effect.

### 6. Deferred, costed: precomputed frozen DINOv3 features

The image transform is fully deterministic (CenterCrop/Resize/ToDtype/Normalize — no
augmentation), so caching is exactly equivalent. Removes the ViT from the step entirely
(~8.5 TFLOP), removes all JPEG decode, and cuts PCIe 752 -> 151 MiB/batch.

Storage: ~1.97M distinct frames x 256 patches x 384 dims x 2 B ~= **390 GB** bf16
(~195 GB at int8/fp8). Compelling because runs are 10–25 epochs — the ViT cost is paid once
instead of 25 times.

Prior art for the shard format is `src/rmind/scripts/offset_head_retrain.py`
(`_flush_shard`, `CACHE_DTYPES`), but note its `_load_cache` pulls a whole split into RAM
and will not scale. The right extension point is a new `TensorSource`-shaped class
(`__getitem__` + `__len__`, it is a structural `Protocol`) dropped into
`streams.cam_front_left.sources.*._target_` — exactly how `ConcurrentPathTensorSource`
replaced the upstream one.

______________________________________________________________________

## 6. Verification protocol

1. `nix develop --command uv run --all-extras --group test pytest tests/test_patch_policy.py -q`
   — correctness gate for every trunk change (synthetic, CPU, ~1s).
1. `RMIND_BENCH_BATCH=128 ... pytest tests/test_patch_policy_benchmark.py -s -q`
   — 40s A/B for any checkpointing/attention/compile change.
1. Short end-to-end A/B (needs NAS + wandb artifacts for the two frozen tokenizers):
   ```bash
   WANDB_MODE=disabled just train-unsafe \
     experiment=yaak/patch_policy/dinov3_dropout \
     datamodule=yaak/train_debug \
     +trainer.limit_train_batches=50 +trainer.limit_val_batches=10 \
     paths.data=/mnt/verda-nas \
     paths.rbyte.cache=/mnt/verda-nas/alex/.rbyte_cache_patch_policy_dropout
   ```
   Record steps/s and `system/train/max_memory_allocated_gb`. **Read
   `max_memory_allocated`, never `max_memory_reserved`** — the latter is a process-wide
   monotone ratchet (`src/rmind/callbacks/memory_stats.py`) and will not show a reduction
   within one process.
1. **Still to build (Phase 0 item, not yet done):** gate
   `torch.cuda.set_sync_debug_mode("warn")` behind an env var next to `TORCH_PROFILER` in
   `src/rmind/utils/profiling.py`. Turns every implicit device sync into a warning with a
   Python traceback — settles item 2 definitively instead of by op-name inference.
1. **If you re-profile, fix the window first.** `maybe_profile` currently wraps only
   `_compute_metrics`, so it cannot see backward (§1). To profile a whole step, the profiler
   has to wrap the Lightning training step, e.g. via a `Callback` around
   `on_train_batch_start`/`end`, or PL's own `profiler=` trainer argument.
1. Append results to `docs/gpu_utilization_profiling.md` as "Update 8+", matching its
   evidence-first style.

## Success criterion

Trunk fwd+bwd at batch 128 under ~500 ms (from 1129 ms), peak allocated 35–45 GB, and
unchanged loss curves over the first ~1k steps versus the current arm — with the caveat
that if option A in §4 is taken, "unchanged" becomes "A/B'd and accepted", since dropping
attention dropout is a real modeling change.

## Aside

`commands.sh` (untracked, repo root) still contains a live plaintext W&B API key. Rotate
it and move it to `.env`, independent of all the above.
