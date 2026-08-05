# Decoder-only PatchPolicy with a bounded KV cache

Implements `/nasa/max/docs/decoder_only_handoff.md`. Turns the block-causal
temporal trunk (fixed 6-frame sliding window, recomputed from scratch every tick)
into a causal decoder over frame blocks with a reusable, bounded KV cache.

Per tick the model encodes **one** new frame (257 tokens: 1 speed token prepended
to 256 goal-fused patch tokens), runs those 257 queries against the cached K/V of
past frames, and never recomputes or re-attends old frames to each other.

| | file |
| --- | --- |
| trunk + mask + RoPE + cache | `src/rmind/components/transformer/causal_frame.py` |
| one-tick export wrapper | `src/rmind/models/patch_policy_decoder.py` |
| correctness gate (CPU) | `tests/test_causal_frame.py` |
| ONNX export (all cache variants) | `src/rmind/scripts/decoder_only_export.py` |
| ONNX-level parity between variants | `src/rmind/scripts/decoder_only_cache_parity.py` |
| training arm | `config/experiment/yaak/patch_policy/dinov2_dinowm_causal.yaml` |

**If you are here for the latency verdict, read §10 and §12.** In short: the
per-tick KV `Concat` is *not* the bottleneck it was reported to be (§9's 34 % was a
bucketing artifact; removing it is worth 3.6 %), and **fp16 is** — it fuses each
trunk layer's whole attention into one kernel and takes `_big` at 64 frames from
364 ms to 97 ms. Keep `cache_attention="concat"`: the `Concat` is what the fused
kernel needs.

Everything outside the trunk is unchanged: the frozen DINOv2 ViT-S encoder, the
frozen goal encoder and its RVQ, the fusion, `patch_projection`,
`speed_embedding`, `norm`, and the VQ-BeT `code_head` / `offset_head` / tokenizer.
`CausalSelfAttention` even keeps `nn.MultiheadAttention`'s parameter names, so a
trunk trained with `BlockCausalTransformer` loads into it 1:1 — this is a
positional-encoding change, not a re-parameterization.

## 1. The positional encoding, which is the whole problem

`BlockCausalTransformer` adds `nn.Embedding(1542, d)` indexed over the flattened
sequence. That index is **window-absolute**: slot 0 always means "oldest frame
currently in the window". Slide the window and every frame's positional input
changes, so every cached key is stale. Nothing is reusable even though 5 of 6
frames are identical.

Replaced by a **factorized** scheme, each factor stable under a sliding window.

**Intra-frame — a learned 257-slot embedding, tiled onto every frame.** Token *k*
of frame *f* always gets `intra_pos[k]`, for every *f*. Frame-relative by
construction, never stale. This keeps exactly the intra-frame capacity the
1542-slot table had (patch identity; speed-token vs patch-token identity), so
nothing about what the model sees is reduced — §4 of the hand-off forbids buying
latency with representation.

**Inter-frame — RoPE at *frame* granularity**, applied to q and k in every layer.
All 257 tokens of a frame share one rotation `R_f`, which gives both properties
the hand-off asks for:

* intra-frame attention is **exactly unrotated**, `(R_f q)ᵀ(R_f k) = qᵀk`, so it
  stays fully bidirectional and is ordered only by the intra-frame embedding;
* inter-frame logits depend only on `f_q − f_k`, so a key rotated with its own
  episode-absolute frame index is valid **forever**, at any future window
  position. That is precisely cache-safety.

`rope_base = 1000`, not the customary 10000: over a 64-frame range, base 10000
leaves most frequency pairs rotating <0.01 rad and therefore inert.

**Why not ALiBi.** ALiBi is equally cache-safe and, being a pure additive bias,
folds into the existing mask with no new ops. It remains the better choice for a
low-precision *serving* engine, since RoPE normally introduces `Sin`/`Cos` and the
trt-export skill flags trigonometric ops as the most fp16-fragile part of this
model family. Two reasons it is not the default here:

1. RoPE's `Sin`/`Cos` are avoided anyway — `rope_cos`/`rope_sin` are **graph
   inputs**, computed host-side in float64 from the episode frame counter. The
   exported engine contains zero trigonometric nodes, so ALiBi's advantage
   evaporates.
2. ALiBi's bias is per-head and query-dependent, so the training path must
   materialize an `(H, S, S)` tensor. At 32 frames that is 12 × 8224² × 4 B ≈
   3.2 TB. RoPE has no such term.

**Episode-absolute learned embeddings** were rejected: they bound the episode
length and reintroduce a table that must be re-derived if the context changes.

## 2. Mask

`frame_block_causal_mask(num_frames, tokens_per_frame, window=N)` — bidirectional
within a frame, causal across frames, and additionally blocking frames more than
`N-1` older than the query. With `window=None` it is bit-identical to the existing
`patch_policy.block_causal_mask`.

The `window` is not cosmetic. **Streaming one frame per tick against a ring of
capacity `N-1` past frames is exactly one full forward under
`frame_block_causal_mask(window=N)`** — in both, frame *f*'s layer-*l* input
attends to frames `[f-N+1 .. f]`. That equivalence is the correctness gate and it
is also the operational meaning of "train the way you infer" (§6): train with
`window=N`, serve with a ring of `N-1`.

At serving time the step needs **no causal mask at all**: every cached key is in
the past by construction and every own-frame key is visible. The only masking is
`cache_bias`, which zeroes out unfilled ring slots.

## 3. Cache layout and the runtime contract

Single stacked tensor per side:

```
past_k, past_v : (num_layers, batch, num_heads, cache_frames * 257, head_dim)
new_k,  new_v  : (num_layers, batch, num_heads,            257, head_dim)
cache_bias     : (1, 1, 1, cache_frames * 257)   0 = filled, -1e4 = empty
rope_cos/sin   : (1, head_dim)
```

`past_k`/`past_v` are **read-only** graph inputs; only the new frame's K/V come
out. **The ring buffer lives in the host, not the graph.** No in-graph scatter, no
`ScatterElements` in TRT, and the cache is ordinary engine I/O. Between ticks the
host shifts by one frame block and writes 257 tokens per layer.

ONNX input names (`torch.export` flattens the `Mapping` argument):
`inputs_image`, `inputs_speed`, `inputs_waypoints`, `inputs_past_k`,
`inputs_past_v`, `inputs_cache_bias`, `inputs_rope_cos`, `inputs_rope_sin`.
Outputs: `policy.joint_actions`, `new_k`, `new_v`.

Two silent-failure hazards on the drivr side:

* `TRTEngine.run` binds via `set_tensor_address` — a raw pointer with **no size
  validation** (§3.4). A cache allocated for a different `cache_frames`, layer
  count, head count or dtype is not an error; TRT reinterprets the buffer and the
  model merely looks weak. **Validate every binding against
  `engine.get_tensor_shape(name)` before the first `run`.**
* Cache and frame counter must be **reset on every episode boundary** — engage,
  disengage, manual override. drivr already clears the action plan on those
  transitions; hook the same paths. A stale cache is not detectable from the
  output.

`Resize=0` and in-graph ImageNet normalization are unchanged, so the host still
owes an exact 224×224 (dinov2) `[0,1]` frame and `--image-norm unit`.

## 4. Gather before the final block (hand-off §3.3)

The head reads one token per frame, so in the final trunk block the attention
output and the MLP for the other 256 positions are discarded.
`readout_only_final_block=True` (the default in `PatchPolicyDecoderStep`) computes
them for the readout position only, while still emitting K/V for all 257 — future
frames attend to those. Verified equivalent in
`test_readout_only_final_block_matches_full_final_block`.

## 5. Correctness gate — `tests/test_causal_frame.py`

Runs on CPU in <1 s. Every equivalence is paired with a **negative control** on a
window-absolute variant of the same trunk, driven through the same harness, so a
pass is falsifiable.

The literal §5.1 phrasing — "run on `[0..5]` cold, then `[1..6]` warm vs a full
recompute of `[1..6]`" — is exact only when "full recompute" means the
sliding-window recompute described in §2. Re-running `[1..6]` as a *fresh isolated
6-frame episode* is a **different computation** and cannot match for any bounded
window with more than one layer: there frame 5 never saw frame 0, whereas in the
stream its layer-2+ K/V were produced with frame 0 in context. That residual is
recorded as a quantified diagnostic, and corroborated by the fact that it
vanishes at `num_layers=1`.

**Result at production sizes** (257 tokens/frame, max over every frame's readout):

| configuration | float64 | float32 |
| --- | --- | --- |
| small 8L/512d, window 6, T=7 | 1.78e-15 | 1.43e-06 (3.2e-07 rel) |
| `_big` 12L/768d, window 6, T=7 | 2.67e-15 | 1.55e-06 (4.0e-07 rel) |
| `_big` 12L/768d, window 32, T=40 | 4.44e-15 | 1.91e-06 (4.2e-07 rel) |
| **negative control** (window-absolute, small, window 6) | — | **9.37e-02 (2.1e-02 rel)** |

The float64 residual is at machine epsilon, which proves the equivalence is exact
and the ~1.5e-6 float32 residual is purely floating-point accumulation order —
streaming computes a 257×1542 attention where the recompute computes 1542×1542, so
the sums are ordered differently. The window-absolute control misses by **five
orders of magnitude**, which is what the gate is there to catch.

Isolated-episode diagnostic, same trunk, float64: **7.1e-02** at `L=8` and
**4.4e-16** at `L=1` — exactly the layer-≥2 leakage explanation, and not a
positional or cache defect.

The ONNX graph agrees with eager to **1.4e-05 absolute / 4e-06 relative** on the
new K/V and 5.7e-07 on the action chunk (ORT CPU fp32 vs PyTorch). Note
`torch.onnx.export(verify=True)` reports a "large relative difference of 2.87" for
`new_k`; that is per-element relative error on near-zero entries and is a false
alarm — compare absolute error against the tensor scale.

### The cache-attention variants are gated the same way

§10 introduces three graph formulations of the *same* attention and two cache
bindings. Each is gated at every level it could break at, and the negative control
is re-run through each so a pass stays falsifiable:

| level | check | result |
| --- | --- | --- |
| eager, step vs step, warm / **cold** / **half-filled** cache, x50 garbage behind every masked slot | `test_split_attention_matches_concat` | ≤1e-12 float64, ≤1e-5 float32, all finite |
| eager, the §5.1 recompute gate itself, through the restructured step | `test_split_attention_passes_the_recompute_gate` | 1e-6 |
| ... and its window-absolute control | `..._recompute_gate_negative_control` | diverges, as required |
| eager, per-layer binding vs stacked | `test_per_layer_cache_binding_is_the_same_computation` | 1e-6 |
| **production size**, `_big` 12L/768d, 63-frame cache, one step | `prodsize` probe (§10) | **2.2e-15** float64, 1.43e-06 (4.1e-07 rel) float32 |
| **exported ONNX**, through ORT, warm and cold | `decoder_only_cache_parity.py` | `split_kt` 3.5e-07 rel on `policy.joint_actions`; per-layer binding **bit-identical (0.0)** |

Cold and half-filled are not decoration: an online-softmax merge that mishandles a
fully-masked block produces `0 * inf`, and the garbage-behind-the-mask fixture is
what turns that into a NaN the test can see instead of a zero it cannot.

## 6. Memory budget on Orin — weights and KV cache

### Weights (measured, fp32 engines)

| arm | parameters | fp32 TRT engine | ONNX initializers |
| --- | --- | --- | --- |
| small (8L/512d) | 52.6 M | **207.2 MiB** | 213.0 MB |
| `_big` (12L/768d) | ~115 M | **440.2 MiB** | 453.3 MB |

The decoder-step engine is marginally *smaller* than the baseline's at the same
arm (434.2 vs 440.2 MiB for `_big`) — the 1542-slot positional table is gone and
the 257-slot one replaces it. Weights are context-independent, so a longer window
costs cache, never weights.

### KV cache

`bytes = 2 (K,V) × L × (cache_frames × 257) × D × sizeof(dtype)`, with
`cache_frames = context_frames − 1`.

Per cached frame: small (8L/512d) **8.03 MiB** fp32 / 4.02 MiB fp16; `_big`
(12L/768d) **18.07 MiB** fp32 / 9.04 MiB fp16.

| context (frames) | cached keys | small fp32 | `_big` fp32 | small fp16 | `_big` fp16 |
| --- | --- | --- | --- | --- | --- |
| 6 | 1285 | 40 MiB | 90 MiB | 20 MiB | 45 MiB |
| 16 | 3855 | 120 MiB | 271 MiB | 60 MiB | 136 MiB |
| 32 | 7967 | 249 MiB | 560 MiB | 124 MiB | 280 MiB |
| 64 | 16191 | 506 MiB | 1138 MiB | 253 MiB | 569 MiB |
| 128 | 32639 | 1020 MiB | 2295 MiB | 510 MiB | 1147 MiB |

### Total resident

delta-dev1 has **15 GiB** of LPDDR5 shared between CPU and GPU, ~12 GiB
practically available (measured `free`). Weights + cache, fp32:

| | 6 frames | 16 | 32 | 64 |
| --- | --- | --- | --- | --- |
| small | 247 MiB | 327 MiB | 456 MiB | 713 MiB |
| `_big` | 530 MiB | 711 MiB | **1000 MiB** | 1578 MiB |

**Memory is not the binding constraint at any context length worth training** —
`_big` at 64 frames is 1.5 GiB, 13 % of what is available, and halves again with an
fp16 cache. Latency binds first, and it binds well before memory does: `_big`
leaves the 333 ms tick at 64 frames (§9) while still using only an eighth of RAM.

## 7. drivr serving sketch

```python
# --- once, at engine load: the graph is the authority on every cache dimension
for name in ("inputs_past_k", "inputs_past_v", "inputs_cache_bias",
             "inputs_rope_cos", "inputs_rope_sin", "new_k", "new_v"):
    expected = tuple(engine.get_tensor_shape(name))
    if tuple(buffers[name].shape) != expected:      # set_tensor_address does NOT check
        raise ValueError(f"{name}: engine wants {expected}, got {buffers[name].shape}")
layers, _, heads, cache_tokens, head_dim = engine.get_tensor_shape("inputs_past_k")
tokens_per_frame = engine.get_tensor_shape("new_k")[3]      # 257
cache_frames = cache_tokens // tokens_per_frame             # context - 1

# --- on engage / disengage / manual override, wherever the action plan is cleared
def reset_cache():
    cache_bias.fill_(-1e4)     # every slot invalid; K/V contents then do not matter
    frame_index = 0            # RoPE counter; fp64 host-side, monotone per episode

# --- per tick
rope_cos, rope_sin = frame_rope_cos_sin(frame_index, head_dim=head_dim, base=1000.0)
out = engine.run(image=frame, speed=..., waypoints=...,
                 past_k=past_k, past_v=past_v, cache_bias=cache_bias,
                 rope_cos=rope_cos, rope_sin=rope_sin)

# ring advance: write into ONE slot and move nothing else
slot = slice((frame_index % cache_frames) * tokens_per_frame,
             (frame_index % cache_frames + 1) * tokens_per_frame)
past_k[..., slot, :] = out["new_k"]
past_v[..., slot, :] = out["new_v"]
cache_bias[..., slot] = 0.0     # after the first `cache_frames` ticks: all zeros
frame_index += 1
```

**Do not shift the cache.** The obvious form,
`past_k[..., :-257, :] = past_k[..., 257:, :]`, is wrong twice: it is an
*overlapping* in-place copy on one storage (undefined in torch, a genuine
read/write race on GPU), and it moves the entire cache every tick — ~1129 MiB in
and out, ~2.3 GiB of device traffic, order 20 ms at `_big`/64 frames, versus
18 MiB for a slot write.

The slot write is valid because **attention is permutation-invariant over keys**,
and each key carries its own position (RoPE-rotated with its absolute frame index)
and its own `cache_bias` entry — so the *order* of the cache carries no
information. Verified, not assumed:
`test_ring_slot_write_matches_shift_left` streams both policies and requires the
readouts to agree to 1e-6.

Three failure modes, all silent, all worth an explicit check:

1. **Wrong cache shape** — `set_tensor_address` takes a raw pointer, so a
   mismatched cache is reinterpreted rather than rejected. Validate at load.
2. **Cache not reset at an episode boundary** — the model conditions on frames
   from before the disengage. The output stays plausible. Hook the paths that
   already clear the action plan.
3. **`frame_index` not monotone** (e.g. reset mid-episode, or wrapped) — RoPE
   offsets become wrong for the frames still in the ring. Reset the counter
   **only** together with the cache.

Optional but cheap: assert the number of valid slots in `cache_bias` equals
`min(frame_index, cache_frames) * 257`. That single invariant catches all three.

## 8. rbyte / dataset considerations

`clip_length = episode_length + clip_horizon - 1`, so moving from 6 to 16 frames
takes clips from 11 to 21 samples, and 64 frames would need 69. Consequences:

* the dataset must be **rebuilt** — `clip_period` is derived from `clip_length`
  and the existing clip-11 build is not reusable;
* at `episode_step = 10` (≈3 Hz), 21 samples span ~7 s of driving and 69 span
  ~23 s. Long clips are cut at session boundaries, so the number of usable
  windows falls roughly linearly with `clip_length` — expect a materially
  smaller train set at 64 frames, on top of the compute cost;
* per-sample decode/IO grows linearly with `clip_length`, so the loader becomes a
  real cost at long context, not just the GPU.

This is the practical reason to step 6 → 16 → 32 rather than jumping to 64.

## 9. Measured latency

Method: fp32 engines (the hand-off's 194.8 / 448.8 ms baseline is fp32, and fp32
is the only precision that has ever reached 0/200 on parity), built and
benchmarked on an idle delta-dev1 (AGX Orin 16 GB, 8 cores, TRT 10.7) with the
GPU clock **pinned at 918 MHz** — `governor=performance`, `min_freq=918000000`,
verified by sampling `cur_freq` continuously *through* a 5 s idle gap (all 25
samples at 918 MHz, so the `nvhost_podgov` 306 MHz artifact is not present).
`trtexec --iterations=60 --avgRuns=20 --useSpinWait --warmUp=1000`, median GPU
compute time.

These are **engine GPU compute** and therefore exclude the host-side ring update,
as `inference_ms` in drivr's logs also does. With the slot write of §7 that update
is one 257-token copy per layer — ~18 MiB at `_big`/64 frames, sub-millisecond. With
the naive shift-left it would be ~2.3 GiB and order 20 ms, which is the other reason
not to write it that way.

**Gate zero — the baseline reproduces.** Randomly-initialized fp32 exports of the
*existing* block-causal architecture at 6 frames:

| arm | measured here | hand-off §1 | delta |
| --- | --- | --- | --- |
| small (8L/512d) | **200.85 ms** | 194.8 ms | +3.1 % |
| `_big` (12L/768d) | **420.16 ms** | 448.8 ms | −6.4 % |

Close enough that the comparison is sound, and it confirms `_big` is **12** layers
(§7 says 8; an 8-layer 768-d trunk could not cost 420 ms). Speedups below are
quoted against *these* numbers, measured on the same host on the same day through
the same export path, not against the hand-off's.

**Decoder step, per tick, fp32, median GPU compute:**

| context (frames) | small (8L/512d) | vs 6-frame baseline | `_big` (12L/768d) | vs 6-frame baseline |
| --- | --- | --- | --- | --- |
| 6 | **41.79 ms** | **4.81×** | **87.21 ms** | **4.82×** |
| 16 | 59.58 ms | 3.37× | 127.79 ms | 3.29× |
| 32 | 92.26 ms | 2.18× | 205.05 ms | 2.05× |
| 64 | 160.51 ms | 1.25× | 364.22 ms | 1.15× |

At the same 6 frames of context the step is **4.8× cheaper in both arms**. The
hand-off estimated ~75 ms for `_big`; the measurement is 87.2 ms, i.e. the estimate
was optimistic by ~16 %.

⚠️ An earlier version of this paragraph attributed that 16 % to the in-graph `Concat`
of `past_k` with the new frame's keys. **That is wrong too** — at 6 frames the whole
cache-copy path is ~0.46 ms per layer, ~5.5 ms of the 87.2, and a graph with the
`Concat` removed entirely measures 87.00 ms, i.e. **0.2 ms** cheaper (§10). Whatever
the 12 ms is, cache management is not it.

### Correcting the hand-off: per-tick cost does NOT stop scaling with context

§2 claims "per-tick cost stops scaling with window length". It does not. The step
runs 257 queries against `(N-1) × 257 + 257` keys, so both the attention terms
and the cache traffic are **linear in N**. Measured, over 6 → 64 frames:

| | fixed cost (N→1) | marginal cost per extra frame | R² of the linear fit |
| --- | --- | --- | --- |
| small | ~31.5 ms | **2.05 ms/frame** | >0.999 |
| `_big` | ~63.3 ms | **4.78 ms/frame** | >0.999 |

The slope tracks `L × D` and nothing else, which makes it predictable: the extra
attention work per cached frame is `2 × 257 × 257 × D × 2 × L` FLOPs — 1.08 GFLOP
(small), 2.43 GFLOP (`_big`) — and the two measured slopes correspond to the same
**~510–530 GFLOP/s effective** (~16 % of the module's ~3.3 TFLOP/s fp32 peak) in
both arms.

**Where the marginal time goes — corrected, and the correction matters.** An
earlier version of this section reported a bucket called "fused kernels containing
the cache `Concat`" at **68.2 ms / 34 %** of small at N=64, and concluded from it
that a paged/in-place attention plugin was the highest-value follow-up. **That
number was a name-matching artifact and the conclusion drawn from it is wrong.**

TRT fuses and renames everything to `__myl_<Op><Op>...`, so buckets have to be
derived from those concatenated op abbreviations — and `Con` in a myelin name means
*any* `Concat`. Three unrelated `Concat`s exist per layer: the KV cache concat, the
`F.pad(cache_bias)` that appends the own-frame keys' zero bias, and RoPE's own
`cat((-x2, x1))`. The single most expensive kernel in the whole `_big`/N=64 profile,
`__myl_RepRepConAddMaxSubExpSumDivMul` at 9.75 ms, is the **softmax** — its `Con` is
the 1-row bias pad. Matching `Con` before the softmax pattern put it, and its
siblings, in the `Concat` bucket.

Re-bucketed with the softmax pattern taking precedence
(`rmind.scripts.decoder_only_profile_buckets`, which keeps the naive rule too so the
artifact reproduces rather than being asserted — it prints 74.5 ms / 36.8 % naive on
the same file):

| bucket | small N=6 | small N=64 | `_big` N=64 |
| --- | --- | --- | --- |
| MatMul (qkv / proj / MLP / head / attention GEMMs) | 28.8 ms (62 %) | 65.5 ms (32 %) | 143.8 ms (40 %) |
| `scaled_dot_product_attention` (the P·V GEMM) | 4.1 ms (9 %) | 35.0 ms (17 %) | 82.6 ms (23 %) |
| fused softmax (incl. the `cache_bias` pad) | 3.4 ms (7 %) | 30.6 ms (15 %) | 68.5 ms (19 %) |
| KV cache copy — K path (slice + rope + cat + transpose) | 2.0 ms (4 %) | 20.5 ms (10 %) | 46.1 ms (13 %) |
| KV cache copy — V path (slice + cat) | 2.5 ms (5 %) | 26.1 ms (13 %) | 12.8 ms (4 %) |
| other fused elementwise | 5.2 ms (11 %) | 24.7 ms (12 %) | 4.0 ms (1 %) |
| conv2d (ViT patch embed) | 0.15 ms | 0.15 ms | 0.15 ms |

The naive rule reproduces the old **74.5 ms / 36.8 %** on the same file, so this is
the same measurement read two ways, not a new one.

⚠️ **And do not trust the corrected table either.** TRT splits the same work across
different fusions per arm, so ~17 ms of small/N=64's "other fused elementwise" is
also cache-copy work that the pattern does not catch — the corrected KV total for
that arm is somewhere in 47–64 ms, which is a range, not a number. **Bucketing a
myelin profile cannot settle this question.** What settles it is building the graph
without the `Concat` and measuring it, which is §10, and the answer there is 12.5 ms
at small/N=64.

⚠️ The old warning that `--dumpProfile` inflates large-tensor kernels most does not
survive either. It inflates the *small* arm (202.5 ms profiled vs 160.5 benchmarked
at N=64) but **not** `_big`: 357.9 profiled vs 363.9 benchmarked, i.e. slightly
*under*. Whatever the small-arm gap is, it is not proportional to tensor size.

Even the corrected table overstates the prize, and §10 measures why: those cache-copy
kernels are dominated by a **slice of the stacked cache tensor**, which any
formulation pays, not by the `Concat`, which is fused into it for free. §10 removes
the `Concat` from the graph outright and gains 3.6 %.

What *does* improve superlinearly is the comparison against recomputing an
N-frame window, which is quadratic. The right way to state the prize:

* **`_big` attends to 32 frames for 205 ms — less than half of what it costs today
  to attend to 6** (420 ms). Recomputing a 32-frame window block-causally would be
  ~28× the trunk work of the 6-frame one, i.e. several seconds.
* **small attends to 32 frames for 92 ms, versus 201 ms today for 6.**
* Every configuration measured except `_big` at 64 frames (364 ms) fits inside one
  333 ms tick, and `_big` clears the ~270 ms threshold that removes the
  plan-execution distortion (hand-off §7) at **up to 32 frames** — where today it
  does not clear it at 6.

Practical reading of the slope: 32 frames is the point where `_big` still fits
comfortably (205 ms, half of today's cost) and 64 frames is where it stops being
free (364 ms, no longer inside a tick at fp32). If 64 frames is wanted, the lever is
the fp32-encoder + fp16-trunk serving engine from the trt-export skill §9 — which
halves the cache and its traffic as well as the GEMM cost. **Not** an in-place/paged
attention rewrite: §10 built one and it is worth 3.6 %. And not more caching: the
cache is already exact.

⚠️ The decoder numbers include §3.3's gather-before-final-block, which the
baseline does not have. That optimization is free and applies to the baseline
independently; the hand-off measured it at 30.6 ms (6.8 %) there. Subtracting it
from the comparison would put the like-for-like `_big` speedup at ~4.5× rather
than 4.8×.

Note **cold vs warm cache costs the same** for a static-shape engine: the graph
does identical work at any fill level, and correctness at cold start comes from
`cache_bias`, not from a smaller computation. A cheaper first tick would need a
dedicated small-context engine; it is not worth an engine to save one tick per
episode.

## 10. Paged / in-place attention: what removing the copy is actually worth

The question this section answers: **how much of the linear-in-context cost is the
per-tick KV copy, and what is the cheapest mechanism that removes it?** Same host,
same day, same methodology as §9 (fp32, GPU pinned 918 MHz, idle,
`--iterations=60 --avgRuns=20 --useSpinWait --warmUp=1000`). **Every** §9 point that
anything below is compared against was re-measured first, as a second gate zero
against the same engines:

| | §9 recorded | re-measured | delta |
| --- | --- | --- | --- |
| `_big` N=6 | 87.21 ms | **87.21 ms** | 0.00 % |
| `_big` N=32 | 205.05 ms | **205.22 ms** | +0.08 % |
| `_big` N=64 | 364.22 ms | **363.93 ms** | −0.08 % |
| small N=6 | 41.79 ms | **41.78 ms** | −0.02 % |
| small N=32 | 92.26 ms | **92.27 ms** | +0.01 % |
| small N=64 | 160.51 ms | **160.58 ms** | +0.04 % |

Six for six inside 0.1 %, so the deltas below are real at the ~0.5 ms level.

### The mechanisms

All are the *same attention*, differing only in what the graph materializes;
`cache_attention` on `CausalFrameTransformer` selects between them, and
`tests/test_causal_frame.py` plus `rmind.scripts.decoder_only_cache_parity` gate
the equivalence (float64 residual **2.2e-15** at production size — machine
epsilon — and **3.5e-7** relative on `policy.joint_actions` through ORT).

* **`concat`** (§3, the shipped default) — `cat(past_k, k)` then one SDPA.
* **`split`** — two attentions, one over the cache and one over the own frame,
  combined by online-softmax renormalization (`merge_attention`): the flash trick
  applied once, at frame granularity. **No cache-sized tensor is created**: the
  largest `Concat` in the exported graph drops from 12.4 M elements (16448 × 768)
  to 0.197 M (RoPE's own `cat` on 257 tokens).
* **`split_kt`** — `split`, plus `past_k` bound **pre-transposed**
  `(L, b, H, head_dim, cache_tokens)`, so `q @ past_k` needs no transpose. Free
  for the host: the ring slot write is 257 tokens per layer in either layout.

No custom CUDA, no plugin, no `ScatterElements` — a graph restructure and an I/O
layout change.

### Measured

**Removing the `Concat` entirely is worth 3.6–7.8 % at 64 frames and nothing at 32.**

fp32, median GPU compute. `concat` is the re-measured baseline above.

| arm | context | `concat` | `split` | `split_kt` | best delta |
| --- | --- | --- | --- | --- | --- |
| `_big` | 6 | 87.21 ms | — | 87.00 ms | −0.2 ms (−0.2 %) |
| `_big` | 32 | 205.22 ms | 212.55 ms | 205.18 ms | −0.04 ms (**0 %**) |
| `_big` | 64 | 363.93 ms | 365.10 ms | **350.78 ms** | **−13.2 ms (−3.6 %)** |
| small | 6 | 41.78 ms | — | 41.97 ms | +0.2 ms (worse) |
| small | 32 | 92.27 ms | — | 87.65 ms | −4.6 ms (−5.0 %) |
| small | 64 | 160.58 ms | — | **148.09 ms** | **−12.5 ms (−7.8 %)** |

**`split` is never worth building.** It is 1.2 ms worse at `_big`/64 and 7.3 ms
worse at `_big`/32 — the transposed copy TRT inserts in its place costs slightly
*more* than the `Concat` it replaced. Only the pre-transposed layout wins, which is
the actionable half of this result.

**What `split_kt` really changes is the slope, not the level.** Marginal cost per
extra cached frame:

| | `concat` | `split_kt` |
| --- | --- | --- |
| `_big`, 6→32 | 4.54 ms/frame | 4.55 ms/frame |
| `_big`, 32→64 | **4.97 ms/frame** | **4.55 ms/frame** |
| small, 6→32 | 1.94 ms/frame | 1.76 ms/frame |
| small, 32→64 | **2.13 ms/frame** | **1.89 ms/frame** |

`concat`'s marginal cost *grows* with context (4.54 → 4.97 on `_big`); `split_kt`'s
is flat (4.55 → 4.55). The copy is what makes the baseline superlinear, so the gain
is ~0 at 32 frames and widens with N — extrapolating the two slopes, ~7 % at 128
frames. That is a much weaker and much more specific claim than "the `Concat` is
34 % of runtime", and it is the one the measurements support.

Two things fall out of the profiles, both of which contradict §9's old conclusion.

**1. The `Concat` was never an extra copy.** `past_k` arrives as one stacked
`(L, b, H, S, D)` tensor, so every layer must first take `past_k[i]`. TRT
materializes that slice as a **copy** — 24 `__myl_Sli` kernels, one per (layer,
side), 1.05 ms each, **25.3 ms** at `_big`/N=64, all hoisted to the start of the
engine. In the `concat` graph the `Concat` is *fused into that same copy*
(`__myl_SliResTraResCon`), so it costs nothing on top of a copy that happens
anyway. Delete the `Concat` and the copy remains. That is the whole reason
`split` gains nothing.

**2. The transpose is the part that was actually costing something.** With
`past_k` in its natural layout, TRT does **not** fold `k.transpose(-1,-2)` into the
GEMM — it emits 12 extra `__myl_Tra`/`__myl_SliRes` kernels (14.6 + 12.5 ms) to
materialize a transposed copy, which is exactly what the `Concat` kernel had been
doing before. Pre-transposing the cache removes them, and that 13-14 ms *is* the
entire measured win. Cross-checked against DRAM traffic: one layer's K side at
`_big`/N=64 is 49.7 MiB, so a read+write copy is ~100 MiB, and 100 MiB / 1.05 ms
= **95 GB/s** — right at what this LPDDR5 delivers. The copies are not
inefficient; there are just no free ones.

### Where the marginal cost really is, per layer, at `_big`/N=64

| | `concat` | `split_kt` |
| --- | --- | --- |
| qkv projection | 0.91 | 0.90 |
| KV copy / rope / transpose | 5.01 | 0.13 |
| **q · Kᵀ** | **8.19** | **8.21** |
| **softmax (+ bias)** | **5.80** | **7.08** |
| **P · V** | **7.37** | **7.26** |
| online-softmax merge | — | 0.05 |
| own-frame attention (257×257) | (in the above) | 0.40 |
| out projection + MLP | 2.67 | 2.67 |
| **per layer** | **~27.6** | **~24.3** |
| plus, once per engine: stacked-cache slice | (fused) | 25.3 total |

**22.5 of the 24.3 ms/layer is the three cache-attention kernels, and no graph
restructure touches them.** They are irreducible in fp32 for a structural reason:
the 257 × 16448 score matrix is written by `q · Kᵀ`, read and rewritten by the
softmax, and read by `P · V` — ~812 MiB of DRAM traffic per layer, ~9.7 GiB per
tick. Avoiding that requires never materializing the scores, i.e. a fused
FlashAttention-style kernel. TRT 10.7 has none for this shape (§12).

### What this buys against the budget

| | `concat` | `split_kt` | fits 333 ms? | fits 270 ms? |
| --- | --- | --- | --- | --- |
| `_big` N=32 | 205.2 | 205.2 | yes | yes |
| `_big` N=64 | 363.9 | **350.8** | **no** (both) | no |
| small N=32 | 92.3 | 87.7 | yes | yes |
| small N=64 | 160.6 | **148.1** | yes | yes |

`_big` at 64 frames goes 364 → 351 ms: still outside the 333 ms tick and well
outside 270 ms. **The copy was not what put it there.** The honest statement of the
prize is that `split_kt` alone is a free 3.6–7.8 % at 64 frames with no numerical
cost, and *not* a mechanism that changes which context lengths are servable — it
leaves `_big`/64 ~18 ms short of the tick. §11 closes that remaining gap (318 ms,
inside the tick) by removing the copy that actually costs something, and §12 makes
the whole comparison moot by changing the dtype.

⚠️ Read the latency column, not trtexec's `H2D Latency`. That column moves a lot
between these variants (51.3 ms `concat`, 73.6 `split`, 64.2 `split_kt` at
`_big`/N=64) purely because trtexec re-uploads the whole cache every iteration and
the declared input layouts differ. drivr keeps the cache resident on the device, so
none of it is a real per-tick cost, and `split`'s +22 ms there is not a regression.

## 11. The mechanisms that were *not* prototyped, and why

### A stacked cache costs a copy per layer — measured, and removable

The 25.3 ms of `__myl_Sli` kernels (§10) are TRT materializing `past_k[i]` out of
the stacked `(L, b, H, S, D)` input. That slice is a **contiguous view** — a pointer
offset, not data movement — so it is TRT choosing to copy, not arithmetic. Binding
the cache as `2 × L` separate inputs (`inputs_past_k_0 …`) would leave TRT nothing
to slice, and costs the host nothing: the ring write is 257 tokens per layer either
way, into per-layer buffers instead of into slices of one buffer.

Probed on a trunk-only graph (12L/768d, N=64, `split_kt`, cache bound both ways)
before changing `PatchPolicyDecoderStep`, because TRT could simply re-insert an
equivalent reformat per binding. It does not:

| trunk-only graph, fp32 | latency | profiled `__myl_Sli` |
| --- | --- | --- |
| stacked `(L, b, H, ...)` cache | 336.14 ms | **25.28 ms, n=24** |
| one binding per layer (24 cache inputs) | **303.27 ms** | **0.00 ms, n=0** |

**−32.9 ms (−9.8 %), and the diff is otherwise empty** — the 24 slice kernels vanish
and nothing else in the graph changes. The benchmark saves 7.6 ms *more* than the
profile attributes to those kernels, which is the launch and serialization overhead
of 24 hoisted copies.

So `per_layer_cache=True` is implemented (`empty_cache`, `write_slot` and the export
all follow the layout, and `tests/test_causal_frame.py` gates the equivalence). It
costs the host nothing: the ring write is 257 tokens per layer either way, into 24
buffers instead of 24 slices of two buffers.

**Confirmed on the full graph**, `_big`/N=64, fp32 — and the trunk-only probe
predicted it to 0.4 ms:

| full decoder step, `_big` N=64, fp32 | latency | vs baseline | fits 333 ms? |
| --- | --- | --- | --- |
| `concat`, stacked cache (shipped) | 363.93 ms | — | no |
| `split_kt`, stacked cache | 350.78 ms | −13.2 ms | no |
| `split_kt` + **per-layer bindings** (30 bindings) | **318.33 ms** | **−45.6 ms (−12.5 %)** | **yes** |

So the two graph-level mechanisms together do bring `_big`/64 inside the tick at
fp32 — 12.5 %, of which **three quarters is the stacked-cache slice, not the
`Concat`**. §12 then makes the whole question moot at fp16, but if an fp32-only
engine is ever required, this is the configuration.

⚠️ **But read §12 before reaching for it.** This measurement is `split_kt`/fp32, and
at fp16 the shipped `concat` graph fuses its whole attention into one kernel per
layer, which changes both the baseline and the size of this prize (~12 ms, not ~33).

### A custom fused-attention (FlashAttention) plugin — effort estimate, not prototyped

This is the only mechanism that attacks the real cost: the 257 × 16448 score matrix,
~812 MiB of DRAM traffic per layer, ~9.7 GiB per tick at `_big`/N=64.

* **Upper bound on the gain.** The two GEMMs are 12.98 GFLOP per layer; at the
  ~800 GFLOP/s the current unfused GEMMs achieve, a kernel that eliminated *all*
  score traffic and nothing else would land near 16 ms/layer against the measured
  22.5 — **~6.5 ms/layer, ~78 ms at `_big`/N=64**, i.e. 351 → ~273 ms. Realistically
  50–70 % of that, because a hand-written fp32 kernel will not match cuBLAS-class
  GEMM efficiency on Orin's non-tensor-core fp32 path.
* **Effort: 3–6 engineer-days.** Kernel (tiled online-softmax over the key range,
  with a flash-decoding-style split over keys to get enough blocks for 16 SMs;
  257 queries × 12 heads is only 48 tiles at 64-query granularity), `IPluginV3`
  boilerplate (creator, serialization, shape/type inference, autotuning),
  a custom-op export path so `torch.onnx.export` emits it, and parity per skill §5.
* **Maintenance risk, which is the real objection.** A plugin is a `.so` tied to the
  TRT version *and* GPU arch, loaded at runtime by drivr on the car. It becomes a
  second artifact that has to be version-matched to every engine, and the skill's
  existing rule — engines are valid only because dev1 and the car share TRT 10.7 —
  now applies to a hand-written binary as well. Do not take this on for a 20 % trunk
  win; take it on only if 64+ frames at fp32 is a product requirement.

### TensorRT-LLM's paged attention — ruled out, with specifics

Not a close call, for four independent reasons:

1. **Wrong entry point.** TRT-LLM does not consume ONNX. Its `gptAttentionPlugin`
   is reachable only through TRT-LLM's own Python graph builder, so using it means
   reimplementing PatchPolicy — ViT encoder, goal RVQ, VQ-BeT heads — in that API.
2. **Wrong attention shape.** The plugin has a context phase (whole prompt, causal)
   and a generation phase (one query token). Ours is a 257-query block that is
   **bidirectional within itself** and full-attention over the cache; that is
   neither, and the causal mask is not optional.
3. **Wrong positional encoding.** Its RoPE is per *token* with its own convention;
   ours is per *frame* (all 257 tokens share one rotation), which is the property
   that makes the cache valid at all (§1). Expressing it would mean modifying the
   plugin.
4. **Wrong dtype.** The fused MHA kernels are fp16/bf16/fp8/int8. fp32 — the only
   precision that has ever reached 0/200 on parity for this model family — has no
   fused path.

Plus the deployment objection: it pins the car to a TRT-LLM build matched to
JetPack 6.1 / TRT 10.7 / SM 8.7.

### What TRT 10.7 already ships, checked on the device

`trt.get_plugin_registry()` on delta-dev1 (73 creators) contains exactly one fused
attention plugin: **`CustomQKVToContextPluginDynamic` v1/v2/v3** — the BERT fMHA
plugin. It takes *packed* QKV with **equal Q and K/V sequence lengths**, i.e.
self-attention only, and its fused kernels are fp16/int8. It cannot express
257 queries against 16448 cached keys. The other candidates
(`DisentangledAttention_TRT`, `MultiscaleDeformableAttnPlugin_TRT`) are unrelated.
**So there is no off-the-shelf fused attention for this shape on this stack** —
which is why the fp16 question below is the one that decides whether a custom
plugin is worth days of CUDA.

## 12. fp16 — and the reason to keep the `Concat`

Latency-only, and deliberately so: every measurement in this document uses random
weights, so parity (skill §4/§5) is unrunnable and **nothing here is a ship
recommendation**. The question was narrow and architectural: TRT's fused MHA kernels
exist only for fp16/int8, so does myelin pick a fused path for a 257 × 16448 block
once the dtype allows it?

**It does, and it changes the answer to the whole investigation.** Same host and
methodology, `--precision fp16`, median GPU compute:

| arm | context | fp32 | **fp16** | speedup |
| --- | --- | --- | --- | --- |
| `_big` | 6 | 87.21 ms | **19.70 ms** | 4.4× |
| `_big` | 32 | 205.22 ms | **54.15 ms** | 3.8× |
| `_big` | 64 | 363.93 ms | **97.42 ms** | 3.7× |
| small | 6 | 41.78 ms | **10.49 ms** | 4.0× |
| small | 32 | 92.27 ms | **27.16 ms** | 3.4× |
| small | 64 | 160.58 ms | **45.70 ms** | 3.5× |

And the marginal cost per cached frame falls by the same factor — `_big`
**4.78 → 1.34 ms/frame**, small **2.05 → 0.61** — so the slope problem §9 identified
is a slope problem *at fp32 only*.

The `split_kt` graph does **not** get this. At `_big`/N=64 it goes 350.78 → 253.92 ms,
1.4×, and ends up **2.6× slower than the `concat` graph it was built to improve on.**

At fp16 the *entire* attention block of each trunk layer — `q · Kᵀ`, softmax, `P · V`
— collapses into **one `_gemm_mha_v2` kernel at 3.39 ms**, against 21.4 ms for the
same three kernels at fp32. That is a **6.3×** attention speedup and it is what
carries the whole 3.7×.

### The `Concat` is load-bearing

`_gemm_mha_v2` needs contiguous K and V for the full key set, which is exactly what
`cat(past_k, k)` produces. `split`/`split_kt` decompose the attention into two
matmul-softmax-matmul chains plus an online-softmax merge, and **myelin no longer
recognizes attention at all**: the fp16 `split_kt` engine has zero fused MHA for the
trunk (only 0.73 ms total, the own-frame 257 × 257 blocks), and instead 78.2 ms of
hand-rolled softmax kernels and 85 ms of slices and copies.

So the mechanism that the profile-bucket artifact pointed at — remove the
`Concat` — is not merely worth 3.6 %; **at fp16 it is a 2.6× regression.** The
`Concat` is the pattern that buys the fused kernel. This is the single most
important operational conclusion in this document:

> **Keep `cache_attention="concat"` as the default. Do not ship `split_kt`** unless
> the engine is fp32 *and* stays fp32. A 13 ms fp32 win that costs 156 ms at fp16 is
> a trap, and the serving pair in the trt-export skill §9 is explicitly fp16 for the
> trunk.

### What is left on the table at fp16

Two items, both visible in the fp16 profile of the `concat` engine (84.6 ms
profiled / 97.4 ms benchmarked):

* **19.3 ms of "Reformatting CopyNode for Input Tensor 2/3"** — TRT converting the
  fp32 `past_k`/`past_v` engine inputs into the fp16 layout the fused kernel wants,
  once per tick. **An fp16 cache would remove it**, and it halves cache memory too
  (§6): 569 MiB instead of 1138 at `_big`/64. That is the cheapest remaining ~20 %.
* **11.9 ms of KV copy** (12 × ~0.54 ms) — the same stacked-cache slice fused with
  the `Concat` as in fp32, but 4× cheaper because the tensors are half the size and
  the kernel is no longer the bottleneck. Per-layer bindings (§11) would target this,
  so their fp16 ceiling is ~12 ms, not the ~33 ms measured at fp32.

### Against the budget

`_big` at 64 frames of context, fp16, is **97.4 ms** — inside the 333 ms tick with
room for the ~55 ms DVFS and ~52 ms recorder overheads the skill documents, and
inside 270 ms as well. Every arm and context measured fits, with margin; even a
128-frame `_big` extrapolates to ~183 ms. **fp16, not paged attention, is what makes
long context servable.**

What that costs is the parity work the skill §4/§4a/§5 mandates on a real
checkpoint: margin screen first, then `--trials 200` against an fp32 control, and
per §4 the encoder pinned to fp32 (`--precision mixed --fp32-index-ranges`) rather
than the pure fp16 measured here — so treat these as the fp16 *floor*, with the
mixed engine somewhere between it and fp32.

⚠️ Two honest limits on this table. The engines were built with `--precision fp16`
from a **randomly initialized** export, so (a) nothing about accuracy is established
— this model family's `ArgMax` code head is exactly the fp16-fragile part
(skill §4/§4a), and 2 of 7 real checkpoints flip ~4 % of actions at fp16; and (b)
the only correctness statement here is that the engine produces finite, plausible
outputs (`--dumpOutput`: `policy.joint_actions` in [−0.59, 0.39], no NaN), i.e. the
speed is not an artifact of a degenerate graph.

## 13. Recommendation

Ranked by measured value per unit of risk, for the question "how do we serve long
context on the Orin".

1. **Serve fp16 for the trunk, and keep the `Concat`.** 3.7× at `_big`/N=64
   (364 → 97 ms), because the fused `_gemm_mha_v2` kernel does the whole attention
   block in one pass. Cost: the parity work the trt-export skill mandates — margin
   screen (§4a), then `--trials 200` against an fp32 control on a **real
   checkpoint**, and per §4 build it as `fp32 encoder + fp16 trunk`, not pure fp16.
   Nothing here validates numerics; these are random weights.
2. **Make the cache fp16 while you are there.** It removes the ~19 ms of per-tick
   input reformat in the fp16 engine *and* halves cache memory (§6). No accuracy
   argument needed beyond (1) — the cache holds the same activations the trunk is
   already computing in fp16.
3. **Bind the cache per layer** (`per_layer_cache=True`). −45.6 ms on the full graph
   at `_big`/N=64 in fp32 when combined with `split_kt` (364 → 318 ms, which *does*
   fit the tick); ceiling ~12 ms in fp16. Pure I/O-binding change, ONNX
   bit-identical, no numerical cost. Worth taking whenever the shape validation of
   §7 is in place, and it is strictly a serving-side change.
4. **Do not ship `split`/`split_kt`.** Worth 3.6–7.8 % at 64 frames in fp32, worth
   **−156 ms** at fp16. They stay in the tree because they are what proved the
   `Concat` is not the cost, and because they are the right formulation if a future
   engine is fp32-only — not because they should be served.
5. **Do not write a fused-attention plugin, and do not reach for TensorRT-LLM.**
   §11 has the effort estimate and the specific incompatibilities. fp16 already
   gets the fused kernel that a plugin would have been written to provide.

The honest summary of the original question — *how much of the linear-in-context
cost is avoidable by eliminating the per-tick KV concat/copy* — is **3.6 % from the
`Concat`, ~9 % more from the stacked-cache slice (12.5 % together, which does bring
`_big`/64 inside the tick at fp32), and neither is the reason 64 frames was
expensive.** The dtype was: fp16 is 3.7× on the same graph, and it needs the
`Concat` kept.

## 14. Open risks

1. **Training cost is untouched, and it is now the bottleneck.** The cache makes
   *inference* per-tick cost linear in context. Training still does one dense
   forward over the whole clip, and the attention terms are quadratic in the
   flattened length: 6 frames = 1542 tokens, 32 frames = 8224 (5.3× the tokens,
   **28×** the attention work), 64 frames = 16448 (10.7× / **114×**). The `window`
   mask makes attention block-sparse but plain SDPA still materializes the dense
   score matrix, so the saving is not realized without a block-sparse kernel
   (FlexAttention / `create_block_mask`). At `clip_length == window` the mask is
   dense anyway. **This is the single largest practical risk to the bonus track**;
   16 frames is the defensible first step, not 64.
2. **Behaviour change, not just speed** (hand-off §6). Conditioning on 16–64
   frames is a different model. Needs rsim and road, not val L1 — PR #248
   established that aggregate L1 is blind to this failure class.
3. **Train/infer mismatch if the window is not matched.** Serving with a ring of
   `N-1` is only equivalent to training if training used
   `frame_block_causal_mask(window=N)`. `CausalFrameTransformer` rejects
   `max_sequence_length < window * tokens_per_frame` (a clip shorter than the
   window trains a narrower context than will be served), but **nothing prevents
   serving a checkpoint against the wrong cache size** — the trunk has no intrinsic
   maximum length any more, so a 32-frame-trained model will happily run against a
   64-frame cache and silently extrapolate. Validate the engine's `inputs_past_k`
   shape against the checkpoint's `window` at load time.
4. **No parity/precision verification yet.** All measurements are fp32 with random
   weights, so `parity_matrix.py --trials 200` has not been and cannot be run — it
   needs a trained checkpoint. The 5 in-graph `ArgMax` nodes and the code-flip
   defect are unchanged by this work (hand-off §6), and the margin screen must be
   re-run on any real checkpoint before serving anything below fp32.
5. **RoPE base is unvalidated.** `rope_base=1000` is reasoned (base 10000 leaves
   most frequency pairs inert over 64 positions) but not measured. It is a
   trainable-arm hyperparameter, and changing it after training invalidates the
   checkpoint.
6. ~~**The `Concat` of cache and new keys is unavoidable without a plugin**, and it
   is already a first-order cost at 64 frames — the largest single bucket. A paged /
   in-place attention plugin is the escape hatch.~~ **Withdrawn — measured false
   (§10).** The `Concat` was removed from the graph outright, with no plugin, and it
   is worth **3.6 %** at `_big`/64 frames. It was fused into the stacked-cache slice
   copy, which any formulation pays. The remaining risk is the opposite one: *this
   was the cheap lever, and it is now spent.* Anything further at long context has to
   attack the 257 × 16448 score matrix, which needs either fp16 (§12) or a custom
   fused-attention plugin (days of CUDA, and a `.so` pinned to the car's TRT
   version).
7. **`_big` at 64 frames does not fit a tick at fp32** on the shipped graph (364 ms;
   351 with `split_kt`, 318 with `split_kt` + per-layer bindings, 97 at fp16). Not a
   defect, but it bounds what is servable without either the §11 restructure or the
   mixed-precision engine.
8. **`split_kt` changes the runtime contract**, and `set_tensor_address` will not
   tell you: `inputs_past_k` becomes `(L, b, H, head_dim, cache_tokens)` and `new_k`
   comes out matching. Bound the old layout, the engine silently reinterprets the
   buffer. The §7 shape validation is what catches it, and it is not optional for
   this variant.
