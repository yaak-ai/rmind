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
| correctness gate | `tests/test_causal_frame.py` (cache/positional gates on CPU; the FlexAttention fwd+bwd parity gate needs CUDA and skips without it) |
| block-sparse training benchmark | `tests/bench_causal_frame.py` |
| training arms | `config/experiment/yaak/patch_policy/dinov2_dinowm_causal.yaml`, `..._causal_80gb.yaml` |

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

The cache, mask and positional gates run on CPU in ~2 s (27 passed, 8 skipped);
with a CUDA device the 8 skips are the FlexAttention parity gates of §11.2 and the
whole file is 35 passed in ~9 s. Every equivalence is paired with a **negative
control** on a window-absolute variant of the same trunk, driven through the same
harness, so a pass is falsifiable.

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
  ~23 s;
* **the train-set SIZE barely changes**, which is worth stating because the
  opposite is the intuitive guess. The sampler is
  `rbyte.io.DataFrameGroupByDynamic(every=${episode_stride} = 10i,
  period=${clip_period}, gather_every=${episode_step})`: clip *starts* are strided
  by `every`, independently of `period`, so a longer clip does not thin the
  windows — it only truncates each drive's tail. Going from `clip_length` 11 to 37
  costs ~26 windows of ~3.1k per drive, <1 % of the ~1.97M samples (assuming short
  trailing windows are dropped; confirm on the first cache rebuild). Steps/epoch is
  therefore `~1.97M / batch_size` at any `episode_length`;
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
hand-off estimated ~75 ms for `_big`; the measurement is 87.2 ms, i.e. the
estimate was optimistic by ~16 % — close, and the reason is exactly the "no new
overhead from cache management" assumption it flagged (§8): the in-graph
`Concat` of `past_k` with the new frame's keys is real work.

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

**Where the marginal time actually goes is split, and the `Concat` is a first-order
term.** Bucketing `trtexec --dumpProfile` by kernel kind (TRT fuses and renames
everything to `__myl_*`, so the module hierarchy is *not* recoverable — this is why
the hand-off cross-checked its §1 split against an independent FLOP count instead):

| bucket | small, N=6 | small, N=64 |
| --- | --- | --- |
| MatMul (qkv / proj / MLP / head / unfused attention GEMMs) | 25.5 ms (55 %) | 62.2 ms (31 %) |
| fused kernels containing the cache `Concat` | 6.6 ms (14 %) | **68.2 ms (34 %)** |
| `scaled_dot_product_attention` | 4.1 ms (9 %) | 35.0 ms (17 %) |
| standalone fused softmax | 3.4 ms (7 %) | 30.5 ms (15 %) |
| other fused elementwise | 6.4 ms (14 %) | 6.5 ms (3 %) |
| conv2d (ViT patch embed) | 0.15 ms | 0.15 ms |

⚠️ Treat this as *indicative only*: `--dumpProfile` inflates the total (46.2 ms
against a benchmarked 41.8 ms at N=6, 202.5 against 160.5 at N=64) and it inflates
large-tensor kernels most, which is exactly the `Concat` bucket. Both sources agree
on the actionable conclusion though: **at long context the in-graph `Concat` of
`past_k` with the new frame's keys is comparable to the attention itself**, so an
in-place / paged attention plugin is the highest-value follow-up if 64+ frames are
wanted — bigger than any further caching, because the cache is already exact.

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
free (364 ms, no longer inside a tick at fp32). If 64 frames is wanted, the levers
are the fp32-encoder + fp16-trunk serving engine from the trt-export skill §9 —
which halves the cache and its traffic as well as the GEMM cost — or a
block-sparse/paged attention kernel. Not more caching: the cache is already exact.

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

## 10. Open risks

1. ~~**Training cost is untouched, and it is now the bottleneck.**~~ **Resolved —
   see §11.** `attention_impl: flex` realizes the block-sparsity the `window` mask
   always had, and with `episode_length > window` training cost becomes linear in
   context length instead of quadratic. At a constant 768 frame-slots per step,
   32-frame clips with a 16-frame window cost **1.08×** today's 6-frame trunk step
   (dense SDPA: 2.74×), and 64 frames costs 1.15×. What remains of this risk is
   narrower and listed in §11.6: `attn_dropout` must go to 0, the block-sparse
   path is CUDA-only, and it loses to SDPA below roughly 1000 frame-slots per step.
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
6. **The `Concat` of cache and new keys is unavoidable without a plugin**, and the
   profile says it is already a first-order cost at 64 frames (§9) — the largest
   single bucket, comparable to the attention. A paged / in-place attention plugin
   is the escape hatch; an in-graph `ScatterElements` deliberately is not (TRT
   support risk, and the host-side ring is free).
7. **`_big` at 64 frames does not fit a tick at fp32** (364 ms). Not a defect, but
   it bounds the context length that is servable today without the mixed-precision
   engine.

## 11. Block-sparse training — FlexAttention vs FlashAttention-2 vs dense

**Recommendation: FlexAttention (`attention_impl: flex`), and it is the only exact
option.** Everything below was measured on 2026-08-05 on an idle RTX 5090
(sm_120, torch 2.12.1+cu130) with `tests/bench_causal_frame.py`; correctness is
`tests/test_causal_frame.py`.

### 11.1 The problem

`CausalSelfAttention.forward` handed a materialized bool mask to
`F.scaled_dot_product_attention`. Any `attn_mask` disqualifies the flash backend,
so every masked position is computed anyway: cost `O((F·257)²)` no matter how
narrow `window` is. The mask, however, is blocky in 257-token frame units — dense
bidirectional blocks on the diagonal, a causal band below — which is exactly what a
block-sparse kernel wants.

### 11.2 Correctness (the gate)

fp32, tf32 disabled, production geometry (257 tokens/frame, `window` 6 and 16,
512-d/8-head and 768-d/12-head), forward **and** backward:

| tensor | max abs diff | max abs value | scale-relative |
| --- | --- | --- | --- |
| output | 1.4e-6 | 5.2 | 2.7e-7 |
| d/d input | 1.2e-6 | 5.0 | 2.4e-7 |
| d/d `in_proj_weight` | 1.8e-5 | 12.8 | 1.4e-6 |
| d/d `out_proj.weight` | 3.6e-5 | 43.3 | 8.4e-7 |
| d/d `intra_position_embedding` | 3.3e-6 | 20.3 | 1.6e-7 |

The gate is applied **scale-relative** (`max|diff| / max|ref|`), not as an absolute
1e-5, and that is a deliberate choice worth stating plainly: a parameter gradient
is a sum over all 1542–4112 sequence positions, so its entries are O(10) and its
fp32 accumulation noise alone exceeds 1e-5 in absolute terms whichever kernel
produced it. That the residual **is** that noise is shown independently, by
comparing both fp32 arms against the same trunk run in float64 on the CPU:

| | sdpa vs exact fp64 | flex vs exact fp64 |
| --- | --- | --- |
| worst tensor (scale-rel) | 6.8e-7 | 9.4e-7 |

flex sits 1.4× sdpa's own distance from the exact answer, not 10×. Parity is also
verified in `.train()`, i.e. with the compiled kernel inside
`checkpoint(use_reentrant=False)` and its recompute, and the frame-offset
shift-invariance of §1 still holds on the flex path — RoPE is applied to q/k before
attention, so the positional scheme and the kernel are orthogonal.

### 11.3 The 257-token tile-alignment finding

`create_block_mask` tiles at 128, and **128 is the only usable block size**:
`BLOCK_SIZE=64` fails to lower on sm_120/torch 2.12 (`ValueError: Q and KV block
size must be divisible by BLOCK_M and BLOCK_N`). 257 = 2·128 + 1, so a frame block
can never tile exactly and every frame boundary yields partial blocks, which the
kernel computes in full and then masks elementwise. Measured computed area against
the exact mask area:

| frames | window | BLOCK 128 | BLOCK 64 | BLOCK 32 | frame padded to 384, BLOCK 128 |
| --- | --- | --- | --- | --- | --- |
| 6 | 6 | **1.288×** | 1.137× | 1.064× | 2.233× |
| 16 | 6 | **1.191×** | 1.091× | 1.041× | 2.233× |
| 16 | 16 | **1.111×** | 1.051× | 1.022× | 2.233× |
| 32 | 16 | **1.074×** | 1.033× | 1.013× | 2.233× |
| 64 | 16 | **1.063×** | 1.027× | 1.014× | 2.233× |
| 64 | 64 | **1.023×** | 1.008× | 1.004× | 2.233× |

Two conclusions. The waste is a **boundary effect that amortizes**: 29 % at 6
frames, 6 % at 64, because the number of straddling blocks grows with `F` while the
computed area grows with `F·window`. And **padding each frame to 384 to align it is
strictly worse** — exactly `(384/257)² = 2.233×` at every geometry, since it pays
1.49× more queries against 1.49× more keys to remove a ≤29 % overhead. Reordering
tokens so frames tile 128 would need the 256 patches kept together and the speed
token moved elsewhere; not worth it against a 6–11 % overhead in the regime that
matters. The BLOCK 64/32 columns are what the hardware would allow if the kernel
did — they are not available, and they would trade tile efficiency for the
alignment anyway.

### 11.4 Kernel benchmark — one attention layer, fwd+bwd (RoPE included), bf16

`tests/bench_causal_frame.py attention --batches 4,16`:

#### 512-d / 8 heads

| frames | window | seq | b4 sdpa | b4 flex | b16 sdpa | b16 flex | b16 peak sdpa / flex |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | 6 | 1542 | 1.26 ms | **1.40** | 4.24 | **2.11** | 396 / 350 MiB |
| 16 | 6 | 4112 | 6.68 | **1.75** | 25.05 | **6.97** | 1083 / 933 |
| 16 | 16 | 4112 | 6.68 | **2.47** | 25.06 | **9.51** | 1083 / 933 |
| 32 | 8 | 8224 | 24.32 | **4.05** | 94.71 | **16.97** | 2263 / 1866 |
| 32 | 16 | 8224 | 24.35 | **5.96** | 94.95 | **24.44** | 2263 / 1866 |
| 32 | 32 | 8224 | 24.37 | **7.68** | 95.16 | **30.70** | 2263 / 1866 |
| 64 | 16 | 16448 | 95.32 | **13.53** | 377.40 | **54.65** | 4914 / 3745 |
| 64 | 64 | 16448 | 95.70 | **27.37** | 379.92 | **108.25** | 4914 / 3745 |

#### 768-d / 12 heads

| frames | window | seq | b4 sdpa | b4 flex | b16 sdpa | b16 flex | b16 peak sdpa / flex |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | 6 | 1542 | 1.85 ms | **1.26** | 6.34 | **3.15** | 594 / 525 MiB |
| 16 | 6 | 4112 | 9.66 | **2.53** | 37.29 | **10.59** | 1602 / 1399 |
| 16 | 16 | 4112 | 9.66 | **3.55** | 37.26 | **14.33** | 1602 / 1399 |
| 32 | 8 | 8224 | 36.00 | **6.18** | 142.08 | **25.50** | 3295 / 2798 |
| 32 | 16 | 8224 | 36.08 | **9.04** | 142.43 | **36.67** | 3295 / 2798 |
| 32 | 32 | 8224 | 36.11 | **11.50** | 142.69 | **45.85** | 3295 / 2798 |
| 64 | 16 | 16448 | 142.25 | **20.38** | 568.75 | **82.35** | 6979 / 5602 |
| 64 | 64 | 16448 | 143.18 | **40.81** | 572.26 | **163.13** | 6979 / 5602 |

Three things to read off. **Dense SDPA is flat in `window`** — 24.32 / 24.35 / 24.37
ms across windows 8/16/32 at 32 frames — which is the defect, stated as a
measurement: the mask is doing nothing for cost. **flex is proportional to
`F·window`**: 6.97 → 9.51 ms as the window goes 6 → 16 at 16 frames, and 54.65 ms
at 64 frames/window 16 against 377.40 dense, a **6.9×** kernel speedup. And **flex
uses less memory**, 24 % less at 64 frames, because there is no dense mask tensor
and fewer score tiles in flight.

The one row where flex loses is batch 4 at 6 frames (1.40 vs 1.26 ms): 13 kv blocks
and 4 sequences cannot fill a 5090. It wins by 2× at the same geometry with batch
16. See §11.6.5.

For scale: the frozen DINOv2 ViT-S forward over the 768 images a step consumes
costs **79.9 ms / 2031 MiB** on the same GPU (`bench_causal_frame.py vit
--batches 768`). It is linear in frame-slots, identical for every row of §11.5, and
about a tenth of the trunk step — so the trunk really is the term worth attacking.

### 11.5 What the training step actually costs

Attention is only part of a step; the MLP is linear in tokens and dilutes the
saving, and the frozen ViT is linear in `batch × frames` and is untouched. The row
that answers "is long-context training affordable" is the **full trunk**, in
`.train()` (gradient checkpointing on), normalized to a constant **768 frame-slots
per step** — today's `batch 128 × 6 frames`, which is what fixes the ViT cost, the
readouts per step and the activation memory:

| frames | window | 512-d/8L sdpa | 512-d/8L flex | vs today | 768-d/12L sdpa | 768-d/12L flex | vs today |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | 6 | 652 ms | 497 ms | 0.76× | 1695 ms | 1323 ms | 0.78× |
| 16 | 6 | 1085 | 550 | 0.84× | 2649 | 1447 | 0.85× |
| 16 | 16 | 1085 | 626 | 0.96× | 2652 | 1616 | 0.95× |
| 32 | 8 | 1787 | 593 | 0.91× | 4223 | 1538 | 0.91× |
| 32 | 16 | 1787 | **705** | **1.08×** | 4237 | **1792** | **1.06×** |
| 32 | 32 | 1791 | 800 | 1.23× | 4245 | 2001 | 1.18× |
| 64 | 16 | 3215 | **749** | **1.15×** | 7469 | **1897** | **1.12×** |
| 64 | 64 | 3233 | 1151 | 1.77× | 7501 | 2809 | 1.66× |

("vs today" is against the 6-frame step of **the same trunk with
`attention_impl: sdpa`**, 652 / 1695 ms — the honest denominator for "what does
flex buy". It is not literally the production `BlockCausalTransformer`, which has
the same widths and the same dense-mask cost structure but no per-layer RoPE.
Reproduce with `tests/bench_causal_frame.py trunk`.)

**32-frame training lands inside today's 6-frame budget**: 1.08× at 512-d, 1.06× at
768-d, against 2.74× / 2.50× if the mask stays dense. 64 frames is 1.15× / 1.12×.
The cost of context has gone from quadratic to essentially free, and what is left
is set by `window`, not by clip length: `32/8` (0.91×) is *cheaper* than today.

And the recipe point itself, measured directly at the batch each arm actually
uses — no normalization, and one fresh process per cell
(`bench_causal_frame.py trunk --only 32x16 --batches 24`, plus the same for the
6-frame/batch-128 baseline: a shared process that has already peaked at tens of GiB
fragments the allocator for whatever runs next, which shows up as a spurious OOM):

| arm | geometry | batch | impl | step | trunk peak |
| --- | --- | --- | --- | --- | --- |
| 512-d / 8L | 6 frames, window 6 | 128 | sdpa (today) | 670.7 ms | 9.60 GiB |
| 512-d / 8L | **32 frames, window 16** | **24** | **flex** | **701.7 ms** | **9.60 GiB** |
| 768-d / 12L | 6 frames, window 6 | 128 | sdpa (today) | 1713.5 ms | 16.80 GiB |
| 768-d / 12L | **32 frames, window 16** | **24** | **flex** | **1779.6 ms** | **16.79 GiB** |

**1.05× the step time and the same memory to the last 10 MiB, for 5.3× the
context.** Both arms. That is the whole result, and it agrees with the normalized
table above to within 3 %, which is the check on the normalization.

Peak activation memory at a constant 768 frame-slots is flat because the trunk's
memory is dominated by the per-token residual/MLP activations that gradient
checkpointing keeps, and those scale with `batch × frames`, not with either factor
alone. Block-sparsity is not the memory lever (flex is only 2–5 % under sdpa at
equal shape: 9825 vs 10017 MiB at 32 frames/batch 24, 22.8 vs 23.6 GiB at 64
frames/batch 16) — **holding frame-slots constant is**.

For the 80 GB arm, batch 64 × 32 frames scales to ~25.6 GiB (512-d) / ~44.8 GiB
(768-d) of trunk activations. It **OOMs on a 32 GB 5090** — measured, in a fresh
process — which is exactly why it is a separate yaml. Note also that the first
compile of a new shape needs headroom *beyond* steady state (inductor benchmarks
several kernel configs while the activations are live), so leave more margin than
the steady-state figure suggests.

The recipe built from this is
`config/experiment/yaak/patch_policy/dinov2_dinowm_causal.yaml` (32 frames, window
16, batch 24, 5090-32 GB) and `..._80gb.yaml` (batch 64); both carry their own
step-count arithmetic, because the cosine LR schedule oscillates past
`num_training_steps`.

### 11.6 Operational hazards (all of these bit or nearly bit)

1. **No `dropout_p` in FlexAttention.** `attn_dropout` must be 0 on the flex path;
   the constructor raises rather than silently dropping it. This is a
   regularization change riding along with a kernel change — an A/B against
   today's arm must zero `attn_dropout` on the sdpa side too.
2. **An in-place op in a `mask_mod` kills the path entirely.** Inductor lowers a
   `mask_mod` as a pointwise subgraph, in which no buffer may be created, so
   `keep &= x` fails with `SubgraphLoweringException: Buffers cannot be created
   while lowering a pointwise subgraph` — at *every* shape. `ruff check` in this
   repo runs with `fix = true, unsafe-fixes = true` and rewrites
   `keep = keep & x` into `keep &= x` under PLR6104, which is exactly that. It
   happened during this work and cost a benchmark run.
   `test_mask_mod_is_free_of_in_place_ops` guards it on CPU.
3. **dynamo's 8-entry compile cache.** One compiled `flex_attention` serves every
   shape and `dynamic=False` specializes per shape; past 8 specializations dynamo
   silently falls back to **eager** flex, which materializes the score matrix. Fine
   in training (2–3 shapes), fatal in a sweep — raise
   `torch._dynamo.config.cache_size_limit` there.
4. **Build the `BlockMask` once.** `create_block_mask` costs 1.6–9 ms; per layer per
   step that is ~25 ms of pure launch overhead at 8 layers, more than the attention
   it feeds. `frame_block_causal_block_mask` is memoized on
   `(num_frames, tokens_per_frame, window, device)`.
5. **flex is not free at small shapes.** At batch 4 / 6 frames the whole trunk is
   *slower* under flex (58.7 vs 52.6 ms at 512-d; 84.2 vs 71.6 at 768-d): 13 kv
   blocks and a batch of 4 do not fill the GPU. The crossover is around 1000
   frame-slots per step; below that, keep `sdpa`. This is why the default stays
   `sdpa` and the flex path is opt-in per experiment.
6. **CUDA only.** There is no CPU backward for FlexAttention, so the CPU path is
   the eager reference implementation (correct, not fast) and the fwd+bwd parity
   tests are CUDA-gated. A CPU-only CI run skips them; it still runs the mask-mod,
   block-accounting and CPU-forward-parity tests.
7. **Serving is untouched, on purpose.** `step` stays SDPA: it is the ONNX/TRT
   export target (a `torch.compile`d Triton kernel is not exportable) and it has
   257 queries against a fully-visible cache, i.e. no sparsity to exploit.
   `attention_impl` therefore cannot change what an exported engine computes.

### 11.7 Why not FlashAttention-2

`flash_attn` is not installed in the repo venv and installing it was out of scope,
so this is an analytic rejection — but it does not depend on a measurement, because
the mismatch is in what FA2 can *express*.

FA2's windowed attention is **token-granular and strictly causal**
(`window_size=(left, right)` over token indices). Our mask is frame-granular with
**bidirectional** intra-frame blocks. The closest approximation is a token window of
`window · 257`, and it is wrong in two independent ways. Counting mask entries:

| frames | window | exact keeps | missing | of which intra-frame future | spurious |
| --- | --- | --- | --- | --- | --- |
| 16 | 6 | 5,349,969 | 526,336 (9.8 %) | **100 %** | 328,960 (6.1 %) |
| 32 | 16 | 25,891,208 | 1,052,672 (4.1 %) | **100 %** | 526,336 (2.0 %) |
| 64 | 16 | 59,708,296 | 2,105,344 (3.5 %) | **100 %** | 1,579,008 (2.6 %) |

* Every missing pair is an **intra-frame future** pair: the current frame's 257
  tokens stop being mutually visible, so a patch at intra-frame position 3 cannot
  see position 200 of its own frame. That destroys the one property the whole
  positional scheme is built around (§1: frame-granular RoPE exists so that
  intra-frame attention is *exactly unrotated* and stays bidirectional), and it
  makes the trunk a raster-order scanner over patches rather than a per-frame
  encoder.
* The spurious pairs are a **ragged window edge**: a query late in its frame still
  reaches keys of frame `f - window`, which the ring buffer has already evicted.
  That breaks the streaming/recompute equivalence of §5 — the training mask would
  no longer be any ring capacity's mask, so there is no `N` to serve with.

`is_causal` at "frame level" via reshaping does not exist either: attention would
have to be causal in one index and dense in another within a single softmax, and no
reshape of `(B, H, S, D)` produces that. Options: (b) is not expressible; (a) is
expressible but changes the model into a different one and forfeits the correctness
gate; therefore (c) reject. Note the comparison is not even a performance question —
torch's own flash backend with plain `is_causal` is the lower bound FA2 could reach,
and flex at 64 frames/window 16 already computes only 23.5 % of the dense area.

### 11.8 Not done

**Readout-only final block during training.** Only each frame's last token is read
by the heads, so the final layer could run `F` queries instead of `F·257` — worth
about `1/L` of attention plus `1/L` of MLP, ~10 % of the trunk at `L = 8`. It is
expressible in FlexAttention (`Q_LEN = F`, `KV_LEN = F·257`, `mask_mod` mapping the
query index straight to a frame index), and §4 already proves the equivalence for
the decode path. Not wired up, because it changes `PatchPolicy`'s readout indexing
and the win is small next to the 2.5–4× already realized.
