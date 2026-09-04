# Decoder-only PatchPolicy with a bounded KV cache

Implements `/nasa/max/docs/decoder_only_handoff.md`. Turns the block-causal
temporal trunk (fixed 6-frame sliding window, recomputed from scratch every tick)
into a causal decoder over frame blocks with a reusable, bounded KV cache.

Per tick the model encodes **one** new frame per camera (`tokens_per_frame = cam * 256 + 1`: 1 speed token prepended to each camera's 256 goal-fused patch
tokens, `cam = len(policy.cameras)` -- 257 at `cam=1`, 769 at `cam=3`), runs
those queries against the cached K/V of past frames, and never recomputes or
re-attends old frames to each other.

**Everything measured in this doc (§6, §9, §11, §12) is `cam=1`** (257
tokens/frame, the production arm at time of writing) unless a section says
otherwise. `tokens_per_frame` scaling non-linearly affects attention cost (it's
quadratic in the intra-frame width, linear in the inter-frame one), so none of
those latency/memory/margin numbers can be scaled by "×cam" -- they need
re-measuring at whatever camera count is actually served
(`config/experiment/yaak/patch_policy/dinov2_dinowm_causal_3cam.yaml`).

|                                 | file                                                                                                                                |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| trunk + mask + RoPE + cache     | `src/rmind/components/transformer/causal_frame.py`                                                                                  |
| one-tick export wrapper         | `src/rmind/models/patch_policy_decoder.py`                                                                                          |
| correctness gate                | `tests/test_causal_frame.py` (cache/positional gates on CPU; the FlexAttention fwd+bwd parity gate needs CUDA and skips without it) |
| block-sparse training benchmark | `tests/bench_causal_frame.py`                                                                                                       |
| training arms                   | `config/experiment/yaak/patch_policy/dinov2_dinowm_causal.yaml`, `..._causal_80gb.yaml`                                             |

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

- intra-frame attention is **exactly unrotated**, `(R_f q)ᵀ(R_f k) = qᵀk`, so it
  stays fully bidirectional and is ordered only by the intra-frame embedding;
- inter-frame logits depend only on `f_q − f_k`, so a key rotated with its own
  episode-absolute frame index is valid **forever**, at any future window
  position. That is precisely cache-safety.

`rope_base = 1000`, not the customary 10000: over a 64-frame range, base 10000
leaves most frequency pairs rotating \<0.01 rad and therefore inert.

**Why not ALiBi.** ALiBi is equally cache-safe and, being a pure additive bias,
folds into the existing mask with no new ops. It remains the better choice for a
low-precision *serving* engine, since RoPE normally introduces `Sin`/`Cos` and the
trt-export skill flags trigonometric ops as the most fp16-fragile part of this
model family. Two reasons it is not the default here:

1. RoPE's `Sin`/`Cos` are avoided anyway — `rope_cos`/`rope_sin` are **graph
   inputs**, computed host-side in float64 from the episode frame counter. The
   exported engine contains zero trigonometric nodes, so ALiBi's advantage
   evaporates.
1. ALiBi's bias is per-head and query-dependent, so the training path must
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

Single stacked tensor per side (`k = tokens_per_frame = cam * 256 + 1`):

```
past_k, past_v : (num_layers, batch, num_heads, cache_frames * k, head_dim)
new_k,  new_v  : (num_layers, batch, num_heads,               k, head_dim)
cache_bias     : (1, 1, 1, cache_frames * k)   0 = filled, -1e4 = empty
rope_cos/sin   : (1, head_dim)
```

`past_k`/`past_v` are **read-only** graph inputs; only the new frame's K/V come
out. **The ring buffer lives in the host, not the graph.** No in-graph scatter, no
`ScatterElements` in TRT, and the cache is ordinary engine I/O. Between ticks the
host shifts by one frame block and writes `k` tokens per layer.

ONNX input names (`torch.export` flattens the `Mapping` argument): one
`inputs_image_<camera>` per configured camera (`policy.cameras`, e.g.
`inputs_image_cam_front_left` alone at `cam=1`, plus
`inputs_image_cam_left_forward` / `inputs_image_cam_right_forward` at `cam=3`)
-- a flat binding per camera rather than a nested mapping, so every camera is an
ordinary TRT/ONNX I/O tensor like the ones below. Plus `inputs_speed`,
`inputs_waypoints`, `inputs_past_k`, `inputs_past_v`, `inputs_cache_bias`,
`inputs_rope_cos`, `inputs_rope_sin`. Outputs: `policy.joint_actions`, `new_k`,
`new_v`. Camera **order** (`policy.cameras`) is load-bearing -- it's the order
the trained `patch_projection` saw, not a config convenience the host can
reshuffle.

Two silent-failure hazards on the drivr side:

- `TRTEngine.run` binds via `set_tensor_address` — a raw pointer with **no size
  validation** (§3.4). A cache allocated for a different `cache_frames`, layer
  count, head count or dtype is not an error; TRT reinterprets the buffer and the
  model merely looks weak. **Validate every binding against
  `engine.get_tensor_shape(name)` before the first `run`.**
- Cache and frame counter must be **reset on every episode boundary** — engage,
  disengage, manual override. drivr already clears the action plan on those
  transitions; hook the same paths. A stale cache is not detectable from the
  output.

`Resize=0` and in-graph ImageNet normalization are unchanged, so the host still
owes an exact 224×224 (dinov2) `[0,1]` frame **per camera** and `--image-norm unit`.

## 4. Gather before the final block (hand-off §3.3)

The head reads one token per frame, so in the final trunk block the attention
output and the MLP for the other `tokens_per_frame - 1` positions are
discarded.
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

| configuration                                           | float64  | float32                    |
| ------------------------------------------------------- | -------- | -------------------------- |
| small 8L/512d, window 6, T=7                            | 1.78e-15 | 1.43e-06 (3.2e-07 rel)     |
| `_big` 12L/768d, window 6, T=7                          | 2.67e-15 | 1.55e-06 (4.0e-07 rel)     |
| `_big` 12L/768d, window 32, T=40                        | 4.44e-15 | 1.91e-06 (4.2e-07 rel)     |
| **negative control** (window-absolute, small, window 6) | —        | **9.37e-02 (2.1e-02 rel)** |

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

| arm               | parameters | fp32 TRT engine | ONNX initializers |
| ----------------- | ---------- | --------------- | ----------------- |
| small (8L/512d)   | 52.6 M     | **207.2 MiB**   | 213.0 MB          |
| `_big` (12L/768d) | ~115 M     | **440.2 MiB**   | 453.3 MB          |

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
| ---------------- | ----------- | ---------- | ----------- | ---------- | ----------- |
| 6                | 1285        | 40 MiB     | 90 MiB      | 20 MiB     | 45 MiB      |
| 16               | 3855        | 120 MiB    | 271 MiB     | 60 MiB     | 136 MiB     |
| 32               | 7967        | 249 MiB    | 560 MiB     | 124 MiB    | 280 MiB     |
| 64               | 16191       | 506 MiB    | 1138 MiB    | 253 MiB    | 569 MiB     |
| 128              | 32639       | 1020 MiB   | 2295 MiB    | 510 MiB    | 1147 MiB    |

### Total resident

delta-dev1 has **15 GiB** of LPDDR5 shared between CPU and GPU, ~12 GiB
practically available (measured `free`). Weights + cache, fp32:

|        | 6 frames | 16      | 32           | 64       |
| ------ | -------- | ------- | ------------ | -------- |
| small  | 247 MiB  | 327 MiB | 456 MiB      | 713 MiB  |
| `_big` | 530 MiB  | 711 MiB | **1000 MiB** | 1578 MiB |

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
1. **Cache not reset at an episode boundary** — the model conditions on frames
   from before the disengage. The output stays plausible. Hook the paths that
   already clear the action plan.
1. **`frame_index` not monotone** (e.g. reset mid-episode, or wrapped) — RoPE
   offsets become wrong for the frames still in the ring. Reset the counter
   **only** together with the cache.

Optional but cheap: assert the number of valid slots in `cache_bias` equals
`min(frame_index, cache_frames) * 257`. That single invariant catches all three.

## 8. rbyte / dataset considerations

`clip_length = episode_length + clip_horizon - 1`, so moving from 6 to 16 frames
takes clips from 11 to 21 samples, and 64 frames would need 69. Consequences:

- the dataset must be **rebuilt** — `clip_period` is derived from `clip_length`
  and the existing clip-11 build is not reusable;
- at `episode_step = 10` (≈3 Hz), 21 samples span ~7 s of driving and 69 span
  ~23 s;
- **the train-set SIZE barely changes**, which is worth stating because the
  opposite is the intuitive guess. The sampler is
  `rbyte.io.DataFrameGroupByDynamic(every=${episode_stride} = 10i, period=${clip_period}, gather_every=${episode_step})`: clip *starts* are strided
  by `every`, independently of `period`, so a longer clip does not thin the
  windows — it only truncates each drive's tail. Going from `clip_length` 11 to 37
  costs ~26 windows of ~3.1k per drive, \<1 % of the ~1.97M samples (assuming short
  trailing windows are dropped; confirm on the first cache rebuild). Steps/epoch is
  therefore `~1.97M / batch_size` at any `episode_length`;
- per-sample decode/IO grows linearly with `clip_length`, so the loader becomes a
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

| arm               | measured here | hand-off §1 | delta  |
| ----------------- | ------------- | ----------- | ------ |
| small (8L/512d)   | **200.85 ms** | 194.8 ms    | +3.1 % |
| `_big` (12L/768d) | **420.16 ms** | 448.8 ms    | −6.4 % |

Close enough that the comparison is sound, and it confirms `_big` is **12** layers
(§7 says 8; an 8-layer 768-d trunk could not cost 420 ms). Speedups below are
quoted against *these* numbers, measured on the same host on the same day through
the same export path, not against the hand-off's.

**Decoder step, per tick, fp32, median GPU compute:**

| context (frames) | small (8L/512d) | vs 6-frame baseline | `_big` (12L/768d) | vs 6-frame baseline |
| ---------------- | --------------- | ------------------- | ----------------- | ------------------- |
| 6                | **41.79 ms**    | **4.81×**           | **87.21 ms**      | **4.82×**           |
| 16               | 59.58 ms        | 3.37×               | 127.79 ms         | 3.29×               |
| 32               | 92.26 ms        | 2.18×               | 205.05 ms         | 2.05×               |
| 64               | 160.51 ms       | 1.25×               | 364.22 ms         | 1.15×               |

At the same 6 frames of context the step is **4.8× cheaper in both arms**. The
hand-off estimated ~75 ms for `_big`; the measurement is 87.2 ms, i.e. the
estimate was optimistic by ~16 % — close, and the reason is exactly the "no new
overhead from cache management" assumption it flagged (§8): the in-graph
`Concat` of `past_k` with the new frame's keys is real work.

### Correcting the hand-off: per-tick cost does NOT stop scaling with context

§2 claims "per-tick cost stops scaling with window length". It does not. The step
runs 257 queries against `(N-1) × 257 + 257` keys, so both the attention terms
and the cache traffic are **linear in N**. Measured, over 6 → 64 frames:

|        | fixed cost (N→1) | marginal cost per extra frame | R² of the linear fit |
| ------ | ---------------- | ----------------------------- | -------------------- |
| small  | ~31.5 ms         | **2.05 ms/frame**             | >0.999               |
| `_big` | ~63.3 ms         | **4.78 ms/frame**             | >0.999               |

The slope tracks `L × D` and nothing else, which makes it predictable: the extra
attention work per cached frame is `2 × 257 × 257 × D × 2 × L` FLOPs — 1.08 GFLOP
(small), 2.43 GFLOP (`_big`) — and the two measured slopes correspond to the same
**~510–530 GFLOP/s effective** (~16 % of the module's ~3.3 TFLOP/s fp32 peak) in
both arms.

**Where the marginal time actually goes is split, and the `Concat` is a first-order
term.** Bucketing `trtexec --dumpProfile` by kernel kind (TRT fuses and renames
everything to `__myl_*`, so the module hierarchy is *not* recoverable — this is why
the hand-off cross-checked its §1 split against an independent FLOP count instead):

| bucket                                                     | small, N=6     | small, N=64        |
| ---------------------------------------------------------- | -------------- | ------------------ |
| MatMul (qkv / proj / MLP / head / unfused attention GEMMs) | 25.5 ms (55 %) | 62.2 ms (31 %)     |
| fused kernels containing the cache `Concat`                | 6.6 ms (14 %)  | **68.2 ms (34 %)** |
| `scaled_dot_product_attention`                             | 4.1 ms (9 %)   | 35.0 ms (17 %)     |
| standalone fused softmax                                   | 3.4 ms (7 %)   | 30.5 ms (15 %)     |
| other fused elementwise                                    | 6.4 ms (14 %)  | 6.5 ms (3 %)       |
| conv2d (ViT patch embed)                                   | 0.15 ms        | 0.15 ms            |

⚠️ Treat this as *indicative only*: `--dumpProfile` inflates the total (46.2 ms
against a benchmarked 41.8 ms at N=6, 202.5 against 160.5 at N=64) and it inflates
large-tensor kernels most, which is exactly the `Concat` bucket. Both sources agree
on the actionable conclusion though: **at long context the in-graph `Concat` of
`past_k` with the new frame's keys is comparable to the attention itself**, so an
in-place / paged attention plugin is the highest-value follow-up if 64+ frames are
wanted — bigger than any further caching, because the cache is already exact.

What *does* improve superlinearly is the comparison against recomputing an
N-frame window, which is quadratic. The right way to state the prize:

- **`_big` attends to 32 frames for 205 ms — less than half of what it costs today
  to attend to 6** (420 ms). Recomputing a 32-frame window block-causally would be
  ~28× the trunk work of the 6-frame one, i.e. several seconds.
- **small attends to 32 frames for 92 ms, versus 201 ms today for 6.**
- Every configuration measured except `_big` at 64 frames (364 ms) fits inside one
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
1. **Behaviour change, not just speed** (hand-off §6). Conditioning on 16–64
   frames is a different model. Needs rsim and road, not val L1 — PR #248
   established that aggregate L1 is blind to this failure class.
1. **Train/infer mismatch if the window is not matched.** Serving with a ring of
   `N-1` is only equivalent to training if training used
   `frame_block_causal_mask(window=N)`. `CausalFrameTransformer` rejects
   `max_sequence_length < window * tokens_per_frame` (a clip shorter than the
   window trains a narrower context than will be served), but **nothing prevents
   serving a checkpoint against the wrong cache size** — the trunk has no intrinsic
   maximum length any more, so a 32-frame-trained model will happily run against a
   64-frame cache and silently extrapolate. Validate the engine's `inputs_past_k`
   shape against the checkpoint's `window` at load time.
1. ~~**No parity/precision verification yet.**~~ **Resolved for the first trained
   checkpoint — see §12.** The margin screen and a 200-trial decision-parity ladder
   have now been run on `do8m9ot8:v0`. What remains of this risk is narrower and
   unavoidable: margins are **per-checkpoint**, so §12 must be re-run for every
   checkpoint before it is served below fp32. A verdict cannot be inherited from a
   sibling — 2 of 7 checkpoints in the baseline family genuinely flip ~4 % while
   their own parents pass.
1. **RoPE base is unvalidated.** `rope_base=1000` is reasoned (base 10000 leaves
   most frequency pairs inert over 64 positions) but not measured. It is a
   trainable-arm hyperparameter, and changing it after training invalidates the
   checkpoint.
1. **The `Concat` of cache and new keys is unavoidable without a plugin**, and the
   profile says it is already a first-order cost at 64 frames (§9) — the largest
   single bucket, comparable to the attention. A paged / in-place attention plugin
   is the escape hatch; an in-graph `ScatterElements` deliberately is not (TRT
   support risk, and the host-side ring is free).
1. **`_big` at 64 frames does not fit a tick at fp32** (364 ms). Not a defect, but
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

| tensor                         | max abs diff | max abs value | scale-relative |
| ------------------------------ | ------------ | ------------- | -------------- |
| output                         | 1.4e-6       | 5.2           | 2.7e-7         |
| d/d input                      | 1.2e-6       | 5.0           | 2.4e-7         |
| d/d `in_proj_weight`           | 1.8e-5       | 12.8          | 1.4e-6         |
| d/d `out_proj.weight`          | 3.6e-5       | 43.3          | 8.4e-7         |
| d/d intra-frame position table | 3.3e-6       | 20.3          | 1.6e-7         |

Measured on the flat intra-frame position table, whose single parameter is
`intra_position_embedding.weight`. The gate now enumerates the table's parameters
from the trunk (`CausalFrameTransformer.intra_position_parameters()`) and compares
each, so it covers the factorized arms' `view_position_embedding` /
`patch_position_embedding` / `special_position_embedding` too; on a flat trunk
that set is exactly the one row above.

The gate is applied **scale-relative** (`max|diff| / max|ref|`), not as an absolute
1e-5, and that is a deliberate choice worth stating plainly: a parameter gradient
is a sum over all 1542–4112 sequence positions, so its entries are O(10) and its
fp32 accumulation noise alone exceeds 1e-5 in absolute terms whichever kernel
produced it. That the residual **is** that noise is shown independently, by
comparing both fp32 arms against the same trunk run in float64 on the CPU:

|                          | sdpa vs exact fp64 | flex vs exact fp64 |
| ------------------------ | ------------------ | ------------------ |
| worst tensor (scale-rel) | 6.8e-7             | 9.4e-7             |

flex sits 1.4× sdpa's own distance from the exact answer, not 10×. Parity is also
verified in `.train()`, i.e. with the compiled kernel inside
`checkpoint(use_reentrant=False)` and its recompute, and the frame-offset
shift-invariance of §1 still holds on the flex path — RoPE is applied to q/k before
attention, so the positional scheme and the kernel are orthogonal.

### 11.3 The 257-token tile-alignment finding

`create_block_mask` tiles at 128, and **128 is the only usable block size**:
`BLOCK_SIZE=64` fails to lower on sm_120/torch 2.12 (`ValueError: Q and KV block size must be divisible by BLOCK_M and BLOCK_N`). 257 = 2·128 + 1, so a frame block
can never tile exactly and every frame boundary yields partial blocks, which the
kernel computes in full and then masks elementwise. Measured computed area against
the exact mask area:

| frames | window | BLOCK 128  | BLOCK 64 | BLOCK 32 | frame padded to 384, BLOCK 128 |
| ------ | ------ | ---------- | -------- | -------- | ------------------------------ |
| 6      | 6      | **1.288×** | 1.137×   | 1.064×   | 2.233×                         |
| 16     | 6      | **1.191×** | 1.091×   | 1.041×   | 2.233×                         |
| 16     | 16     | **1.111×** | 1.051×   | 1.022×   | 2.233×                         |
| 32     | 16     | **1.074×** | 1.033×   | 1.013×   | 2.233×                         |
| 64     | 16     | **1.063×** | 1.027×   | 1.014×   | 2.233×                         |
| 64     | 64     | **1.023×** | 1.008×   | 1.004×   | 2.233×                         |

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

| frames | window | seq   | b4 sdpa | b4 flex   | b16 sdpa | b16 flex   | b16 peak sdpa / flex |
| ------ | ------ | ----- | ------- | --------- | -------- | ---------- | -------------------- |
| 6      | 6      | 1542  | 1.26 ms | **1.40**  | 4.24     | **2.11**   | 396 / 350 MiB        |
| 16     | 6      | 4112  | 6.68    | **1.75**  | 25.05    | **6.97**   | 1083 / 933           |
| 16     | 16     | 4112  | 6.68    | **2.47**  | 25.06    | **9.51**   | 1083 / 933           |
| 32     | 8      | 8224  | 24.32   | **4.05**  | 94.71    | **16.97**  | 2263 / 1866          |
| 32     | 16     | 8224  | 24.35   | **5.96**  | 94.95    | **24.44**  | 2263 / 1866          |
| 32     | 32     | 8224  | 24.37   | **7.68**  | 95.16    | **30.70**  | 2263 / 1866          |
| 64     | 16     | 16448 | 95.32   | **13.53** | 377.40   | **54.65**  | 4914 / 3745          |
| 64     | 64     | 16448 | 95.70   | **27.37** | 379.92   | **108.25** | 4914 / 3745          |

#### 768-d / 12 heads

| frames | window | seq   | b4 sdpa | b4 flex   | b16 sdpa | b16 flex   | b16 peak sdpa / flex |
| ------ | ------ | ----- | ------- | --------- | -------- | ---------- | -------------------- |
| 6      | 6      | 1542  | 1.85 ms | **1.26**  | 6.34     | **3.15**   | 594 / 525 MiB        |
| 16     | 6      | 4112  | 9.66    | **2.53**  | 37.29    | **10.59**  | 1602 / 1399          |
| 16     | 16     | 4112  | 9.66    | **3.55**  | 37.26    | **14.33**  | 1602 / 1399          |
| 32     | 8      | 8224  | 36.00   | **6.18**  | 142.08   | **25.50**  | 3295 / 2798          |
| 32     | 16     | 8224  | 36.08   | **9.04**  | 142.43   | **36.67**  | 3295 / 2798          |
| 32     | 32     | 8224  | 36.11   | **11.50** | 142.69   | **45.85**  | 3295 / 2798          |
| 64     | 16     | 16448 | 142.25  | **20.38** | 568.75   | **82.35**  | 6979 / 5602          |
| 64     | 64     | 16448 | 143.18  | **40.81** | 572.26   | **163.13** | 6979 / 5602          |

Three things to read off. **Dense SDPA is flat in `window`** — 24.32 / 24.35 / 24.37
ms across windows 8/16/32 at 32 frames — which is the defect, stated as a
measurement: the mask is doing nothing for cost. **flex is proportional to
`F·window`**: 6.97 → 9.51 ms as the window goes 6 → 16 at 16 frames, and 54.65 ms
at 64 frames/window 16 against 377.40 dense, a **6.9×** kernel speedup. And **flex
uses less memory**, 24 % less at 64 frames, because there is no dense mask tensor
and fewer score tiles in flight.

The one row where flex loses is batch 4 at 6 frames (1.40 vs 1.26 ms): 13 kv blocks
and 4 sequences cannot fill a 5090. It wins by 2× at the same geometry with batch
16\. See §11.6.5.

For scale: the frozen DINOv2 ViT-S forward over the 768 images a step consumes
costs **79.9 ms / 2031 MiB** on the same GPU (`bench_causal_frame.py vit --batches 768`). It is linear in frame-slots, identical for every row of §11.5, and
about a tenth of the trunk step — so the trunk really is the term worth attacking.

### 11.5 What the training step actually costs

Attention is only part of a step; the MLP is linear in tokens and dilutes the
saving, and the frozen ViT is linear in `batch × frames` and is untouched. The row
that answers "is long-context training affordable" is the **full trunk**, in
`.train()` (gradient checkpointing on), normalized to a constant **768 frame-slots
per step** — today's `batch 128 × 6 frames`, which is what fixes the ViT cost, the
readouts per step and the activation memory:

| frames | window | 512-d/8L sdpa | 512-d/8L flex | vs today  | 768-d/12L sdpa | 768-d/12L flex | vs today  |
| ------ | ------ | ------------- | ------------- | --------- | -------------- | -------------- | --------- |
| 6      | 6      | 652 ms        | 497 ms        | 0.76×     | 1695 ms        | 1323 ms        | 0.78×     |
| 16     | 6      | 1085          | 550           | 0.84×     | 2649           | 1447           | 0.85×     |
| 16     | 16     | 1085          | 626           | 0.96×     | 2652           | 1616           | 0.95×     |
| 32     | 8      | 1787          | 593           | 0.91×     | 4223           | 1538           | 0.91×     |
| 32     | 16     | 1787          | **705**       | **1.08×** | 4237           | **1792**       | **1.06×** |
| 32     | 32     | 1791          | 800           | 1.23×     | 4245           | 2001           | 1.18×     |
| 64     | 16     | 3215          | **749**       | **1.15×** | 7469           | **1897**       | **1.12×** |
| 64     | 64     | 3233          | 1151          | 1.77×     | 7501           | 2809           | 1.66×     |

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

| arm         | geometry                 | batch  | impl         | step          | trunk peak    |
| ----------- | ------------------------ | ------ | ------------ | ------------- | ------------- |
| 512-d / 8L  | 6 frames, window 6       | 128    | sdpa (today) | 670.7 ms      | 9.60 GiB      |
| 512-d / 8L  | **32 frames, window 16** | **24** | **flex**     | **701.7 ms**  | **9.60 GiB**  |
| 768-d / 12L | 6 frames, window 6       | 128    | sdpa (today) | 1713.5 ms     | 16.80 GiB     |
| 768-d / 12L | **32 frames, window 16** | **24** | **flex**     | **1779.6 ms** | **16.79 GiB** |

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
1. **An in-place op in a `mask_mod` kills the path entirely.** Inductor lowers a
   `mask_mod` as a pointwise subgraph, in which no buffer may be created, so
   `keep &= x` fails with `SubgraphLoweringException: Buffers cannot be created while lowering a pointwise subgraph` — at *every* shape. `ruff check` in this
   repo runs with `fix = true, unsafe-fixes = true` and rewrites
   `keep = keep & x` into `keep &= x` under PLR6104, which is exactly that. It
   happened during this work and cost a benchmark run.
   `test_mask_mod_is_free_of_in_place_ops` guards it on CPU.
1. **dynamo's 8-entry compile cache.** One compiled `flex_attention` serves every
   shape and `dynamic=False` specializes per shape; past 8 specializations dynamo
   silently falls back to **eager** flex, which materializes the score matrix. Fine
   in training (2–3 shapes), fatal in a sweep — raise
   `torch._dynamo.config.cache_size_limit` there.
1. **Build the `BlockMask` once.** `create_block_mask` costs 1.6–9 ms; per layer per
   step that is ~25 ms of pure launch overhead at 8 layers, more than the attention
   it feeds. `frame_block_causal_block_mask` is memoized on
   `(num_frames, tokens_per_frame, window, device)`.
1. **flex is not free at small shapes.** At batch 4 / 6 frames the whole trunk is
   *slower* under flex (58.7 vs 52.6 ms at 512-d; 84.2 vs 71.6 at 768-d): 13 kv
   blocks and a batch of 4 do not fill the GPU. The crossover is around 1000
   frame-slots per step; below that, keep `sdpa`. This is why the default stays
   `sdpa` and the flex path is opt-in per experiment.
1. **CUDA only.** There is no CPU backward for FlexAttention, so the CPU path is
   the eager reference implementation (correct, not fast) and the fwd+bwd parity
   tests are CUDA-gated. A CPU-only CI run skips them; it still runs the mask-mod,
   block-accounting and CPU-forward-parity tests.
1. **Serving is untouched, on purpose.** `step` stays SDPA: it is the ONNX/TRT
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

| frames | window | exact keeps | missing           | of which intra-frame future | spurious          |
| ------ | ------ | ----------- | ----------------- | --------------------------- | ----------------- |
| 16     | 6      | 5,349,969   | 526,336 (9.8 %)   | **100 %**                   | 328,960 (6.1 %)   |
| 32     | 16     | 25,891,208  | 1,052,672 (4.1 %) | **100 %**                   | 526,336 (2.0 %)   |
| 64     | 16     | 59,708,296  | 2,105,344 (3.5 %) | **100 %**                   | 1,579,008 (2.6 %) |

- Every missing pair is an **intra-frame future** pair: the current frame's 257
  tokens stop being mutually visible, so a patch at intra-frame position 3 cannot
  see position 200 of its own frame. That destroys the one property the whole
  positional scheme is built around (§1: frame-granular RoPE exists so that
  intra-frame attention is *exactly unrotated* and stays bidirectional), and it
  makes the trunk a raster-order scanner over patches rather than a per-frame
  encoder.
- The spurious pairs are a **ragged window edge**: a query late in its frame still
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

## 12. Trained-checkpoint verification (resolves §10.4)

Everything above §11 was measured with **randomly initialized** weights, which is
sufficient for latency and memory but cannot speak to precision: an `ArgMax`
margin is a property of what the head learned. This section is the first
measurement of this architecture with a real checkpoint.

**Checkpoint.** `yaak/rmind/model-do8m9ot8:v0` — run `do8m9ot8` ("vague-shape-587"),
**epoch 0, step 80797**, `dinov2_dinowm_causal`: 8L/512d/8H, `window: 16`,
`rope_base: 1000`, `fusion_norm: true`, `attention_impl: flex`, 52.78 M
parameters. It is past warmup and **far from converged**. That is fine for
everything here — latency depends on shapes, and parity depends on weight
*statistics* — but **no quality conclusion may be drawn from it**.

`attention_impl: flex` does not reach the export: `CausalFrameTransformer.step`
is always SDPA (§11.6.7), so the trained trunk exports through the same path a
`sdpa` checkpoint would.

Tooling: `rmind.scripts.decoder_only_export --artifact ...` (the architecture then
comes from the checkpoint's hparams and the trunk is used as trained) and
`rmind.scripts.decoder_only_verify {gates,onnx-vs-eager,margins}`.

### 12.1 The equivalence gate still holds with trained weights

Same gate as §5, on the trained trunk at its trained `window=16`, T=17 frames
(one frame past a full window, so the sliding case is exercised):

|                                              | max abs  | scale-relative |
| -------------------------------------------- | -------- | -------------- |
| float64                                      | 2.66e-15 | **6.41e-16**   |
| float32                                      | 1.19e-06 | **2.87e-07**   |
| negative control (RoPE counter not advanced) | 2.29e-01 | **5.50e-02**   |

float64 sits at machine epsilon, so the equivalence is **exact** and the float32
residual is purely accumulation order — the same conclusion as §5, and the
float32 figure (2.87e-07) lands on §5's random-weight 3.2e-07. The negative
control is five orders of magnitude worse, so the gate is falsifiable rather
than vacuous. Note the control chosen here is the *operational* one — a runtime
that resets the cache but forgets `frame_index` (§7 failure mode 3) — rather
than §5's architectural window-absolute trunk.

### 12.2 ONNX agrees with eager

ORT CPU fp32 vs PyTorch, warm cache, 3 input sets. Scored as **absolute error
against each tensor's own scale**, never per-element relative:

| output                 | max abs  | scale-relative |
| ---------------------- | -------- | -------------- |
| `policy.joint_actions` | 8.9e-09  | 2.6e-06        |
| `new_k`                | 1.55e-05 | 2.2e-06        |
| `new_v`                | 1.13e-05 | 3.1e-06        |

Worst 3.11e-06 scale-relative, reproducing §5's random-weight 1.4e-05 absolute on
the new K/V. `torch.onnx.export(verify=True)` again reports a "large relative
difference" of ~2.7 on `new_k`; it is again the documented false alarm — per-element
relative error on near-zero entries.

### 12.3 Latency: the random-weight architecture numbers were right

Rebuilt and re-benchmarked on delta-dev1, TRT 10.7.0.23, **GPU clock pinned at
918 MHz** (`governor=performance`, verified: every clock sample taken *through*
each benchmark was 918 MHz), load ~0.1, no other users. `trtexec --iterations=60 --avgRuns=20 --useSpinWait --warmUp=1000`, median GPU compute.

**fp32 here means TF32 CLEARED** (`--noTF32`). trtexec's default "fp32" is the
TF32 tensor-core path — a different numeric configuration, and the one that has
never been decision-exact.

| context         | fp32 trained | fp32 predicted (§9, random) | fp16 trained | fp16 predicted | fp16 speedup |
| --------------- | ------------ | --------------------------- | ------------ | -------------- | ------------ |
| 6               | **41.79 ms** | 41.79                       | **10.55 ms** | 10.5           | 3.96×        |
| 16 (**served**) | **59.82 ms** | 59.58                       | **16.97 ms** | —              | **3.53×**    |
| 32              | **92.20 ms** | 92.26                       | **27.19 ms** | 27.2           | 3.39×        |

**Every random-weight prediction was confirmed, none corrected** — the largest
disagreement anywhere in the table is 0.4 % (59.82 vs 59.58 at window 16) and the
6- and 32-frame cells agree to 0.1 % or better. So §9's central claim, that random
weights are sufficient to measure this *architecture*, is now itself measured
rather than argued.

The linear-in-context correction of §9 also survives. Least-squares over the three
contexts:

|                        | marginal cost per extra frame | fixed cost (N→1) |
| ---------------------- | ----------------------------- | ---------------- |
| fp32 trained           | **1.95 ms/frame**             | 31.5 ms          |
| fp32 §9 (random, 6→64) | 2.05 ms/frame                 | ~31.5 ms         |
| fp16 trained           | **0.64 ms/frame**             | 7.4 ms           |

Per-tick cost still does **not** stop scaling with context — §2's claim remains
wrong — but fp16 cuts the *slope* by 3.0× as well as the fixed term by 4.3×.

⚠️ **Do not explain that slope drop as halved cache bandwidth.** All three fp16
engines in this fit are plain `--fp16` builds, and §12.7 shows those keep their KV
cache in **float32** — there is no halved data width anywhere in their cache path.
The marginal term is cache traffic *and* attention against `(N-1)·257` keys, and
only the latter moved to tensor cores, so the mechanism is not established by these
three points. Separating the two would need fp16-cache engines at 6 and 32 frames,
which were not built; at window 16 the fp16 cache is worth a further 3.4 ms (§12.7),
which is a fixed-cost saving at that one context, not a slope measurement.

The **controlled** statement is the same-host, same-day, same-flags randomly
initialized control, exported from the same script at the same context and built
minutes apart:

| window 16 | trained  | random control | delta |
| --------- | -------- | -------------- | ----- |
| fp32      | 59.82 ms | 59.65 ms       | 0.3 % |
| fp16      | 16.97 ms | 16.93 ms       | 0.2 % |

Weight values do not move a static-shape engine's latency — now measured rather
than assumed, and measured at *both* precisions, which matters because fp16
tactic selection could in principle have been value-sensitive. (§9's 59.58 ms is
separate, weaker corroboration: a different build path, built four days earlier.)

fp16 is **3.53× faster than fp32 at window 16** (59.82 → 16.97 ms) and the
engine is half the size (203 → 107 MiB).

### 12.4 The fused MHA kernel is present — the fp16 win is real

`_gemm_mha_v2` was the thing to check, because without it the entire fp16
speedup evaporates, and because the in-graph `Concat` of `past_k` with the new
frame's keys is **load-bearing** for the fusion (§9, §10.6) — it must not be
"optimized away".

Built with `--profilingVerbosity=detailed` so tactic names survive into the
engine (profiling a default-verbosity engine returns stripped names, and
absence-of-name would read as absence-of-kernel — a false alarm on exactly this
check). Grepped in four places:

| source                                    | `_gemm_mha_v2` name occurrences                    |
| ----------------------------------------- | -------------------------------------------------- |
| fp16 build log                            | 38                                                 |
| fp16 `--exportLayerInfo`                  | 38                                                 |
| fp16 `--exportProfile`                    | 19                                                 |
| fp16 profile log                          | 57                                                 |
| **fp32 build log / layer info / profile** | **0** (expected: it is an fp16 tensor-core kernel) |

**Verdict: present.** No alarm.

⚠️ Those are **name occurrences, not kernel counts**, and they are not comparable
between engines or between files: detailed verbosity emits hash-suffixed tactic
names that recur several times per kernel in one dump, and an engine built at
default verbosity has them partly stripped. The attributed figure — the one to
quote — comes from the profile JSON, where each instance carries its own time:
**19 instances, 5.34 ms, 29.4 % of the step** at window 16 fp16, i.e. the fused
attention is the single largest bucket in the graph.

The same profile shows standalone `cat`/`concat` kernels at **0.00 ms**. The
`Concat` has not vanished — TRT absorbed it *into* `_gemm_mha_v2`, which is exactly
why §9 and §10.6 call it load-bearing. Removing it to "save a copy" would take the
fusion with it.

### 12.5 Margin screen — and which `ArgMax` is actually yours

Five in-graph `ArgMax` nodes, screened with `decoder_only_verify margins`
(ORT-only, no GPU, 25 trials over 3 real camera frames). `margin = top1 − top2`
expressed in **fp16 ULPs at that magnitude**: below ~1 ULP the two codes are not
distinguishable in fp16 at all.

Running the identical screen on the **randomly initialized** export of the same
graph separates the frozen subgraph from the trained one — the skill's trick, and
it works exactly as advertised:

| probe           | decisions/trial | trained min ULP | random min ULP | trained median | random median |
| --------------- | --------------- | --------------- | -------------- | -------------- | ------------- |
| `node_argmax`   | 1               | 2.688           | **2.688**      | 412.9          | **412.9**     |
| `node_argmax_1` | 1               | 11.557          | **11.557**     | 44.4           | **44.4**      |
| `node_argmax_2` | 1               | 2.875           | **2.875**      | 49.3           | **49.3**      |
| `node_argmax_3` | 1               | 3.317           | **3.317**      | 41.1           | **41.1**      |
| `node_argmax_4` | **4**           | **1.938**       | 21.362         | 162.9          | 678.7         |

Probes 0–3 are **byte-identical** between a trained checkpoint and a random one,
so they are the **frozen waypoints-tokenizer RVQ** — four quantizer stages, one
decision each, fixed for every checkpoint in this family. They are not a
checkpoint risk and they are not fixable by training. `node_argmax_4` emits four
decisions per trial and is the only probe that moved: it is the **trained VQ-BeT
`code_head`** (4 quantizers × 16 codes).

So the number that matters is the code head's, and reading the aggregate
`min ULP` is misleading — the aggregate floor of 2.688 belongs to a frozen table.

**The trained code head, at epoch 0:** min **1.94 ULP**, p1 4.22, median 162.9,
**0 % of decisions under 1 ULP**, 1 % under 4. Against the skill's validated
thresholds (`<1 ULP` predicted every measured fp16 failure across 16 checkpoints;
`≥4 ULP` has never failed) that is the **marginal** band: not a predicted failure,
but low precision must be verified at n≥200 rather than assumed.

Worth noting for the training side: **training has tightened these margins**, from
21.4 ULP min / 679 median at initialization to 1.9 / 163 after 80.8 k steps. That is
the tie geometry the code-flip defect is made of, forming early. It is a quantity
worth tracking per checkpoint, and this is a baseline for it.

### 12.6 The parity ladder — n=200, real frames, decision changes

`decoder_parity_orin.py`, run on delta-dev1 so the reference and the engines see
byte-identical inputs. 10 histories, each a genuinely **streamed** 15-frame warm
cache (cold start, ring slot writes, monotone frame counter — what drivr does),
× 20 current-frame/speed/waypoint variations = 200 trials. Reference is ORT CPU
fp32 on the same ONNX. A trial counts as a **decision change** when any action
channel moves more than 0.02 — float noise is ~1e-4 here and a flipped code is
~0.1, so the threshold is not delicate.

| engine                  | latency  | decision changes  | of which control-channel only | worst \|d\| | mean \|d\| |
| ----------------------- | -------- | ----------------- | ----------------------------- | ----------- | ---------- |
| **fp32** (TF32 cleared) | 59.82 ms | **0/200**         | 0                             | 4.2e-07     | 1.8e-08    |
| fp16                    | 16.97 ms | **5/200** (2.5 %) | **5**                         | 0.1827      | 5.0e-04    |

**fp32 is 0/200, so the harness is sound** — that is the skill's gate on the
harness itself, and it passes. Its residual is 4.2e-07 max, i.e. fp32 round-off
between ORT CPU and TRT, not a modelling difference.

Per channel, fp16 vs the fp32 reference:

| channel            | max \|d\|  | mean \|d\| |
| ------------------ | ---------- | ---------- |
| gas_pedal          | 0.1065     | 1.18e-03   |
| brake_pedal        | 0.0068     | 6.02e-05   |
| **steering_angle** | **0.1827** | 7.09e-04   |
| turn_signal        | 0.0041     | 5.46e-05   |

**This arm's fp16 failure is different in kind from the baseline's, and worse.**
On the block-causal baseline the headline magnitude was entirely `turn_signal`
(0.546) — an indicator state, not a trajectory. Here `turn_signal` never even
crosses tolerance (max 0.0041) and **all five flips are control channels**: up to
**18.3 % of steering range** and **10.6 % of throttle** on 2.5 % of plans. There is
no "it was only the indicator" reading available.

### 12.7 KV-cache memory — the halving is real but is NOT the default

§6's arithmetic is confirmed against the actual graph: at window 16 the cache is
`8 layers × 15 frames × 257 × 512 × 4 B × 2 (K,V)` = **120.47 MiB**, and the
engine's own `inputs_past_k` binding reports exactly that.

The part §6 did not say is that **a plain `--fp16` build does not halve it.** TRT
preserves the dtypes the ONNX declares for network *I/O* and casts internally, so
the fp16 engine's cache bindings come back **float32**:

| engine                | `inputs_past_k` dtype | KV cache       | latency      |
| --------------------- | --------------------- | -------------- | ------------ |
| fp32                  | float32               | 120.47 MiB     | 59.82 ms     |
| fp16 (plain `--fp16`) | **float32**           | **120.47 MiB** | 16.97 ms     |
| fp16 + fp16 cache I/O | float16               | **60.23 MiB**  | **13.57 ms** |

The halving has to be *asked for*, per tensor:

```
trtexec --onnx=M.onnx --fp16 \
  --inputIOFormats=fp32:chw,fp32:chw,fp32:chw,fp16:chw,fp16:chw,fp16:chw,fp32:chw,fp32:chw \
  --outputIOFormats=fp32:chw,fp16:chw,fp16:chw
```

i.e. `past_k`/`past_v`/`cache_bias` and `new_k`/`new_v` in fp16, while the camera
frame, speed, waypoints, RoPE and the action chunk stay fp32 — so the only host
contract that changes is the cache itself.

**And it is not only memory: it is 20 % of the step.** Profiling the plain fp16
engine (`--dumpProfile`, detailed verbosity) the two most expensive kernels in the
whole graph are not compute at all:

| kernel                                                | time        | share  |
| ----------------------------------------------------- | ----------- | ------ |
| `Reformatting CopyNode for Input Tensor 3` (`past_k`) | 1.264 ms    | 7.0 %  |
| `Reformatting CopyNode for Input Tensor 2` (`past_v`) | 1.250 ms    | 6.9 %  |
| 19 × `_gemm_mha_v2` (fused MHA, the actual attention) | 5.34 ms     | 29.4 % |
| kernels naming `cat`/`concat`                         | **0.00 ms** | 0 %    |

2.51 ms — 14 % of the step — is spent converting a 120 MiB fp32 cache to fp16
every tick, and it disappears when the cache is handed over as fp16 already:
16.97 → 13.57 ms, a 3.40 ms saving that matches the reformat cost plus the halved
cache traffic. **So the default `--fp16` build leaves 60 MiB and 3.4 ms on the
table for nothing.**

Note also the `Concat` line: **0 ms as a standalone kernel.** It has not
disappeared — TRT has absorbed it *into* `_gemm_mha_v2`, which is precisely why §9
called it load-bearing for the fusion. Do not "optimize" it away.

Total resident at the served window 16, measured: 203 MiB of fp32 weights +
120 MiB cache = **323 MiB** (§6 predicted 327 MiB), or 107 MiB + 60 MiB =
**167 MiB** on the fp16 + fp16-cache engine. Memory remains nowhere near binding
on a 15 GiB Orin.

### 12.8 The serving decision

All four engines at the served window 16, on the same host at 918 MHz, scored
against one shared ORT fp32 reference over the same 200 trials:

| engine                    | latency      | vs fp32   | engine  | KV cache   | decisions | worst \|d\| | mean \|d\| | `_gemm_mha_v2` |
| ------------------------- | ------------ | --------- | ------- | ---------- | --------- | ----------- | ---------- | -------------- |
| **fp32** (TF32 cleared)   | 59.82 ms     | 1.00×     | 203 MiB | 120 MiB    | **0/200** | 4.2e-07     | 1.8e-08    | n/a            |
| `encfp32-fp16trunk`       | 21.79 ms     | 2.75×     | 155 MiB | 120 MiB    | 2/200     | 0.0427      | 1.2e-04    | present        |
| fp16                      | 16.97 ms     | 3.53×     | 107 MiB | 120 MiB    | 5/200     | 0.1827      | 5.0e-04    | present        |
| **fp16 + fp16 cache I/O** | **13.63 ms** | **4.39×** | 105 MiB | **60 MiB** | 2/200     | 0.0426      | 1.6e-04    | present        |

**Serve fp32.** Not because low precision looks bad, but because at this window
there is no latency pressure to trade anything for: 59.82 ms is **3.36× cheaper
than the 200.85 ms 6-frame baseline that is served today** (§9's gate zero), while
attending 16 frames instead of 6. The tick carrying an inference costs
`333 + 59.8 = 393 ms` against today's `333 + 200.9 = 534 ms`. fp32 is the only
0/200 configuration and it is *also* the faster-than-today one, so accepting even
1 % of flipped control decisions buys nothing.

**If a faster engine is ever needed** — the `_big` arm, window 32/64, or a tighter
tick — the answer for *this* architecture is **fp16 with fp16 cache I/O**, and that
is a **departure from the skill's standing recommendation**. `encfp32-fp16trunk` is
the skill's designated "fast one", but here it is dominated on every axis at once:

|                     | latency      | KV cache   | decisions | worst \|d\| |
| ------------------- | ------------ | ---------- | --------- | ----------- |
| `encfp32-fp16trunk` | 21.79 ms     | 120 MiB    | 2/200     | 0.0427      |
| fp16 + fp16 cache   | **13.63 ms** | **60 MiB** | 2/200     | 0.0426      |

1.6× faster, half the cache, indistinguishable parity. The reason is structural
and specific to the decoder step: **the step encodes only ONE frame**, so the image
encoder is 69 % of the graph's layers (1950 of 2818) instead of the ~18 % of
runtime it is in the recompute-everything baseline. Pinning it is no longer cheap.
Meanwhile the fp16 cache lever does not exist in the baseline at all, because the
baseline has no cache.

Two honesty notes on those parity counts:

- **2/200 vs 5/200 is not a distinguishable difference.** The 95 % intervals
  overlap (≈0.1–3.6 % against ≈0.8–5.7 %), and the skill's own warning about
  2/50 vs 4/50 applies with less force but still applies. What *is* a clean signal
  is the magnitude, which is not a counting statistic: worst |d| falls 4.3×
  (0.183 → 0.043) and mean |d| 3–4× when either the encoder is pinned or the
  cache is handed over in fp16. Both flagged trials in both better engines are
  control-channel-only.
- ⚠️ **The fp16-cache parity number is optimistic and must not be shipped on.** The
  harness streams each history through ORT in **fp32** and casts to fp16 once per
  trial. A real fp16 ring accumulates the cache *in fp16 across every tick*, so
  rounding compounds over an episode in a way this measures nothing about. Before
  serving that engine, re-run the ladder with a genuinely fp16 ring — the harness
  needs one change, to keep `past_k`/`past_v` in fp16 between ticks.

**The fusion is present in every fp16-containing engine**, and that is the whole
verdict — do not read the raw hit counts in the table as kernel counts. They are
not comparable across engines: `build_mixed.py` never sets
`profiling_verbosity`, so the mixed engine was built at TRT's default and its
tactic names are partly stripped, while the trtexec builds used
`--profilingVerbosity=detailed` and emit hash-suffixed names that appear several
times per kernel in the same dump. The one quantity that is defensible is from the
profile JSON of the plain fp16 engine, where instances and times are attributed:
**19 instances, 5.34 ms, 29.4 % of the step**.

Two traps this check walked into, both worth remembering:

- `build_mixed.py`'s **build log showed 0 hits**, purely because its TRT logger is
  not verbose. Reading that as "the fusion is gone" would have fired the alarm on
  the one check that matters. The engine has to be interrogated with
  `--dumpLayerInfo` / `--dumpProfile`, never only the build log.
- An earlier version of `decoder_only_trt_measure.sh` grepped `[fm]mha`, which
  matches `fmha`/`mmha` but **not** `_gemm_mha_v2`, so its "here is what fused MHA
  exists" fallback printed an empty list next to a primary check reporting 38 hits.
  Fixed to `mha`.

### 12.9 What is NOT established

- **Nothing about driving quality.** epoch 0, step 80797. Latency and parity are
  properties of shapes and weight statistics; behaviour is not, and PR #248
  established that aggregate val L1 is blind to this failure class anyway.
- **Windows 6 and 32 are latency artifacts only.** They were built from a
  window-16 checkpoint, which `step` will run against any cache size while
  silently extrapolating (§10.3). The engines on delta-dev1 are renamed
  `LATENCY-ONLY-WRONG-WINDOW.*` with a `PARITY-NOTES.md` beside them.
- **The harness shares 10 caches across 200 trials** (20 current-frame variants
  each) rather than using 200 independent histories, because a cache is 126 MiB.
  The cache is the slowest-moving real input, but this is a genuine narrowing of
  the input distribution.
- **Speed and waypoints are synthetic**, which makes this harness ~7× harsher than
  the road on the baseline arm. Do not quote 2/200 as a deployment rate.
- **Margins are per-checkpoint.** Re-run §12.5 and §12.6 for every checkpoint
  before serving it below fp32. A verdict cannot be inherited from a sibling.

## 13. cam=3 training OOMs: `window` is not the lever, `batch_size` is

`ead564b` ("add cam_left_forward/cam_right_forward to the causal PatchPolicy
arm") generalized `PatchPolicy` from a single `image` path to a `cameras: tuple[str, ...]` hparam, and added `dinov2_dinowm_causal_3cam.yaml` on top of
`dinov2_dinowm_causal.yaml`. Per its own comment, the change is contained to
`tokens_per_frame`: `256 * num_cameras + 1` (257 -> 769 at `cam=3`) and
`max_sequence_length` — "the trunk itself needed no changes, since it only ever
sees an opaque `tokens_per_frame`". Everything else (`window: 16`,
`episode_length: 32`, `batch_size: 24`, `attention_impl: flex`) is inherited
unmodified from the cam=1 arm, whose §11.5 recipe was measured — not
guessed — at `tokens_per_frame=257`. §0's warning ("none of those
latency/memory/margin numbers can be scaled by ×cam") is exactly what bit here:
the `batch_size: 24` survived the inheritance uncorrected, 3× too large for
`tokens_per_frame=769`.

### 13.1 What the run actually hit

wandb `alex-tmp/035euhok` / container `71ee898ee9fc`, command:

```
just train-unsafe experiment=yaak/patch_policy/dinov2_dinowm_causal_3cam \
  ++datamodule.train.batch_size=16 datamodule.train.num_workers=2 \
  wandb.project=alex-tmp paths.rbyte.cache=.rbyte_cache_32step \
  "++wandb.tags=[patch_policy,dinov2_dinowm_causal,3cam,window_10]"
```

Two things about this command before the OOM itself:

- **The `window_10` tag is a false record.** No `model.encoder.window=10`
  override is in the command, so the run trained at the inherited `window: 16`
  — the tag describes an experiment that was not actually run. Caught from the
  container's own `docker inspect .Config.Cmd`, not from wandb config (worth
  checking wandb's logged `model/encoder/window` on any run before trusting a
  tag).
- **`datamodule.val.batch_size` was never overridden.** `train_3cam.yaml` sets
  it to `32` (the cam=1-tuned value, same as `train.yaml`) and nothing in
  `dinov2_dinowm_causal_3cam.yaml` or the command touches it.

The traceback (`docker logs 71ee898ee9fc`) OOMs inside `on_validation_batch_end`
-> `PatchPolicy.predict_step` -> `CausalFrameTransformer.forward`, on a
`[24608, 64]` tensor (`24608 = 32 (val batch) × 769 (tokens_per_frame)`) trying
to allocate 6.01 GiB with only 4.78 GiB free — **25.82 GiB was already
allocated by PyTorch before validation's own forward pass finished**. So this
was not a training-step OOM; training at `batch=16` survived. Validation, still
at the untouched default of `batch=32`, is what didn't.

### 13.2 Measured: `window` does not move peak memory at fixed batch

Re-ran the exact arm (512d/8H/8L, `tokens_per_frame=769`, `episode_length=32`,
`attention_impl=flex`, bf16, gradient checkpointing, AdamW) inside
`rmind:b900e8e...` on the same RTX 5090, one fresh container per data point
(a shared process re-triggers dynamo's 8-shape recompile-limit fallback to
eager `flex_attention` and the allocator-fragmentation trap §11.5 already
warns about — both hit and discarded before these numbers):

| window | batch | train step peak (fwd+bwd+ckpt+AdamW) |
| ------ | ----- | ------------------------------------ |
| 16     | 24    | **OOM**                              |
| 16     | 20    | 27792 MiB                            |
| 16     | 16    | 22311 MiB                            |
| 16     | 12    | 16830 MiB                            |
| 16     | 8     | 11348 MiB                            |
| 10     | 24    | **OOM**                              |
| 10     | 20    | 27793 MiB                            |
| 10     | 16    | **22311 MiB**                        |

`window: 16 -> 10` changes peak memory by **0 MiB** at every batch tested
(22311 vs 22311 at batch 16; 27792 vs 27793 at batch 20 — noise). This is §11.5
restated, not contradicted: *"holding frame-slots constant is \[the memory
lever\]... block-sparsity is worth only another 2-5%."* `window` gates how many
past frames the block-sparse mask lets each frame attend to — it moves
*attention compute* (§9's linear-in-context slope) — but under gradient
checkpointing the dominant term is the per-token residual/MLP activation at
checkpoint boundaries, which scales with `batch × frames × tokens_per_frame`
and has no `window` in it at all. **Dropping `window` to 10 will not fix this
OOM.** It's a legitimate lever for serving latency/context depth (§9) or for a
deliberate train/infer behavior change (§10.2) — not for VRAM.

### 13.3 Measured: what validation actually costs

Isolated no-`grad` forward, trunk only (excludes the frozen ViT and every other
`PatchPolicy` head, so these are a *floor*, not the full validation cost):

| val batch | peak (no_grad forward, trunk only) |
| --------- | ---------------------------------- |
| 32        | 17874 MiB                          |
| 8         | 7037 MiB                           |

`val.batch_size=32` alone needs almost 18 GiB just in the trunk, on top of
whatever training already left resident — which is exactly the shape of the
crash: `train.batch_size=16` alone peaks at 22.3 GiB (§13.2), and the real
pipeline (ViT + trunk + VQ-BeT heads) landed at the observed 26.54 GiB before
validation added its own forward on top of that.

### 13.4 The fix

Neither knob that was actually touched (`train.batch_size=16`, an untouched
`window_10` tag) addressed the real problem. Two batch sizes need scaling down
together, matched to `cam=3`'s 3× `tokens_per_frame`, not to `window`:

```
just train-unsafe experiment=yaak/patch_policy/dinov2_dinowm_causal_3cam \
  datamodule.train.batch_size=8 \
  datamodule.val.batch_size=8 \
  datamodule.train.num_workers=8 \
  wandb.project=alex-tmp \
  "++wandb.tags=[patch_policy,dinov2_dinowm_causal,3cam,docker]"
```

`batch=8` trunk-only peaks at 11.3 GiB train / 7.0 GiB val (§13.2/§13.3);
adding the ViT forward and heads on top of either has multiple GiB of slack
before the 31.36 GiB the container actually reports as available (the driver
and other host processes already account for the gap to the card's nominal
32 GiB). `batch=12` (16.8 GiB train) is a plausible next step up if a run at 8
shows comfortable headroom in practice, but `batch=16` measured only ~4.8 GiB
free *before* validation ran at all, which is too tight to also survive a val
pass safely — that margin, not `window`, is what a real fix has to buy back.
If throughput at low batch is a problem, `+trainer.accumulate_grad_batches=N`
recovers the effective batch size without adding peak activation memory
(gradient accumulation reuses the same activation buffer per micro-batch);
the schedule arithmetic (`lr_total_steps`, `lr_warmup_steps`) still needs
rederiving against whatever batch is finally used, same as `_80gb.yaml` does
for its batch bump.

## 14. Register-compression measurement gate (Step 0)

Per-camera register compression (plans/effervescent-chasing-seahorse.md)
proposes shrinking the 3-cam causal frame block from 769 tokens
(`speed + 3×256` patches, §13's arm) to 50 (`speed + 3×16` DrivoR-style
registers + 1 readout), at the cost of making the ViT backward-capable
(LoRA rank 32 on `attn.qkv`/`attn.proj`, 16 trainable registers per camera).
Before writing any model code, this measures whether that trade is worth it:
does the ViT's new backward cost less than the trunk saves, on the actual
training GPU?

Machine: **RTX 5090, 32607 MiB** (`nvidia-smi`), the "32 GB card" §13.4
sizes `batch_size: 8` against. Commands and full harness in
`tests/bench_causal_frame.py` (`vit --mode register`, `trunk --tokens-per-frame`). Everything below is bf16, `iters=5, warmup=2`,
geometry `32 frames / window 16` (`attention_impl: flex`, per
`dinov2_dinowm_causal.yaml:230`), the arm's actual settings.

### 14.1 ViT: frozen forward vs LoRA + register fwd+bwd

| images | mode                   | ckpt | ms      | peak MiB |
| ------ | ---------------------- | ---- | ------- | -------- |
| 768    | frozen (no_grad)       | --   | 78.83   | 2031     |
| 768    | LoRA+register, fwd+bwd | on   | 378.29  | 5279     |
| 768    | LoRA+register, fwd+bwd | off  | 272.34  | 22542    |
| 2304   | frozen (no_grad)       | --   | 331.58  | 5944     |
| 2304   | LoRA+register, fwd+bwd | on   | 1595.86 | 15615    |
| 2304   | LoRA+register, fwd+bwd | off  | OOM     | --       |

768/79.9ms/2031MiB reproduces §11.4's frozen-ViT reference number almost
exactly -- the harness change is measuring what it says it's measuring.
Grad checkpointing is load-bearing exactly as predicted: without it, even
768 images costs 22.5 GiB (all 12 blocks' activations resident at once) and
2304 images (`batch 24 × 32 frames × 3 cams`, the configured training shape)
OOMs outright. With it, both fit comfortably.

### 14.2 Trunk: `tokens_per_frame` 769 vs 50, `flex` attention

At `batch=8` (§13.4's actual working `train.batch_size`):

| tpf | width     | ms      | peak MiB |
| --- | --------- | ------- | -------- |
| 769 | 512/8/8   | 1208.26 | 9804     |
| 50  | 512/8/8   | 72.43   | 795      |
| 769 | 768/12/12 | 2927.97 | 17158    |
| 50  | 768/12/12 | 93.51   | 1503     |

769/17158 MiB (sdpa: 11535 MiB, see raw log) is consistent with §13.2's
11.35 GiB at batch 8. `tokens_per_frame=50` cuts trunk step time
17-31x and peak memory 11-12x at this geometry.

At `batch=24` (the base config's stated `batch_size`, never actually
reachable per the config comment §13 already flags): `tpf=769` **OOMs
outright**, both `sdpa` and `flex`, at both widths. `tpf=50` runs easily
(2054-3677 MiB peak, 91-323 ms).

### 14.3 Gate verdict

Full step (ViT + trunk) at `batch=8`, `768/12/12` (the wider, more
conservative width):

- **Today**: `78.83 + 2927.97 = 3006.8 ms`, ViT+trunk peak ~2.0 + 17.2 = 19.2
  GiB.
- **Proposed**: `378.29 + 93.51 = 471.8 ms`, ~5.3 + 1.5 = 6.8 GiB.
- **Ratio: 0.16x -- a 6.4x net speedup**, not the hoped-for "flat", and
  comfortably inside the plan's ≤1.5x threshold. `512/8/8` gives the same
  direction (1287.1 ms -> 450.7 ms, 2.9x).

At `batch=24` the comparison is even more one-sided: today's arm cannot run
at all on this card (OOM), while the proposed arm fits with several GiB of
headroom (`1595.86 + 248.32 = 1844.2 ms`, ~15.6 + 3.7 = 19.3 GiB peak, well
under 32607 MiB) -- i.e. register compression, even paying for a trainable
ViT, is what would let the 3-cam causal arm's *configured* batch size
actually run on a 32 GB card, which it does not do today (§13.4).

**PASS.** Proceed to the full implementation
(plans/effervescent-chasing-seahorse.md, steps 1-6): LoRA port into
`src/rmind/components/lora.py`, generalized backbone, `PatchPolicy` opt-in
trainable encoder, `SelectiveAdamW` cases, new experiment config.

Freeze-contract spot check (`_BenchRegisterViT`, 223 params): 49 trainable,
all and only `lora_A`/`lora_B`/`reg_token` -- matches the intended freeze
contract with no mismatches.

Not measured here (deferred to the real port): the accuracy side of the
trade (DrivoR's own 86.9-vs-90.0 frozen-vs-LoRA gap), export/serving parity,
and camera-identity regression (`patch_policy_camera_probe.py`) -- all
require an actual trained checkpoint, which this step doesn't produce.

### 14.4 Re-measured against the REAL model (Verification 3)

§14.1--14.3 measured a `_BenchRegisterViT` stand-in, because the backbone did
not exist yet. It does now (`RegisterViTBackbone`, a77e624 + per-camera
registers), so `bench_causal_frame.py`'s `vit --mode register` was pointed at
the shipped class -- same LoRA rank 32 on `attn.qkv`/`attn.proj`, 16 registers
per camera, but `num_cameras=3` (a real `(3, 16, 384)` parameter) and the real
`(n/3, 3, 3, 224, 224)` input shape rather than a flat image batch. The
stand-in and its `_BenchLoRALinear` are deleted; the bench now measures
production code.

Same machine, **but not the same conditions**: an unrelated CarlaUE4 process
held 8595 MiB for the whole re-run, leaving 22.5 of 31.4 GiB free. That
matters for two rows below and for nothing else.

| images / tpf | row                                | Step 0 (stand-in) | real model      | peak MiB       |
| ------------ | ---------------------------------- | ----------------- | --------------- | -------------- |
| 768          | frozen ViT                         | 78.83 ms          | 85.69 ms        | 2031 / 2031    |
| 2304         | frozen ViT                         | 331.58 ms         | 257.04 ms       | 5944 / 5944    |
| 768          | LoRA+register fwd+bwd, ckpt on     | 378.29 ms         | 417.49 ms       | 5279 / 5279    |
| 2304         | LoRA+register fwd+bwd, ckpt on     | 1595.86 ms        | 1271.09 ms      | 15615 / 15613  |
| 769 tpf      | trunk, 512/8/8, b=8, flex          | 1208.26 ms        | 1324.01 ms      | 9804 / 9804    |
| 769 tpf      | trunk, 768/12/12, b=8, flex        | 2927.97 ms        | 3206.89 ms      | 17158 / 17158  |
| 50 tpf       | trunk, 512/8/8, b=8, flex          | 72.43 ms          | 87.94 ms        | 795 / 796      |
| 50 tpf       | trunk, 768/12/12, b=8, flex        | 93.51 ms          | 129.55 ms       | 1503 / 1504    |
| 50 tpf       | trunk, 768/12/12, b=24, flex       | (2054--3677 MiB)  | 273.33 ms       | -- / 3681      |
| 769 tpf      | trunk, b=24, every width/impl      | OOM               | OOM             | --             |

**Peak memory reproduces to within 2 MiB on every row** -- the 2 MiB being the
per-camera register parameter the stand-in did not have. That is the number the
gate actually rests on, and it transferred exactly. Timings scatter ±20% in both
directions (the real model is *faster* at 2304 images, slower at 768), which is
the Carla process contending for SMs, not a systematic difference: nothing in
the real backbone's forward differs from the stand-in's beyond `run_layer_stack`
replacing an inline checkpoint loop.

Two rows read as regressions and are not. `768 images, ckpt off` OOMs here where
Step 0 measured 22542 MiB -- 22.5 GiB does not fit in 22.5 GiB free. `2304
images, ckpt off` OOMs in both runs, on an idle card too. Grad checkpointing
remains load-bearing exactly as §14.1 found.

Recomputing the gate verdict on real-model numbers, 768/12/12, `flex`:

- **b=8, today**: `85.69 + 3206.89 = 3292.6 ms`, ~19.2 GiB.
- **b=8, proposed**: `417.49 + 129.55 = 547.0 ms`, ~6.8 GiB. **6.0x faster**
  (Step 0 projected 6.4x). At 512/8/8: `1409.7 -> 505.4 ms`, 2.8x (projected 2.9x).
- **b=24, today**: OOM at every width and attention impl.
- **b=24, proposed**: `1271.09 + 273.33 = 1544.4 ms`, ~19.3 GiB peak.

The gate holds against the shipped model. Note the b=24 proposed row now has
~3 GiB of headroom on a card that is *already* 8.6 GiB occupied, which is a
stronger result than Step 0's, not a weaker one.

Environment note for whoever re-runs this: inside `nix develop` the flake's `uv`
wrapper overwrites `LD_LIBRARY_PATH`, and its `TRITON_LIBCUDA_PATH` points at
NixOS's `/run/opengl-driver/lib`. On a non-NixOS host both must be pointed at a
directory holding *only* the host driver libs (`libcuda`, `libnvidia-*`) --
`NIX_LD_LIBRARY_PATH=$DIR TRITON_LIBCUDA_PATH=$DIR uv run python ...`. Without
the first, `torch.cuda.is_available()` is False; without the second, every
`flex` (Triton-compiled) row dies with `InductorError: SubprocException`. Do
not put `/usr/lib/x86_64-linux-gnu` itself on the path -- its glibc shadows
nix's and every nix binary stops running.

### 14.5 The combined step, both arms, real config (Verification 4)

§14.1--14.4 measure the ViT and the trunk separately and add their peaks. That
sum is a lower bound on the real step: it leaves out the action tokenizer, the
heads and losses, the input transform, and the fp32 master weights and Adam
state the trainer actually carries. `tests/bench_patch_policy_step.py` runs the
whole `PatchPolicy` from its own experiment config -- forward, backward,
optimizer step -- on a synthetic batch at the yaak batch's real paths, dtypes
and shapes.

RTX 5090, bf16 autocast, `episode_length=32`, 3 cameras, lr forced to 1e-3
(see below), steady-state step time excluding step 0's compile/warmup:

| arm                            | tpf | batch | ms/step | peak MiB |
| ------------------------------ | --- | ----- | ------- | -------- |
| `dinov2_dinowm_causal_3cam`    | 769 | 8     | ~1900   | 11267    |
| `dinov2_registers_causal_3cam` | 50  | 8     | ~730    | 8135     |
| `dinov2_dinowm_causal_3cam`    | 769 | 24    | **OOM** | --       |
| `dinov2_registers_causal_3cam` | 50  | 24    | ~2030   | 23146    |

**2.6x faster and 28% less memory at matched batch.** §14.4 projected 2.8x for
this arm's width (512/8/8) from the separated benches -- the combined step lands
at 2.6x, so the model-level overhead the separated benches omit costs about 7%
of the projected win and nothing more. The ratio held stable across re-runs
while absolute times moved ±40% with GPU contention, which is the number to
trust.

At the configured `batch_size: 24` the register arm runs a full step in roughly
the time the baseline needs for a batch of 8 -- ~2.8x the samples/second -- and
the baseline does not run at all. (Its OOM here had ~6.5 GiB held by unrelated
processes; it reached 23.11 GiB allocated before failing, so this is not proof
it would OOM on a completely idle card. It is consistent with §14.2's OOM at
769/b=24 across both attention impls and both widths, and with the base config's
own comment.)

Both arms overfit the fixed batch at a comparable rate (baseline 10.32 -> 6.32,
register 10.25 -> 6.65 over 8 steps), with finite losses throughout. **Read
nothing into which is lower**: the batch is random noise and the register arm's
LoRA has barely left zero-init after 8 steps. What these numbers establish is
that the arm trains at all and does not diverge -- the accuracy question is
Verification 5's, and needs real data.

**Gradient routing (the actual point of Verification 4).** Per-step encoder
gradient norms:

- `camera_reg_token`: non-zero on **all three cameras** from step 0
  (~1.0 each), so `trainable_image_encoder` really does lift
  `PatchPolicy`'s `no_grad`, and the per-camera registers are each independently
  trained rather than one being carried by the others.
- `lora_B`: non-zero from step 0 (~0.4).
- `lora_A`: **exactly 0.0 on step 0**, non-zero from step 1 onward. This is
  correct, not a disconnected graph: `lora_B` is zero-initialized, so `dL/dA`
  carries a factor of `B` and vanishes until the first optimizer step moves `B`
  off zero. It is worth stating explicitly because it makes the obvious
  assertion ("every trainable parameter has non-zero gradient") fail on a
  healthy model, and its opposite ("`lora_A` stays zero") pass on a broken one.
  `tests/test_backbone_registers.py::test_trainable_encoder_receives_gradient`
  pins both halves.

**Why `--lr` exists.** The cosine-with-warmup `LambdaLR` steps itself once at
construction, so every param group's lr is `0.0` until the Lightning trainer
advances the scheduler. A bare `configure_optimizers()` outside a trainer
therefore never moves a weight: the loss sits flat, `lora_B` never leaves zero,
and `lora_A`'s gradient stays 0 forever -- which reads exactly like a broken
adapter. This is normal Lightning behaviour, not a defect, but it silently
voids the check, so the harness forces a usable lr by default.

Still open: Verification 5 (a real A/B against `kughoqfi` on real data) and 6
(the camera-identity probe on the winner). Both need a real training run; the
rbyte sample-index caches on this host are empty, so that starts with a full
index build. `fusion_goal_rms` recalibration rides along with 5 -- register
outputs come off the same final LayerNorm as patch features but need not share
their statistics, so `quality/token_norm/*` is the thing to watch on the first
real run.
