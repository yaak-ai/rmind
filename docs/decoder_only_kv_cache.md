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
| training arm | `config/experiment/yaak/patch_policy/dinov2_dinowm_causal.yaml` |

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

## 6. KV memory budget on Orin

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

delta-dev1 has **15 GiB** of LPDDR5 shared between CPU and GPU, ~12 GiB
practically available. Memory is not the binding constraint at any context length
worth training: `_big` at 64 frames is 9.3 % of available RAM in fp32, 4.6 % in
fp16. Latency and the per-tick cache read bandwidth bind first.

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
# ring-buffer advance: shift one frame block left, write 257 tokens per layer
past_k[..., :-tokens_per_frame, :] = past_k[..., tokens_per_frame:, :]
past_k[..., -tokens_per_frame:, :] = out["new_k"]
# ... same for V ...
cache_bias[..., -tokens_per_frame:] = 0.0     # the tail is now filled
frame_index += 1
```

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

The marginal term is attention against the cache. Per extra cached frame it is
`2 × 257 × 257 × D × 2 × L` FLOPs — 1.08 GFLOP (small) and 2.43 GFLOP (`_big`) —
so the measured slopes correspond to **528 and 509 GFLOP/s effective**, ~16 % of
the module's ~3.3 TFLOP/s fp32 peak, and *identical between arms*. That agreement
says the scaling is dominated by a thin (257-query) attention GEMM rather than by
the `Concat` or by cache bandwidth, and it means the slope is predictable: it
scales with `L × D`, nothing else.

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
   `frame_block_causal_mask(window=N)`. `CausalFrameTransformer` cross-checks
   `max_sequence_length == window * tokens_per_frame` so a config mismatch raises,
   but nothing prevents serving a checkpoint against the wrong cache size —
   validate the engine's `inputs_past_k` shape against the checkpoint's `window`.
4. **No parity/precision verification yet.** All measurements are fp32 with random
   weights, so `parity_matrix.py --trials 200` has not been and cannot be run — it
   needs a trained checkpoint. The 5 in-graph `ArgMax` nodes and the code-flip
   defect are unchanged by this work (hand-off §6), and the margin screen must be
   re-run on any real checkpoint before serving anything below fp32.
5. **RoPE base is unvalidated.** `rope_base=1000` is reasoned (base 10000 leaves
   most frequency pairs inert over 64 positions) but not measured. It is a
   trainable-arm hyperparameter, and changing it after training invalidates the
   checkpoint.
6. **The `Concat` of cache and new keys is unavoidable without a plugin.** It costs
   an extra read+write of the cache per tick. Measured slopes say it is not
   dominant, but at 128 frames it would be. A paged/in-place attention plugin is
   the escape hatch; `ScatterElements` in the ONNX graph is deliberately not.
7. **`_big` at 64 frames does not fit a tick at fp32** (364 ms). Not a defect, but
   it bounds the context length that is servable today without the mixed-precision
   engine.
