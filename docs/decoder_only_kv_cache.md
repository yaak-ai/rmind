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

## 7. Measured latency

See the measurement table in the accompanying report. Method: fp32 engines (the
hand-off's 194.8 / 448.8 ms baseline is fp32, and fp32 is the only precision that
has ever reached 0/200 on parity), built and benchmarked on an idle delta-dev1
with the GPU clock pinned at 918 MHz, `trtexec --iterations=60 --avgRuns=20
--useSpinWait --warmUp=1000`.

Note **cold vs warm cache costs the same** for a static-shape engine: the graph
does identical work at any fill level, and correctness at cold start comes from
`cache_bias`, not from a smaller computation. A cheaper first tick would need a
dedicated small-context engine; it is not worth an engine to save one tick per
episode.
