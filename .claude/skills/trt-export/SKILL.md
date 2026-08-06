---
name: trt-export
description: Build, validate and deploy TensorRT engines from rmind ONNX exports for the yaak test car — fp16 / fp16-strict / fp32, ONNX-Runtime parity verification, Orin latency benchmarking, and the two-hop transfer to the car. Use whenever asked to export/convert an ONNX to TRT, build engines for serving, benchmark model latency on Orin, or ship a model to the car.
---

# TRT export for the drivr car

Turn an rmind ONNX export into a TRT engine that is **verified equivalent to the ONNX**,
benchmarked against the control-loop budget, and deployed to the car.

Building the engine is the easy part. Every expensive failure here was silent: an engine
that loads, runs fast, and decodes **different actions**; an engine built under CPU load
that is needlessly slow; a runtime that feeds the wrong image size without complaining.
Each rule below exists because one of those happened.

## 1. Hosts

| Host | Role | Notes |
| --- | --- | --- |
| `sisyphos.ml`, `tresor.ml` | ONNX source | `/nasa/...` shared NAS (same files on both) |
| `max@delta-dev1.kit` | preferred build host | **IP moves** — was `172.30.0.62`; if the alias fails, ask |
| `max@delta-emc1.kit` | the car, serving target | `172.30.0.40` |

**Routing — verified 4 Aug, and only the car needs a relay:**

- **sisyphos ↔ dev1 is direct**, over ZeroTier, using dev1's `172.30.0.62` (its ZeroTier
  address) — *not* `192.168.144.35`, which has no route from sisyphos. So push ONNX
  straight from the NAS to the build host; no hop through your machine.
  **Use `ConnectTimeout=30`.** A cold ZeroTier path starts out in `RELAY` state and takes
  many seconds to establish; `ConnectTimeout=8` fails and looks exactly like "no route".
  That false negative is why this doc previously claimed no route existed.
- **The car is reachable only from your machine**, via the AWS Client VPN (`utun5`,
  `172.30.0.0/24`). Both sisyphos and dev1 get "No route to host" for `172.30.0.40`
  despite sharing the ZeroTier network with it. So the *engine* leg does need two
  `rsync` hops through your machine — `md5sum` both ends.
- Cloudflare WARP carries `192.168.207.0/24` and `192.168.144.0/24` for enrolled
  **clients** only; sisyphos and dev1 have no `warp-cli`, so it is not a host-to-host
  path. (`ProxyJump 10.19.17.255` in sisyphos's ssh_config is stale — it times out.)
- The car link is flaky: expect rsync to die mid-transfer on large engines. Use
  `--partial --append-verify` plus a retry loop, and treat `md5sum` as the acceptance
  gate — an interrupted `--partial` run leaves a **truncated file under the final
  engine name**, which is worse than no file.

Use `/home/max/Code/drivr/.venv` on dev1 and the car — it has `tensorrt`, `onnxruntime`,
`torch`, `structlog`. `/home/nvidia/workspace/drivr/.venv` on dev1 has none of them; a
build that "fails" in **0 seconds** is this, not a real build error.

## 2. Inspect the graph — do not trust the manifest

```
scripts/inspect_onnx.py MODEL.onnx [MODEL2.onnx ...]
```

Exports ship a `MANIFEST.md`; trust it for intent, not facts (names get flattened at
export, so `data.cam_front_left` becomes `batch_data_cam_front_left` and the manifest may
predate that). Four things decide everything downstream:

- **Input image size, and the `Resize` count.** Zero `Resize` nodes means the graph does
  **not** rescale — the host must deliver the exact size. PatchPolicy: DINOv2 = 224×224,
  DINOv3 = 256×256.
- **First ops.** `Sub`/`Div` at the top = ImageNet normalization is **in-graph**, so the
  host feeds `[0,1]` (`--image-norm unit`). Normalizing host-side too double-normalizes
  every pixel and degrades the model silently.
- **`ArgMax` count.** Non-zero = codebook/VQ decoding in-graph. **This is the model class
  where fp16 is dangerous** (§4) — a tiny numeric error flips a discrete code, so the
  action changes by a *step*, not a nudge.
- **`Sin`/`Cos`.** Indicates RoPE. DINOv3 has them, DINOv2 does not, and DINOv3 is by far
  the most fp16-fragile of the family.

## 3. Build — only on an idle host

```
/home/max/Code/drivr/.venv/bin/python \
  /home/max/Code/drivr/scripts/build_trt_engine.py \
  --onnx MODEL.onnx --precision {fp32|fp16|fp16-strict} --workspace-gb 6
```

Suffixes: `fp32` → `.trt`, `fp16` → `.fp16.trt`, `fp16-strict` → `.fp16strict.trt`.
`fp16-strict` = FP16 with computation layers pinned to FP32 via
`OBEY_PRECISION_CONSTRAINTS`.

**TRT selects kernels by timing them**, so building on a loaded machine bakes in slower
tactics. `scripts/build_and_bench.sh` gates on sustained low load (3 checks under 3.0,
45 s apart) and **exits without building** if it never settles — that is deliberate;
a mistimed engine is worse than no engine.

Engines are tied to **TRT version + GPU arch**. Building on dev1 and serving on the car
is only valid because both are Orin on the same TRT. Check before shipping:

```
dpkg -l | grep -m1 libnvinfer-bin                                    # must match
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq            # and clocks,
cat /sys/devices/platform/bus@0/17000000.gpu/devfreq/*/max_freq      # or latency
grep -m1 PM_CONFIG /etc/nvpmodel.conf                                # won't transfer
```

## 4. Pick precision by measurement, never by default

Measured on PatchPolicy (ViT-S encoder, 6 frames, Orin @ GPU 918 MHz):

Full Pareto curve, **all at n=200 with an fp32 control**, on `zt39kjn4-v1` (the
widest-margin checkpoint in a 16-model screen — so these are best-case numbers):

| config | fp32-pinned layers | latency | decisions |
| --- | --- | --- | --- |
| **fp32** (TF32 flag *cleared*) | all | **420 ms** | **0/200 — the only decision-exact option** |
| `tf32` (TRT's default fp32 path) | all | 226 ms | 1/200 |
| fp16-strict | ~all compute | 232 ms | 1/200 |
| bf16 everywhere + fp32 decode | decode | 182 ms | 19/200 |
| fp32 encoder + **bf16** trunk | encoder + decode | 98 ms | 12/200 |
| **fp32 encoder + fp16 trunk** | encoder + decode | **98 ms** | **1/200 ← best non-fp32** |
| AMP-faithful + fp32 encoder | 86 + encoder + decode | 97 ms | 2/200 |
| fp32 encoder + int8 trunk | encoder + decode | 96 ms | 1/200 |
| **AMP-faithful** (norms/softmax/reduce) | **86 layers** + decode | **74 ms** | 5/200 |
| int8 everywhere but decode | decode | 72 ms | 7/200 |
| pure fp16 | none | 72 ms | ~16/200 |

**What to take from this, in order of how much time it saves you:**

1. **`fp16-strict` is dominated — stop reaching for it as the "safe middle".** It is
   232 ms at 1/200, while **fp32-encoder + fp16-trunk is 98 ms at the same 1/200** — 2.4×
   faster for identical parity. fp16-strict's design is *backwards* relative to how these
   models train: it pins the **compute** (GEMMs) to FP32 and leaves data movement in fp16,
   whereas AMP does the opposite. That is why it is both slow (the GEMMs are the cost) and
   not actually safer.
2. **Pin the image encoder, not the tail.** Error is created in the encoder (relative error
   9e-4 → 3.2e-3) and is flat through the temporal trunk (2.46e-3 → 2.63e-3). Every attempt
   to pin the tail — TopK, ArgMax, constants, decode head — produced engines **bit-identical**
   to the unpinned build, because the readout feature was already corrupted. The encoder is
   only ~18 % of runtime, so protecting it is cheap.
3. **bf16 is strictly worse here, despite being the training precision.** 12/200 vs 1/200 on
   the trunk *at identical latency*, and 19/200 network-wide at 182 ms. bf16 has 8 mantissa
   bits vs fp16's 11 (~8× coarser ULP), and this failure is mantissa-limited. Do not
   "match training precision" reflexively.
4. **Trunk precision barely matters** once the encoder is fp32 — fp16, int8 and bf16 all land
   within a few ms, because the trunk is bandwidth-bound. Choose fp16 (best parity).
5. **A cheap partial:** pinning just the 86 `NORMALIZATION`/`SOFTMAX`/`REDUCE` layers to fp32
   gives 5/200 at **74 ms** — 3× better than pure fp16 for 2.7 ms. Good if you need ~74 ms
   and can accept ~2.5 %.
6. **naive int8 PTQ (logits quantized) = 82 % wrong.** Never ship it.
7. **`tf32` — TRT's *default* — is also dominated, and worth understanding.** Our `fp32`
   build calls `config.clear_flag(trt.BuilderFlag.TF32)`, i.e. it disables the tensor-core
   path to get *true* fp32. That is the entire reason fp32 costs 420 ms; leave TF32 on and
   the same network is **226 ms**. But it scores **1/200**, so it buys nothing over the
   98 ms mixed engine and is 2.3× slower. Build it only to reason about *why* fp16 fails:
   TF32 rounds matmul **inputs** to a 10-bit mantissa (same as fp16) while accumulating in
   fp32 and keeping fp32 storage between ops, so it isolates input-mantissa loss from
   intermediate rounding. Keep using the TF32-*cleared* build as the reference — it is the
   only 0/200 config, and a reference must be exact.

### Read `max|d|` PER CHANNEL — the single scalar is misleading

`parity_matrix.py` reports one `max|d|` over the whole `(horizon x action_features)` chunk.
For PatchPolicy the channels are **`[gas_pedal, brake_pedal, steering_angle, turn_signal]`**
(the tokenizer's continuous normalizer is `Identity`, so gas/brake/steering are already in
their `*_normalized` units). `turn_signal` has a much larger dynamic range than the pedals and
**dominates the headline number**.

Measured on `zt39kjn4-v5`, `encfp32-fp16trunk` vs native, n=200
(`scripts/per_channel_parity.py`):

| channel | max \|d\| | mean \|d\| |
| --- | --- | --- |
| gas_pedal | 0.0843 | 4.5e-04 |
| brake_pedal | 0.0355 | 1.9e-04 |
| steering_angle | 0.0541 | 2.5e-04 |
| **turn_signal** | **0.5461** | 2.6e-03 |

The engine's headline `max|d| = 0.546` is entirely the **turn signal** — an indicator state
change, not a trajectory change. Reporting it as a "coarse mode change" was wrong.

Do **not** over-correct the other way either: both flagged trials also exceed tol on a
*control* channel (`decision changes CONTROL only: 2/200`). The defensible statement is
"worst case 8.4 % throttle / 5.4 % steering / 3.6 % brake on 1 % of plans, on a harness ~7x
harsher than the road", not "only the indicator changed".

**Always run `per_channel_parity.py` before quoting a magnitude to anyone.**

### The ONNX fp32 reference is validated against rmind native — don't re-litigate it

Every parity number here is scored against ORT fp32 on the exported ONNX. That baseline was
checked directly against **rmind PyTorch native** (`PatchPolicy.load_for_export` — the same
callable the graph is traced from, `sample_codes=False`), same harness inputs, same seed:

| | decisions | max \|d\| | mean \|d\| |
| --- | --- | --- | --- |
| native vs ONNX fp32 (`zt39kjn4-v5`, n=200) | **0/200** | 7.75e-07 | 5.10e-08 |

Float32 round-off, not a modelling difference. Re-scoring the engines against a **native**
reference cache instead of the ORT one reproduced the TRT numbers *exactly* — 0/200 for fp32,
and 2/200 at `max|d| = 0.546103`, `mean = 8.72e-04` for the mixed engine, identical to six
digits and on the identical trials. So the ORT reference is sound and results hold
transitively. `scripts/native_vs_ort.py` re-runs this and emits a native ref-cache in
`parity_matrix.py`'s npz format (feed it via `--ref-cache`) if you ever need to re-check a
new arm.

### The residual `1/200` is a boundary tie, not precision damage

Worth knowing before you spend anything trying to reach 0/200 on a fast engine. Four
different numeric paths — `tf32`, `encfp32-fp16trunk`, and the derived-range rebuild — all
flip **the same trial (t82)** by the **same magnitude** (0.0952 / 0.0950 / 0.0950). And
`tf32` is measurably *more* accurate than the fp16 trunk (mean |d| 1.23e-4 vs 1.39e-4) yet
flips the identical decision. An input that flips under every perturbation regardless of the
perturbation's size is sitting **on** an `ArgMax` boundary — it is a tie, not damage.

Its consequence is small *by construction*: a tight margin means the two codes are
near-equivalent, so the decode barely moves. Measured, that flip is 0.095 in normalized
action units, matching the ~0.105 mean consequence of tie states from the code-bimodality
study (decisive decisions carry ~0.50). Real-data ULP-risk rate is **0.136 %**, ~7× rarer
than this synthetic harness suggests.

So: **read the serving engine's `1/200` as "clean apart from one degenerate input", not as a
0.5 % defect rate**, and do not conclude a training change is needed to widen margins — at
this configuration there is almost nothing left for it to fix. (Bounded honestly: n=200 over
3 real frames with synthetic speed/waypoints. This rules out *additional* precision-driven
flips at that sample size, not all of them. Re-check per checkpoint with §4a — margins are
per-checkpoint, and 2 of 7 checkpoints genuinely do flip ~4 %.)

⚠️ **Only the TF32-cleared fp32 build has ever reached 0/200.** Do not conclude parity from
n=50: `2/50` and `4/50` are indistinguishable and both are consistent with 3.5 %. Every
low-precision config that looked acceptable at n=50 failed at n=200.

⚠️ **Caveat on the reference.** These models train *and validate* under `torch.autocast(bf16)`
(`patch_policy_eval.py`), while `export_onnx.py` exports in plain fp32. So the fp32 ONNX we
score against is a precision the model was never trained or evaluated at. At a top1−top2 gap
of 0.35 fp16 ULP the model never learned a preference at that resolution, so a 1/200
disagreement with fp32 is not self-evidently a defect. If you need to settle this, build a
bf16-faithful engine (export under autocast + `trtexec --stronglyTyped`) and score it against
a **bf16** reference.

⚠️ **At n=200, fp32 is the only config that has ever passed.** Do not conclude from an
n=50 run that a low-precision engine is at parity: `2/50` and `4/50` are indistinguishable,
and both are consistent with a true rate of 3.5 %. Every low-precision config that looked
acceptable at n=50 failed when re-measured at n=200.

**Two things to take from this table.**

**(1) fp16-strict is per-checkpoint, not per-architecture.** 2 of 7 checkpoints flip ~4 % of
actions, and failure does **not** follow the arm — `b2itoon8-v3` passes while its own parent
`wxyp0bzq-v9` fails on the same head. **You cannot inherit a verdict from a sibling.** Root
cause is `code_head` ArgMax margin geometry: a top1−top2 gap under ~1 fp16 ULP is not
resolvable in fp16 at all. Error accumulates in the *encoder* (9e-4 → 3.2e-3) and is flat
through the head, so **no mixed-precision layer pinning fixes it** — pinning TopK/ArgMax/
constants to FP32 produced engines *bit-identical* to the unpinned one.

**(2) int8 is not usable on this model family, even with the decode head protected.**
Quantizing the logits too is catastrophic (82 % wrong). Pinning the code-head MLP, codebook
gather, offset table and `tokenizer.decoder` to FP32 (`build_mixed.py --precision int8-mixed
--fp32-index-ranges …`) rescues it to *look* fine at n=50 — but at **n=200 it flips 3.5 %
even on the checkpoint with the widest margins in the whole fleet** (13.7 fp16 ULP), and
7.0 % on a tight one. int8 also buys nothing over pure fp16 on speed (both ~70 ms — the
model is bandwidth-bound), so there is no reason to prefer it.

**The margin screen predicts fp16 safety, NOT int8 safety.** ULPs are an fp16 unit; int8
activation quantization error is far larger than one fp16 ULP, so a 13.7-fp16-ULP margin is
not remotely a guarantee under int8. If you want int8, the margin requirement must be
re-derived in int8 quantization steps — expect it to be orders of magnitude larger.

**Also measured, so don't re-derive:** DLA is a dead end for a transformer (201 layers on
DLA vs 1024 on GPU, all LayerNorms and Softmaxes falling back, DLA subgraphs only 6.3 % of
runtime, and **19 % slower** overall from DLA↔GPU reformat overhead).

### 4a. Screen the margins BEFORE building — it predicts the failure

`scripts/margin_screen.py` probes every `ArgMax`'s input, measures `top1 − top2`, and
expresses it in fp16 ULPs at that magnitude. 25 trials, ONNX-Runtime only, **no GPU, ~40 s**.

Validated on 16 checkpoints: **`min ULP < 1` predicted every measured fp16-strict failure,
and no checkpoint with `min ULP ≥ 4` has ever failed.** `<4 ULP` is *not* the discriminator —
a checkpoint can carry 1 % of decisions under 4 ULP and still score 0/50.

```
scripts/margin_screen.py --onnx A.onnx,B.onnx --labels a,b --trials 25 --frames f1,f2,f3
```

| min ULP | meaning |
| --- | --- |
| **< 1** | fp16-strict **will** flip actions. Serve fp32, or pick another checkpoint. |
| 1–4 | marginal; verify parity at ≥200 trials before serving low precision |
| **≥ 4** | fp16-strict has always been clean here |

Frozen sub-codebooks give identical margins across checkpoints — if several ArgMax probes
report byte-identical numbers for different models, that part of the graph is frozen and is
not your problem. Only the trained head varies.

**The real fix is upstream**, not in TRT: near-tied code logits mean the model is genuinely
torn between actions that decode ~0.08 apart (max 0.82), and fp16 turns that into a coin
flip. That is a training-side robustness bug. Note the ties are an *overfitting* symptom —
train `p_gt ≈ 0.6` vs val `≈ 0.13` — so a margin loss computed on training batches may not
move val margins at all.

Benchmark median GPU compute:

```
/usr/src/tensorrt/bin/trtexec --loadEngine=ENGINE.trt \
  --iterations=60 --avgRuns=20 --useSpinWait --warmUp=1000
```

**Budget** — measured on the car from `reports/*_predictions.jsonl`, 31 Jul, not assumed:

- Inference runs **once per ~2 s plan**, not once per 333 ms step (a 6-step joint-actions
  plan dispatches between inferences). So "does the model fit in a step?" is the wrong
  question; the cost lands on one tick in six.
- The control loop *blocks* during inference, and the tick carrying one costs exactly
  `333 ms (frame wait) + inference_ms`: 610 ms at fp32 (268 ms), 512 ms at fp16-strict
  (172 ms). 95 ms less GPU time bought 96 ms of tick — it pays back **1:1**, with no
  hidden overhead. Plain ticks hold 332–333 ms.
- Consequence of the overrun: ~35 % of plan cycles dispatch only 5 of their 6 steps at
  fp32, ~27 % at fp16-strict. Real, but a weaker effect than the latency gap suggests.
- **Idle `trtexec` is a floor, by ~+55 to +60 ms.** Attributed on dev1 (4 Aug) by running
  real drivr and bracketing the *same* launch with `torch.cuda.Event`, across 15
  configurations. Two independent GPU-side mechanisms, and they are **not additive**
  (both together measured +88 ms, not +107):

  1. **GPU DVFS — ~+55 ms, the dominant term.** `nvhost_podgov` really does drop to
     **306 MHz** between inferences: during a realistic 2 s-gap run, **70.9 %** of 20 ms
     samples sat at 306 MHz and only **10.8 %** at 918 MHz.
     `corr(mean clock during inference, inference_ms) = −0.971`. Inferences starting at
     918 MHz took ~229 ms; starting at 306 MHz, 269–318 ms. Pinning the clock with a CUDA
     keep-alive dropped latency 280.9 → 247.1 ms.
     **Fix: pin the clock** — `jetson_clocks`, or devfreq `governor=performance` /
     `min_freq=918000000`. Needs root. Recovers ~55 ms for free, no code change.
     ⚠️ **An earlier version of this doc claimed the clock stays pinned at 918 MHz and told
     you not to re-derive it. That was WRONG.** It came from sampling `cur_freq` only
     *after* each inference, when the clock has already ramped, and from a synthetic
     duty-cycle test that kept the GPU warmer than drivr does. Sample `cur_freq`
     continuously *through* the idle gap, or join a clock trace to individual inferences.
  2. **The 8 secondary mp4 recorders — ~+52 ms, independent of the clock.** Present even
     with the clock at 918 MHz for every measured inference. Each `_SecondaryRecorder`
     opens its own **1080p** V4L2 stream and encodes with cv2 `mp4v`, a **CPU** encoder;
     a bandwidth-bound model loses ~23 %. **Fix: NVENC instead of `mp4v`, and/or fewer /
     lower-resolution recorders while AI control is engaged.**
     Beware the measurement trap: feeding pre-made small frames from memory shows only
     +0.8 ms. You must include real capture at real resolution.
- **Refuted suspects, so nobody re-spends the time:** rerun logging costs **0 ms**
  (229.0 ms active vs 229.5 ms absent — and check `rerun-sdk` is actually installed, it was
  silently missing from dev1's venv). GIL / thread count: +0.2 ms with 12 extra threads at
  50/30/10 Hz. Flask: invisible. The whole camera path — defish 1080p, pageable H2D, CUDA
  `preprocess_image` — is only **+4.8 ms**, so preprocessing offload is a minor lever.
- **It is GPU time, not CPU scheduling.** `inference_ms − gpu_ms` (CUDA events around the
  identical launch) stayed **≤1.21 ms in all 15 configs** while `inference_ms` ranged
  225→314 ms. Don't chase descheduling or the GIL.
- `inference_ms` in the logs brackets only `execute_async_v3` + `synchronize()`, so it is
  directly comparable to `trtexec` GPU-compute, and does **not** include capture,
  preprocessing or the H2D/D2H copies.

## 5. Validate parity — not optional

```
scripts/verify_trt_parity.py --onnx M.onnx --engine M.fp16strict.trt \
  --trials 50 --frames /tmp/a.png,/tmp/b.jpg,/tmp/c.jpg
```

Compares TRT against ONNX Runtime (CPU fp32) and counts **decision changes** — samples
where any action channel moves more than `--code-tol` (default 0.02). Float noise is
~1e-4 and a code flip is ~0.1, so the threshold is not delicate. Non-zero exit on failure.

`scripts/parity_matrix.py` does the same across several engines against **one shared** ORT
reference (`--ref-cache`), which is how you compare precisions without paying for the
reference N times.

Four rules, each learned by getting it wrong:

- **Real camera frames, several of them.** Noise understated one failure (4/10 vs 8/10)
  and produced a false failure elsewhere. Vary speed and waypoints too, but the image
  must be real. Grab frames off the car (`/tmp/video*.jpg`) or from an rrd.
- **≥50 trials to detect, ≥200 to decide.** 10 trials passed engines that flip 6–8 %.
  And **2/50 vs 4/50 is not a distinguishable difference** — if a ship decision turns on
  which of two configs is better, 50 trials cannot support it. n=200 bounds a clean run
  near ~1.5 %; n=50 only bounds it near 7 %.
- **Always run an fp32 control.** Without it you cannot distinguish precision loss from
  `ArgMax` boundary ties — inputs near a decision boundary flip under *any* numeric
  difference, so a nonzero rate is not automatically a defect. When we ran it, fp32 scored
  0/50 **bit-exact** (max abs diff `0.000000`), proving the flips were precision, not ties.
- **Quarantine by renaming, and record why.** Rename to
  `QUARANTINE-FAILED-PARITY.<name>.trt` and drop a `PARITY-NOTES.md` beside the engines
  with the measured numbers. A failed engine under a plausible name will eventually be
  served by someone.

## 6. Deploy

ONNX to the build host goes **direct** (§1); only the engine leg to the car needs a relay:

```
# NAS -> build host, no hop through your machine
ssh max@sisyphos.ml 'rsync -a --partial \
  -e "ssh -o ConnectTimeout=30" /nasa/max/models/<family>/MODEL.onnx \
  max@172.30.0.62:onnx_exports/<family>/'

# engine -> car, two hops via your machine, retried, md5-gated
rsync -a --partial --append-verify --timeout=120 max@172.30.0.62:onnx_exports/<family>/ENGINE.trt ./
rsync -a --partial --append-verify --timeout=120 \
  -e 'ssh -o ServerAliveInterval=10 -o ServerAliveCountMax=3' \
  ./ENGINE.trt max@delta-emc1.kit:onnx_exports/<family>/
# then md5sum on BOTH ends and compare - this is the acceptance gate, not optional
```

Wrap the car leg in a retry loop; it dies mid-transfer regularly. **A killed `--partial`
transfer leaves a truncated file under the final engine name** — verify or delete it, never
leave it. Same for engines that failed parity: delete or clearly quarantine, or someone
will serve one.

## 7. Tell the user how to serve it — both failure modes are silent

- **`--image-norm unit`** when normalization is in-graph (§2).
- **Input size must match the engine.** `TRTEngine.run` binds via `set_tensor_address`,
  which takes a raw pointer with **no size validation**, so a 256×256 tensor handed to a
  224×224 binding is *not* an error: TRT reinterprets the buffer as 224-wide rows,
  producing a diagonally sheared, partially truncated image. The model runs and merely
  looks weak. Branch `feat/engine-derived-image-size` (PR #27) reads the size from the
  engine; without it only models matching drivr's hardcoded 256×256 are evaluated fairly.

## 8. Host quirks

- **Bare PATH over non-interactive ssh.** `uv`/`just` live in `~/.nix-profile/bin` on
  sisyphos/tresor — prefix `export PATH=$HOME/.nix-profile/bin:$PATH` or use `bash -lc`.
- **Background jobs die on disconnect.** `( … ) &` inside ssh takes SIGHUP. Use
  `setsid nohup script.sh > log 2>&1 < /dev/null &`.
- **`git diff` may go through an external differ** (difftastic-style) and produce output
  `git apply` rejects with "unrecognized input". Use `git --no-pager diff --no-ext-diff`.
- **Heredocs over ssh get mangled.** Write the script locally, `scp`, then run it.

## 9. Recommended workflow — build the SERVING PAIR, nothing else

**Every model gets exactly two engines. This is the standard; do not deviate without a
measurement that justifies it.**

| # | engine | build | role |
| --- | --- | --- | --- |
| 1 | `MODEL.trt` | `--precision fp32` | **the reference.** Bit-exact (0/200). Serve it wherever it fits the tick — for a ViT-S arm that is ~195–200 ms, so it fits. |
| 2 | `MODEL.encfp32-fp16trunk.trt` | `--precision mixed --fp32-index-ranges <encoder>,<decode>` | **the fast one.** ~4.3× faster at the *same* measured parity as fp16-strict. Serve it when fp32 does not fit the tick. |

```
scripts/build_serving_pair.sh MODEL.onnx        # does all of 1-5 below
```

**Do NOT build these** — each was measured and each is dominated:

- **`fp16-strict`** — 232 ms at 1/200, versus 98 ms at 1/200 for engine #2. It pins the
  GEMMs to FP32 and leaves data movement in fp16, the *inverse* of how these models train
  (`bf16-mixed` autocast: GEMMs low, softmax/layernorm/reductions fp32). Hence slow *and*
  not safer. **If you find one being served, replace it with the pair.**
- **`tf32`** — 226 ms at 1/200, so it is dominated by engine #2 on both axes (2.3× slower,
  same parity). Note this is TRT's **default**: engine #1 is only decision-exact because
  `--precision fp32` explicitly *clears* `BuilderFlag.TF32`. Diagnostic value only (§4).
- **`int8` / `int8-mixed`** — no speed gain over fp16 (both ~72 ms; the model is
  bandwidth-bound) and worse parity (7/200). With the logits quantized: 82 % wrong.
- **`bf16` anything** — 12–19× more flips than fp16 at *identical* latency (8 mantissa bits
  vs 11), despite being the training precision. Training precision was chosen for
  throughput on a 4090/5090, not fidelity on an Orin; it has no claim as a reference.
  Confirmed by pairwise scoring: bf16 sits 20–23/200 from *every* other config while fp32
  and the whole fp16 family cluster at 2–11.

### Steps

1. `inspect_onnx.py` → input size, norm contract, `ArgMax`/RoPE risk.
2. **`margin_screen.py` — 40 s, no GPU, before you build anything.** `min ULP < 1` ⇒ low
   precision will flip actions; serve fp32 or pick another checkpoint. Predicted every
   measured failure across 16 checkpoints, including one nobody had tested.
3. Check TRT version on build host vs car, confirm the build host is idle, and **pin the GPU
   clock** (§4) or your numbers are ~55 ms pessimistic.
4. **`precision_ranges.py` to derive the fp32 ranges for *this* architecture.** Never
   hardcode them — a dinov2 224² arm and a dinov3 256² arm have entirely different layer
   counts, and a wrong range silently mispins.
5. `parity_matrix.py --trials 200` across both engines off one shared reference. **fp32 must
   be 0/200**; if not, the harness is broken, stop.
6. Two-hop rsync the passing engines; md5 both ends; quarantine failures by renaming to
   `QUARANTINE-FAILED-PARITY.*` and leave a `PARITY-NOTES.md`.
7. Report a **latency × parity** table with a recommendation. State sample sizes, and
   **compare against what is actually served, not against fp32** — quoting a speedup versus
   a config nobody runs overstates the win.

⚠️ **The harness is ~7× harsher than reality.** It feeds real frames but *synthetic* speed
and waypoints. On 2,210 real validation states the within-1-fp16-ULP rate is **0.136 %**, not
the ~1 % measured synthetically, and the flips concentrate in fine residual levels whose
consequence is small. Good conservatism for a gate; do not quote harness rates as deployment
rates.
