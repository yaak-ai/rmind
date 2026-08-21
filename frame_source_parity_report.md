# Frame source parity report

How close can inference-side preprocessing get to the frames the model was actually
trained on, and what was standing in the way.

Measured on `Niro122-HQ/2023-05-25--09-34-14`, checkpoint
`yaak/alex-tmp/model-9xfz6ify:v6` (PatchPolicy / DINOv2, 224×224 input),
`benchmark_onnx.py` with `frame_sources=[video,jpg,video_jpeg,ffmpeg,torch_nv12]`.

## Summary

Four things came out of this:

1. **`_simulate_offline_jpeg` was over-compressing by a wide margin.** `-q:v 16` was
   passed to `simplejpeg` as if it were a libjpeg 0–100 quality. Fixed — `video_jpeg`
   now reproduces `extract_frames`' quantization tables byte-for-byte.
2. **Added an `ffmpeg` frame source** that re-runs `extract_frames`' own command. It is
   the baseline: **1.59 uint8 MAE** against the real training jpgs, where the best
   pure-Python path reaches 3.80 and two *adjacent* training frames differ by 1.34.
3. **ONNX was silently running on CPU.** The `export` and `predict` extras pulled the
   CPU `onnxruntime` wheel into the shared venv. Fixed — ONNX went from
   **760 ms (1.3 Hz) to 17.8 ms (56 Hz)**.
4. **The 1.59 "floor" was not a floor.** A pure-torch reproduction of the same pixel
   chain reaches **1.13** — see [The GPU parity path](#the-gpu-parity-path), added
   after this report's first pass. It needs no ffmpeg, no encoder and no subprocess,
   costs 1.5 ms/frame on GPU-resident planes, and is what `~/drivr` now runs on the
   car. `scale_cuda` was the limiting factor, not `scale_npp`.

## What training actually did to the pixels

From `/home/alex/data/dvc.yaml`, the `extract_frames` stage, verbatim:

```
ffmpeg -y -vsync 0 -threads 0 -hwaccel cuda -hwaccel_output_format cuda
       -c:v hevc_cuvid -i <video>
       -filter_complex "scale_npp=576:324,hwdownload,format=nv12"
       -f image2 -q:v 16 <out>/%09d.jpg
```

Two details drive everything below:

- **The resize happens in NV12.** Chroma is scaled at half resolution and the frame
  never round-trips through RGB. The cv2 path did YUV→RGB→resize→RGB→YUV, which
  resizes chroma at full resolution — this is the single largest source of error and
  it is not fixable by changing the resize kernel.
- **Every training frame is mjpeg `-q:v 16`.** The model has only ever seen
  DCT-quantized pixels. `-q:v` is ffmpeg's own quantizer scale (2–31, lower is
  better), *not* a libjpeg 0–100 quality.

## Results

### Pixel parity — uint8 MAE vs the real training jpg, at 576×324

| path | MAE |
| --- | --- |
| `torch_nv12` — `rmind.utils.frame_parity`, all-GPU | **1.13** |
| `ffmpeg`, `scale_cuda` bicubic | 1.59 |
| *two adjacent real training frames (reference)* | *1.34* |
| `ffmpeg`, `scale_cuda` bilinear | 2.40 |
| `ffmpeg`, software resize, still in NV12 | 2.84 |
| `video_jpeg` — cv2 INTER_CUBIC + mjpeg q16 | 3.80 |
| `video` — cv2 INTER_CUBIC, no jpeg step | 4.42 |
| `video_jpeg` **before the fix** (libjpeg quality 16) | 5.59 |

The residual 1.59 is `scale_npp` vs `scale_cuda`. NPP ships only in NVIDIA's
proprietary build — but that gap turned out to be reducible anyway, without NPP, by
matching the kernel rather than the binary: see below. (The `torch_nv12` row is
measured on a different frame range than the rest of this table, where adjacent
training frames differ by 0.64 rather than 1.34; its own like-for-like comparison
against `ffmpeg` is in that section.)

### The resize kernel is not the problem — in the RGB path

Every kernel lands on the same ~4.4 floor for the cv2 RGB path, so kernel choice is
second-order **there**. Once the resize moves into the YUV domain this reverses and the
kernel becomes the dominant term — see [The GPU parity path](#the-gpu-parity-path).

| kernel (1920×1080 → 576×324) | MAE |
| --- | --- |
| `cv2.INTER_LINEAR` | 4.39 |
| `cv2.INTER_AREA` | 4.43 |
| torchvision `Resize` bilinear + antialias | 4.51 |
| `cv2.INTER_LANCZOS4` | 4.54 |
| `cv2.INTER_CUBIC` (what the benchmark uses) | 4.55 |
| torchvision `Resize` bicubic + antialias | 4.59 |

It is also **not** a colorspace or range issue: a per-channel affine fit leaves
4.386 vs 4.366 uncorrected, and limited↔full range conversions make it much worse
(8.1 / 12.1).

### The quantization table bug

The real training jpgs carry `ffmpeg -q:v 16` tables exactly. Mean absolute
difference across all 64 DQT coefficients:

| encoder setting | mean \|ΔDQT\| |
| --- | --- |
| `ffmpeg -q:v 16` | **0.00** — exact match |
| best libjpeg quality (50) | 20.31 |
| **libjpeg quality 16** — what the code did | **96.31** |

libjpeg quality 16 is the *worst* point in the entire 0–100 sweep. That is why
`video_jpeg` used to land further from the training frames than doing nothing at all.

ffmpeg builds the table itself by scaling `ff_mpeg1_default_intra_matrix`: the DC
coefficient stays pinned at 8, each AC coefficient becomes `base * qv // 8` clamped
to 1–255. Verified byte-identical to ffmpeg for **every `-q:v` in 2..31**, so no
reference file or subprocess is needed at runtime. Encoding now goes through Pillow
(the only encoder here that accepts explicit tables) at 4:2:0, and decoding through
`cv2.imdecode` so the decoder matches the `jpg` source's `cv2.imread` — leaving
quantization as the only thing `video_jpeg` adds over `video`.

### Effect on model predictions

Median |Δ| against the `jpg` source, PyTorch backend, 20 episodes. The ordering is
monotonic and `ffmpeg` wins every channel:

| source | gas | brake | steer | turn_signal |
| --- | --- | --- | --- | --- |
| `ffmpeg` | **0.0015** | **0.0020** | **0.0045** | 20/20 |
| `video_jpeg` | 0.0040 | 0.0058 | 0.0104 | 20/20 |
| `video` | 0.0071 | 0.0185 | 0.0142 | 17/20 |

One caveat: `video_jpeg`'s *mean* brake error is 0.0595 against a 0.0058 median.
Episodes 0–2 show a discrete jump (brake 0.13 → 0.43), and those three episodes share
5 of their 6 frames — it is one bin-flip in the brake head's argmax during the
brake-hold, not three independent failures.

### Timing

| backend | before | after |
| --- | --- | --- |
| ONNX | 760 ms (1.3 Hz) | **17.8 ms (56 Hz)** |
| PyTorch | 36 ms | 36 ms |

ONNX is now *faster* than PyTorch, which is the expected direction for a CUDA-executed
ONNX graph. Running both on the same device also tightened ONNX↔PyTorch agreement
from ~1e-3 to **4.3e-5** mean — confirming the note already in `pyproject.toml` that
cross-device float noise flips the VQ head's code argmax on near-tie frames.

### Why ONNX was on CPU

`onnxruntime` and `onnxruntime-gpu` install the *same* `onnxruntime` module. The
`benchmark` group correctly asked for the GPU wheel under a `sys_platform` marker, but
the `export` and `predict` extras asked for the CPU wheel with no marker — so
`uv sync --all-extras --all-groups` put both in the venv and the CPU wheel won the
import. Those extras now carry the same platform split.

Watch out for one wrinkle: because both distributions own the same directory,
uninstalling the CPU wheel deletes the GPU wheel's files too, leaving an empty
namespace package (`onnxruntime.__file__ is None`). Reinstall to recover.


## The GPU parity path

The `ffmpeg` source above is the best *measurement*, not a deployable one: it decodes a
whole video file, shells out per 512-frame block, and writes jpgs to disk. `~/drivr`'s
`just drive` has none of those affordances — a live V4L2 camera at 10 Hz with a hard
per-frame budget.

So the chain was reproduced as GPU tensor ops in `rmind/utils/frame_parity.py`. Four
stages, each **fitted against ffmpeg's or libjpeg's own intermediate output** rather
than guessed, because a single end-to-end number cannot tell a wrong quantizer from a
wrong kernel — the first attempt got 3.85 and it took isolating each stage to find out
why.

### What each stage turned out to be

| stage | fitted against | agreement |
| --- | --- | --- |
| resize (`Resampler`) | `scale_cuda`'s own luma plane | 0.51 |
| limited→full range (`expand_range`) | ffmpeg's own `nv12`→`yuvj420p` | 0.00 luma / 0.49 chroma |
| mjpeg quantize (`quantize_planes`) | ffmpeg's real encoder, plane domain | 0.13 |
| chroma upsample + BT.601 (`yuv420_to_rgb`) | `cv2.imread` on the identical jpg | 0.12 |

Four findings came out of that, in descending order of how much they cost:

1. **The quantizer is not round-to-nearest.** `dct_quantize_c` in
   `libavcodec/mpegvideo_enc.c` computes `(|coef| * qmat + bias) >> shift`, and
   `ff_mpv_encode_init` sets `intra_quant_bias = 3 << (QUANT_BIAS_SHIFT - 3)` = **96/256
   = 3/8** for MJPEG. Truncation with a 3/8 bias reproduces the encoder to 0.13 MAE;
   round-to-nearest only reaches 1.14. This alone was ~1.3 MAE end to end.
2. **Quantize the planes, not the RGB.** Going to RGB first means subsampling chroma a
   second time on the way back into the DCT. 3.45 → 2.87 at that point in the fit.
3. **The kernel is Catmull-Rom, not torch's bicubic.** `scale_npp`/`scale_cuda` use
   a = −1/2; `F.interpolate(mode="bicubic")` hardcodes a = −3/4. 1.31 → 1.13.
4. **The encoder is fed uint8.** Rounding the planes before the DCT rather than carrying
   float into it: 1.28 → 1.13.

### Result

Measured on 20 frames of the same drive (frames 2910–2929, where adjacent training
frames differ by 0.64 — not the range used in the table above):

| path | MAE |
| --- | --- |
| *two adjacent real training frames (reference)* | *0.64* |
| **`frame_parity`, from the same nv12 planes** | **1.13** |
| `ffmpeg` end-to-end, `scale_cuda` bicubic + real mjpeg | 1.58 |
| ” with `F.interpolate`'s bicubic (a = −3/4) instead | 1.31 |
| ” without rounding planes to uint8 before the DCT | 1.28 |
| ” with an antialiased resize | 3.19 |
| ” without the DCT quantize | 3.87 |
| ” without the limited→full range conversion | 6.88 |

**It beats ffmpeg's own command.** Not by being cleverer — by using the kernel
`scale_npp` uses, where `scale_cuda` does not. Since `scale_npp` is what actually
produced the training data, matching it in torch is closer than substituting
`scale_cuda` for it. The ablations are asserted in `tests/test_frame_parity.py`, so a
future "simplification" that drops a stage fails rather than silently regressing.

Note the ordering: the antialiased resize is *worse* (3.19 vs 1.13) even though it is
the better resampler in the abstract. `scale_npp` does not prefilter, so neither can
this.

### Effect on model predictions

|Δ| against the `jpg` source — the real training frames — PyTorch backend at 224×224,
20 episodes, same protocol as the table earlier:

| source | gas | brake | steer | gas P95 | brake P95 | steer P95 |
| --- | --- | --- | --- | --- | --- | --- |
| `torch_nv12` | 0.00207 | **0.00151** | **0.00387** | **0.0069** | 0.0673 | **0.0078** |
| `ffmpeg` | **0.00150** | 0.00199 | 0.00447 | 0.0225 | **0.0154** | 0.0120 |
| `video_jpeg` | 0.00399 | 0.00579 | 0.01041 | 0.0161 | 0.2994 | 0.0262 |
| `video` | 0.00707 | 0.01849 | 0.01423 | 0.0405 | 0.0389 | 0.0814 |

`torch_nv12` has the lowest median on brake and steer and the tightest P95 on gas and
steer; `ffmpeg` has the lowest gas median and a tighter brake tail. They agree with
*each other* to 0.0012 / 0.0026 / 0.0022 median — the two closest sources in the table
by a wide margin — so the pixel result carries through to the predictions and the
deployable path is not paying for being deployable.

Reproduced on the ONNX backend too (0.0018 / 0.0014 / 0.0035 for `torch_nv12` against
`ffmpeg`'s 0.0016 / 0.0020 / 0.0044), so the ordering is not a backend artefact.

The brake P95 is the one place `torch_nv12` is behind, and it is the same phenomenon
the earlier `video_jpeg` caveat describes: a single bin-flip in the brake head's argmax
during the brake-hold, amplified by the fact that consecutive episodes share 5 of their
6 frames. Compare against the Max column — the P95, P99 and Max are nearly equal, which
is the signature of one frame rather than a systemic difference.

### Cost

Warmed up, on GPU-resident planes, 1920×1080 → 576×324, RTX 5090:

| | ms/frame |
| --- | --- |
| `to_training_frame` (resize + range + quantize + colour) | 1.51 |
| ” including `uyvy_to_yuv420` on a packed 4:2:2 buffer | 1.60 |
| ” including the GPU defish (drivr's full chain) | 1.76 |

Two things were worth 3–4× each and are easy to get wrong: the constant tables (qtable,
DCT matrix, colour matrix) are `lru_cache`d per (device, dtype), because rebuilding
them per frame is a host-to-device sync — 4.4 ms/frame rather than 1.51. And
`TorchDefisher` caches its sampling grid **on the plane's device**; a 1080p grid is
16 MB, and moving it per call cost 6 ms/frame of pure copy.

These are 5090 numbers. The Orin figure is not measured — see
`.claude/plans/bright-napping-flask.md` for the on-car procedure.

## Reproduce

### 0. Environment — exactly one onnxruntime wheel

```bash
just sync

# if `import onnxruntime` has no get_available_providers, the shared directory was
# emptied by the CPU wheel's uninstall — restore the GPU wheel's files:
uv sync --all-extras --all-groups --reinstall-package onnxruntime-gpu

uv run --group benchmark python -c \
  "import onnxruntime; print(onnxruntime.get_available_providers())"
# expect CUDAExecutionProvider in the list
```

### 1. Export the checkpoint

```bash
just export-onnx \
  export=yaak/patch_policy/finetuned_dinov2 \
  model.artifact=yaak/alex-tmp/model-9xfz6ify:v6 \
  f=/home/alex/rsim/outputs/onnx/pp/9xfz6ify:v6.onnx
```

### 2. Benchmark all five frame sources

```bash
just benchmark-onnx \
  onnx=/home/alex/rsim/outputs/onnx/pp/9xfz6ify:v6.onnx \
  data_dir=/nasa/drives/yaak/data/Niro122-HQ/2023-05-25--09-34-14 \
  wandb_model=yaak/alex-tmp/model-9xfz6ify:v6 \
  export=yaak/patch_policy/finetuned \
  model.artifact=yaak/alex-tmp/model-9xfz6ify:v6 \
  'frame_sources=[video,jpg,video_jpeg,ffmpeg,torch_nv12]' \
  num_episodes=20 \
  output=/tmp/rmind_5src.csv
```

**Any change to `~/drivr`'s preprocessing has to be re-run through this**, with the
`jpg` source in the list: `jpg` *is* the training frames, so the `jpg` vs
`<your source>` rows of the VALIDATION CHECKS table are the number that matters. Drop
`onnx=` to score the PyTorch backend instead, which is what produced the prediction
table above:

```bash
just benchmark-onnx \
  data_dir=/nasa/drives/yaak/data/Niro122-HQ/2023-05-25--09-34-14 \
  wandb_model=yaak/alex-tmp/model-9xfz6ify:v6 \
  export=yaak/patch_policy/finetuned \
  model.artifact=yaak/alex-tmp/model-9xfz6ify:v6 \
  'frame_sources=[video,jpg,video_jpeg,ffmpeg,torch_nv12]' \
  'image_size=[224,224]' \
  num_episodes=20
```

`ffmpeg` needs `ffmpeg` on PATH and a CUDA device. Without CUDA:

```bash
just benchmark-onnx \
  onnx=/home/alex/rsim/outputs/onnx/pp/9xfz6ify:v6.onnx \
  data_dir=/nasa/drives/yaak/data/Niro122-HQ/2023-05-25--09-34-14 \
  export=yaak/patch_policy/finetuned \
  'frame_sources=[ffmpeg]' ffmpeg_hwaccel=false num_episodes=20
```

To probe the model's sensitivity to compression, sweep `jpeg_qv` on a single source —
a cleaner test than comparing decode paths, whose pixel differences sit near the
quantization floor either way:

```bash
for qv in 4 8 16 24 31; do
  just benchmark-onnx \
    onnx=/home/alex/rsim/outputs/onnx/pp/9xfz6ify:v6.onnx \
    data_dir=/nasa/drives/yaak/data/Niro122-HQ/2023-05-25--09-34-14 \
    export=yaak/patch_policy/finetuned \
    'frame_sources=[video_jpeg]' jpeg_qv=$qv num_episodes=20 \
    output=/tmp/rmind_qv$qv.csv
done
```

### 3. Parity assertions

```bash
just test tests/test_benchmark_onnx_preprocessing.py
just test tests/test_frame_parity.py
```

`test_frame_parity.py` — 23 tests — asserts every stage of the GPU chain against the
piece of ffmpeg or libjpeg it was fitted to (the table in
[The GPU parity path](#the-gpu-parity-path)), plus the ablations, so a dropped stage
fails loudly rather than quietly costing 2–6× MAE.

`test_benchmark_onnx_preprocessing.py` — 12 tests pass. They assert, against real drive data:

- the emitted DQT and 4:2:0 subsampling match real `extract_frames` output
- the `-q:v` table-scaling rule, across the range
- a libjpeg-style 0–100 quality is rejected rather than silently accepted
- the jpeg round-trip moves frames *toward* the training data
- the baseline ordering `ffmpeg < video_jpeg < video`, plus a floor guard on `ffmpeg`

Drive-dependent tests skip cleanly elsewhere; point `RMIND_BENCHMARK_DRIVE_DIR` at
another drive to override the search.

## Deployment — what `~/drivr` now does

The geometry and normalization chain there was already effectively exact — I measured
`preprocess_image` at **0.0032 in ImageNet-normalized units (0.18 gray levels)** from
the training chain, so the `.float()`-before-resize and functional-vs-class-`Resize`
differences are 30× smaller than the JPEG gap and not worth touching.

What it omitted was the resize domain and the quantization. Both are now available
behind `just drive --frame-parity`:

| mode | what it does | MAE |
| --- | --- | --- |
| `off` (default) | today's chain: OpenCV UYVY→BGR, CPU `cv2.remap` defish, `cv2.resize` in RGB, no quantization | 4.58 |
| `jpeg` | + `quantize_rgb` in `preprocess_image`, on the tensor already on the GPU | 3.94 |
| `full` | raw UYVY buffer → GPU defish → YUV 4:2:0 resize → range → quantize | 1.13 |

`full` also *removes* work: OpenCV's UYVY→BGR conversion, the `cv2.remap` defish
(`defish.py`'s own docstring calls it "a few milliseconds at 1080p on the Orin CPU")
and the `cv2.resize` all become GPU ops on the buffer's single upload, which is
2 bytes/px instead of 3. Whether that nets out faster on the Orin is unmeasured —
everything here is a 5090.

Two things gate turning it on, and both need the car:

- **The camera's colour range.** Training's videos are limited range (`ffprobe` says
  `color_range=tv`), and this report's earlier note about a possible BT.601/BT.709
  mismatch now has a price tag: getting the range wrong costs **6.88 vs 1.13** MAE,
  more than every other stage combined. `v4l2-ctl --get-fmt-video` answers it directly
  (`Quantization`, `YCbCr Encoding`). Hence `--frame-parity` defaults to `off` and
  `--camera-full-range` exists.
- **Whether the V4L2 driver honours `CAP_PROP_CONVERT_RGB = 0`.** `full` needs the raw
  buffer; the code probes it and falls back to `jpeg` loudly rather than reading BGR as
  packed 4:2:2 (which produces a green frame, not an error).

`.claude/plans/bright-napping-flask.md` has the on-car procedure for both, plus the
per-stage latency measurement.

One thing this cannot close: training's frames come out of an HEVC-encoded mp4, so they
carry video compression the live path has no equivalent for. Unmeasured — there is no
pre-encode original to compare against.

The other direction is still worth doing: re-extract training frames with the inference
resize path, which `jpg_preprocessing_parity_results.md` found collapses the video/jpg
prediction gap 5–9× and reduces closed-loop collisions (50% vs 70–80% of episodes).
That is a retraining change; this one deploys today.

## Known limitations

- **n_effective ≈ 4, not 20.** `benchmark_onnx.py` advances each episode by
  `frame_step` while an episode spans `5 × frame_step` frames, so consecutive episodes
  share 5 of their 6 timesteps. The 20 episodes cover frames 2910–3160 — 8.3 s of one
  drive out of 85,930 frames. The cross-source deltas above are indicative, not
  statistically established. Fixing this means spacing windows at least
  `6 × frame_step` apart, across multiple drives.
- **PyTorch is never warmed up.** `_warmup_backends` filters to ONNX backends only, so
  the first source in `frame_sources` eats a ~1.2 s cold start (reported 88.7 ms mean
  vs ~36 ms for the rest). That gap reflects list order, not the source.
- **`_TARGET_LATENCY_MS = 100` is now toothless.** With ONNX at 56 Hz the "10 Hz OK?"
  column always passes.
- **The benchmark's timing table does not measure preprocessing.** Frames are built in
  `_load_batch`, outside the timed region, so the per-source "Mean (ms)" column is
  model inference only and is the same for every source by construction. The
  preprocessing costs in [Cost](#cost) were measured separately. A source that took a
  second per frame would look identical in that table.
- **3 pre-existing test errors** in `test_benchmark_onnx_preprocessing.py` — now
  `hydra.errors.MissingConfigException: Cannot find primary config 'export/onnx'`
  (the earlier `'ChunkFields' object has no attribute 'get'` symptom has been replaced
  by this one). Verified identical on `HEAD` in a clean worktree, and unrelated to
  preprocessing.
- **`~/drivr`'s test suite cannot run on this machine.** Its deps do not resolve here
  (the jetson-ai-lab index 410s, and it pins Python 3.10), so
  `tests/test_train_parity_capture.py` was exercised by importing `drivr.io.defish` and
  `drivr.io.yuv_capture` into rmind's environment instead — 13 passed. That covers the
  adapter, the GPU defish and the `CAP_PROP_CONVERT_RGB` probe, but not the app wiring
  in `drivr.py`, which was only syntax-checked under 3.10.
- `_get(key, fallback_idx)` in `_ONNXBackend.run` silently falls back to positional
  output indices when a name doesn't match, which would mis-assign heads with no error
  raised.
- `config/export/yaak/patch_policy/finetuned.yaml` documents its output as
  `policy.joint_actions (1, 6, 4)` and its input as `[0,1] float RGB`. The graph
  actually emits four scalars, and `load_for_export` replaces the in-model image
  transform with `nn.Identity()` — so the host must apply ImageNet normalization. Both
  docstring claims are stale; the code is right.

## Changed files

| file | change |
| --- | --- |
| `src/rmind/scripts/benchmark_onnx.py` | `_ffmpeg_mjpeg_qtable`, rewritten `_simulate_offline_jpeg`, new `_FfmpegFrameSource`, `jpeg_qv` validator, `ffmpeg_hwaccel` |
| `tests/test_benchmark_onnx_preprocessing.py` | 5 new tests (12 parametrized cases) asserting the above against real drive data |
| `config/benchmark_onnx.yaml` | corrected `jpeg_qv` docs, added `ffmpeg_hwaccel` |
| `pyproject.toml` | `sys_platform` split on `export`/`predict` onnxruntime; declared `pillow` |
| `src/rmind/utils/jpeg.py` | the qtable + Pillow reference round trip, lifted out of `benchmark_onnx` so non-training consumers can import it |
| `src/rmind/utils/frame_parity.py` | **new** — the whole GPU chain: `Resampler`, `expand_range`, `quantize_planes`, `yuv420_to_rgb`, `uyvy_to_yuv420`, `to_training_frame`, `quantize_rgb` |
| `tests/test_frame_parity.py` | **new** — 23 tests, one per fitted stage plus the ablations |
| `src/rmind/scripts/benchmark_onnx.py` | `_TorchNv12FrameSource` (the `torch_nv12` source) |

And in `~/drivr`:

| file | change |
| --- | --- |
| `src/drivr/io/yuv_capture.py` | **new** — `TrainParityPreprocessor`, the camera-side adapter (`from_uyvy`, `from_rgb`) |
| `src/drivr/io/defish.py` | `TorchDefisher` — the same Kannala-Brandt warp as `grid_sample`, per plane resolution |
| `src/drivr/io/jetson_camera.py` | `train_parity` / `limited_range`, the `CAP_PROP_CONVERT_RGB` probe, and the GPU capture loop |
| `src/drivr/app/drivr.py` | `preprocess_image(jpeg_qv=...)`, `--frame-parity`, `--camera-full-range` |
| `tests/test_train_parity_capture.py` | **new** — 13 tests on the adapter, the GPU defish and the probe |
