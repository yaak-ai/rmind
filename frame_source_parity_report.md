# Frame source parity report

How close can inference-side preprocessing get to the frames the model was actually
trained on, and what was standing in the way.

Measured on `Niro122-HQ/2023-05-25--09-34-14`, checkpoint
`yaak/alex-tmp/model-9xfz6ify:v6` (PatchPolicy / DINOv2, 224×224 input),
`benchmark_onnx.py` with `frame_sources=[video,jpg,video_jpeg,ffmpeg]`.

## Summary

Three things came out of this:

1. **`_simulate_offline_jpeg` was over-compressing by a wide margin.** `-q:v 16` was
   passed to `simplejpeg` as if it were a libjpeg 0–100 quality. Fixed — `video_jpeg`
   now reproduces `extract_frames`' quantization tables byte-for-byte.
2. **Added an `ffmpeg` frame source** that re-runs `extract_frames`' own command. It is
   the baseline: **1.59 uint8 MAE** against the real training jpgs, where the best
   pure-Python path reaches 3.80 and two *adjacent* training frames differ by 1.34.
3. **ONNX was silently running on CPU.** The `export` and `predict` extras pulled the
   CPU `onnxruntime` wheel into the shared venv. Fixed — ONNX went from
   **760 ms (1.3 Hz) to 17.8 ms (56 Hz)**.

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
| `ffmpeg`, `scale_cuda` bicubic — **the floor** | **1.59** |
| *two adjacent real training frames (reference)* | *1.34* |
| `ffmpeg`, `scale_cuda` bilinear | 2.40 |
| `ffmpeg`, software resize, still in NV12 | 2.84 |
| `video_jpeg` — cv2 INTER_CUBIC + mjpeg q16 | 3.80 |
| `video` — cv2 INTER_CUBIC, no jpeg step | 4.42 |
| `video_jpeg` **before the fix** (libjpeg quality 16) | 5.59 |

The ideal case lands just above one frame of scene motion. The residual 1.59 is
`scale_npp` vs `scale_cuda` — NPP ships only in NVIDIA's proprietary build, so that
gap is irreducible locally.

### The resize kernel is not the problem

Every kernel lands on the same ~4.4 floor for the cv2 RGB path, so kernel choice is
second-order:

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

### 2. Benchmark all four frame sources

```bash
just benchmark-onnx \
  onnx=/home/alex/rsim/outputs/onnx/pp/9xfz6ify:v6.onnx \
  data_dir=/nasa/drives/yaak/data/Niro122-HQ/2023-05-25--09-34-14 \
  wandb_model=yaak/alex-tmp/model-9xfz6ify:v6 \
  export=yaak/patch_policy/finetuned \
  model.artifact=yaak/alex-tmp/model-9xfz6ify:v6 \
  'frame_sources=[video,jpg,video_jpeg,ffmpeg]' \
  num_episodes=20 \
  output=/tmp/rmind_4src.csv
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
```

12 tests pass. They assert, against real drive data:

- the emitted DQT and 4:2:0 subsampling match real `extract_frames` output
- the `-q:v` table-scaling rule, across the range
- a libjpeg-style 0–100 quality is rejected rather than silently accepted
- the jpeg round-trip moves frames *toward* the training data
- the baseline ordering `ffmpeg < video_jpeg < video`, plus a floor guard on `ffmpeg`

Drive-dependent tests skip cleanly elsewhere; point `RMIND_BENCHMARK_DRIVE_DIR` at
another drive to override the search.

## Deployment implication

The same measurement applies to `~/drivr`'s `just drive`, which never JPEG-compresses.
Its geometry and normalization chain is already effectively exact — I measured
`preprocess_image` at **0.0032 in ImageNet-normalized units (0.18 gray levels)** from
the training chain, so the `.float()`-before-resize and functional-vs-class-`Resize`
differences are 30× smaller than the JPEG gap and not worth touching. `INTER_AREA` in
its capture loop is also the right kernel.

What it omits is the quantization round-trip. Adding it (Pillow with tables lifted off
a reference training frame, `subsampling=2`) costs ~2.2 ms per 576×324 frame and moves
frames from 4.31 → 3.83 MAE. Given the `ffmpeg` baseline reaches 1.59, though, the
durable fix is the other direction: re-extract training frames with the inference
resize path rather than chasing `scale_npp`. The repo's own
`jpg_preprocessing_parity_results.md` already found that collapses the video/jpg
prediction gap 5–9× and reduces closed-loop collisions (50% vs 70–80% of episodes).

One thing left unverified: `jetson_camera._capture_loop`'s live UYVY→BGR→RGB
conversion. OpenCV's V4L2 path uses BT.601 by default; if the training videos are
BT.709 that is a systematic color shift potentially larger than the JPEG gap. It needs
a real captured frame to measure.

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
- **3 pre-existing test errors** in `test_benchmark_onnx_preprocessing.py`
  (`'ChunkFields' object has no attribute 'get'` in fixture setup). They fail
  identically on `HEAD` and are unrelated to preprocessing.
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
