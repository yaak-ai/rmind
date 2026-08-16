# LFG auxiliary supervision for the causal patch policy — implementation brief

**Status:** not started. This is a hand-off brief; everything below has been verified against the
code as of 2026-08-16 unless explicitly marked as an estimate or an open question.

**Goal.** Offline-label the yaak training frames with the released LFG model, and use those labels
as an auxiliary per-patch loss on the `CausalFrameTransformer` trunk in `PatchPolicy`. The 256
patch tokens the trunk produces per frame are currently supervised by nothing — only the last one
is read (`src/rmind/models/patch_policy.py:392`). This brief puts a signal on the other 256.

**Non-goals.** Do not replace the frozen DINOv2 `image_encoder` with LFG. Do not change anything on
the serving path: the aux heads are training-only and `PatchPolicyDecoderStep` must be untouched.
Do not use LFG's geometry (point maps / poses) in the first pass — see §9.

**Licence — read before running anything.** The LFG *code* is Apache 2.0 (headers on every file in
the repo). The *weights* are **CC BY-NC 4.0**, i.e. non-commercial. Labels derived from those
weights, and any policy trained on them, inherit that question. Get this resolved before spending
GPU time; it is the only true blocker in this brief.

---

## 1. Established facts

Verified by reading source. Cite these rather than re-deriving them.

### 1.1 LFG

- Repo: `https://github.com/Applied-Intuition-Open-Source/LFG` (Apache 2.0, ~880 KB, no weights).
- Weights: `https://huggingface.co/AppliedIntuitionResearch/LFG`, single file
  `lfg_seg_motion_m3n3.pt`, 4.87 GB, ungated, no auth. 1.22B params.
- Architecture (`Pi3/pi3/models/pi3.py`): DINOv2 **ViT-L/14 with registers** encoder (1024-d) →
  36-layer alternating-attention decoder (1024-d; even layers intra-frame, odd layers cross-frame)
  → autoregressive transformer for future frames → six task decoders + heads.
- `patch_size = 14`, `patch_start_idx = 5` (five register tokens prepended to every frame block).
- **ImageNet normalization is applied inside `forward`** with the same constants rmind uses. Feed
  `[0, 1]` floats, do NOT pre-normalize.
- Deps are plain: `torch, numpy, opencv-python, pillow, matplotlib, tqdm, yacs`. `RoPE2D` is pure
  PyTorch (`Pi3/pi3/models/layers/pos_embed.py`) and attention falls back to
  `torch.nn.functional.scaled_dot_product_attention` — **no CUDA extension, no flash-attn, no
  xformers required** despite the `cuRoPE2D` error string in `pi3.py`.
- `MAX_TOTAL_FRAMES = 15` (`lfg/model.py:27`) and `forward(imgs, n_future_frames_override=0)` gives
  "current-frames-only mode (no AR work)". **Use m=15, n=0 for labelling** — more context per pass
  and the AR transformer is skipped entirely.
- `model_config_from_checkpoint(cfg, state_dict)` (`lfg/config.py`) infers the architecture from the
  state dict, so you do not have to hard-code it.

**Checkpoint inspected 2026-08-16** — `torch.load` gives top-level keys
`['config', 'global_step', 'model_state_dict']`, `global_step = 10000`, **1392 tensors / 1.218B
params, all float32** (hence the 4.87 GB; cast to bf16 for inference). Resolved `ModelConfig`:

```
m=3  n=3  encoder_name=dinov2  decoder_size=large  ar_n_heads=8  ar_n_layers=4
use_segmentation_head=True  segmentation_num_classes=7  use_motion_head=True
use_flow_head=False  point_head_type=linear
```

Head shapes confirm the class counts directly (`LinearPts3d` emits `output_dim * 14²= output_dim *
196` channels): `segmentation_head.proj.weight (1372, 1024)` → **1372/196 = 7 classes**;
`motion_head (196, 1024)` → 1 channel; `point_head (588, 1024)` → 3; `conf_head (196, 1024)` → 1;
`camera_head.fc_rot (9, 512)` + `fc_t (3, 512)`. **`flow_head` is absent**, as the filename implies.

Parameter budget, which matters for §3.6:

| module | params |
|---|---|
| `decoder` (36 alternating layers) | 453.6M |
| `encoder` (DINOv2 ViT-L/14 reg) | 304.4M |
| `autoregressive_transformer` | 125.9M |
| `point_decoder` / `conf_decoder` / `segmentation_decoder` / `motion_decoder` | 66.1M each |
| `camera_decoder` | 65.6M |
| heads | ~4.5M total |

**Two free compute savings for the labelling job**, worth ~35% of the forward pass:
`n_future_frames_override=0` skips the 126M-param AR transformer (already in §3.3), and
`point_decoder` + `camera_decoder` (132M params of full-sequence transformer) are computed
unconditionally in `forward` but unused by pass 1 — patch `forward` to skip them, or accept the
cost. Do not skip `conf_decoder`; §3.4 uses its output as the loss weight.

### 1.2 LFG outputs (`LFG.forward` return dict)

Every head is `LinearPts3d`, which projects to `output_dim * 14²` and `pixel_shuffle`s — so
**outputs are at full pixel resolution H×W, not the patch grid**.

| Key | Shape | Used here |
|---|---|---|
| `segmentation` | `[B, N+M, H, W, 7]` | **yes** |
| `motion` | `[B, N+M, H, W, 1]` | **yes** |
| `conf` | `[B, N+M, H, W, 1]` | **yes** (as a loss weight) |
| `local_points` / `points` | `[B, N+M, H, W, 3]` | not in pass 1 |
| `camera_poses` | `[B, N+M, 4, 4]` | not in pass 1 |
| `flow` | `[B, N+M, H, W, 2]` | head absent in this ckpt |
| `dino_features`, `pi3_features`, `autonomy_features`, `point_features`, `conf_features`, `camera_features`, `all_decoder_features` | `[B*(N+M), 5+P, D]` | not in pass 1 |

Point maps are up to **scale and shift**; poses are up to **scale**. Neither is needed for pass 1,
which is exactly why pass 1 is the cheap one — segmentation and motion are categorical and carry no
gauge ambiguity at all.

### 1.3 yaak data

- Frames are **pre-extracted JPEGs**, not video:
  `/nasa/drives/yaak/data/{drive_id}/frames/cam_front_left.pii.mp4/576x324/{:09d}.jpg`
  (source-of-truth: `config/_templates/dataset/yaak/train.yaml:671`). Native size **576×324**.
- Read by `rbyte.io.PathTensorSource` with
  `simplejpeg.decode_jpeg(colorspace="rgb", fastdct=True, fastupsample=True)`, indexed by
  `meta/ImageMetadata.cam_front_left/frame_idx`.
- `paths.data` = `/nasa/drives/yaak/data` (`config/paths/yaak/default.yaml`);
  `/mnt/verda-nas` on Verda (`config/paths/yaak/verda.yaml`).
- **655 train drives.** Total ≈1.97M training samples.
- Model input pipeline (`config/model/yaak/patch_policy/raw.yaml:38-57`):
  `Rearrange(h w c -> c h w)` → `CenterCrop([320, 576])` → `Resize([224, 224])` →
  `ToDtype(scale=True)` → `Normalize(ImageNet)`.
- **Only every 10th raw frame is ever loaded.** `episode_stride: 10` and `episode_step: 10` mean
  clip starts are at row multiples of 10 and frames within a clip step by 10 — i.e. exactly 3 Hz off
  a 30 Hz source. **Label only frames whose `frame_idx % 10 == 0`.** This is the single biggest
  compute saving available; do not label all 30 Hz frames.

### 1.4 Field of view lines up

`CenterCrop([320, 576])` is a **wide 1.80-aspect crop** (not square). LFG's own
`preprocess_frames(target_size=518, mode="crop", patch_size=14)` applied to a 320×576 image yields
`new_width=518`, `new_height=round(320*(518/576)/14)*14 = 294`, with no further cropping — i.e.
**294×518, the paper's resolution, same field of view, ≤2% anisotropic squash.**

This makes label↔patch alignment exact: the LFG output grid, the 320×576 crop, and the 224×224
model input are all related by pure resizes, so a spatial pooling from the LFG output straight to
16×16 lands on the model's patch grid regardless of the intermediate aspect.

### 1.5 The trunk side

- `dinov2` arm: `image_resize: [224, 224]`, `num_patches: 256` (16×16 grid of 14×14 px),
  `policy_embedding_dim: 512`, `num_layers: 8`, `num_heads: 8`.
- `tokens_per_frame = 257`: `_frame_tokens` concatenates `[speed_token, patches]`, speed **first**,
  so patches are indices **1..256** and the readout is index 256 (`patch_policy.py:316`).
- Patch order is `Rearrange("... c h w -> ... (h w) c")` → **row-major 16 rows × 16 cols**. Labels
  must be flattened with the same convention.
- Trunk output already carries a final `LayerNorm` (`CausalFrameTransformer.norm`). `self.norm` on
  `PatchPolicy` is the *policy readout* norm — the aux heads must NOT reuse it.

---

## 2. Environment

```bash
# LFG inference package — keep it OUT of the rmind venv to avoid dependency drift
git clone --depth 1 https://github.com/Applied-Intuition-Open-Source/LFG.git /nasa/tools/lfg
uv venv /nasa/tools/lfg/.venv --python 3.12
uv pip install --python /nasa/tools/lfg/.venv/bin/python -r /nasa/tools/lfg/requirements.txt simplejpeg
curl -L -o /nasa/tools/lfg/lfg_seg_motion_m3n3.pt \
  https://huggingface.co/AppliedIntuitionResearch/LFG/resolve/main/lfg_seg_motion_m3n3.pt
```

A partial copy of the checkpoint and a clone of the repo already exist under this session's
scratchpad; prefer re-downloading to a durable path rather than depending on scratchpad.

Hardware: one RTX 5090 (32 GB) is present. bf16 autocast; the 15-frame global-attention layers see
15 × 782 = 11730 tokens, which fits comfortably.

---

## 3. Stage 1 — the labelling job

Write `scripts/lfg_label_drives.py` (new file; `scripts/` is already an untracked working dir in
this repo).

### 3.1 Frame selection

For each of the 655 drives in `config/_templates/dataset/yaak/train.yaml` (plus the val drive list
in `val.yaml`):

```
frames = sorted(glob(f"{paths.data}/{drive}/frames/cam_front_left.pii.mp4/576x324/*.jpg"))
targets = [f for f in frames if int(f.stem) % 10 == 0]
```

Chunk `targets` into consecutive non-overlapping windows of 15. A trailing partial window is fine —
`decode()` handles any N; just pad the last window by repeating its final frame and discard the
padded slots' outputs.

### 3.2 Preprocessing — must match training byte-for-byte upstream of the crop

There is prior evidence in this repo (`jpg_preprocessing_parity_results.md`) that resize-kernel
differences between the training JPEGs and the inference path materially changed closed-loop
behaviour. Do not improvise here.

```python
import simplejpeg, torch, torch.nn.functional as F

rgb = simplejpeg.decode_jpeg(path.read_bytes(), colorspace="rgb",
                             fastdct=True, fastupsample=True)      # (324, 576, 3) uint8
x = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0          # (3, 324, 576)
x = torchvision.transforms.v2.functional.center_crop(x, [320, 576]) # (3, 320, 576)  <- SAME op as training
x = F.interpolate(x[None], size=(294, 518), mode="bicubic",
                  align_corners=False).clamp_(0, 1)[0]              # (3, 294, 518)
```

- The decoder settings and the `CenterCrop([320, 576])` must be identical to
  `config/model/yaak/patch_policy/raw.yaml`. Everything after the crop is ours to choose, because
  the label is pooled back onto the patch grid.
- `bicubic` matches LFG's own `_resize_image`. Do not pre-normalize — `LFG.forward` does it.

### 3.3 Inference

```python
imgs = torch.stack(window)[None].to("cuda")            # (1, 15, 3, 294, 518)
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    out = model(imgs, n_future_frames_override=0)      # skips the AR transformer
```

Build the model via `lfg.model.build_model(lfg.config.model_config_from_checkpoint(cfg, state))` —
do not hard-code the architecture.

### 3.4 Reduction to the 16×16 patch grid

Pool directly from the LFG pixel grid to 16×16. `adaptive_avg_pool2d` *is* the correct area-average
resize; there is no need to go via 224×224.

```python
seg  = out["segmentation"][0].permute(0, 3, 1, 2).float()   # (15, 7, 294, 518)
mot  = out["motion"][0].permute(0, 3, 1, 2).float()         # (15, 1, 294, 518)
conf = out["conf"][0].permute(0, 3, 1, 2).float()           # (15, 1, 294, 518)

p    = F.adaptive_avg_pool2d(seg.softmax(dim=1), (16, 16))  # (15, 7, 16, 16) class fractions
seg_label  = p.argmax(dim=1).to(torch.uint8)                # (15, 16, 16)  dominant class
seg_purity = (p.max(dim=1).values * 255).round().to(torch.uint8)
motion     = (F.adaptive_avg_pool2d(mot.sigmoid(),  (16, 16))[:, 0] * 255).round().to(torch.uint8)
confidence = (F.adaptive_avg_pool2d(conf.sigmoid(), (16, 16))[:, 0] * 255).round().to(torch.uint8)

packed = torch.stack([seg_label, seg_purity, motion, confidence], dim=1)  # (15, 4, 16, 16) uint8
```

`seg_purity` is the fraction of the patch covered by its dominant class — used downstream to drop
patches that straddle a boundary. `conf` is passed through `sigmoid` purely to get a bounded
relative weight; it is never interpreted as a calibrated probability.

### 3.5 Output format and path

One file per labelled frame, raw uint8, **no header**:

```
/nasa/drives/yaak/lfg_labels/v1/{drive_id}/{frame_idx:09d}.bin      # exactly 1024 bytes
```

Layout: `(4, 16, 16) uint8` C-order — channel 0 `seg_label`, 1 `seg_purity`, 2 `motion`,
3 `confidence`. Row-major within each 16×16 plane, matching the patch order in §1.5.

Rationale for headerless `.bin`: the decoder becomes a one-liner
(`np.frombuffer(b, np.uint8).reshape(4, 16, 16)`), there is no npy/npz parsing per sample, and the
file is a fixed size so corruption is detectable by `stat` alone.

- **Write to a new root, not into `paths.data`** — the drive tree is shared and should be treated
  as read-only.
- Volume: ~1.97M frames × (1024 B payload, 4 KiB block) ≈ **2 GB logical / ~8 GB on disk**, ~3000
  files per drive directory. The sibling JPEG directories already hold ~30k files each, so this
  access pattern is not new to the filesystem.
- If small-file IO over NFS turns out to be the training bottleneck, the fallback is one memmapped
  array per drive plus a custom `TensorSource`; do not do this pre-emptively.

Also write, per drive, `/nasa/drives/yaak/lfg_labels/v1/{drive_id}/manifest.json`:
`{"n_frames": int, "frame_indices": [...], "lfg_sha256": str, "script_git_sha": str,
"crop": [320,576], "lfg_resolution": [294,518], "grid": [16,16], "created": "..."}`.
The training job must fail loudly if the manifest is missing or its geometry disagrees with the
config.

### 3.6 Run it as a pilot first

Do **not** launch 655 drives blind. Run the 30-drive list from
`config/dataset/yaak/train_subset30.yaml` first (the same subset
`config/datamodule/yaak/predict_train_subset.yaml` uses), then execute §7.1 and §7.2. Only then
scale out.

Compute is an **estimate**: ~1.97M frames ÷ 15 per window ≈ 131k forward passes of a 1.22B model.
Measure windows/s on the pilot and extrapolate before committing; order-of-days on a single 5090 is
plausible and the job is embarrassingly parallel across drives.

---

## 4. Stage 2 — plumbing the labels into the batch

### 4.1 New path config key

Add to `config/paths/yaak/default.yaml` and `verda.yaml`:

```yaml
lfg_labels: /nasa/drives/yaak/lfg_labels/v1
```

### 4.2 Decoder helper

New file `src/rmind/utils/lfg_labels.py`:

```python
import numpy as np
import numpy.typing as npt

LFG_LABEL_SHAPE = (4, 16, 16)
LFG_LABEL_NBYTES = 4 * 16 * 16


def decode_lfg_label(data: bytes) -> npt.NDArray[np.uint8]:
    """Decode a packed LFG per-patch label blob written by scripts/lfg_label_drives.py.

    Layout is `(4, 16, 16)` uint8, C-order: seg_label, seg_purity, motion, confidence.

    Raises:
        ValueError: if the blob is not exactly `LFG_LABEL_NBYTES` long.
    """
    if len(data) != LFG_LABEL_NBYTES:
        msg = f"expected {LFG_LABEL_NBYTES} bytes, got {len(data)}"
        raise ValueError(msg)
    return np.frombuffer(data, dtype=np.uint8).reshape(LFG_LABEL_SHAPE)
```

### 4.3 New rbyte stream

In `config/_templates/dataset/yaak/train.yaml` (and `val.yaml`), alongside `cam_front_left`:

```yaml
streams:
  cam_front_left: { ... unchanged ... }

  lfg_labels:
    index: meta/ImageMetadata.cam_front_left/frame_idx     # same index as the image
    sources:
      #@ for/end drive_id in drives:
      (@=drive_id@):
        _target_: rbyte.io.PathTensorSource
        path: "${paths.lfg_labels}/(@=drive_id@)/{:09d}.bin"
        decoder:
          _target_: rmind.utils.lfg_labels.decode_lfg_label
          _partial_: true
```

Edit the **template**, then `just generate-config`. Sharing `index` with the image stream is what
guarantees the label and the frame are the same frame.

### 4.4 Remap into the batch

In `config/model/yaak/patch_policy/raw.yaml`, extend the `Remapper` `context` group:

```yaml
        context:
          waypoints: [data, waypoints/xy_normalized]
          lfg: [data, lfg_labels]
```

`context` already maps to `torch.nn.Identity` in the per-modality `ModuleDict`, so the tensor passes
through untouched as `(b, t, 4, 16, 16)` uint8.

### 4.5 Dataset cache

Adding a stream changes the dataset build. Rebuild the rbyte cache with a **single writer** — two
concurrent builds corrupt the samples store, and the cache is keyed only by output name (there is a
known contamination risk with concurrent runs).

---

## 5. Stage 3 — the aux loss in `PatchPolicy`

All edits in `src/rmind/models/patch_policy.py` unless noted.

### 5.1 Expose the patch tokens

`_features` currently discards everything but the readout. Change it to return the reshaped block:

```python
def _features(
    self, batch: Any, *, require_chunk: bool = True
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Readout features `(b, t, d)`, the full token block `(b, t, k, d)`, and chunks."""
    ...
    blocks = rearrange(embedding, "b (t k) d -> b t k d", t=num_frames)
    features = blocks[:, :, -1]
    if self.norm is not None:
        features = self.norm(features)
    return features, blocks, chunk
```

Update the three call sites — `_compute_metrics`, `forward`, `predict_step` — to unpack three
values (`features, _blocks, chunk`). `blocks` is the trunk output *after* `CausalFrameTransformer`'s
own `LayerNorm` and *before* `self.norm`; the aux heads read `blocks`, the policy head reads
`features`. Keep it that way.

### 5.2 New constructor parameters

```python
aux_heads: HydraConfig[ModuleDict] | InstanceOf[ModuleDict] | None = None,
aux_weights: dict[str, float] | None = None,
aux_purity_min: float = 0.6,
lfg_labels: Path = ("context", "lfg"),
```

Register with `init_hydra_param(hparams, "aux_heads", aux_heads)` like the others, and record
`aux_weights` / `aux_purity_min` / `lfg_labels` in `hparams`. `@validate_call` will reject a plain
`dict` where a `ModuleDict` is expected — pass the right type.

### 5.3 The loss

New method, called from `_compute_metrics` when `self.aux_heads is not None`:

```python
def _aux_metrics(
    self, blocks: Tensor, labels: Tensor
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Per-patch auxiliary losses against the cached LFG labels.

    `blocks` is `(b, t, 257, d)`; patches are indices 1..256 in row-major 16x16 order,
    matching the label planes. `labels` is `(b, t, 4, 16, 16)` uint8.
    """
    tokens = blocks[:, :, 1:]                                   # (b, t, 256, d)

    seg_target = labels[:, :, 0].flatten(-2).long()              # (b, t, 256)
    purity     = labels[:, :, 1].flatten(-2).float() / 255.0
    motion     = labels[:, :, 2].flatten(-2).float() / 255.0
    conf       = labels[:, :, 3].flatten(-2).float() / 255.0

    # confidence-weighted, and boundary-straddling patches dropped entirely
    weight = conf * (purity >= self.aux_purity_min)
    denom = weight.sum().clamp(min=1.0)

    losses: dict[str, Tensor] = {}
    metrics: dict[str, Tensor] = {}

    seg_logits = self.aux_heads["segmentation"](tokens)          # (b, t, 256, 7)
    seg_nll = F.cross_entropy(
        rearrange(seg_logits, "b t p c -> (b t p) c"),
        seg_target.flatten(),
        reduction="none",
    ).view_as(weight)
    losses["segmentation"] = (seg_nll * weight).sum() / denom

    motion_logit = self.aux_heads["motion"](tokens)[..., 0]      # (b, t, 256)
    motion_bce = F.binary_cross_entropy_with_logits(
        motion_logit, motion, reduction="none"
    )
    losses["motion"] = (motion_bce * weight).sum() / denom

    with torch.no_grad():
        correct = (seg_logits.argmax(dim=-1) == seg_target).float()
        metrics["segmentation_acc"] = (correct * weight).sum() / denom
        metrics["motion_mae"] = (
            (motion_logit.sigmoid() - motion).abs() * weight
        ).sum() / denom
        metrics["supervised_fraction"] = (weight > 0).float().mean()

    return {k: v * self.aux_weights[k] for k, v in losses.items()}, metrics
```

Return it as a second top-level group so the existing machinery picks it up unchanged:

```python
result = {"policy": {"loss": losses, "metric": metrics}}
if self.aux_heads is not None:
    aux_losses, aux_metrics = self._aux_metrics(blocks, self._get(inputs, self.lfg_labels))
    result["aux"] = {"loss": aux_losses, "metric": aux_metrics}
return TensorDict(result)
```

`_step` computes `metrics.select(*((k, "loss") for k in metrics.keys()))` and sums, so the aux terms
enter `loss/total` automatically and log as `train/aux/loss/segmentation` etc. **Weights must be
applied inside `_aux_metrics`** (as above) because that sum is unweighted.

`_compute_metrics` needs `inputs` in scope — it currently only receives `_features`' return value.
Either have `_features` also return `inputs`, or fetch the labels in `_compute_metrics` via a small
second call to `self.input_transform`. Prefer the former; do not run `input_transform` twice.

### 5.4 What must NOT change

- `PatchPolicyDecoderStep` (`src/rmind/models/patch_policy_decoder.py`) reads only the trunk plus
  `code_head`/`offset_head`/tokenizer. Aux heads are simply absent from it. Confirm the ONNX export
  still runs after the change; nothing should require editing there.
- `readout_only_final_block=True` means the export step does not compute the final block's
  non-readout positions. That is fine — aux is train-only — but it does mean the aux head could
  never be served without disabling that flag. Note it, don't design around it.
- Gradients flow into the trunk, `patch_projection`, the fusion gains and `speed_embedding`. They do
  **not** reach `image_encoder`, `goal_encoder` or `tokenizer` — all three are permanently frozen
  and forced to `eval()` in `train()`.

---

## 6. Stage 4 — the experiment arm

New file `config/experiment/yaak/patch_policy/dinov2_dinowm_causal_lfgaux.yaml`:

```yaml
# @package _global_
defaults:
  - /experiment/yaak/patch_policy/dinov2_dinowm_causal
  - _self_

model:
  aux_heads:
    _target_: rmind.components.containers.ModuleDict
    modules:
      segmentation:
        _target_: rmind.components.nn.Linear
        in_features: ${policy_embedding_dim}
        out_features: 7
      motion:
        _target_: rmind.components.nn.Linear
        in_features: ${policy_embedding_dim}
        out_features: 1
  aux_weights:
    segmentation: 0.1
    motion: 0.1
  aux_purity_min: 0.6

wandb:
  tags: [patch_policy, dinov2_dinowm_causal, lfg_aux]
```

Start with linear probes. If they fit too easily (segmentation accuracy saturating early), the
signal is not shaping the trunk — deepen to a 2-layer MLP rather than raising the weight.

`aux_weights` is the one hyperparameter that matters. Sweep `{0.03, 0.1, 0.3}` after the
`aux_weights: 0` control passes §7.3.

---

## 7. Validation gates

Pass each before proceeding to the next stage.

**7.1 Spatial alignment (pilot labels).** Render a labelled frame's 16×16 `seg_label` plane as a
heatmap over the 224×224 model input for ~20 frames. Road must dominate the lower-centre patches and
sky the upper band. A vertical flip or a transpose here is the single most likely bug and is
invisible in the loss — it just trains slightly worse. Also assert numerically that the road-class
fraction in the bottom two patch rows exceeds that in the top two.

**7.2 Round-trip through the datamodule.** For a known `(drive_id, frame_idx)`, assert the tensor
delivered at `("context", "lfg")` equals `decode_lfg_label(Path(...).read_bytes())` exactly, and
that the image delivered at `("image", "cam_front_left")` came from the same `frame_idx`.

**7.3 Zero-weight control.** An arm with `aux_weights: {segmentation: 0.0, motion: 0.0}` must
reproduce the baseline `dinov2_dinowm_causal` training curve. Any divergence means the plumbing
perturbed batch composition or RNG, which invalidates every subsequent comparison.

**7.4 Selection metric.** Judge arms on `offset_argmax_recon_last` and `code_acc_joint_last`, **not**
on `val/loss/code_*`. This is already documented in `_readout_metrics` (`patch_policy.py:534-548`):
on `dashing-dream-514` val `code_0` rose 256% while the deployment-aligned argmax recon *improved*
13%, and the arm with the best val `code_0` had the worst argmax recon and underperformed in sim.
Also watch `code_partial_window` vs `code_full_window` — if the aux loss helps only the partial
bucket it is fixing cold-start, not steady-state.

**7.5 A/B hygiene.** `dinov2_dinowm_causal.yaml` §4 already warns that `attn_dropout` must be held
equal across arms or the kernel change confounds the comparison. The same applies here: change
*only* `aux_weights` between the control and the treatment.

---

## 8. Risks and known traps

| Risk | Mitigation |
|---|---|
| **Licence (CC BY-NC 4.0)** | Resolve before any large run. Blocks productionisation, not experimentation. |
| Preprocessing drift between labeller and trainer | §3.2 pins the decoder settings and the crop; `jpg_preprocessing_parity_results.md` documents why this matters. |
| Patch-order mismatch (flip/transpose) | Gate 7.1. |
| 7-class imbalance (road + sky dominate) | Log per-class accuracy, not just the mean. Consider class weights only if a minority class is at 0 recall. |
| Motion masks are the noisiest teacher (SAM2 + CoTracker3 pseudo-labels, distilled) | If the motion term hurts, drop it and keep segmentation — they are independent terms. |
| Aux loss trades away policy accuracy | Gates 7.3 + 7.4; the weight sweep exists for this. |
| Small-file IO on NFS | Measure step time against the control; fallback in §3.5. |
| Concurrent rbyte cache builds | Single-writer build only. |

---

## 9. Deferred to a second pass

Only after pass 1 shows a measurable effect:

- **Geometry.** `local_points` (up to scale *and* shift → use a scale-invariant loss such as SILog,
  which sidesteps both) and `camera_poses` (up to scale only → fix from
  `meta/VehicleMotion/speed`, which is in **km/h**, so divide by 3.6; `Δt = 1/3 s`; least-squares
  one scalar per 15-frame window using only frames above a speed threshold, then apply it to the
  stationary frames in that window — a stopped vehicle gives an unidentifiable scale on its own).
- **Feature distillation** against `autonomy_features` instead of task labels. Much larger storage
  (~512 KB/frame vs 1 KB), and it should only be attempted if task-label supervision proves the
  representation is the bottleneck.
- **LFG's future frames** (`n_future_frames_override=3`) as a forward-prediction target on the
  trunk. This is the most interesting direction and the most likely to interact badly with the
  windowed causal mask; treat it as its own investigation.

---

## 10. Open questions for the implementer

1. Does `paths.lfg_labels` want to live on the same NFS volume as the drives, or on local NVMe? The
   training-time read pattern is random-access over ~2M 1 KB files.
2. Should the val drives be labelled with the same script and the aux loss evaluated on val, or is
   the aux term train-only? Recommendation: label both, log the val aux metrics, but exclude the aux
   term from the val loss so `val/loss/total` stays comparable to prior arms.
3. `seg_purity` threshold `0.6` is a guess. Report the resulting `supervised_fraction` from the
   pilot and tune it so that roughly 70–85% of patches are supervised.
