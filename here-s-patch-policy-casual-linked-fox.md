# Task brief: camera identity in the 3-camera causal PatchPolicy

**Audience:** an agent picking this up cold, in `/home/alex/rmind` on branch
`feat/patch-policy-decoder-causal-3cam`. Everything you need is below; you should
not have to re-derive the analysis. Read §1-§2 before touching code.

---

## 1. Background — what the code does today

### 1.1 The paper's scheme

Patch Policy (https://arxiv.org/html/2607.18236v1) §2.1-2.2: a frozen ViT turns
each frame into `P x D` patch features; the `T x P` tensor is flattened, a
**learned 1-D positional embedding indexed by the token's position in the
flattened sequence** is added, and a **block-causal** mask is applied — patches
attend bidirectionally within a frame, causally across frames. Multi-view
(LIBERO: third-person + wrist; real tasks: two wrist cams) gets no special
treatment. View identity is carried **entirely by the absolute slot index**.

### 1.2 What this repo does

`ead564b` added `cam_left_forward` / `cam_right_forward` to the causal arm.

- `src/rmind/models/patch_policy.py:425-427` — `torch.stack([image_by_camera[c]
  for c in self.cameras], dim=2)` → `(b, t, cam, c, h, w)`. `self.cameras` order
  is the only thing fixing which camera is which.
- `src/rmind/models/patch_policy.py:330-338` — one frozen ViT for all cameras,
  then `rearrange(patches, "b t cam p d -> b t (cam p) d")`.
- `:340-350` — shared `fusion_patch_norm` (LayerNorm) x scalar `fusion_patch_gain`,
  goal concat, then a **shared** `patch_projection` = `Linear(384+g -> 512)`,
  xavier-uniform init (`config/model/yaak/patch_policy/raw.yaml:88-91`,
  `src/rmind/components/nn.py:25`).
- `:352-355` — speed token prepended; per-frame block is `[speed, *patches]`,
  `tokens_per_frame = 256*cam + 1` → **769 at cam=3**.
- `src/rmind/components/transformer/causal_frame.py:616-622` —
  `intra_position_embedding = nn.Embedding(tokens_per_frame, dim_model)`,
  `trunc_normal_(std=0.02)`, tiled onto every frame by `_intra` (`:641-643`).
- RoPE is **frame-granular only** (`causal_frame.py:16-34`, `:669-673`), so
  intra-frame attention is exactly unrotated. The intra-frame table is therefore
  the *only* intra-frame positional signal.
- The mask (`causal_frame.py:143-167`) is `frames[:, None] - frames[None, :]`
  with an optional window — it has **no camera-level structure at all**.

Config: `config/experiment/yaak/patch_policy/dinov2_dinowm_causal_3cam.yaml`
(`cameras: [cam_front_left, cam_left_forward, cam_right_forward]`, `num_cameras: 3`,
`tokens_per_frame: ${eval:'${num_patches} * ${num_cameras} + 1'}`). It loads no
artifact — the 3cam arm trains from scratch.

---

## 2. The finding

**Cameras are not indistinguishable.** Camera `c`'s patch `j` sits at intra-frame
slot `1 + 256c + j`, and each slot has its own row in `intra_position_embedding`.
The tests at `tests/test_patch_policy.py:208-262` and
`tests/test_patch_policy_decoder.py:224-268` confirm a camera swap changes the
output. The implementation is faithful to the paper.

**But the signal is in the weakest possible form.** Five reasons, in order of
importance:

1. **No dedicated view direction.** "I am the left camera" is not a parameter — it
   is a property the trunk must infer from 768 mutually-unrelated rows. No vector
   receives gradient from all 256 tokens of a camera at once. Nothing is shared
   between slot `1+j` and slot `1+256+j` either, so spatial structure learned on
   the front camera does not transfer to the sides.
2. **Magnitude.** Table init std = 0.02. Token content at the same point is
   LayerNorm'd patches (element RMS ≈ 1) through xavier `patch_projection`
   → element RMS ≈ 0.9-1.0. Roughly a **45x ratio**: the view/position code is
   ~0.05% of token variance at init. Blocks are pre-LN, which preserves the ratio.
   *(This estimate is derived, not measured — Phase 1 measures it.)*
3. **The frozen ViT works against it.** `src/rmind/components/timm_backbone.py:54-77`
   folds `(b, t, cam)` into the batch dim, so patch `j` of *every* camera gets the
   *identical* pretrained ViT positional embedding. The encoder actively pulls
   same-slot patches from different cameras together in feature space.
4. **Zero per-camera trainable capacity** anywhere — shared ViT, shared
   `fusion_patch_norm`, scalar `fusion_patch_gain`, shared `patch_projection`.
5. **Regime mismatch with the paper.** Wrist vs third-person views are trivially
   separable from *content alone*, so the paper never stresses its
   positional-only scheme. Three forward-facing road cameras have near-identical
   appearance statistics — precisely the case where content cannot disambiguate
   and the weak positional code has to carry it.

### Two incidental findings

6. **The deployed readout changed camera.** Readout is `features[:, :, -1]`,
   "last patch token per frame" (`patch_policy.py:437`), and the speed token is
   prepended so the block ends on a patch (`:354`). At `cam=1` that is
   `cam_front_left`'s last patch; at `cam=3` it is **`cam_right_forward`'s
   bottom-right patch**. Harmless for the from-scratch 3cam run, but it silently
   invalidates readout semantics on the `*_cont.yaml` / `finetuned.yaml`
   warm-start paths. **Decision: out of scope — document only, do not change the
   readout.** Changing it would move `tokens_per_frame` to 770 and break KV-cache
   and ONNX binding shapes.
7. **Camera order is load-bearing and unenforced.** Only a docstring guards it
   (`patch_policy_decoder.py:20`, `:152-153`). `num_cameras` in the 3cam config is
   never cross-checked against `len(model.cameras)`. `ead564b` renamed the hparam
   `image` → `cameras` with no migration. `config/export/yaak/patch_policy/
   finetuned.yaml` still lists only `cam_front_left`, so a cam=3 checkpoint
   KeyErrors at `patch_policy.py:426`.

### Risk register

| Risk | Impact | Likelihood | Action |
|---|---|---|---|
| Flat 769-slot table: no dedicated view direction, no cross-view sharing | High | Certain by construction | Phase 2 |
| View code ~45x quieter than content at init | High | Med-High | Phase 1 measures, Phase 2 fixes |
| Frozen ViT pos-emb pulls same-slot patches together | Med | Certain | amplifier; no direct fix |
| No per-camera trainable capacity | Med | Certain | optional, see §5 |
| Readout anchored on `cam_right_forward` corner | High | Certain | document only (out of scope) |
| `cameras` order unenforced train↔serve; export cfg 1-cam | High (silent) | Med | Phase 3 |
| Tests assert only init-time numerical difference | Med | Certain | Phase 4 |
| 769² intra-frame area (9x vs cam=1); 769 = 6·128+1 flex tiling | Cost | Certain | note only |

---

## 3. Decisions already made (do not revisit)

- **Diagnostics first**, then the fix. Phase 1 gates Phase 2.
- **Clean re-init.** No warm-start decomposition of the trained flat table; the
  3cam arm retrains from scratch.
- **Readout unchanged.** No new readout token, no `tokens_per_frame` change,
  no change to export/KV-cache shapes.

---

## 4. Phase 1 — Diagnostics (no training, no architecture change)

Goal: a defensible number for *"does the trained 3cam trunk distinguish cameras?"*
Run against an existing 3cam checkpoint on the val set.

### New script `src/rmind/scripts/patch_policy_camera_probe.py`

Model it on `src/rmind/scripts/patch_policy_eval.py` — it already does
`--artifact` wandb loading, `initialize_config_dir` / `compose` / `instantiate`
of the datamodule, `_to_device`, batch iteration, and per-frame-position
reporting. Copy that skeleton and reuse its `_default_cluster_fn` so numbers are
comparable with the training-time predict metrics.

Four measurements, reported per camera:

**(a) Camera-swap sensitivity — the decisive test.** Before `model._features`,
swap `cam_left_forward` ↔ `cam_right_forward` in the batch dict; separately swap
`cam_front_left` ↔ `cam_left_forward`. Report Δ in `offset`,
`code_acc_joint_last`, and steering error. Reference scale: Δ from replacing one
camera with an unrelated frame. **Δ ≈ 0 on the left/right swap confirms the model
is genuinely mixing them up.** The swap idiom already exists at
`tests/test_patch_policy.py:218-228` — lift it.

**(b) Per-camera ablation.** Replace one camera's images with (i) zeros, (ii) the
front camera's images, (iii) a per-camera mean frame; report Δ loss per camera.
Drive this through the existing `FeaturePermutator`
(`src/rmind/callbacks/feature_permutation.py`) — it already permutes batch paths
by `MappingKey` and hooks `on_predict_batch_start`; a camera is just the path
`[data, cam_left_forward]`. **Extend it with a `mode: permute | zero | copy_from`
rather than writing a second callback.**

**(c) Linear probe for camera identity.** Per trunk layer, fit logistic
regression from a patch token's hidden state → camera id (3 classes) on held-out
frames. Layer-0 accuracy answers "does identity reach the trunk at all"; the
curve across layers answers "is it preserved". Target >95% at layer 0.
`src/rmind/scripts/residual_unimodality_probe.py` is the closest in-repo
precedent for a probe script's shape.

**(d) Embedding-table audit** (checkpoint only, no data). Row norms of
`encoder.intra_position_embedding.weight` vs the measured RMS of the trunk input
(this is where the ~45x estimate gets confirmed or refuted); the mean vector of
each 256-slot camera band and the pairwise cosines between bands; a 2-D PCA of
the 769x512 table coloured by band. **If the bands do not separate, the table
never learned a view code.**

*Optional if cheap:* attention mass from the readout position onto each camera
band, per layer/head. `src/rmind/callbacks/loggers/patch_similarity.py` has the
per-camera selection and logging scaffolding to borrow.

### Gate

Proceed to Phase 2 if **any** of: left/right swap Δ within noise of zero;
layer-0 probe accuracy < 95%; camera-band cosines > ~0.9. Report all numbers to
the user either way, before starting Phase 2.

---

## 5. Phase 2 — Factorize the intra-frame table

A re-parameterization, not a new mechanism. `_intra` still returns a
`(tokens_per_frame, dim_model)` tensor, so shapes, KV-cache layout, ONNX
bindings and the export/parity path are **untouched**.

### `src/rmind/components/transformer/causal_frame.py`

Replace `intra_position_embedding` (`:619`) with factors, gated on a new ctor arg
so single-camera arms are unaffected:

- `num_cameras: int = 1`; derive `num_patches = (tokens_per_frame - 1) // num_cameras`
  and validate it divides exactly (raise in `__init__`, next to the existing
  `max_sequence_length` guard at `:587-601`).
- `view_embedding = nn.Embedding(num_cameras, dim_model)`
- `patch_position_embedding = nn.Embedding(num_patches, dim_model)`
- `speed_position = nn.Parameter(torch.zeros(1, dim_model))` for slot 0

Composed table (write **one** helper; both `forward` at `:669` and `step` at
`:751-764` must call it, so there is exactly one definition):

```python
table = torch.cat([
    self.speed_position,
    self.view_embedding.weight.repeat_interleave(num_patches, 0)
    + self.patch_position_embedding.weight.repeat(num_cameras, 1),
])
# _intra: table.repeat(num_frames, 1)
```

Params drop 393,728 → 133,632. Camera identity becomes a rank-1 direction fed by
256 tokens x every frame x every sample; patch `j`'s spatial code is now shared
across views.

At `num_cameras=1` this is behaviourally identical to today (a single view vector
added uniformly is absorbed into the patch table), so **no existing arm changes**.

### Scale balancing

The repo already has the idiom: `fusion_patch_gain` / `fusion_goal_gain` and the
measured-RMS calibration in `PatchPolicy._init_fusion_norm`
(`patch_policy.py:357-380`). Mirror it — a learnable scalar gain on the composed
table, init 1.0, logged each epoch — and raise `view_embedding`'s init std above
0.02 (0.05-0.1). **Phase 1(d) sets the exact value; do not guess before it runs.**

### Config

`config/experiment/yaak/patch_policy/dinov2_dinowm_causal_3cam.yaml:100-103` —
add `num_cameras: ${num_cameras}` under `encoder:`, alongside the existing
`tokens_per_frame` / `max_sequence_length` interpolations.

---

## 6. Phase 3 — Ordering and config guards

Independent of Phases 1-2; land regardless of the gate outcome.

- `PatchPolicy.__init__` (`patch_policy.py:178-260`): assert `cameras` has no
  duplicates, and that `encoder.tokens_per_frame == len(cameras) * num_patches + 1`
  when the encoder exposes `tokens_per_frame`. Today a `num_cameras` /
  `len(cameras)` mismatch only surfaces as a length error deep inside
  `CausalFrameTransformer.forward` (`causal_frame.py:665-667`).
- `PatchPolicyDecoderStep.__init__` (`patch_policy_decoder.py:85-90`): assert
  `policy.hparams["cameras"] == policy.cameras`, so an export can never silently
  reorder bands relative to the trained weights.
- `src/rmind/scripts/decoder_only_verify.py`: `_image_input_names` (`:145-161`) is
  already name-based — add an assertion that the ONNX `inputs_image_*` order
  matches `policy.cameras`.
- `config/export/yaak/patch_policy/finetuned.yaml`: add the two side cameras to
  its `input.data` block (or fail loudly) — today a cam=3 checkpoint KeyErrors at
  `patch_policy.py:426`.
- Add a comment at `patch_policy.py:354` and a note in
  `docs/decoder_only_kv_cache.md` recording finding 6 (readout relocates from
  `cam_front_left` to the last camera as `cam` grows), so the continuation /
  finetune arms are not warm-started blind.

*Aside worth flagging to the user, not fixing here:* `docs/decoder_only_kv_cache.md`
ends at §12.9, but `dinov2_dinowm_causal.yaml` and
`src/rmind/callbacks/predict_metrics.py:62` both cite a §13 / §13.1 that was never
written — the cam=3 memory measurements those comments point at are missing.

---

## 7. Phase 4 — Tests

- `tests/test_patch_policy.py:208-228` and `:231-262` currently assert only a
  *numerical* difference at random init, which **any** positional table produces —
  they cannot detect a trained model that ignores camera identity. Add: with the
  factorized embedding, the composed table's per-camera band means are pairwise
  distinct and the per-patch component is shared across bands.
- `tests/test_causal_frame.py`: assert `num_cameras=1` reproduces current
  single-factor behaviour, and that streamed `step` and full-window `forward`
  build the identical composed table (the existing streamed-vs-full equivalence
  test should cover this once both paths call the one helper).
- `tests/test_patch_policy_decoder.py:224-268` must stay green; add the Phase 3
  ordering assertion.

---

## 8. Verification

```bash
just test tests/test_patch_policy.py tests/test_causal_frame.py \
          tests/test_patch_policy_decoder.py
just lint && just typecheck

# Phase 1, against an existing 3cam checkpoint
uv run python -m rmind.scripts.patch_policy_camera_probe \
    --artifact yaak/rmind/model-<run_id>:latest \
    --config-dir "$PWD/config" \
    --experiment yaak/patch_policy/dinov2_dinowm_causal_3cam \
    --batches 200 --device cuda

# Phase 2, end-to-end sanity before a full run
just train-debug experiment=yaak/patch_policy/dinov2_dinowm_causal_3cam

# Phase 3, serving contract unchanged
uv run python -m rmind.scripts.decoder_only_export --arm small_3cam ...
uv run python -m rmind.scripts.decoder_only_verify ...
```

Note: training/eval on this box goes through the nix env, not a bare `uv run` —
check with the user before the first long run.

**Success criterion.** On the re-trained 3cam arm: the Phase 1 left/right
camera-swap Δ is clearly above noise, layer-0 probe accuracy > 95%, and val
`offset` / `code_acc_joint_last` are no worse than the current 3cam baseline.
