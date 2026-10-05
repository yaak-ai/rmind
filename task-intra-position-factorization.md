# Task brief: configurable intra-frame position embedding for the 3-cam causal PatchPolicy

**Audience: an implementing agent with no prior context on this work.** Everything you
need is here or cited by `file:line`. Companion document:
`here-s-patch-policy-casual-linked-fox.md` (the camera-identity brief) — this task
un-shelves its §5 and makes its §4.2 open question configurable. Read that document's
§2, §4.2, §4.6 and §5 before starting; they are the evidence base and are not repeated
in full here.

______________________________________________________________________

## 0. Decisions already made — do not revisit

- **Everything is a hydra flag. Nothing is deleted.** `b846a4f`'s norm+gain stays in the
  code as the default; the new modes are opt-in.
- **Defaults must preserve today's behaviour bit-for-bit** — same parameter names, same
  state-dict keys, same init RNG draw order. This is a testable gate, not an aspiration.
- **One new training arm**, all three changes on. No dedicated `b846a4f`-isolating run:
  `03tuy3q9` (pre-fix) vs `kughoqfi` (post-fix) already bounds the direction, and since
  every axis is a flag, further arms are CLI overrides at zero code cost.
- **Clean re-init.** The factorized arms train from scratch. No warm-start decomposition
  of the trained flat table (this matches the camera-identity brief §3).
- **The composed table shape never changes.** It stays `(tokens_per_frame, dim_model)`,
  so the KV-cache layout, ONNX bindings and the export/parity path are untouched. Any
  design that changes `tokens_per_frame` is out of scope.
- **Camera overlap is handled explicitly, not ignored.** See §3.2.

______________________________________________________________________

## 1. Why

`CausalFrameTransformer` gives every one of the 3-cam arm's 769 intra-frame slots an
independent row in a flat `nn.Embedding` (`src/rmind/components/transformer/causal_frame.py:619`).
The camera-identity brief §2 shows why that is the weakest possible encoding:

1. **No dedicated view direction.** "I am the left camera" is not a parameter — it is a
   property the trunk must infer from 768 mutually-unrelated rows. No vector receives
   gradient from all 256 tokens of a camera at once.
1. **No cross-view sharing.** Nothing is shared between slot `1+j` and slot `1+256+j`, so
   spatial structure learned on the front camera does not transfer to the sides.
1. **`b846a4f`'s side effect.** Its `nn.LayerNorm(dim_model, elementwise_affine=False)`
   forces *every* row to norm 11.15, including slots that had learned to be quiet. §4.2
   measured the speed token going from 1.74x content/position to **0.19x** — the speed
   token entering the trunk is now ~84% a fixed position vector and ~16% actual speed.
   The brief left "scope the norm+gain to the patch band only, or accept" undecided.

Three changes are wanted:

1. Back out `intra_position_norm` + `intra_position_gain`, **as a config option**.
1. Add a camera ("view") embedding — the brief's §5 factorization, un-shelved.
1. Add 2D positional structure spanning all three cameras.

All three are the same object — how the `(tokens_per_frame, dim_model)` table is
parameterized and scaled — so they land as two orthogonal, hydra-selectable axes.

______________________________________________________________________

## 2. Established facts

Verified against the tree at `6c9e61d` on branch `feat/patch-policy-3cam-retile-patch`.
**Do not re-derive these.** Do re-check any `file:line` you are about to edit.

### 2.1 Token layout

The 3-cam arm is `tokens_per_frame = 256*3 + 1 = 769`, laid out **camera-major**:

```
[ speed(1) | cam_front_left x256 | cam_left_forward x256 | cam_right_forward x256 ]
```

- Built by `PatchPolicy._frame_tokens` (`src/rmind/models/patch_policy.py:448-524`).
  Speed is prepended deliberately (`:501`, "speed first so the frame block ENDS on the
  readout position"). Camera-major comes from
  `rearrange(patches, "b t cam p d -> b t (cam p) d")` (`:483`).
- The canonical band-slice helper is `_band_slices`
  (`src/rmind/scripts/patch_policy_position_audit.py:63-78`), whose docstring says it
  "must mirror `PatchPolicy._frame_tokens`'s `torch.cat` order exactly". It is
  **duplicated** as `_slot_layout` in `src/rmind/scripts/patch_policy_camera_probe.py:112-129`.
- **Register and readout tokens are opt-in** (`use_readout_token` / `num_register_tokens`,
  `patch_policy.py:223-224`, defaults `False`/`0`) and sit **after** all patch bands,
  registers first, readout strictly last (`:503-510`), because the head reads
  `[:, :, -1]` (`:628-631`). They are **OFF** in the 3-cam config but **ON** in
  `dinov2_dinowm_causal_readout.yaml` (1-cam, 2 registers + readout, 260 slots). Your
  design must represent both.
- `PatchPolicy._features` (`:609-623`) already raises if the produced layout length
  differs from `encoder.tokens_per_frame`.

### 2.2 Patch geometry

- **16x16 = 256 patches per camera, row-major** — patch index `i` maps to
  `(row, col) = divmod(i, 16)`.
- DINOv2 ViT-S/14 @ 224 (`config/experiment/yaak/patch_policy/dinov2.yaml:10-11`,
  `image_embedding_dim: 384`). Prefix (CLS/register) tokens are dropped in
  `src/rmind/components/timm_backbone.py:67-72`, which reshapes with an implicit
  **square-grid assumption** (`grid = isqrt(tokens.shape[1])`).
- **There is no grid-aware position handling in the PatchPolicy path today.** The trunk's
  table is flat 1-D over the 769 slots. Inside the ViT, timm's own `pos_embed` is applied
  and — per the brief §2 finding 3 — patch `j` of *every* camera gets the **identical**
  pretrained ViT positional embedding, actively pulling same-slot patches from different
  cameras together in feature space.

### 2.3 Rig geometry (`/nasa/alex/docs/yaak-data-samples.md`, "Camera placement(s)")

Frames are stored as **576x324** JPEGs
(`config/dataset/yaak/train_3cam.yaml`, paths end `.../576x324/{:09d}.jpg`), and the
transform is `CenterCrop([320, 576])` then `Resize([224, 224])`
(`config/experiment/yaak/patch_policy/dinov2_dinowm_causal_3cam.yaml:64-67`). The crop
takes 320 of 324 rows and **the full 576-pixel width**, so each camera's full horizontal
FOV survives into the model. The 576->224 resize is a uniform horizontal resample, so
column `j` still corresponds to a fixed fraction of the original width.

| camera              | yaw     | FOV | pitch  | bearing span    | mount                 |
| ------------------- | ------- | --- | ------ | --------------- | --------------------- |
| `cam_left_forward`  | -70 deg | 90  | 0 deg  | -113.2 .. -26.8 | x=-85.7cm, 35cm lower |
| `cam_front_left`    | 0 deg   | 90  | +4 deg | -43.2 .. +43.2  | origin                |
| `cam_right_forward` | +70 deg | 90  | 0 deg  | +26.8 .. +113.2 | x=+85.7cm, 35cm lower |

Two consequences that drive the design:

- **Physical left-to-right order is `left_forward, front_left, right_forward`** — a
  *permutation* of the config's `cameras` order
  (`[cam_front_left, cam_left_forward, cam_right_forward]`). Derive it from yaw; see
  §3.2 for why hardcoding it is a trap.
- **Adjacent views overlap by 16.4 deg, about 4 of 16 columns** (25% of each edge). The
  three views tile a continuous ~226 deg strip; they do not abut.

Bearing of column `j` (`j = 0..cols-1`), for a rectilinear camera:

```
x_j     = (2*j + 1) / cols - 1                 # in (-1, 1), column centre
bearing = yaw_c + atan(x_j * tan(hfov / 2))    # NOT linear in j
```

*Caveat to record in the docstring:* the doc gives "FOV 90" without saying horizontal or
diagonal. If diagonal on 16:9, horizontal is ~82.6 deg and the seam overlap is ~12.6 deg
instead of 16.4 deg. Both are overlapping; make `camera_hfov_deg` a config value so this
can be corrected without a code change.

### 2.4 The code you will edit

**`CausalFrameTransformer`** (`src/rmind/components/transformer/causal_frame.py:541-805`):

- `__init__` (`:556-660`) — all keyword-only, `# noqa: PLR0913`. **It does NOT use
  `@validate_call`**, unlike most of the repo (`BlockCausalTransformer.__init__`,
  `PatchPolicy.__init__` do). **Do not add it** — it would start pydantic-coercing
  `checkpoint: bool | int` (a real union-coercion behaviour change) and the
  `AttentionImpl` Literal. Normalize hydra's `ListConfig -> tuple` by hand instead.
- `_intra` (`:662-665`) and `step` (`:784-786`) **each duplicate the norm+gain
  expression**. This duplication already caused one silent bug — see §2.6.
- `step` uses `torch.arange(src.shape[1])`, **not** `self.tokens_per_frame`, so a
  wrong-width `src` silently indexes a prefix of the table rather than erroring (unlike
  `forward`, which validates at `:686-688`).
- The module docstring's "intra-frame position" bullet (`:19-24`) describes the flat
  table and will need updating.

**`PatchPositionEmbedding2D`** (`src/rmind/components/position_encoding.py:58-71`) —
**reuse this.** It already does additive row+col composition with a row-major flatten,
over `rmind.components.nn.Embedding`, whose `default_weight_init_fn` is
`trunc_normal_(std=0.02, a=-0.04, b=0.04)` — *the same init the trunk uses today*. It is
currently used only by ControlTransformer
(`config/model/yaak/control_transformer/raw.yaml:408`,
`src/rmind/components/objectives/forward_dynamics.py:37,78`). The only thing missing is a
way to get the table without adding it to `x`.

### 2.5 `SelectiveAdamW` will reject a badly-named parameter

`src/rmind/components/optimizers/selective_adamw.py:65` derives `param_type` from the
**last dot-separated component** of the parameter name, and `:101-103` **raises
`NotImplementedError`** on anything unrecognized — at `configure_optimizers`, before
step 1.

- `intra_position_gain` is whitelisted **by literal name** at `:73`. Renaming it raises.
- `<module>.weight` hits `case "weight"` (`:75`) and is no-decay iff the submodule is in
  the blacklist — for patch_policy that is `torch.nn.Embedding` and `torch.nn.LayerNorm`
  (`config/model/yaak/patch_policy/raw.yaml:154-164`). `rmind.components.nn.Embedding`
  subclasses `torch.nn.Embedding`, so it qualifies.
- **A bare `nn.Parameter` named e.g. `speed_position` — as the brief's §5 sketch proposed
  — would fall to `case _` and raise.** Use an `nn.Embedding` submodule instead.
- Param-group membership is built from `sorted(...)` name sets, and torch maps optimizer
  state **positionally** (`:126-128`). Changing which group a param lands in silently
  corrupts Adam moments on a `ckpt_path` full-resume.
- There is **no test** guarding the `NotImplementedError` path. Add one (§5.7).

### 2.6 Diagnostics that read the position table directly

`b846a4f` already produced exactly one bug of this class: the audit script read the raw
table while the model applied a scaled one, so it reported the position signal had become
2.3x *weaker* when it was in fact 4.8x *stronger* (brief §4.3). Keep one definition
authoritative.

- `src/rmind/scripts/patch_policy_position_audit.py:94-115` (`_applied_position_table`),
  `:171-181`, `:326+`.
- `src/rmind/scripts/patch_policy_camera_probe.py:145-190` and
  `src/rmind/scripts/patch_policy_temporal_consistency.py:140-185` each have a
  `strict=False` checkpoint-loading fallback keyed on the string `intra_position_gain`,
  which monkeypatches `encoder.intra_position_norm = nn.Identity()` so pre-`b846a4f`
  checkpoints run with their real behaviour.

### 2.7 Warm start

`src/rmind/scripts/warm_start_ckpt.py` converts a 1-cam checkpoint into a 3-cam
warm start. It hardcodes `INTRA_POSITION_KEY = "encoder.intra_position_embedding.weight"`
(`:64`), tiles it (`tile_intra_position_embedding`, `:67-91`), self-checks its first dim
(`:162-180`) and loads `strict=True` (`:205`). Exercised by
`tests/test_patch_policy.py:703,717,724,754`.

### 2.8 Config

**`config/experiment/yaak/patch_policy/dinov2_dinowm_causal_3cam.yaml` is hand-written.**
Only `config/dataset/**` and `config/logger/**` are ytt-generated (`.gitignore:145-146`,
`config/_templates/` contains only those two subtrees). CLAUDE.md's "edit templates, not
generated files" does **not** apply to this file.

- It defines `num_cameras: 3` (`:14`), used **only** in its own two `${eval:...}`
  expressions at `:101-103`. Nothing links it to `model.cameras` (`:17`); they can
  silently disagree.
- Encoder base: `config/experiment/yaak/patch_policy/dinov2_dinowm_causal.yaml:224-238`
  — `dim_model=512`, `num_layers=8`, `num_heads=8`, `window=16`, `attention_impl=flex`,
  `attn_dropout=0.0`, `rope_base=1000.0`, `drop_path_rate=0.1`, `episode_length=32`.
- `dinov2_dinowm_causal_readout.yaml` overrides the same two keys with a *different*
  formula; **readout + 3cam is not composable today** and no config expresses it.

### 2.9 Export and tests

- `src/rmind/scripts/decoder_only_export.py` calls `trunk.step()` and `empty_cache()` and
  derives every shape from the trunk (`:282`), so a reparameterization preserving the
  composed table shape is export-safe. It uses `torch.export` — **your composed-table
  helper must be traceable**. `config/export/yaak/patch_policy/finetuned_3cam.yaml`
  exists and needs no change.
- `tests/test_causal_frame.py` gates, all paired with negative controls:
  `test_stream_equals_full_recompute_unbounded` (`:328`),
  `test_stream_equals_sliding_window_recompute` (`:341`), `test_shift_invariance`
  (`:287`), and the CUDA `test_flex_matches_sdpa_forward_and_backward` (`:674`), whose
  `_fwd_bwd` helper (`:540-566`) **hardcodes `"intra_position_embedding.weight"`** for its
  `d_intra_position` key. Flex tests hardcode `tokens_per_frame=257`;
  `TOKENS_PER_FRAME = 17` elsewhere.
- `tests/test_patch_policy_decoder.py:152,224` run the full `PatchPolicy` -> trunk ->
  `step` -> cache-advance path at float64. `:224` is
  `test_streamed_decode_is_sensitive_to_camera_identity` — thematically exactly what
  change 2 is about.
- **`tests/test_training_step_snapshot.py` covers ControlTransformer only** — it is *not*
  a safety net for the trunk. It *is* a genuine safety net for the
  `PatchPositionEmbedding2D` edit.
- `tests/test_overfit_regularizers.py:237` asserts
  `any("encoder.intra_position_embedding" in n for n in grads)` on a flat trunk — passes
  unchanged, but relax to `"position_embedding" in n` so it stays meaningful.

______________________________________________________________________

## 3. Design

### 3.1 The two axes

New keyword-only arguments on `CausalFrameTransformer.__init__`. Defaults reproduce
today exactly.

```python
IntraPositionScaling = Literal["norm_gain", "patch_norm_gain", "gain", "none"]
IntraPositionFactorization = Literal["flat", "view", "view_2d", "pano_col", "pano_bearing"]

    intra_position_scaling: IntraPositionScaling = "norm_gain",
    intra_position_factorization: IntraPositionFactorization = "flat",
    intra_position_target_norm: float | None = None,   # None -> std 0.02 (today)
    num_cameras: int | None = None,
    patch_grid: tuple[int, int] | None = None,
    num_prefix_tokens: int = 1,        # speed
    num_suffix_tokens: int = 0,        # registers + readout
    camera_yaw_deg: tuple[float, ...] | None = None,   # in `cameras` order
    camera_hfov_deg: float = 90.0,
    num_bearing_bins: int | None = None,               # pano_bearing; default C*cols
```

Put the body in a private `_init_intra_position(...)` called from `__init__`, mirroring
`PatchPolicy._init_readout_tokens` / `_init_fusion_norm`
(`src/rmind/models/patch_policy.py:386-441`).

**Validation** (only when `intra_position_factorization != "flat"`):

- `num_cameras` and `patch_grid` are required -> `ValueError` naming both.
- `num_prefix_tokens + num_cameras*rows*cols + num_suffix_tokens == tokens_per_frame`,
  else `ValueError` quoting the `_frame_tokens` layout and the arithmetic.
- `camera_yaw_deg` required for `pano_col`/`pano_bearing`, length `num_cameras`, all
  distinct.

**Do not** try to infer `num_cameras` by divisibility of `tokens_per_frame - 1`: 768
divides by 1, 2, 3, 4, 6, ... and registers make it worse. The trunk genuinely cannot
know; it must be config-supplied.

Store the settings as public attributes (`self.intra_position_scaling`,
`self.num_cameras`, `self.patch_grid`, ...) so the diagnostics can read the arm off the
trunk instead of re-deriving it from `PatchPolicy`.

**Axis 1 — scaling.** `T` is the raw composed table.

| mode              | applied table                                                  | rationale                                           |
| ----------------- | -------------------------------------------------------------- | --------------------------------------------------- |
| `norm_gain`       | `LN(T) * gain`, all rows                                       | today; the default                                  |
| `patch_norm_gain` | `LN * gain` on the patch band only; speed/register/readout raw | brief §4.2's cheap fix, left undecided there        |
| `gain`            | `T * gain`, no LayerNorm                                       | keeps one scale knob, drops the per-slot flattening |
| `none`            | `T`                                                            | `b846a4f` fully off                                 |

Create `intra_position_norm` / `intra_position_gain` **only in the modes that use them**,
so the arm is legible in the state dict and `SelectiveAdamW`'s literal-name whitelist has
nothing to match in the others.

`intra_position_target_norm` sets the init std so that "scaling off" does not silently
also mean "back to a 0.45 row norm". For `M` additive factors at `d` dims, a
`trunc_normal_(std=s)` row at the repo's +/-2sigma convention has norm
`~= sqrt(d*M) * s * 0.8796` (the +/-2sigma truncation realizes 0.7737 of the nominal
variance). Solve for `s` per factor; special (non-patch) rows always have `M = 1`.
Reference points at `d = 512`: **11.15** = `kughoqfi`'s settled applied norm (brief §4.2),
**23.0** = content patch-token parity, std **0.02** = pre-`b846a4f` (row norm 0.45).
Pin the constant with a test (§5.5) rather than trusting the arithmetic.

**Axis 2 — factorization.** `P = rows*cols`, `pre = num_prefix_tokens`,
`suf = num_suffix_tokens`, `C = num_cameras`.

| mode           | patch row `[c, r, j]`                        | learned rows    |
| -------------- | -------------------------------------------- | --------------- |
| `flat`         | `flat[pre + c*P + r*cols + j]`               | 768             |
| `view`         | `view[c] + patch[r*cols + j]`                | 3 + 256         |
| `view_2d`      | `view[c] + row[r] + col[j]`                  | 3 + 16 + 16     |
| `pano_col`     | `view[c] + row[r] + gcol[order[c]*cols + j]` | 3 + 16 + 48     |
| `pano_bearing` | `view[c] + row[r] + interp(bearing(c,j))`    | 3 + 16 + n_bins |

Non-patch slots (speed, registers, readout) **always** get their own free rows from a
`special_position_embedding`, indexed `[0:pre]` and `[pre:pre+suf]`. They never
participate in the factorization — a readout token has no camera and no grid position.

### 3.2 Camera overlap, and why `pano_bearing` exists

`pano_col` lays 48 free learned columns out in physical order. It gets the *ordering*
right but the *metric* wrong: it spaces the seams as if the three views abut, when in
fact 4 of every 16 edge columns are duplicated content (§2.3). A free learned table can
absorb that — adjacency is made *available* to the trunk, not imposed.

`pano_bearing` handles the overlap properly. Map each patch column to its true bearing
(§2.3's formula, including the rectilinear `atan` nonlinearity), then linearly
interpolate into **one shared bearing table** spanning `[min bearing, max bearing]`.
Overlapping columns from adjacent cameras then index the **same bins and literally share
a code** — the strongest available form of "make the encoder's spatial perception
easier".

Implement it as a constant `(C*cols, n_bins)` interpolation matrix built once in
`__init__` as `register_buffer(..., persistent=False)` — non-persistent keeps it out of
the state dict, so `strict=True` loads and `warm_start_ckpt`'s self-check are unaffected.
The column term is then one matmul against the bearing table: static shape,
constant-foldable under `torch.export`.

**Derive the panorama order as `argsort(camera_yaw_deg)`. Never configure it directly.**
For this rig the permutation is `[1, 0, 2]`, which is **its own inverse** — so getting
the direction backwards would be completely invisible at runtime and would only show up
as a slightly worse arm. Deriving it from yaw removes the failure mode; `camera_yaw_deg`
is also self-documenting in the config in a way a bare permutation is not.

**Rows are shared across cameras** (maximum transfer of vertical structure). Known
approximation to record in the docstring: `cam_front_left` has +4 deg pitch and the side
cameras sit ~35 cm lower, so the horizon sits ~1.1 rows apart between centre and sides
(vfov ~58 deg over 16 rows = 3.6 deg/row). Per-camera rows are a one-line change if this
turns out to matter.

### 3.3 The single composed-table helper

This is the part the brief's §7 explicitly asks for: `b846a4f` had to duplicate the
norm+gain expression in `_intra()` and `step()`, and nothing today would catch the two
drifting apart except the equivalence test — which only compares outputs and would pass
if both drifted the *same wrong way*.

```python
def intra_position_table(self) -> Tensor:          # raw composed, (tokens_per_frame, d)
def intra_position_applied_table(self) -> Tensor:  # scaling applied
def _intra(self, num_frames: int) -> Tensor:
    return self.intra_position_applied_table().repeat(num_frames, 1)
```

`forward` becomes `x = src + self._intra(num_frames)`; `step` becomes
`x = src + self.intra_position_applied_table()` (broadcasts over batch). The expression
exists exactly once.

In `flat` mode return `.weight` directly rather than `embedding(arange(n))` — an
embedding lookup on `arange` is an exact gather of the whole weight, so this is
bit-identical, and it removes an `arange`+gather pair from the exported decode graph.

Also add to `step` the width check `forward` already has:

```python
if src.shape[1] != self.tokens_per_frame:
    msg = f"step expects {self.tokens_per_frame} tokens, got {src.shape[1]}"
    raise ValueError(msg)
```

Nothing in the repo feeds a short `src` — `PatchPolicyDecoderStep` builds exactly
`tokens_per_frame` tokens and `decoder_only_export.py:282` derives all shapes from
`step.trunk.tokens_per_frame`.

**Do not** cache the composed table into a buffer on `eval()` — it would go stale during
training. If `torch.onnx.export(dynamo=True, optimize=True)` chokes, the fallback is
caching gated on `torch.compiler.is_exporting()`.

### 3.4 Parameter naming

Every new parameter must be an `nn.Embedding.weight` (§2.5):

```
encoder.special_position_embedding.weight
encoder.view_position_embedding.weight
encoder.patch_position_embedding.weight              # "view" mode
encoder.patch_position_embedding.row_embed.weight    # 2D modes
encoder.patch_position_embedding.col_embed.weight    # 2D modes
encoder.bearing_position_embedding.weight            # pano_bearing
```

All hit `case "weight"`, resolve to `torch.nn.Embedding`, and land in the no-decay group
— the same group `intra_position_embedding.weight` is in today. **No `selective_adamw.py`
edit is required.** Set the per-factor init std via
`rmind.components.nn.Embedding(n, d, weight_init_fn=partial(nn.init.trunc_normal_, mean=0.0, std=s, a=-2*s, b=2*s))`;
the module already supports a custom init fn, so no post-construction mutation.

______________________________________________________________________

## 4. Work, in order

Each step is separately reviewable and leaves the tree green.

1. **`PatchPositionEmbedding2D.table()`** (`src/rmind/components/position_encoding.py`).
   Add a `table() -> Tensor` returning the `(h*w, d)` row+col table and re-express
   `forward` as `return x + self.table()`. Purely additive, bit-identical for the
   existing ControlTransformer user. Run `just test tests/test_training_step_snapshot.py`
   — it is a genuine safety net for this edit. Independent, zero-risk, unblocks the rest.

1. **Pure refactor, no new arguments.** Extract `intra_position_table()` /
   `intra_position_applied_table()`, rewire `_intra` and `step` to call them, add the
   `step` width check. Verify bit-identity against `main`. This alone satisfies the
   "one helper" requirement.

1. **Axis 1** — `intra_position_scaling` + `intra_position_target_norm`, defaulting to
   `norm_gain` / `None`. Tests §5.1, §5.5, §5.6.

1. **Geometry arguments** + promote `_band_slices` to a shared public free function in
   `causal_frame.py` (pure index arithmetic, no torch) and have both diagnostic scripts
   import it instead of their two copies. Then implement `view`. Tests §5.4, §5.6, §5.7.

1. **`view_2d`**, then **`pano_col`** (with the `argsort(yaw)` order), then
   **`pano_bearing`** (with the interpolation buffer). Tests §5.4.

1. **Parametrize the existing gates** over the arms — §5.2, §5.3, §5.8, §5.9.

1. **`warm_start_ckpt.py`.** *Scope cut, deliberate:* make it **raise a clear
   "factorized target arms train from scratch — see task-intra-position-factorization.md
   §0" error** when the target trunk's factorization is not `flat`. Do **not** build an
   ANOVA seeding path; the arms train from scratch by decision. The flat->flat path and
   its four tests stay untouched. Add a `logger.warning` that optimizer-state resume
   across an arm change is unsafe (§2.5).

1. **Diagnostics.** `_applied_position_table`
   (`patch_policy_position_audit.py:94-115`) delegates to
   `encoder.intra_position_applied_table()` when present, keeping the existing raw +
   `getattr` path as the pre-refactor fallback. Add per-camera `view_position_embedding`
   row norms and pairwise cosines as a *direct* read — the "is camera identity a rank-1
   direction" quantity stops being an estimate. Keep `camera_band_cosine_centered` and
   `patch_row_variance` computed on the **composed** table so flat and factorized arms
   stay comparable on identical metrics. In both `strict=False` fallbacks
   (`patch_policy_camera_probe.py:145-190`,
   `patch_policy_temporal_consistency.py:140-185`), guard the
   `intra_position_norm = nn.Identity()` monkeypatch on `hasattr` so it does not create a
   dead attribute on a `none`-scaling arm.

1. **Config.** In `dinov2_dinowm_causal_3cam.yaml`, add the geometry block under
   `model.encoder` with `${oc.select:<var>,<default>}` indirection so every axis is a CLI
   override (the `neighbor_smoothing_tau: ${oc.select:neighbor_smoothing_tau,null}` idiom
   in `config/model/yaak/patch_policy/raw.yaml`). This finally gives `num_cameras: 3` a
   second real consumer. Add the new sibling arm config
   `dinov2_dinowm_causal_3cam_pano.yaml` — `intra_position_scaling: none`,
   `intra_position_target_norm: 11.15`, `intra_position_factorization: pano_bearing`,
   plus `wandb.tags` — modelled on `dinov2_dinowm_causal_readout.yaml`. Comment
   `camera_yaw_deg` with the yaws and the resulting physical order.

1. **Guard** (camera-identity brief §6, independent of everything else): in
   `PatchPolicy._features` (`patch_policy.py:614-623`), alongside the existing
   `tokens_per_frame` cross-check, raise if the trunk declares `num_cameras` and it
   differs from `len(self.cameras)`.

1. **Export smoke run** (§6) — the real risk gate. Then update the `causal_frame.py`
   module docstring (`:19-24`), the `docs/decoder_only_kv_cache.md:507`
   gradient-table row label, and add a note to
   `here-s-patch-policy-casual-linked-fox.md` §5 recording that it is un-shelved,
   with the four-mode scaling axis and the overlap-aware panoramic variant §5 did not
   consider.

______________________________________________________________________

## 5. Tests

Follow the existing paired-with-a-negative-control style in `tests/test_causal_frame.py`.

1. **`test_default_intra_position_is_bit_identical`** — for the default arm,
   `trunk.intra_position_applied_table()` equals `LN(embedding.weight) * gain` under
   `torch.equal` (not `assert_close`). This is *the* "defaults preserve today" gate.
1. **Parametrize `test_stream_equals_full_recompute_unbounded` (`:328`) and
   `test_stream_equals_sliding_window_recompute` (`:341`)** over the factorization and
   scaling arms. They already compare `forward` against `step` at every frame's readout,
   so they *are* the helper-unification gate — no new test needed, just parametrization.
   At `TOKENS_PER_FRAME = 17`, use `num_cameras=2, patch_grid=(2,4), pre=1, suf=0`
   (1 + 2\*8 = 17).
1. **Parametrize `test_shift_invariance` (`:287`)** over the arms — every factorization
   is frame-relative by construction, and `test_shift_invariance_negative_control` keeps
   the assertion falsifiable.
1. **`test_composed_table_structure`**, per mode, under `torch.equal`:
   - shape is `(tokens_per_frame, dim_model)` in every mode;
   - `view`: `T[pre + c*P + k] - T[pre + c'*P + k]` is the **same vector for every `k`**
     — the literal statement of "camera identity is a rank-1 direction", and the thing
     the current tests at `tests/test_patch_policy.py:208-262` cannot detect (they assert
     only an init-time numerical difference, which *any* positional table produces);
   - `view_2d`: the camera delta is independent of `r`;
   - `pano_col`: global column order matches `argsort(camera_yaw_deg)`;
   - `pano_bearing`: `cam_left_forward` col 15 and `cam_front_left` col 0 land in
     overlapping bins with nonzero shared interpolation weight.
1. **`test_target_norm_is_hit`** — with `intra_position_target_norm` set, the mean
   composed patch-row norm is within ~5% of the target for every factorization at
   `dim_model=512`. Pins the trunc-normal constant empirically so it cannot silently rot.
1. **`test_step_rejects_a_wrong_width_src`**; **`test_geometry_mismatch_raises`** (match
   on the layout wording); **`test_camera_yaw_required_for_panoramic_modes`**.
1. **`test_selective_adamw_accepts_every_position_arm`** — construct
   `SelectiveAdamW(trunk, weight_decay=0.1, weight_decay_module_blacklist=(nn.Embedding, nn.LayerNorm), lr=1e-4)`
   for each arm; assert it does not raise and that every position parameter lands in the
   `weight_decay == 0.0` group. This tests `selective_adamw.py:101-103` directly rather
   than by proxy; model it on `tests/test_overfit_regularizers.py:143`'s `_opt` helper.
   **No such test exists today.**
1. **Parametrize `tests/test_patch_policy_decoder.py:_make_model()`** over the arms
   (`NUM_PATCHES = 1` per camera -> `patch_grid=(1,1)`, `num_cameras=3`,
   `tokens_per_frame=4`) so `test_streamed_decode_matches_full_windowed_forward` (`:152`)
   and `test_streamed_decode_is_sensitive_to_camera_identity` (`:224`) run on each. This
   is the cheapest high-value gate in the plan — it covers `PatchPolicy` -> trunk ->
   `step` -> cache advance in one shot at float64.
1. **Make `_fwd_bwd` arm-aware** (`tests/test_causal_frame.py:540-566`). Add
   `CausalFrameTransformer.intra_position_parameters() -> dict[str, Parameter]`; replace
   the single `"d_intra_position"` entry with one per position parameter, keyed
   `f"d_position/{name}"` (for a flat trunk that is exactly one key, so the gate is
   unchanged in substance). Parametrize the CUDA `test_flex_matches_sdpa_forward_and_backward`
   (`:674`) over `flat` + `pano_bearing` at a single `(dim, heads)` to bound runtime — a
   factorized table routes gradients through `repeat` + broadcast-add + matmul, a
   genuinely different autograd path, especially under `checkpoint` (which
   `test_flex_matches_sdpa_under_gradient_checkpointing` covers). The flex tests hardcode
   257, so use `num_cameras=1, patch_grid=(16,16), pre=1, suf=0` -> 257 exactly.
1. Relax `tests/test_overfit_regularizers.py:237` to `"position_embedding" in n`.

______________________________________________________________________

## 6. Verification

```bash
just lint && just typecheck
just test tests/test_causal_frame.py tests/test_patch_policy.py \
          tests/test_patch_policy_decoder.py tests/test_overfit_regularizers.py
just test tests/test_training_step_snapshot.py   # guards the PatchPositionEmbedding2D edit

# defaults unchanged: the existing arm still boots and trains
just train-debug experiment=yaak/patch_policy/dinov2_dinowm_causal_3cam

# the new arm boots, and SelectiveAdamW accepts its parameter names
just train-debug experiment=yaak/patch_policy/dinov2_dinowm_causal_3cam_pano

# THE REAL RISK GATE: the new index_select / matmul in the decode graph must
# constant-fold. Run this before committing to the arm.
nix develop --command uv run python -m rmind.scripts.decoder_only_export \
    --mode decoder --arm small_3cam --out /tmp/pano.onnx --verify
```

After the run, re-run the Phase 1 diagnostics against the new checkpoint —
`patch_policy_position_audit.py` and `patch_policy_camera_probe.py`. The
camera-identity brief §8 has the exact invocations; two operational notes from §4.6 that
will otherwise cost you an hour each:

- `--override datamodule.val.batch_size=8` — the val split's default `batch_size=32`
  OOMs a 32GB GPU at fp32 / `episode_length=32`.
- `--override ++datamodule.train.dataset.samples.resume=true --override ++datamodule.val.dataset.samples.resume=true`
  on any run after the first — rebuilding the rbyte sample table costs ~15-25 min.

The §4.6 (a)/(b)/(c) numbers on `kughoqfi` are the comparison baseline: left `permute`
Δrecon_l1 +15.5%, right +13.3%, swap ratio 0.842, probe depth-0 -> last 0.996 -> 0.916.

______________________________________________________________________

## 7. Risks

| Risk                                                                                                                                                                                                                                                                                                                                      | Impact                                                  | Mitigation                                                                                                                                                       |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`norm_gain` x factorized is not a clean ablation.** The LayerNorm renormalizes every composed row, so the relative weight of the `view` term against the patch term inside a row is not controllable. The rank-1 property survives (gradient still reaches `view[c]` from all 256 of that camera's rows), but the two axes are coupled. | High — shapes the experiment matrix, not just the code  | Run factorization arms with scaling `none` + an explicit `intra_position_target_norm`. Treat `norm_gain` x factorized as a secondary cell.                       |
| **`torch.export`.** `pano_bearing` adds a constant matmul, `pano_col` an `index_select`, to the decode graph.                                                                                                                                                                                                                             | Med — would block serving                               | Verify with the real export run in §6 *before* committing to the arm. Fallback: `is_exporting()`-gated precomputed table. Never cache unconditionally.           |
| **Gauge freedom.** Additive factorizations are non-identifiable (add δ to every `view` row, subtract it from every patch row -> identical table).                                                                                                                                                                                         | Low — benign; `PatchPositionEmbedding2D` already has it | Weight decay picks the minimum-norm gauge. Report *centered* factor statistics in the audit; per-factor norms are otherwise meaningless.                         |
| **Optimizer-state resume** across an arm change silently mis-assigns Adam moments (positional group mapping, `selective_adamw.py:126-128`).                                                                                                                                                                                               | High if it happens, but out of the normal path          | From-scratch arms only. `warm_start_ckpt` writes `optimizer_states: []` by design; add the `logger.warning`. Never `ckpt_path`-full-resume across an arm change. |
| **Overlap is still approximated in `pano_bearing`** — bins are uniform in bearing, rows are shared despite the +4 deg pitch difference, and "FOV 90" may be diagonal rather than horizontal.                                                                                                                                              | Low                                                     | All three are documented approximations, and `camera_hfov_deg` is a config value.                                                                                |
| **`PatchPositionEmbedding2D` is shared with ControlTransformer.**                                                                                                                                                                                                                                                                         | Med (silent)                                            | Only add `table()` and re-express `forward` through it; `tests/test_training_step_snapshot.py` covers exactly this.                                              |
| **Diagnostic drift** — `b846a4f` already shipped an audit that read the raw table while the model applied a scaled one (brief §4.3).                                                                                                                                                                                                      | High (silent, wrong-direction)                          | Delegate `_applied_position_table` to the trunk so one definition is authoritative.                                                                              |

______________________________________________________________________

## 8. Reading the result

`03tuy3q9` (pre-`b846a4f`) vs `kughoqfi` (post) is the reference pair for the scaling
axis. **State the confound when you report:** brief §4.2 and §4.6 both note `kughoqfi`
changed three things at once — the position balance, `window` 16 -> 6, and
`drop_path_rate` 0.1 -> 0.3 — so that pair bounds the *direction* of `b846a4f`'s effect,
not its magnitude. The decision to skip a dedicated isolating arm was taken knowingly
(§0); because every axis is a flag, one can be added later as a CLI override with no code
change.
