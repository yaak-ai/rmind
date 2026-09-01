# Task brief: camera identity in the 3-camera causal PatchPolicy

**Audience:** an agent picking this up cold, in `/home/alex/rmind` on branch
`feat/patch-policy-3cam-retile-patch` (the brief was written on
`feat/patch-policy-decoder-causal-3cam`). Everything you need is below; you
should not have to re-derive the analysis. Read §1-§2 before touching code.

**Progress: Phase 1 CLOSED 2026-08-31 (§4.7). Phase 2 is SHELVED, not started.**

- `src/rmind/scripts/patch_policy_position_audit.py` exists and has been run
  (checkpoint-only, item (d) + a new causal effect audit — see §4).
- `src/rmind/scripts/patch_policy_camera_probe.py` exists and has been run
  against **both** `kughoqfi` and `03tuy3q9` on `val_3cam` — items (a)/(b)/(c)
  are DONE. **See §4.6 for the results, §4.7 for the closure decision.**
  Headline: the §4.5 gate does **not** fire on `kughoqfi` — side cameras are
  used (b) and identity looks *resolved*, not confused (a, c) — and the
  pre/post-`b846a4f` comparison shows the fix measurably strengthened that
  resolution (probe accuracy at the final layer: 61% pre-fix → 92% post-fix,
  identical-frame arm). **User-confirmed: close Phase 1, do not start Phase 2**
  (§4.7). §6/§7 (gate-independent) and §9 (separate future work) are
  unaffected.
- **§4.4 and §4.5 were rewritten on 2026-08-31.** All three items go in **one
  new script on the existing `val_3cam` loader — no new config, no
  `FeaturePermutator` change, no predict-harness work** (§4.4.1). (b) is a
  *precondition* for interpreting (a) rather than a secondary check (§4.4.0),
  (a) gets a paired three-arm design, (c) gets the controls it was missing, and
  the gate criteria are restated — two of the three original ones could not
  fire. Read §4.4.0 first. §9 records the general patch_policy predict harness
  as separate future work.
- Finding 6 (§2) does **not** apply to the audited checkpoints; see the
  correction there.
- **`b846a4f` landed the §5 "Scale balancing" idea early, ahead of the
  factorization and ahead of the gate** — `intra_position_norm` (LayerNorm,
  `elementwise_affine=False`) + a learnable scalar `intra_position_gain`, applied
  in both `_intra()` and `step()`. It closes the magnitude gap in §2 item 2 and
  is confirmed to work (§4.2), but it also flattens the per-slot amplitude the
  old table had learned, which costs the speed token — see §4.2.
- **`patch_policy_position_audit.py` is now STALE and will report the wrong
  numbers on any post-`b846a4f` checkpoint. Fix it before re-running — see
  §4.3.**

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
- `src/rmind/components/transformer/causal_frame.py:616-643` —
  `intra_position_embedding = nn.Embedding(tokens_per_frame, dim_model)`,
  `trunc_normal_(std=0.02)`, then (since **`b846a4f`**) `intra_position_norm`
  = `nn.LayerNorm(dim_model, elementwise_affine=False)` and a learnable scalar
  `intra_position_gain` init 1.0. `_intra` (`:663-665`) returns
  `(intra_position_norm(table) * intra_position_gain).repeat(num_frames, 1)`;
  `step()` (`:781-786`) applies the identical expression on the KV-cached path.
  `intra_position_gain` is blacklisted from weight decay
  (`selective_adamw.py:73`), so it moves on gradient alone.
  **Consequence to keep in mind:** the LayerNorm forces *every* row to norm
  `sqrt(dim_model) * gain` — the per-slot amplitude the old table learned
  (speed 0.61, patch 2.30, register 1.11, readout 1.44) is gone. See §4.2.
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
2. **Magnitude — ADDRESSED by `b846a4f`; kept here for the reasoning, see §4.2
   for the numbers.** Table init std = 0.02. Token content at the same point is
   LayerNorm'd patches (element RMS = 1.0 exactly) through xavier `patch_projection`
   → element RMS ≈ 1.10 (Xavier's own variance-preservation formula, confirmed by
   direct simulation of the untrained modules). Roughly a **~55x ratio at init**:
   the view/position code is ~0.3% of token variance. Blocks are pre-LN, which
   preserves the ratio (LayerNorm normalizes by one scalar per-token statistic
   dominated by content, so it rescales both terms together rather than closing
   the gap).
   **MEASURED on a trained checkpoint** (`patch_policy_position_audit.py`, run
   `yaak/alex-tmp/03tuy3q9`, checkpoint `model-03tuy3q9:latest`): the ratio has
   closed to **10.0x** (`quality/token_norm/train/patch` = 23.04 vs mean patch-band
   row norm 2.30 in a 512-dim table) — training grows the table's norm roughly
   5.8x while content barely moves. Still content-dominated, but far less starved
   than the init-time estimate implied.
   **Post-`b846a4f` the ratio is 2.09x** (run `yaak/alex-tmp/kughoqfi`) —
   see §4.2 for the full before/after and its caveats.
   **This magnitude comparison is per-patch-token only, and is NOT the whole
   story** — see the position-effect audit below, which measures the actual
   causal effect on the readout token (the one the loss reads) and finds it
   far larger than the magnitude ratio alone would suggest. That is why the
   magnitude fix should NOT be assumed to have improved camera
   discriminability: amplitude was probably not the binding constraint.
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
   **Correction (2026-08-31): this does NOT apply to the audited checkpoints.**
   Both `03tuy3q9` and `kughoqfi` have `tokens_per_frame = 772 = 256*3 + 1 + 2
   registers + 1 readout`, so `features[:, :, -1]` (`patch_policy.py:630-633`)
   is the *dedicated readout token*, not a patch. Finding 6 is real only for the
   configs without `use_readout_token`, i.e. `dinov2_dinowm_causal_3cam.yaml` as
   it stands on this branch (§7) and the `*_cont.yaml` / `finetuned.yaml`
   warm-start paths. Its risk-register severity is downgraded accordingly.
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
| ~~View code ~55x quieter than content at init, 10.0x measured post-training~~ (per-patch-token magnitude only) | **RESOLVED** by `b846a4f` | n/a — 2.09x measured on `kughoqfi` (§4.2) | Had already been downgraded once by the effect audit (readout token was never starved), so **do not assume this bought discriminability**; Phase 1(a)/(c) still needed |
| **`b846a4f` side effect: `intra_position_norm` flattens per-slot amplitude, so the speed token is now ~84% constant position code** (content/position 1.74x → **0.19x**). Register/readout are unaffected in substance (constant+constant = reparameterization) | Med | Certain by construction; measured §4.2 | Decide: scope norm+gain to the patch band only, or accept. Not yet decided |
| **`patch_policy_position_audit.py` reads the raw table and bypasses `intra_position_norm`/`intra_position_gain`** — reports 23x where the truth is 2.09x on any post-`b846a4f` checkpoint | High (silent, wrong-direction) | Certain | Fix `:141` and `:285` before re-running — §4.3 |
| No run isolates `b846a4f`: `kughoqfi` also changed `window` 16→6 and `drop_path_rate` 0.1→0.3 | Med | Certain | No performance claim can be attributed to the fix — §4.2 |
| Frozen ViT pos-emb pulls same-slot patches together | Med | Certain | amplifier; no direct fix |
| No per-camera trainable capacity | Med | Certain | optional, see §5 |
| Readout anchored on `cam_right_forward` corner | ~~High~~ **Low for the audited checkpoints** (they carry a dedicated readout token, `tokens_per_frame=772`); High only for the no-readout-token configs | Certain where it applies | document only (out of scope) |
| `cameras` order unenforced train↔serve; export cfg 1-cam | High (silent) | Med | Phase 3 |
| Tests assert only init-time numerical difference | Med | Certain | Phase 4 |
| 769² intra-frame area (9x vs cam=1); 769 = 6·128+1 flex tiling | Cost | Certain | note only |
| **`dinov2_dinowm_causal_3cam.yaml` on this branch does not reproduce `kughoqfi`** — `tokens_per_frame: 769` (no readout/register), inherited `window: 16`, `drop_path_rate: 0.1` | High (invalidates the Phase 2 baseline comparison) | Certain | §7; harmless for Phase 1, which uses the config only for the datamodule |
| **No 3-camera predict dataset** — `config/dataset/yaak/predict.yaml` has one image stream, so a cam=3 `PatchPolicy` KeyErrors in the predict path | Blocks the general harness, **not** Phase 1 (which runs on `val_3cam`) | Certain | §9 |
| **Any future `predict_3cam` copying `predict.yaml`'s `run_folder`** would cross-contaminate the existing rbyte predict cache (keyed by name only, not camera set) | High (silent, wrong data) | Certain if copied verbatim | §9 — re-key the `run_folder`, as `val_3cam.yaml` already does |
| **Dataset templates are duplicated per arm and have already drifted** — `predict.yaml` reads `turn_signal` as `polars.Int8`, `train`/`val`/`val_3cam` as `Int64` | Med (silent dtype skew between train and predict) | Certain, measured | §9 — factor into a ytt library |
| Original gate criteria: two of three unusable (zero noise floor; layer-0 probe cannot fail), and no branch for "side cameras unused" | High (would have mis-gated Phase 2) | Certain | **RESOLVED** — gate revised, §4.5 |

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

**Status: (d) is DONE (checkpoint-only, plus a new effect-audit that goes beyond
what was originally scoped), and has since been re-run against the post-`b846a4f`
checkpoint (§4.2). (a)/(b)/(c) — the ones that need real val data — are still
pending.**

### 4.1 DONE — `src/rmind/scripts/patch_policy_position_audit.py`

Checkpoint-only (no val data, no rbyte cache, no image forward pass), run
against `yaak/alex-tmp/model-03tuy3q9:latest` (run `yaak/alex-tmp/03tuy3q9`,
`dinov2_dinowm_causal_3cam`, `dim_model=512`, `cameras=[cam_front_left,
cam_left_forward, cam_right_forward]`, `num_register_tokens=2`,
`use_readout_token=True`, `tokens_per_frame=772`). Two measurements:

**(d) Embedding-table audit** (`audit_table`) — as originally scoped: row
norms of `encoder.intra_position_embedding.weight` broken out by slot type
(speed / per-camera 256-patch band / register / readout), the mean vector of
each camera band and pairwise cosines between bands, and — via `--run` — the
ratio against the measured `quality/token_norm/train/patch` content scale
logged live during training. (The PCA plot from the original scope was not
built; the row-norm/cosine numbers below answered the question without it.)

Measured results:
- Row norms (mean, per band): speed 0.61, `cam_front_left` 2.23,
  `cam_left_forward` 2.24, `cam_right_forward` 2.44, register 1.11-1.14,
  readout 1.44.
- Patch/table-row ratio: **10.0x** (vs the ~55x derived at init — see §2 item 2).
- **Camera-band mean-vector cosines**: `front_left`↔`left_forward` = **-0.065**,
  `front_left`↔`right_forward` = **+0.001**, `left_forward`↔`right_forward` =
  **-0.751**. At random init three independent rows would sit near 0 (±1/√512 ≈
  0.044). Training has pushed `left_forward`/`right_forward` into a strongly
  opposed relationship while `front_left` stayed near-orthogonal to both —
  consistent with the trunk having learned something like a left/right axis,
  precisely for the pair the paper's regime never stresses (§2 item 5). **This
  is suggestive of a learned view code, not proof of functional
  discriminability** — it doesn't substitute for (a)/(c) below.
  **Correction (§4.2): these cosines are computed on the raw band means, which
  all share a large slot-common component that carries no camera information.**
  Subtracting the grand mean over all 768 patch rows first — the right way to
  isolate camera identity — gives `left_forward`↔`right_forward` = **-0.949**,
  `front_left`↔`right_forward` = **-0.348**, `front_left`↔`left_forward` =
  **+0.035**. The left/right axis is very nearly antipodal, i.e. much
  *stronger* than the uncentered number suggested. `audit_table` should be
  changed to center before taking band means.

**New — position-effect audit** (`audit_effect`), not in the original scope:
measures the *causal* effect of the position table on the trunk's real trained
output, not just input-side magnitude. Builds `x0` (content only) and `x1 = x0
+ intra_position_embedding` from every real trained weight (fusion
norm/gains, `patch_projection`, `speed_embedding`, register/readout tokens,
all 8 trunk layers) — only the pre-fusion image/goal features are synthetic
per-patch noise, since `fusion_patch_norm`/`fusion_goal_gain` normalize away
that upstream distribution by construction regardless. Reports `cos(x0, x1)`
at the input and `||f(x1)-f(x0)|| / ||f(x0)||` after every layer and after
`encoder.norm`.

Measured results — **the readout token (what `code_head`/`offset_head`
actually consume) is NOT quiet**, unlike the individual patch tokens:
- Patch tokens: `rel_diff` stays flat ~4.5-4.9% through all 8 layers — matches
  the magnitude argument (pre-LN preserves the ratio, confirmed causally).
- Readout token: `cos(f(x0), f(x1)) = 0.943`, `rel_diff = 0.33` (33% of its
  output norm) — self-attention aggregates the small per-patch perturbation
  across all 768 patches, and the readout's own position row isn't small to
  begin with, so the aggregate is large even though no single patch's
  perturbation is. **This revises the "View code ~45x/55x quieter → High
  impact" risk-register line down**: the signal isn't starved of gradient
  magnitude at the token the loss reads — whatever limits camera
  discriminability (if anything does) is more likely about *what* the code
  encodes than about it being too quiet to be heard. See the updated risk
  register above.

**Caveats** (apply to both measurements): one checkpoint, one run, no trend
over training and no seed variance; the effect audit's content is
per-patch-independent synthetic noise, not real spatially-correlated ViT
features (the qualitative pattern should be robust since it's driven by
attention aggregating over many similar-magnitude tokens, not content
structure, but hasn't been checked against real images); neither measurement
establishes that the trunk's attention *functionally uses* camera identity for
anything that moves `offset`/`code_acc` — that is exactly what (a)/(c) below
are for, and they have NOT been run yet.

### 4.2 — Post-`b846a4f` comparison: `03tuy3q9` vs `kughoqfi`

Read directly from both checkpoints (`torch.load` + `F.layer_norm(W, (512,)) *
gain`; no data, no forward pass) and cross-checked against each run's own
`quality/*` history.

| | `03tuy3q9` (before) | `kughoqfi` (after) |
|---|---|---|
| `intra_position_gain` | — | **0.494196** |
| raw table Frobenius norm | 65.86 | 28.09 |
| raw row norm (mean) | 2.298 | 1.011 |
| **applied** position-vector norm | 2.298 | **11.154** (= 0.494·√512, σ = 0.0008) |
| content patch token norm | 23.044 | 23.333 |
| **content / position ratio** | **10.0x** | **2.09x** |

**The amplitude gap is closed for patch tokens: ~55x at init → 10.0x unbalanced
→ 2.09x now, a 4.8x improvement in effective position amplitude.**

The strongest evidence is the *trajectory* of the gain, not its endpoint.
`intra_position_gain` carries no weight decay (`selective_adamw.py:73`), so
nothing pulls it down but gradient, and training took it
1.0 → 0.73 (14.5k) → 0.72 (29k) → 0.66 (58k) → 0.54 (87k) → **0.494** (116k).
It settled ~4.8x above where the old run's slow radial growth ended up: the
optimizer chose an operating point the unbalanced parameterization could not
reach in 116k steps. *Caveat:* the gain was still decreasing monotonically at
the end, pinned by the LR anneal (1.8e-6), not converged. It would have to fall
to ~0.10 to re-open the old 10x gap, which the curve does not suggest — but a
longer run or a warm restart could land lower.

**Corroborating W&B metrics** (all readable without the checkpoint):

| metric | before | after | reading |
|---|---|---|---|
| `quality/weight_norm/encoder.intra_position_gain` | *(absent)* | 1.0 → 0.4942 | scalar param, so `weight_norm` *is* the value |
| `quality/weight_norm/encoder.intra_position_embedding` | 65.86 | 28.09 | under the LayerNorm radial growth is a no-op, so the table stopped inflating and only refines direction — the drop is the fix working, not the signal shrinking |
| `quality/grad_to_weight/encoder.intra_position_embedding` | 0.0032 | 0.0312 | **10x more relative gradient** — the commit message's "waiting on gradient descent to grow a tiny table" argument, confirmed |
| `quality/token_norm/train/speed` | 1.068 | 2.102 | the speed embedding fighting back — see the side effect below |
| `quality/weight_norm/speed_embedding` | 23.8 | 42.8 | same |

**Camera identity, centered** (band mean minus the grand mean over all 768
patch rows):

| | before | after |
|---|---|---|
| camera-identity \|v\| / content, per camera | 0.016 / 0.048 / 0.052 | **0.062 / 0.108 / 0.112** |
| cos(`left_forward`, `right_forward`) | −0.949 | −0.840 |
| between-camera vs within-camera variance of patch rows | 0.93 / 4.62 (17%) | 5.08 / 118.0 (4.1%) |

Camera identity is ~2.2x louder against content than before. Note the two
directions of that table disagree: in *absolute* terms the camera code grew,
but as a *fraction* of the position table's total variance it shrank ~4x,
because the uniform amplitude lift raised the per-patch (within-camera)
component just as much. The fix did not preferentially amplify camera identity.

**Side effect — `intra_position_norm` flattens the per-slot amplitude.**
`nn.LayerNorm(elementwise_affine=False)` forces *every* row to norm 11.15,
including slots that had learned to be quiet:

| slot | content/position, before | content/position, after |
|---|---|---|
| patch | 10.00x | 2.09x ✅ |
| speed | 1.74x | **0.19x** |
| register | 0.45x | 0.05x |
| readout | 0.52x | 0.09x |

For **register and readout** this is harmless — both content and position are
learned constants there, so their sum is a single effective vector; it is a
reparameterization, not a loss of information (the effective LR on that vector
does change).

For **speed** it is real. `speed_embedding` is data-dependent (512 bins), so the
speed token entering the trunk is now ~84% a fixed position vector and ~16%
actual speed — a ~9x degradation in its signal-to-constant ratio. The model is
visibly compensating (`speed_embedding` weight norm and `token_norm/train/speed`
both roughly doubled, table above). Given that `kughoqfi` also cut `window`
16 → 6, degrading the speed token is probably not what was wanted. **If this
matters, the cheap fix is to apply the norm+gain to the patch band only and
leave slot 0 / register / readout on the raw table.** Not yet decided.

**No performance claim can be attributed to `b846a4f`.** `kughoqfi` changed
three things at once: the position balance, `window` 16 → 6, and
`drop_path_rate` 0.1 → 0.3. The headline deltas are a textbook drop_path
signature, not a position-embedding one:

- train loss 1.722 → **1.920** (worse), val loss 6.184 → **5.913** (better),
  `quality/gap/loss/total` 4.51 → **4.06**
- val `code_acc_joint_last` 0.112 → 0.125; `quality/repr/effective_rank` 290 → 350
- predict is mixed: aggregate steering `score_l1` 0.0243 → 0.0263 and gas
  0.0561 → 0.0588 (both worse), but `braking_turn` steering 0.221 → **0.195**
  and `cruise_turn` steering 0.0728 → **0.0664** (better)

Every `patch_policy` run in `yaak/alex-tmp` was checked: **`kughoqfi` is the
only `pos_emb_balanced` run**, and there is no `window=6, drop_path=0.3,
balance-off` control. To attribute the fix, the isolating run is either
`window=6, drop_path=0.3` with the balance reverted, or `pos_emb_balanced` at
`window=16, drop_path=0.1`.

### 4.3 — `patch_policy_position_audit.py` is stale, fix before re-running

Both measurements read the raw embedding and bypass the new norm+gain path:

- `patch_policy_position_audit.py:141` — `table =
  encoder.intra_position_embedding.weight.detach().cpu()` → reports row norm
  **1.011** and a ratio of **23x**
- `patch_policy_position_audit.py:285` — `x1 = x0 +
  encoder.intra_position_embedding(idx)` → understates the trunk-input
  perturbation by ~11x, so every `rel_diff` in `audit_effect` is wrong

Run as-is against `kughoqfi` it reports the position signal became **2.3x
weaker** than before the fix, when it is in fact 4.8x stronger. Fix: apply
`encoder.intra_position_norm(...) * encoder.intra_position_gain` at both sites,
guarded with `getattr` so pre-`b846a4f` checkpoints still audit. While in
there, also center the band means before the cosine (see the correction under
"Measured results" above).

### 4.4 DONE 2026-08-31 — design for (a)/(b)/(c) — see §4.6 for results

**Design settled 2026-08-31 (this section rewritten from the original scope).**
The three items do not measure the same thing, and (b) has to be read before
(a) is interpretable:

| item | measures |
|---|---|
| (b) per-camera importance | *is the side camera used at all* |
| (a) camera-swap sensitivity | *is camera identity resolved* |
| (c) probe / attention mass | *where identity is present and whether attention uses it* |

All three go in **one new script**,
`src/rmind/scripts/patch_policy_camera_probe.py`, on the `val_3cam` loader.
**No new config, no `FeaturePermutator` change, no predict-harness work** —
see §4.4.1 for why the predict route was considered and dropped, and §9 for the
general patch_policy harness that is a separate piece of work.

#### 4.4.0 (b) is a PRECONDITION for interpreting (a), not a secondary check

The original scope called (a) "the decisive test" and read `Δ ≈ 0` on the
left/right swap as proof the model mixes the cameras up. **That inference only
holds if the side cameras carry signal in the first place.** This repo has
direct precedent for a modality contributing nothing measurable (the
forward-dynamics permutation-importance work: `turn_signal` ~0%, `waypoints`
~0.2%). So the gate needs a third outcome:

| (b) side-cam importance | (a) swap Δ | reading |
|---|---|---|
| large | ≈ 0 | cameras used, identity unresolved → **Phase 2 is the right fix** |
| large | large | identity resolved → gate does not fire |
| ≈ 0 | ≈ 0 | **side cameras unused** → factorizing the table is not the fix; the question becomes *why* (frozen ViT features, readout attention mass, or the task simply does not need them) |

(b) supplies the denominator that makes (a)'s number dimensionless. Run (b)
first, or at least report them together.

#### 4.4.1 (b) Per-camera importance — same script, NOT the predict harness

**Decided 2026-08-31 (this replaces an earlier draft that routed (b) through
`just predict-policy-with-permutations` + `FeaturePermutator`).** All three
items live in `patch_policy_camera_probe.py`. Reasons, in order:

1. **One ViT pass instead of K.** Every manipulation in the family — swap,
   `copy_from`, batch-permute, zero — is a gather on the `cam` axis of the
   **patch** tensor (§4.4.4). With the caching shim the whole matrix costs one
   encoder pass per batch; the multirun route is K full passes over the data
   plus an offline parquet join.
2. **Pairing is in-batch**, not reconstructed from `(input_id, frame_idx)` join
   keys. Lower variance and far less to get wrong.
3. **`predict_step` cannot report the gate's own metrics.**
   `PatchPolicy.predict_step` (`patch_policy.py:1074-1126`) emits only
   `ground_truth` / `prediction_value` / `score_l1` / `score_signed_error`, and
   collapses to `features[:, -1]`. No `code_acc_joint_last`, no teacher-forced
   `offset`, no readout `rel_diff`, no per-frame-position breakdown.

**No new dataset or datamodule config is needed for Phase 1.**
`patch_policy_eval.py:355-370` already builds its loader as
`instantiate(cfg.datamodule).val_dataloader()`, and `--experiment
yaak/patch_policy/dinov2_dinowm_causal_3cam` resolves through
`datamodule/yaak/train_3cam` to `dataset/yaak/val_3cam` — whose rbyte cache is
already built, being the val set the checkpoint was validated on. That is also
the right data: §4 scopes the diagnostic to the val set. Consequences: no
`predict_3cam` dataset, no new cache, and the cache-contamination risk that a
copied `run_folder` would have carried does not arise.

For the record, `config/dataset/yaak/predict.yaml` would have been the wrong
data anyway — its drive list mixes `Niro096-HQ/2023-01-11--13-47-36` (present in
**both** `train.yaml` and `val_3cam.yaml`) with the `val.yaml` drives and
others. It is a visualization set, not a holdout.

**Ablation ordering matters — permute is the primary, not zero.**
Batch-permute preserves the marginal image distribution exactly and destroys
only the correlation with this sample; it is the correct permutation-importance
ablation, and it keeps the numbers methodologically comparable with the
forward-dynamics permutation-importance work even though the harness differs.
`zero` is out of distribution and its meaning depends on *where* you zero:
zeroing the raw uint8 frame gives a strongly negative constant after
`ImageNormalize`, zeroing post-transform gives ImageNet mean-gray. Either way
the frozen ViT sees a degenerate patch field, so `Δ_zero` **overstates**
importance. Report it as an upper bound only. Order: `permute` → `copy_from`
→ mean-frame → `zero`.

**`FeaturePermutator` is out of scope.** No `mode` argument, no second
callback. The callback stays as it is, for the control_transformer predict path
it was written for.

#### 4.4.2 (a) Camera-swap sensitivity — new script, paired three-arm design

The left↔right swap is the **only** manipulation in this whole family where the
multiset of content vectors entering the trunk is bit-identical and *only* the
position-row assignment is permuted. It is therefore a pure measurement of
positional-code usage, and its **noise floor is exactly zero** under fp32 +
argmax decode (deterministic). The original gate wording, "Δ within noise of
zero", is unusable as written — with a zero floor the criterion has to be a
*ratio* against a reference (§4.5).

Three arms, all forwarded on the same batch:

- **A** baseline
- **B** swap L↔R — content multiset identical, identity permuted → *pure identity signal*
- **C** duplicate (`cam_left_forward` into both side slots) — identity intact, content changed → *reference scale*

Headline number: `Δ(A,B) / Δ(A,C)`.

Metrics — two additions to the original scope:

1. **Readout-feature `rel_diff` and `cos`**, not just loss deltas. Continuous,
   near-zero variance under pairing, and directly comparable to §4.1: the
   *entire* position table moves the readout by `rel_diff = 0.33`. If the swap
   moves it by 0.01, camera identity is ~3% of the position code's causal
   effect. A loss delta can be small merely because the heads are insensitive;
   this cannot.
2. **Break out by `_default_cluster_fn`** (`patch_policy_eval.py:38-95`). Side
   cameras should matter in `cruise_turn` / `braking_turn` and nowhere else —
   and those are precisely the clusters `kughoqfi` improved (§4.2). An
   aggregate Δ can sit at zero while the turning clusters move.

Same harness as (b) (§4.4.1), so the swap arm and the ablation arms land in one
table off one ViT pass. fp32 — `bf16-mixed`'s ~1e-3 relative jitter is the same
order as the Δ being hunted.

The swap idiom to lift is `tests/test_patch_policy.py:270-277`.

#### 4.4.3 (c) Probe — as originally specified it passes trivially and proves nothing

Two problems with "logistic regression from a patch token → camera id, target
>95% at layer 0":

1. **Layer-0 accuracy is already answered analytically.** §4.2 shows the
   centered camera-band means are near-antipodal (cos −0.84 for L↔R) with
   identity at 6-11% of content. A 512-dim linear probe on tens of thousands of
   rows will hit ~100%. A criterion that cannot fail cannot gate anything.
2. **It is confounded by content.** Three cameras pointing in different
   directions have genuinely different image statistics (hood, sky fraction,
   sun position). A probe at 99% may be reading content, not the position code
   — exactly the thing to isolate.

Required controls:

- **Identical-frame arm:** feed the *same* image into all three camera slots.
  Any remaining camera-id decodability is then purely positional/structural.
  This is the arm that actually answers (c)'s question.
- Optionally a second arm with the position table ablated.
- **Split held-out by drive, not by frame.** `val_3cam.yaml` has five drives and
  adjacent frames are near-duplicates; a random frame split inflates accuracy
  badly. Balance is free (3 cameras × equal patch counts); subsample patches to
  keep the fit tractable.

**Promote the "optional if cheap" measurement to the core of (c):** readout-token
attention mass onto each camera band, per layer and head
(`callbacks/loggers/patch_similarity.py` has the per-camera selection and
logging scaffolding). That measures whether the trunk *uses* identity, which is
the real question; the probe only measures whether identity is *present*, which
§4.2 already established it is. **Open question:** whether that capture works
at `window=6` under the flex-attention path, or needs an SDPA fallback —
unverified.

#### 4.4.4 Shared implementation notes for the script

Model it on `src/rmind/scripts/patch_policy_eval.py` — it already does
`--artifact` wandb loading, `initialize_config_dir` / `compose` / `instantiate`
of the datamodule, `_to_device`, batch iteration and per-frame-position
reporting; reuse its `_default_cluster_fn` so the numbers stay comparable with
the training-time predict metrics. `--experiment` is used **only** for
`cfg.datamodule` (model geometry comes from the checkpoint hparams), so pointing
it at `yaak/patch_policy/dinov2_dinowm_causal_3cam` for the val_3cam loader is
safe despite that config's own geometry differing from the checkpoint's (§7).

- **The frozen ViT makes the whole matrix nearly free.** It is applied per
  `(b, t, cam)` independently (`timm_backbone.py:54-77`) and `image_encoder`
  returns `(b, t, cam, p, d)` (`config/model/yaak/patch_policy/raw.yaml:72-80`),
  so *every* interesting manipulation — swap, duplicate, batch-permute — is a
  gather on the `cam` axis of the **patch** tensor. Wrap `model.image_encoder`
  in a script-local caching shim, run the ViT **once per batch**, and the matrix
  costs one ViT pass plus K trunk passes. Only `zero` / mean-frame need extra
  encoder work, and the mean frame's patches cache for the whole run. No model
  changes; `_features(batch)` stays untouched.
- **fp32, argmax only.** No `torch.multinomial` / `sample_codes` — sampling
  noise would swamp the Δ. This also makes the noise floor exactly zero.
- **Report paired per-sample Δ** with bootstrap CI over samples
  (n = batches × b), not two independent aggregates.

### 4.5 Gate — revised

The original criteria were "left/right swap Δ within noise of zero; layer-0
probe accuracy < 95%; camera-band cosines > ~0.9". Two of the three are now
known to be unusable and the set is missing a branch:

- **"swap Δ within noise of zero"** — the floor is exactly zero (§4.4.2), so
  restate as a ratio. Proposed: the gate fires if
  `Δ_swap(readout rel_diff) < 0.1 × Δ_dup(readout rel_diff)`.
- **"layer-0 probe accuracy < 95%"** — cannot fail as specified (§4.4.3).
  Replace with the identical-frame-arm accuracy, and read the *depth curve* plus
  the readout attention mass rather than any single layer.
- **"camera-band cosines > ~0.9"** — already resolved, and it does NOT fire.
  Centered (§4.2), `left_forward`↔`right_forward` is **−0.949** before and
  **−0.840** after: a near-antipodal axis, the opposite of collapse. This
  criterion is settled on both checkpoints and should not be re-run.
- **NEW precondition:** the gate is only meaningful if (b) shows the side
  cameras are used. Proposed threshold: `permute` on a side camera must raise
  the relevant metric by >2% relative. If it does not, do **not** proceed to
  Phase 2 on discriminability grounds — the finding is the third row of the
  §4.4.0 table, and it points somewhere else entirely.

So: **proceed to Phase 2 if (b) shows the side cameras are used AND the swap
ratio is below threshold.** Report all numbers to the user either way, before
starting Phase 2.

**The (d)/effect-audit results already run do NOT resolve the gate**, and
neither does `b846a4f` closing the amplitude gap — the effect audit had already
shown the readout token was never amplitude-starved, so amplitude was probably
not the binding constraint on discriminability in the first place.

Run everything against **`yaak/alex-tmp/model-kughoqfi:latest`**, not
`03tuy3q9` — it is the current 3cam baseline and the only checkpoint that
reflects the shipped `_intra` path.

---

### 4.6 DONE 2026-08-31 — Results: (a)/(b)/(c) on `kughoqfi`, plus a `03tuy3q9` comparison

`src/rmind/scripts/patch_policy_camera_probe.py` (§4.4.4's design, one script,
one shared ViT pass). Run against `val_3cam` (5 drives), `n=320` clips for
(a)/(b) (40 batches × `batch_size=8`), 6 attention-mass batches, 10 probe
batches, `--seed 1337`. Commands in §8.

**Headline — the gate does NOT fire on `kughoqfi`:**

| | `03tuy3q9` (pre-`b846a4f`) | `kughoqfi` (post-`b846a4f`) |
|---|---|---|
| (b) left `permute` Δrecon_l1 | **+1.9%** (NOT clearly used) | **+15.5%** (used) |
| (b) right `permute` Δrecon_l1 | +13.3% (used) | +13.3% (used) |
| (a) swap ratio `Δ(A,B)/Δ(A,C)` | **0.438** [0.430, 0.445] | **0.842** [0.831, 0.855] |
| (c) probe, `identical_frame`, depth 0 → last | 0.959 → **0.612** | 0.996 → **0.916** |
| §4.5 gate (`< 0.1` fires) | does not fire | does not fire |

Reading against §4.4.0's table: (b) large + (a) large → **row 2, "identity
resolved → gate does not fire."** This is the opposite of the hypothesis that
motivated Phase 2 — on `kughoqfi`, camera identity looks *resolved*, not
confused. Per §4.5, the correct call on this evidence is **not** to proceed to
Phase 2's factorization.

**Did `b846a4f` work? The `03tuy3q9` comparison says yes, and cleanly.** Both
checkpoints start near-ceiling at probe depth 0 (input-level identity is
mostly a function of raw structure, not the fix), but pre-fix that positional
signal decays hard through the trunk's depth — 96% → 61%, a ~35-point collapse
— while post-fix it barely decays — 99.6% → 91.6%, ~8 points. That single
depth-curve difference explains the other two numbers: identity that survives
to the final layer moves the readout on a swap almost as much as changing all
the content (ratio 0.84, close to 1); identity that's mostly washed out by
depth barely registers relative to a full content change (ratio 0.44). The
left camera going from "basically unused" (+1.9%) to "used" (+15.5%) is a
third, independent line pointing the same way.

**Caveats, read before acting on this:**

- **The `03tuy3q9`/`kughoqfi` comparison is confounded, exactly as §4.2 already
  warned.** `kughoqfi` changed three things at once — position-balance,
  `window` 16→6, `drop_path_rate` 0.1→0.3 — so this is suggestive, not proof
  the balance fix alone is responsible. The isolating run §4.2 asked for
  (balance-only, holding window/drop_path fixed) still does not exist.
- **Per-cluster breakdown is noisy at this sample size.** §4.4.2 predicted
  `cruise_turn`/`braking_turn` would show the swap hurting more than the
  duplicate; at n=15-25 per cluster the numbers move in both directions and do
  not confirm that pattern. Would need more batches to trust cluster-level
  numbers specifically (the aggregate/bootstrap numbers above are fine at this
  n).
- **`--attention-batches 6` / `--probe-batches 10` are modest** — enough for
  the qualitative pattern above (both are large, consistent effects), not
  enough to trust individual decimal places.
- **One seed, one run per checkpoint.** Bootstrap CIs are tight, so a re-run
  isn't expected to flip the conclusion, but hasn't been done.
- **`03tuy3q9` needed a loading workaround** to run at all:
  `PatchPolicy.load_from_wandb_artifact` fails strict loading on any
  pre-`b846a4f` checkpoint (`Missing key: encoder.intra_position_gain`, since
  `CausalFrameTransformer.__init__` unconditionally creates that parameter in
  the current code). `_load_model` in the script falls back to `strict=False`
  + monkeypatches `intra_position_norm = nn.Identity()` so the checkpoint runs
  with its real pre-fix behavior (verified bit-identical to the raw table),
  not a silently-corrupted hybrid. Same gotcha recorded in the operator's
  training-gotchas memory.
- **The val datamodule's default `batch_size=32` OOMs a 32GB GPU** running
  this script's fp32 (no-autocast, by design — §4.4.4) forward pass at
  `episode_length=32`. Use `--override datamodule.val.batch_size=8` (or
  smaller); the script also releases the caching allocator between batches
  now. Rebuilding the rbyte sample table from scratch costs ~15-25 min per
  invocation — pass `--override ++datamodule.train.dataset.samples.resume=true
  --override ++datamodule.val.dataset.samples.resume=true` on any run after
  the first to skip that.

Full numbers: `results.json` (kughoqfi) / `results_03tuy3q9.json`, both dumped
via `--out` next to their run logs (not checked into the repo — regenerate
with §8's commands).

### 4.7 CLOSED 2026-08-31 — Phase 1 conclusion

**Decision (user-confirmed): close out Phase 1 on §4.6's evidence. Do not
start Phase 2.** The gate does not fire — camera identity is resolved, not
confused, on the checkpoint that matters (`kughoqfi`) — so the factorization
in §5 is not the right next investment. §4.6's caveats (confounded
`03tuy3q9`/`kughoqfi` comparison, modest attention/probe sample sizes) are
noted but don't change the call: none of them point toward identity being
unresolved, they only limit how precisely the *magnitude* of the improvement
can be attributed to `b846a4f` alone.

**What this means for the rest of the doc:**

- **§5 (Phase 2) is shelved, not deleted.** Kept as a ready-to-execute plan in
  case the gate result is later revisited (e.g. after the isolating
  balance-only-vs-window/drop_path experiment §4.2/§4.6 describe but which
  does not exist yet) — not a live recommendation.
- **§6 (Phase 3 — ordering/config guards) and §7 (Phase 4 — tests) are
  independent of the gate outcome** (both sections already say so) and can
  land regardless.
- **§9 (general patch_policy predict harness)** was always separate future
  work, unaffected by this decision.
- If the isolating experiment (§4.2/§4.6) is ever run and changes the
  §4.6 numbers materially, re-open this section rather than starting Phase 2
  on the strength of it alone — re-run §4.4's script against the new
  checkpoint first.

---

## 5. Phase 2 — Factorize the intra-frame table

**STATUS 2026-08-31: SHELVED — see §4.7.** The §4.5 gate does not fire on
`kughoqfi`; Phase 1 is closed without starting this phase. The section below
is kept as a ready-to-execute plan for if that decision is later revisited.

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
table = self.intra_position_norm(table) * self.intra_position_gain  # b846a4f
# _intra: table.repeat(num_frames, 1)
```

The `intra_position_norm`/`intra_position_gain` line is what `b846a4f` added to
`_intra()` and `step()` separately; folding it into the single composed-table
helper is what keeps the two paths from drifting.

Params drop 393,728 → 133,632. Camera identity becomes a rank-1 direction fed by
256 tokens x every frame x every sample; patch `j`'s spatial code is now shared
across views.

At `num_cameras=1` this is behaviourally identical to today (a single view vector
added uniformly is absorbed into the patch table), so **no existing arm changes**.

### Scale balancing

**Mostly done already — `b846a4f` landed this ahead of the factorization.**
`intra_position_norm` + `intra_position_gain` now sit on the composed table in
both `_intra()` and `step()`, mirroring `PatchPolicy._init_fusion_norm`
(`patch_policy.py:357-380`). What is left for Phase 2:

- **Do not add a second gain.** The factorized `view_embedding` /
  `patch_position_embedding` / `speed_position` compose into the same
  `(tokens_per_frame, dim_model)` tensor that `intra_position_norm` +
  `intra_position_gain` already normalize and scale. One scalar knob was the
  whole point of `elementwise_affine=False`; a second one re-opens the
  redundant-degree-of-freedom problem the commit message argues against.
- **The "raise `view_embedding` init std to 0.05-0.1" advice is now moot** for
  *absolute* scale — `intra_position_norm` divides the composed row by its own
  RMS, so any uniform rescale of the factors is normalized away. What the init
  stds now control is only the **relative** weight of the view factor against
  the patch factor *within* a composed row. §4.2 gives the anchor: on the
  trained flat table, centered camera identity is ~11% of content while the
  full position row is ~48% (2.09x ratio), i.e. camera identity is roughly a
  **quarter** of the position row's magnitude. Initializing `view_embedding`
  and `patch_position_embedding` at equal std would start camera identity far
  above that; equal std is still a reasonable, unbiased starting point given
  the factorized table's whole purpose is to make that direction learnable,
  but log both norms and check where training takes the split.
- **Decide the speed-slot question first (§4.2).** If the norm+gain is scoped
  to the patch band only, `speed_position` stays outside it and the composed
  helper has to reflect that. That choice changes the helper's shape, so make
  it before writing the helper, not after.

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
  `cam_front_left` to the last camera as `cam` grows, **unless
  `use_readout_token` is set** — see the correction under finding 6), so the
  continuation / finetune arms are not warm-started blind.
- **Reconcile `dinov2_dinowm_causal_3cam.yaml` with the runs that were actually
  trained.** As it stands on this branch it sets `tokens_per_frame:
  "${eval:'${num_patches} * ${num_cameras} + 1'}"` = 769 and inherits `window:
  16` / `drop_path_rate: 0.1` from `dinov2_dinowm_causal.yaml:164-238`, with no
  `use_readout_token` / `num_register_tokens` — there is no 3cam+readout config
  here at all. But both audited checkpoints are `tokens_per_frame=772` with a
  readout token and two registers, and `kughoqfi` ran `window=6`,
  `drop_path_rate=0.3`. Those runs came from CLI overrides or the
  `feat/patch-policy-decoder-causal-3cam` branch. Either commit a config that
  reproduces `kughoqfi` or record the exact override string in the brief —
  otherwise the Phase 2 comparison against that baseline is void (§8 success
  criterion).

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
  test should cover this once both paths call the one helper). **Also assert the
  composed table goes through `intra_position_norm`/`intra_position_gain` in
  *both* paths** — `b846a4f` had to duplicate that expression at
  `causal_frame.py:663-665` and `:781-786`, and nothing today would catch the
  two drifting apart except the equivalence test, which only compares outputs
  and would pass if both drifted the same wrong way. A direct assertion that
  every composed row has norm `sqrt(dim_model) * intra_position_gain` is cheap
  and pins the invariant.
- `tests/test_patch_policy_decoder.py:224-268` must stay green; add the Phase 3
  ordering assertion.

---

## 8. Verification

```bash
just test tests/test_patch_policy.py tests/test_causal_frame.py \
          tests/test_patch_policy_decoder.py
just lint && just typecheck

# Phase 1(d) + effect audit -- DONE, checkpoint-only, no val data needed.
# NOTE: valid only for PRE-b846a4f checkpoints as the script stands today.
# Fix :141 and :285 (norm+gain) before pointing it at kughoqfi -- see §4.3.
uv run python -m rmind.scripts.patch_policy_position_audit \
    --artifact yaak/alex-tmp/model-03tuy3q9:latest \
    --run yaak/alex-tmp/03tuy3q9 \
    --samples 256 --device cuda --out results.json

# The §4.2 before/after numbers, reproducible without the audit script:
#   ckpt = torch.load(".../model.ckpt", map_location="cpu")["state_dict"]
#   W    = ckpt["encoder.intra_position_embedding.weight"].float()
#   g    = ckpt.get("encoder.intra_position_gain")          # absent pre-b846a4f
#   A    = F.layer_norm(W, (W.shape[1],)) * float(g) if g is not None else W
#   A.norm(dim=-1)  vs  the run's quality/token_norm/train/patch

# Phase 1(a)/(b)/(c) -- ONE script, all three items. DONE 2026-08-31, results
# in §4.6. No new config: --experiment supplies ONLY cfg.datamodule, whose
# val_dataloader() is the already-cached val_3cam loader (4.4.1). fp32 and
# argmax only, one ViT pass per batch for the whole manipulation matrix (4.4.4).
# --override datamodule.val.batch_size=8 avoids OOMing a 32GB GPU (the val
# split's default batch_size=32 does not, at fp32/episode_length=32 -- §4.6).
# The `resume=true` overrides skip rebuilding the rbyte sample table (~15-25
# min) on any run after the first against the same drives.
uv run python -m rmind.scripts.patch_policy_camera_probe \
    --artifact yaak/alex-tmp/model-kughoqfi:latest \
    --config-dir "$PWD/config" \
    --experiment yaak/patch_policy/dinov2_dinowm_causal_3cam \
    --override datamodule.val.batch_size=8 \
    --override ++datamodule.train.dataset.samples.resume=true \
    --override ++datamodule.val.dataset.samples.resume=true \
    --batches 40 --attention-batches 6 --probe-batches 10 \
    --device cuda --seed 1337 --out results.json

# Same, against the pre-b846a4f baseline for the §4.6 comparison -- needs no
# extra flag for the strict-load fallback, _load_model handles it automatically.
uv run python -m rmind.scripts.patch_policy_camera_probe \
    --artifact yaak/alex-tmp/model-03tuy3q9:latest \
    --config-dir "$PWD/config" \
    --experiment yaak/patch_policy/dinov2_dinowm_causal_3cam \
    --override datamodule.val.batch_size=8 \
    --override ++datamodule.train.dataset.samples.resume=true \
    --override ++datamodule.val.dataset.samples.resume=true \
    --batches 40 --attention-batches 6 --probe-batches 10 \
    --device cuda --seed 1337 --out results_03tuy3q9.json

# Phase 2, end-to-end sanity before a full run
just train-debug experiment=yaak/patch_policy/dinov2_dinowm_causal_3cam

# Phase 3, serving contract unchanged
uv run python -m rmind.scripts.decoder_only_export --arm small_3cam ...
uv run python -m rmind.scripts.decoder_only_verify ...
```

Note: training/eval on this box goes through the nix env, not a bare `uv run` —
check with the user before the first long run.

**Success criterion.** On the re-trained 3cam arm: the Phase 1 left/right
camera-swap ratio `Δ_swap / Δ_dup` is materially above the §4.5 threshold, the
identical-frame probe still separates cameras through the depth of the trunk,
and val `offset` / `code_acc_joint_last` are no worse than the current 3cam
baseline — which is now **`kughoqfi`** (`val/loss/total` 5.913,
`val/policy/metric/code_acc_joint_last` 0.1246,
`val/policy/metric/offset_last` 0.00804, `window=6`, `drop_path_rate=0.3`),
*not* `03tuy3q9`. Match its `window`, `drop_path_rate` **and its
`use_readout_token` / `num_register_tokens` (772 tokens per frame, not 769)** or
the comparison is meaningless (§4.2, §6). "Δ clearly above noise" and "layer-0
probe accuracy > 95%" were the original wording; both are dead criteria (§4.5).

---

## 9. Future — a general patch_policy predict harness (NOT Phase 1)

Recorded here because it came up while scoping Phase 1 and was deliberately
kept out of it. `PatchPolicy` has no predict-side configs; the eventual goal is
parity with `control_transformer` (`config/inference/yaak/control_transformer/`:
rerun logging, `DataFramePredictionWriter`, permutation multirun). **Phase 1 does
not need any of it** (§4.4.1). Three things to get right when it happens:

### 9.1 Factor the dataset templates into a ytt library first

The five dataset templates each restate the whole stream + asof-join +
`PathDataFrameBuilder` structure. Measured on this branch: `val.yaml` →
`val_3cam.yaml` differ by 135 lines, `predict.yaml` → `val_3cam.yaml` by 205,
and `train.yaml` is 900+ lines of the same shape. There are no `.lib.yaml` files
under `config/_templates/` at all. **The duplication has already drifted:**
`predict.yaml:112-113` reads `turn_signal` as `polars.Int8` while `train.yaml`,
`val.yaml` and `val_3cam.yaml` all use `polars.Int64`.

Adding a sixth copy for `predict_3cam` is the wrong move. Instead add
`config/_templates/dataset/yaak/_dataset.lib.yaml` with a function along the
lines of

```
#@ def dataset(drives, cameras, run_folder, side_cam_root="${paths.alex_data}"):
```

emitting the streams, sample inputs, `ImageMetadata.<cam>` asof joins and
`PathDataFrameBuilder` pipefuncs once, per camera. Each of `train` / `val` /
`train_3cam` / `val_3cam` / `predict` then reduces to a drive list plus one
call, and `predict_3cam` costs ~10 lines.

**Safe-refactor recipe.** `config/dataset/*` is gitignored (`.gitignore:145`),
so `git diff` will not catch a regression. Snapshot the generated tree, run
`just generate-config`, then diff for byte-identity. Any intentional change —
unifying `turn_signal` to `Int64`, say — then shows up as exactly one deliberate
hunk, and rbyte cache keys stay stable for every existing arm.

### 9.2 `predict_3cam` data

Once 9.1 exists this is a drive list. Two constraints:
- Side-camera frames exist only for the five `val_3cam` drives, under
  `${paths.alex_data}` rather than `${paths.data}`.
- **Re-key `run_folder`.** `predict.yaml`'s is
  `${paths.rbyte.cache}/yaak/predict/samples` — keyed by dataset name only, not
  by camera set — so copying it verbatim cross-contaminates the existing predict
  cache. `val_3cam.yaml` already keys by clip/stride; follow that.

Also note `predict.yaml`'s drive list is not a holdout: it mixes
`Niro096-HQ/2023-01-11--13-47-36` (in **both** `train.yaml` and `val_3cam.yaml`)
with the `val.yaml` drives and others. Decide deliberately what a patch_policy
predict set should contain rather than inheriting that.

### 9.3 The inference config

A `config/inference/yaak/patch_policy/*.yaml` must limit `objectives` to what
`PatchPolicy.predict_step` (`patch_policy.py:1074-1126`) actually emits:
`ground_truth`, `prediction_value`, `score_l1`, `score_signed_error`. The
control_transformer configs also request `prediction_std`, `prediction_probs`
and `score_logprob`; `predict_step`'s `keys` filter drops them silently and
`DataFramePredictionWriter`'s `select` then `KeyError`s. Either implement those
objectives or leave them out of the `select`.
