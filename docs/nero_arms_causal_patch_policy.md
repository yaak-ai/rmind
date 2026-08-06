# nero-arms causal patch policy + SE(3) action tokenizer

Design record for the nero-arms bimanual manipulation arm of the patch policy.
Coded against `nero-arms/DATA_CONTRACT.md` v0.1 (+ §7.2, §10 answers, §11, §13)
and against what the rbyte ingestion side actually emits (`rbyte@feat/nero-arms`).
Where this document and the contract disagree, the contract wins and the
disagreement is listed in "Contract issues" below.

⚠️ **The glove SE(3) parameterisation is a stand-in, not the observation space**
(§10 A1). Iteration-1 teleoperation uses the gloves only as the *input device*;
the recorded observations will be **robot arm/hand state** — Revo2 joint values,
roughly 12 per side rather than 60. Everything below is therefore built so the
action space is a **config value, not a constant**; see "Action-space seam".

Built on `feat/patch-policy-decoder-only`. The decoder-only trunk
(`rmind/components/transformer/causal_frame.py`: frame-RoPE + tiled intra-frame
embedding, bidirectional intra-frame / causal inter-frame, KV-cacheable) is
**reused unchanged**; see `docs/decoder_only_kv_cache.md`. Everything above and
below it is new.

## Files

| file                                                         | what                                                                                                                                                                                                                               |
| ------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/rmind/data/nero.py`                                     | contract §5 shared boundary: `pose_quat_to_9d` / `pose_9d_to_quat`, canonicalisation, 6D rotation, the 60-dim channel **layout**, `PoseStandardizer` (versioned artifact), physical-unit error split, letterbox intrinsics rewrite |
| `src/rmind/components/image.py`                              | `LetterboxResize` (contract §7.2)                                                                                                                                                                                                  |
| `src/rmind/models/nero_pose_tokenizer.py`                    | `NeroPoseTokenizer` — VQ-BeT residual-VQ over one side's action chunk                                                                                                                                                              |
| `src/rmind/models/nero_patch_policy.py`                      | `NeroPatchPolicy`                                                                                                                                                                                                                  |
| `src/rmind/datamodules/nero_random.py`                       | structured synthetic batches at the contract §8 shapes                                                                                                                                                                             |
| `src/rmind/scripts/nero_smoke.py`                            | budget / tokenizer / policy smoke harness                                                                                                                                                                                          |
| `config/model/yaak/nero_patch_policy/raw.yaml`               | policy model config                                                                                                                                                                                                                |
| `config/model/yaak/nero_pose_tokenizer/raw.yaml`             | tokenizer model config                                                                                                                                                                                                             |
| `config/experiment/yaak/nero_arms/{causal,tokenizer}.yaml`   | experiments                                                                                                                                                                                                                        |
| `config/datamodule/yaak/nero_random.yaml`                    | synthetic datamodule                                                                                                                                                                                                               |
| `tests/test_nero_pose.py`, `tests/test_nero_patch_policy.py` | tests                                                                                                                                                                                                                              |

## What the loader actually hands over

rbyte (`feat/nero-arms`, 15,047 samples across 104 episodes) emits shapes that
differ from the §8 table in three ways that matter here. All three are handled at
the model's input boundary:

| | §8 said | rbyte emits | handled by |
| --- | --- | --- | --- |
| images | one `(T, 3, H, W)` per camera, implicitly common H/W | **different grids**: `base` `(T,3,270,480)`, `side_*` `(T,3,300,480)` | `LetterboxResize` |
| state / action | `(T, 2, 60)` | **`(T, 2, 46)`** — the §5.2 quaternion storage form | `state_quat_to_9d` |
| goal image | one `(3, 3, H, W)` | **three keys** `goal.image.{camera}`, each on its own grid | per-camera encode |

`action.commanded` is currently a byte-identical duplicate of
`action.future_state` (~199 MB of a ~470 MB TensorDict). The policy reads exactly
one path, chosen by config, so the duplicate is never consumed.

The 46↔60 conversion is the shared boundary of §5, and it is verified against the
other side rather than assumed: `rmind.data.nero.pose_quat_to_9d` is
**bit-identical** to `rbyte.io.nero.rotation.pose_quat_to_9d` on random poses,
and `state_quat_to_9d` matches structurally (arm `[0:7]`, fingers `[7:42]` in
thumb→little order, hub quaternion `[42:46]`).

## Token layout and sequence length

```
frame block = [ state token (1) ][ base (160) ][ side_left (160) ][ side_right (160) ]
tokens_per_frame = 3 * 160 + 1 = 481
flattened        = episode_length * 481 = 6 * 481 = 2886
```

The ViT input is **140×224**, a 10×16 = 160-patch grid at DINOv2's patch 14. That
is the cameras' own 5:8 aspect, so **nothing is spent on padding except `base`'s
two letterbox rows**. Forcing the usual square 224×224 would have put the same
image content on a 16×16 grid with 6 of 16 rows pure black bars — 769 tokens per
frame and 4614 flattened, i.e. **60% more sequence for zero extra information**.
Since attention is quadratic, that is ~2.6× the attention cost.

`TimmBackbone`'s `norm_patch_tokens` path assumed a square grid (`isqrt(P)`); it
now reads `patch_embed.grid_size`, which is identical for square inputs and
correct for these.

For comparison on the same trunk: the 6-frame driving arm is **1542**, the
16-frame causal driving arm is **4112**.

`head_dim = 512 / 8 = 64`: divisible by 8 (fused SDPA — at head_dim 20 PR #265
fell back to a math kernel that materialises the full `(B, H, L, L)` score matrix
and OOMed) and even (required by the trunk's frame-RoPE). The
`nero_smoke --stage budget` pre-flight asserts both, cross-checks `num_patches`
against the image size, and probes the SDPA backend empirically.

The state token goes **first** so each frame block ends on a patch token, which
is the readout position — unchanged from PR #265, where the speed token played
the same role.

## Design decisions

### Camera conditioning → per-patch concatenation

Each camera's 13-dim vector (§7.1) is concatenated to **that camera's** patch
tokens before `patch_projection`. Alternatives considered: 3 extra tokens per
frame, or FiLM.

1. **Zero sequence cost.** At 481 tokens/frame attention is the binding
   constraint.
1. **It binds the geometry to the tokens it describes.** A `side_left` patch
   carries `side_left`'s extrinsics. Extra tokens or FiLM would deliver all three
   cameras' geometry to all three cameras' patches and make the trunk learn the
   routing.
1. **It doubles as camera identity.** The tiled intra-frame embedding gives each
   slot an index, but that index is setup-specific; the conditioning vector is
   what generalises across camera-setup changes, which is the stated point of
   §7.1.

Tested by `test_per_camera_conditioning_is_bound_to_that_camera` — swapping two
cameras' vectors must change the output, which a shared-broadcast implementation
would not do.

### Goal conditioning → same-index concatenation of goal patch features

The goal image is the episode's final frame **from the same camera**, so goal
patch `(c, p)` and observation patch `(c, p)` are the same ray through the same
lens. Index-aligned concatenation preserves that correspondence — which is
exactly the "where did the object end up" signal. Mean-pooling the goal (the
obvious cheap alternative) discards it, for identical cost. This is the natural
generalisation of the paper's `T × P × (D + G)` scheme with `G = D`, the goal
being constant over `t` (the driving arm's `g_t` varied per frame only because
waypoints are ego-frame).

**Goal dropout** (`goal_dropout = 0.15`, per sample per camera) replaces the goal
features with a **learned** `no_goal` embedding, not zeros, so "no goal supplied"
is distinguishable from "goal that happens to encode near zero".

Goal xyz (§9 alternative) is emitted by the datamodule and left unconsumed; the
second RVQ for it is scaffolded (`NeroGoalXYZTokenizer`) but not trained — it is
only needed if the goal is to be *predicted* rather than conditioned on.

### Image pipeline → letterbox, not resize (§7.2 + the rbyte grids)

rbyte downscales each camera **isotropically to its own grid**, which keeps the
resolution-normalised intrinsics in `camera_cond` exactly valid at zero
propagation cost — and leaves the three streams on different grids. Unifying them
is rmind's job, and it must be a letterbox (uniform scale + symmetric padding),
never an anisotropic resize: that would scale `fx` and `fy` independently so the
conditioning vector stops describing the pixels the policy sees, and would make
the same object a different shape in `base` than in the side views.

`300×480 → 140×224` is exactly isotropic, so the side cameras are resized and not
padded; `base` (`270×480`) is padded by 7 rows top and bottom after scaling.
`letterbox_camera_cond` rewrites the four intrinsic entries to match — for the
side cameras it is a no-op, for `base` `fy` shrinks by `270/300` and the
principal point is re-centred. Extrinsics are untouched; letterboxing does not
move the camera.

§13.2 confirms the conditioning vector should carry **rectified** K plus
extrinsics and *not* distortion coefficients, which is what the 13-dim layout
already assumes.

Consequence to watch: all three cameras share one patch grid, so `base`'s wider
field of view gets coarser real-world coverage per patch than the side cameras,
and `base` additionally spends 2 of its 10 patch rows on letterbox padding. If
the overhead view underperforms, this is the first thing to look at.

### Bimanual `side_valid` (§6.1, §10 A2 — permanent contract)

Consumed in two places, both falsifiable:

- the state token is built from `state * side_valid` with the 2-dim mask
  appended, so perturbing an invalid side's state cannot change any output
  (`test_invalid_side_state_cannot_influence_the_output` asserts **bit**
  identity, plus a control that the *valid* side does matter);
- the action loss **selects** valid `(batch, frame, side)` rows. Normalisation is
  therefore `sum / count`, never `mean` over a zero-padded tensor — the latter
  silently halves the loss on right-only data, which changes the effective LR and
  makes the curve incomparable to a future bimanual run.

Not asserted, deliberately: independence of the *valid* side's prediction from
the invalid side's inputs. The state token is shared, so that dependence is
legitimate.

§10 A2 confirms bimanual is permanent with one arm optionally absent, so this is
contract behaviour rather than dummy-data scaffolding. It is exercised through a
backward pass, not just a forward: a right-only overfit run reaches 1.37 mm /
1.37° with `valid_rows` pinned at 48 = 8 samples × 6 frames × 1 side.

### Per-side, weight-shared tokenizer and head

One tokenizer is fitted on the pooled valid side-chunks of both hands, and one
`code_head`/`offset_head` pair is applied twice with a learned per-side embedding
added to the readout feature. This halves head parameters versus two independent
heads and — more importantly — lets right-only dummy data produce a tokenizer and
a head that are immediately meaningful for the left hand. A single 120-dim
bimanual tokenizer was rejected: on right-only data it would learn "left ≡ 0",
which breaks the moment real bimanual teleop lands.

### Action-space seam (§10 A1, §13.3) — load-bearing

`action_features` (per-side action dimensionality) and `action_horizon` are
config values; every head width derives from
`num_quantizers * codebook_size * action_horizon * action_features`, and
`NeroPoseTokenizer.has_pose_layout` switches the physical-unit metrics off
automatically when `action_features != 60`.

To move to §11 option **(B)** — policy predicts Revo2 joint targets, ~12 dims per
side — what changes:

- **config**: `action_features: 12`, a new tokenizer checkpoint and a new
  `PoseStandardizer` artifact fitted on joint angles. No code change in the
  policy or the tokenizer.
- **rbyte**: the glove-SE(3) → Revo2 retargeting must be applied *at ingestion*
  and `action.commanded` populated (the contract already reserves the slot).
  This is the part that does not exist and cannot be faked.
- **metrics**: `pose_error_metrics` no longer applies; the replacement is
  per-joint degrees, which is a ~10-line addition guarded by `has_pose_layout`.
- **state**: `state.pose` can stay SE(3) (it is a separate config path from the
  action), so the observation side need not move at the same time. That
  asymmetry is the cheapest first experiment if commanded targets arrive.

Offset-head size is the one number that changes materially: at 60 dims it is
`4 × 16 × 6 × 60 = 23040` outputs (11.8M params with the 512-wide penultimate
layer used here); at 12 dims it is 4608 outputs (2.4M).

## Tokenizer recipe (contract §5)

- input is the **6D continuous rotation** form (3 translation + 6 rotation per
  pose), never quaternions — quaternions are discontinuous as a reconstruction
  target;
- quaternions are **canonicalised** (`qw ≥ 0`) at the conversion boundary, so the
  double cover cannot waste codebook capacity even if ingestion misses it;
- **per-channel standardisation** with train-split statistics, shipped as a
  versioned JSON artifact (`PoseStandardizer.save/load`, version-gated on load)
  alongside the checkpoint;
- translation and rotation reconstruction error are reported **separately and in
  physical units** — mm and geodesic degrees. `test_pose_error_metrics_separates_ translation_from_rotation` asserts a translation-only error produces zero
  rotation error and vice versa, so the split cannot silently degrade into one
  scalar.

## Measured (synthetic data — see the caveat)

Hardware: one RTX 5090 (32 GiB). `uv run python -m rmind.scripts.nero_smoke`.

### Pre-flight budget

| | |
| --- | --- |
| patch grid | 10 × 16 |
| tokens per frame | 481 |
| flattened sequence | **2886** |
| head_dim | 64 (÷8 ✓, even ✓) |
| SDPA memory-efficient backend | admissible (probed, not assumed) |
| dense score matrix *if* a math fallback were taken | 0.124 GiB per sample, bf16 |

### Tokenizer reconstruction — translation and rotation, separately

Held-out synthetic chunks, `4 × 16` residual-VQ, 6000 steps. Codebook
perplexity **15.5–15.8 of 16** on held-out data (near-uniform usage).

| path | translation | rotation | standardised L1 |
| --- | --- | --- | --- |
| predict the train mean (baseline) | 14.24 mm | 10.94° | 0.677 |
| **codes only** (what the policy's code head must hit) | **9.76 mm** | **4.89°** | 0.281 |
| unquantized autoencoder (ceiling) | 9.22 mm | 4.12° | 0.241 |

The baseline row is what makes the other two readable. The gap between rows 2
and 3 is the cost of the codebook; the gap between rows 1 and 3 is what the
encoder learned. The policy adds the VQ-BeT offset head on top of row 2, so row 2
is a *coarse* target by construction, not the end-to-end accuracy.

⚠️ These are numbers on **synthetic** SE(3), and the generator's compressibility
was deliberately built in (a ~14-DOF latent, matching a real hand's DOF count).
They demonstrate that the recipe and the split metric work; they are **not** a
prediction of accuracy on real data — which, per §10 A1, will not even be this
action space.

### Policy smoke — 400 steps, batch 8, lr 3e-4

| | fresh random batches | one fixed batch | one fixed batch, right-only |
| --- | --- | --- | --- |
| `side_valid` | both | both | `[False, True]` |
| total loss | 10.11 → 10.04 (flat) | 8.12 → **0.013** | 6.78 → **0.039** |
| offset L1 | 0.295 → 0.302 | 0.237 → 0.013 | 0.235 → 0.029 |
| translation | 16.5 → 17.0 mm | 17.1 → **0.57 mm** | 17.6 → **1.12 mm** |
| rotation | 13.9° → 14.4° | 13.9° → **1.41°** | 13.9° → **1.31°** |
| `valid_rows` | 96 | 96 | **48** |

**The flat fresh-data curve is a property of the synthetic data, not a wiring
bug, and the controls prove it.** The four code losses sit at
2.4316 / 2.4388 / 2.4356 / 2.4368, which is exactly the uniform-prior focal-loss
value: `(1 − 1/16)² · ln 16 = 2.4368`. With a near-uniform codebook (measured
perplexity 15.5–15.8/16) that is the hard floor for any predictor with no
information — and here there is none, because the synthetic images are pure
noise, so 480 of the 481 tokens per frame are distractors and the one informative
token (state) is attenuated ~1/481 at init. Trained on a *single* batch the same
model drives every code loss to 0.000 and reaches 0.57 mm / 1.41°, so the whole
path — readout → per-side embedding → shared head → tokenizer targets →
gradients — is sound.

The third column is the contract's headline bimanual case (§10 A2) exercised
through a **backward** pass, not just a forward: with the left side masked out,
`valid_rows` is pinned at 48 = 8 samples × 6 frames × 1 side and the run still
converges.

**Memory and speed.** Peak **1.74 GiB** at batch 8, median step **0.21 s** —
⚠️ **with the trunk's gradient checkpointing**, which
`rmind.components.transformer.utils.run_layer_stack` applies whenever
`training=True`. That is memory traded for recompute, so it is not the number to
size a non-checkpointed run from. 40.7M trainable / 62.9M total parameters. Wall
time is dominated by synthetic image generation on the CPU, not by the model.

<sub>Footnote: the fresh-batch run composed its config just before the
`ToDtype`/`LetterboxResize` reorder and the other two just after. On pure-noise
images the difference is uint8 rounding, i.e. nothing measurable — but the three
are not a controlled comparison of that change.</sub>

## Contract issues

1. **§8's `(T, 2, 60)` contradicts §5.2's quaternion storage.** rbyte implemented
   §5.2 and emits **46** per side; this side does the 46→60 expansion at the model
   boundary. Resolved in practice, but §8 should be corrected so the next reader
   does not size a tensor from it. **Verified, not assumed**: `pose_quat_to_9d` is
   bit-identical across the two repos and the block order agrees (arm `[0:7]`,
   fingers `[7:42]` thumb→little, hub quaternion `[42:46]`).
2. **The 60-dim channel order was undefined.** §6.1 gave blocks but not their
   order, which makes per-channel standardisation and the translation/rotation
   split unimplementable. It is now pinned on both sides — here as
   `rmind.data.nero.POSE_BLOCK_LAYOUT`, giving 18 translation indices
   `{0-2, 9-11, 18-20, 27-29, 36-38, 45-47}` and 42 rotation indices. The layout
   is also written into the standardisation artifact, so a future mismatch is
   detectable rather than silent. **§6.1 should state it.**
3. **§5 names only the 7↔9 pose pair.** The hub-orientation block is
   rotation-only and needs a 4↔6 pair (`quat_to_rot6d` / `rot6d_to_quat` here,
   and the same pair exists in rbyte). The contract should name it.
4. **§8's image row implies a common H/W across cameras; there is none.** rbyte
   isotropically downscales per camera (`base` 270×480, `side_*` 300×480), which
   is the right call for the intrinsics but makes "letterbox to a common grid" a
   mandatory consumer-side step. §8 should say so explicitly — a consumer that
   assumes a uniform patch grid fails at `torch.cat`, which is at least loud, but
   one that resizes anisotropically fails **silently** and invalidates
   `camera_cond`.
5. **§7's `placeholder: true` makes `camera_cond` a constant** (zero intrinsics,
   identity extrinsics) until the real calibration drops. The generalisation claim
   behind §7.1 is therefore **untestable** on current data, and a model trained
   before the file drop learns nothing from that vector. The smoke harness
   randomises it so the path is at least exercised. §13.2's rectified-K-plus-
   extrinsics choice is what the 13-dim layout already assumes.
6. **`action.commanded` is a byte-identical duplicate** of
   `action.future_state` — ~199 MB of a ~470 MB TensorDict, scaling linearly. The
   policy reads exactly one path (config-selected), so nothing downstream pays
   for it; the duplication is worth removing loader-side before the dataset grows.

## Open items for the user

- the §11 retargeting decision (A vs B) — the single most valuable thing to
  confirm about the incoming teleop format;
- the 60-dim channel order (item 1 above) must be agreed with the rbyte side;
- `lr_warmup_steps` / `lr_total_steps` in the experiment config are placeholders.
  Re-derive from the **true** sample count (contract §8: do not trust
  `.rbyte_cache` counts) before any real run — the cosine schedule rises again
  past `num_training_steps`.
