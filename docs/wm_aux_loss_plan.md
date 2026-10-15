# Task brief: action-conditioned world-model auxiliary loss for `PatchPolicy`

Self-contained brief for an agent with **no prior context** on the
conversation that produced it. Everything needed is here or behind the
file:line references. Sibling brief with the same gating discipline (read it
first, it is the template for this one): `docs/phase3_trajectory_head_plan.md`.
Review of an earlier draft: `docs/wm_aux_loss_plan_verdict.md` -- its
accepted points are folded in below; points rejected on verification are
listed in §9 so they are not re-raised.

## 1. Why this work exists

Goal: **better trajectory planning** from `PatchPolicy`
(`src/rmind/models/patch_policy.py`, Patch Policy arXiv:2607.18236).

AD-E2E-JEPA (arXiv:2609.34085, Zhu/Zhang/LeCun/Choromanska) shows that an
action-conditioned latent world model on frozen DINOv3 patches -- predictor
`(history latents, candidate trajectory) -> future latent` -- learns
representations that rank driving trajectories, and that its learned
representation transfers to an imitation-learning trajectory decoder (EPDMS
80.2 -> 85.4 on NAVSIMv2). Their zero-shot planner uses the *future camera
image* as the goal, which is an oracle we do not have at deployment; we do
**not** copy that part.

What we take: train a world-model (WM) predictor **on top of the causal
trunk's own output tokens**, so its gradient shapes the same trunk that the
`trajectory_head` reads. Intuition: today the trunk is supervised by one
expert action chunk + one trajectory per frame (a few dozen numbers). The WM
loss asks the trunk to support predicting the pooled DINO patches of a future
frame under a given ego-motion -- dense supervision that requires encoding
geometry, free space and other agents' motion, and that speed alone cannot
satisfy (the repo already fights speed-echo shortcuts via `speed_dropout`).

## 2. Decisions already made -- do not revisit

- **Variant (b): the trunk is the WM's history encoder.** No second temporal
  transformer. The block-causal trunk (`BlockCausalTransformer` /
  `CausalFrameTransformer`) already emits a contextualized token at every
  position; `_readout` (`patch_policy.py:861`) keeps only `[:, :, -1]` and
  discards the rest. We additionally use the **patch-position outputs** of
  frame `t` (which, by the block-causal mask, have attended only to frames
  `<= t`).
- **Gradient flows predictor -> trunk** (that is the point). It must NOT flow
  into DINOv2 (frozen, under `no_grad`, both as input and as target).
- **Predictor inputs:** pooled trunk patch tokens of frame `t` + ego
  **relative poses** `P_1..P_k` (cumulative, each relative to the ego at
  anchor `t`; call them `relative_poses` in code, not "deltas"). `P_k` is
  exactly the camera pose at `t+k` relative to `t`, which is what the future
  view depends on. **No pedals/steering/actions** -- the
  trajectory already carries the ego-motion that determines the future view,
  and recorded steering is wrong on `YawShiftCrop`-shifted clips.
- **Target:** frozen DINOv2 patch features of frame `t+k`, **average-pooled
  the same way** (16x16 -> 4x4 per camera), no gradient. No learned
  projector, no SIGReg, no EMA in this phase: a frozen, parameter-free target
  cannot collapse. (Learned projector + SIGReg is Phase 2, §8.)
- **Trajectory input never enters the trunk.** It goes only into the
  predictor. The deploy path (`forward`, `predict_step`,
  `patch_policy_decoder.py`, ONNX/TRT export, KV-cache geometry,
  `tokens_per_frame`) must be **byte-identical**. The WM is training-only.
- **`wm_weight` defaults to `0.0`** (same convention as `trajectory_weight`,
  see `config/experiment/yaak/patch_policy/dinov3.yaml:30-47`): modules are
  built and exercised every step but contribute exactly zero loss/gradient.
  `wm_head=None` (default) must reproduce the current model bit-for-bit.

## 3. Data geometry (verify, don't assume)

- Causal arms: `episode_length: 32`
  (`config/experiment/yaak/patch_policy/dinov2_dinowm_causal.yaml:164`),
  frames spaced `episode_step: 10` raw frames apart (`dinov3.yaml:14`) at
  30 fps (`config/dataset/yaak/train.yaml:9519`), so one clip step is
  ~0.33 s and `wm_horizons=(1, 2, 4)` is ~0.33 / 0.67 / 1.33 s.
  `trajectory_horizon: 5`.
- **Hard limit `k <= trajectory_horizon` (= 5 today).** The predictor needs
  poses up to `t+k`. `k = 8` (~2.67 s) is NOT available without a data
  change: `episode_length + trajectory_horizon <= clip_length` (`dinov3.yaml:38`)
  gives 32 + 8 = 40 > 37, so it needs a dataset rebuild with a longer
  `clip_horizon` (single-writer cache build) or a separate WM pose target.
  Out of scope for the first arm.
- `ChunkFields` narrows images to the first `episode_length` frames
  (`src/rmind/components/nn.py:272-276`). **Use in-window targets only**:
  for anchor `t` and horizon `k`, the target frame `t+k` must satisfy
  `t + k < episode_length`. Those frames are already encoded by the frozen
  `image_encoder` in `_features` (`patches`, `patch_policy.py:839-840`), so
  the target costs no extra ViT pass. Do not extend the image window.
- Relative poses: `trajectory_target` `(b, t, num_poses, 3)` from
  `TrajectoryTarget` / `rolling_dead_reckoned_trajectory`
  (`src/rmind/components/dead_reckoning.py:80`) is cumulative ego-centric
  `(x, y, theta)` relative to the anchor's own pose (meters, no `/100`; see
  `dead_reckon_future_trajectory`, `dead_reckoning.py:11`), at poses
  `t+1 .. t+num_poses` on the same clip-frame grid as the images. For horizon
  `k` feed `trajectory_target[:, t, :k]`. **Verify before relying on it**,
  on a real batch: pose index `i` corresponds to image frame `t+1+i`
  (compare the pose time stamps against the image time stamps, not just
  indices), and the SE(2) convention (axis directions, sign of `theta`)
  matches what the image shows (e.g. a left turn gives `theta > 0` and
  `y` toward the side the scene moves away from).
- Patch grid: 16x16 per camera (dinov2 at 224/patch 14, dinov3 at
  256/patch 16). Frame token layout (`_frame_tokens`, `patch_policy.py:648`):
  `[speed, cam*256 patches, registers..., readout]` -> patch positions are
  `1 : 1 + cam*256`. Reshape per camera to `(16, 16)` before pooling.

## 4. Implementation

### 4.1 Expose all trunk tokens

Refactor `_readout` so the full per-frame embedding `(b, t, k, d)` is
available (e.g. an out-parameter dict like `token_norms` /
`trajectory_target_out` -- that is the established convention in
`_features`; do not widen return tuples). `norm` handling for the readout is
unchanged.

### 4.2 New module `rmind.components.world_model.WorldModelPredictor`

- Input: pooled trunk tokens `(N, cam*16, d_policy)`, relative poses
  `(N, k, 3)`, horizon `k`.
- Pose embedding: MLP on `(x, y, sin theta, cos theta)` per pose, one token
  per pose, plus a learned horizon embedding. Learned positional embedding for
  the `cam*16` spatial tokens (include camera identity).
- Body: small **bidirectional** transformer, 2-4 layers, width `d_policy`
  (512). Bidirectional is fine: the whole pose sequence is a given input and
  no future image is visible to it. Reuse `TransformerBlock` from
  `patch_policy.py:51` with an all-true mask if convenient. **No dropout**
  in the predictor (see the RNG rule in §4.3).
- Output: the `cam*16` spatial positions -> `Linear(d_policy, d_img)`
  (768 for dinov2 ViT-B; read it from `image_embedding_dim`).
- Pooling helper: `adaptive_avg_pool2d` to 4x4 on `(16, 16)` grids, shared
  by trunk-side and target-side so they align exactly.

### 4.3 `PatchPolicy` changes

New ctor args (all opt-in): `wm_head` (Hydra config / module, default
`None`), `wm_weight: float = 0.0`, `wm_horizons: tuple[int, ...] = (1, 2, 4)`,
`wm_detach_trunk: bool = False`, `losses["wm"]` (required when `wm_head` is
set -- raise in `__init__` exactly like `trajectory_head` does at
`patch_policy.py:390`).

**RNG rule (needed for the bit-for-bit gate).** The existing parity test only
reseeds with `torch.manual_seed(0)` before construction
(`tests/test_patch_policy.py:1319-1321`), so:

- construct `wm_head` **last** in `__init__`, after every pre-existing
  module, so their initialization draws are unchanged;
- run the WM forward **after** all policy heads, including `_sample_codes`
  (`sample_codes=True` draws from a multinomial), and keep the predictor
  free of dropout/stochastic ops, so the forward RNG stream is unchanged.

In `_compute_metrics` (`patch_policy.py:1102`), when `wm_head` is set, after
all existing losses:

1. Pooled trunk tokens per frame `(b, T, cam*16, d)`;
   `.detach()` them if `wm_detach_trunk`.
1. Target: pooled frozen `patches` `(b, T, cam*16, d_img)`, under
   `no_grad`, from the same frozen-encoder output the policy already uses
   (no extra ViT pass, no different layer). Apply a **parameter-free
   LayerNorm** (`F.layer_norm` over the feature dim, no affine) to the
   target so the MSE scale is sane for both post-norm (`dinowm`) and pre-norm
   feature arms. Factor this into ONE helper `_wm_target(patches)` (pool ->
   normalize) and use it for the target AND for every baseline in §4.4, so
   all of them are in the identical space.
1. For each `k` in `wm_horizons`, all anchors `t in [0, T-k)`: predict, MSE
   vs target at `t+k`, mean over spatial tokens and feature dims. Average
   **per horizon first, then over horizons** (equal weight per `k`; longer
   horizons have fewer valid anchors and would otherwise be under-weighted).
1. `losses["wm"] = self.wm_weight * mean`.

Masking:

- **1 Hz-heading drives** (Niro101/102/104, 2022-11..2023-01) have
  staircase `headings_denoised`, so their dead-reckoned deltas are wrong --
  this corrupts the predictor's *input*. Check whether the training split
  includes them; if so, add a per-clip validity mask (e.g. drop clips whose
  heading changes in < N distinct steps) or exclude those drives.
- **`YawShiftCrop`** (`src/rmind/components/augment.py:60`) applies a
  per-frame random-walk yaw `psi_t`, so frames `t` and `t+k` have different
  synthetic yaw. Only `..._r1a_yaw.yaml` and `..._r1b_yaw_photo.yaml` use
  it; the first arm's base (`..._r2_heads.yaml`) does not. If an arm
  combines both: the crop is a pure camera rotation (no translation), and
  `YawShiftCrop` already rotates the anchor-`t` poses into the virtual frame
  at `t` (`x, y` rotated, `theta - psi_t`). The only remaining correction is
  the heading at the target frame: `theta_k += psi_{t+k}` (needs the realised
  per-frame `psi` exposed from the augmentation). Simpler alternative: mask
  shifted clips out of the WM loss.

### 4.4 Metrics (val, logged under `policy/metric/`)

- `wm_loss_k{k}` per horizon (unweighted, so it is visible at `wm_weight=0`).
- `wm_persistence_k{k}`: MSE of `_wm_target(z_t)` vs `_wm_target(z_{t+k})`
  -- the same helper, so identical pooling and normalization as the loss.
  **The predictor must beat this**, or it is just copying.
- `wm_action_gap_k{k}`: loss with relative poses shuffled across the batch
  (true trunk tokens) minus loss with the true poses. ~0 means the predictor
  ignores ego-motion -- the "action-conditioned" claim fails.
- `wm_visual_gap_k{k}`: loss with trunk tokens shuffled across the batch
  (true poses) minus loss with the true tokens. ~0 means the predictor
  ignores the visual history and predicts a "typical scene under this
  ego-motion" -- then the WM gradient cannot be teaching the trunk anything
  about the scene. Shuffle with a fixed permutation (e.g. roll by 1) so the
  metric is deterministic.
- `wm_grad_cos`: cosine between the gradient of the existing policy losses
  (action + trajectory + mode, as weighted in the total) and of the
  **unweighted** WM loss w.r.t. the shared trunk parameters (`encoder` +
  `patch_projection`), flattened. Computed periodically (e.g. every N train
  steps, on that step's batch, via `torch.autograd.grad` with
  `retain_graph=True`), not every step. Persistently negative = the two
  objectives fight; the first thing to check if closed loop regresses.
  Must not change the optimizer step (gradients only, no `.backward()`).
- Optional (Phase 3 groundwork): hit rate -- for each anchor, roll out the
  `trajectory_head`'s K hypotheses + GT, rank by distance to the real target
  latent, log top-1/top-K for GT. Fits next to `counterfactual_metrics`
  (`patch_policy.py:1442`) on the same probe subset.

## 5. Tests (`tests/test_patch_policy.py`, mirror the trajectory-head tests at :1238+)

- `wm_head` absent by default; `wm_head` without `losses["wm"]` raises.
- `wm_weight=0.0` reproduces the no-WM baseline total loss **bit-for-bit**
  (and gradients on every pre-existing param identical), with
  `sample_codes=True` so the forward-RNG ordering of §4.3 is exercised.
  Also assert the pre-existing params are initialized identically with and
  without `wm_head` (construction-order rule).
- `wm_weight>0`: trunk (`encoder`, `patch_projection`, `readout_token`)
  receives gradient from `losses["wm"]` alone; `image_encoder`,
  `goal_encoder`, `tokenizer` receive none (extend
  `test_frozen_modules_receive_no_grad`, :465), and the target tensor has
  `requires_grad=False` / no `grad_fn` (no gradient path through the target
  branch).
- `wm_grad_cos` computation leaves `.grad` of every param unchanged.
- `wm_detach_trunk=True`: only `wm_head` params get WM gradient.
- Causality: perturbing frames `> t` does not change the predictor output for
  anchor `t` (model on `test_readout_is_causally_valid`, :682).
- Multi-camera (`cam=3`) shapes; pooling aligns trunk tokens and targets.
- Deploy path unchanged: `forward`/`predict_step` outputs identical with and
  without `wm_head`; decoder step / export test still passes;
  `load_for_export` of a WM-trained checkpoint works (the WM keys must be
  ignored or dropped, not crash strict loading).

## 6. Configs

- `config/experiment/yaak/patch_policy/<base>_wm.yaml` on top of
  `dinov2_dinowm_causal_readout_dropout_depth_anything_v2_r2_heads.yaml`:
  `wm_head` config, `losses.wm: torch.nn.MSELoss`, `wm_weight: 0.0`,
  `wm_horizons: [1, 2, 4]`, wandb tag `world_model`.
- Arms: `wm_weight in {0.0, 0.1, 0.5}`; plus `wm_weight=0.5, wm_detach_trunk=true` (probe-only ablation: does the WM *gradient* help, or
  does the predictor just learn on top?). Keep `trajectory_weight` at the
  base arm's value so the only change is the WM term. Note what the detach
  arm does and does not show: it measures whether the predictor can learn
  from the UNCHANGED trunk; it says nothing about policy improvement.
- Cheap follow-up ablations, only after the first arm passes §7 (2):
  per-step SE(2) deltas `A_i = P_{i-1}^{-1} ∘ P_i` instead of cumulative
  relative poses (same information, different parameterization); 8x8 pooling
  instead of 4x4 (4x4 may wash out small objects -- pedestrians, distant
  lights).
- Generated dataset configs under `config/dataset/*.yaml` are stale ytt
  outputs -- run `just generate-config` if you touch dataset geometry (you
  should not need to).

## 7. Evaluation and acceptance

Run all commands inside `nix develop` (uv/just only work there; GPU needs
`NIX_LD_LIBRARY_PATH` pointed at host driver libs). Check training liveness
with W&B `global_step`, not file growth or `%CPU`.

In order:

1. Tests green; `wm_weight=0.0` bit-for-bit gate passes.
1. WM is learning and uses BOTH inputs: `wm_loss_k*` < `wm_persistence_k*`,
   and both `wm_action_gap_k*` and `wm_visual_gap_k*` clearly > 0 on val.
1. Policy is not hurt: `offset_argmax_recon_last` (the deployed-path metric
   -- select checkpoints on it, not `val/loss/code_*`) within noise of the
   `wm_weight=0` arm.
1. Planning improves: trajectory metrics (`trajectory_loss_xy`,
   `trajectory_loss_heading`, the minADE/FDE-style metrics the head logs) and
   `goal_sensitivity_{yaw,lateral}` vs. the `wm_weight=0` arm.
1. **Closed-loop (rsim) route completion** before any weight > 0.5 or a
   longer run. Round 6 saw an unweighted aux head trend negative closed-loop
   while open-loop looked fine -- do not skip this.

Interpretation: if (2) passes but (4) does not, and the `wm_detach_trunk`
arm matches the coupled arms, the WM learns but the trunk does not benefit
-- stop tuning the weight and go to Phase 2. If (4) regresses, look at
`wm_grad_cos` first: persistently negative means the objectives conflict
under joint training, which is itself an argument for Phase 2's sequential
recipe. Do not optimize WM MSE as a goal in itself; a better DINO-latent
prediction is not evidence of better planning.

## 8. Later phases (out of scope for this brief, design only)

- **Phase 2 -- learned compact projector (the paper's transfer result).**
  Replace average pooling with AD-E2E-JEPA's projector (2x Conv2d k2 s2,
  D -> 256, 16x16 -> 4x4) on the target side, trained with SIGReg against
  collapse; optionally feed its 16 tokens/camera into the trunk instead of the
  256 raw patches (tokens_per_frame 260 -> ~20 at cam=1, ~52 at cam=3; needs
  encoder/decoder/export geometry changes). This is also where the
  **sequential recipe** belongs (WM pretraining -> policy fine-tuning), which
  is what the paper actually measured (80.2 -> 85.4 EPDMS); Phase 1 is joint
  multitask training, which the paper does not test. Phase 1 results do not
  predict Phase 2's.
- **Longer horizons** (`k = 8`, ~2.67 s): needs the data change in §3.
- **Phase 3 -- WM as hypothesis selector.** `trajectory_head` proposes K;
  the predictor rolls each from the same frame-`t` trunk tokens; score
  against a route-conditioned goal latent head (readout + goal latent ->
  estimate of the future pooled latent) as an alternative to `mode_head`.
  This would be the first change that touches the deploy path.

## 9. Review points checked and NOT adopted

From `docs/wm_aux_loss_plan_verdict.md`, rejected after checking the repo --
do not re-raise without new evidence:

- *"YawShiftCrop needs translation corrections, not just yaw."* The crop is
  a pure camera rotation; only the target-frame heading needs `+psi_{t+k}`
  (§4.3). Not relevant to the first arm anyway.
- *"B2: frozen DINO target + learned adapter on the prediction side."* The
  predictor (transformer + `Linear(d_policy, d_img)` output) already is that
  adapter; a separate arm adds nothing. B3 (projector + SIGReg) = Phase 2.
- *"Skip the WM forward entirely at `wm_weight=0` in production."*
  Production is `wm_head=None`, which already skips it; `wm_weight=0` exists
  only as the parity gate, same convention as `trajectory_weight`.
- *"Check that samples don't cross episode boundaries."* Clips are built
  within a single drive; timestamp gaps inside a drive would already corrupt
  `trajectory_target` for the existing trajectory head -- not a new WM risk.
