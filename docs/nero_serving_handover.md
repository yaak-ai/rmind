# nero-arms policy — serving handover

A **random-weight** checkpoint you can build a serving app against today, before any model is
trained. Shapes and the interface are real; the numbers it outputs are meaningless.

## Get it

```bash
git clone -b feat/nero-arms-causal-patch-policy git@github.com:yaak-ai/rmind.git
cd rmind && uv sync
uv run python -m rmind.scripts.nero_serving_stub --out ./nero-serving
```

Writes `policy_random.ckpt` (~250 MB) and `io_contract.json`. Load with
`torch.load(..., weights_only=False)` → `{"state_dict", "experiment"}`, and build the module with
`hydra` from experiment `yaak/nero_arms/causal`.

**62.9M parameters** (40.7M trainable; the rest is the frozen DINOv2 image encoder).

## Inputs — 9 tensors, batch dim first

| key | shape (batch 1) | dtype |
|---|---|---|
| `image.base` | `(1, 6, 3, 270, 480)` | **uint8** |
| `image.side_left` | `(1, 6, 3, 300, 480)` | **uint8** |
| `image.side_right` | `(1, 6, 3, 300, 480)` | **uint8** |
| `goal.image.base` | `(1, 3, 270, 480)` | uint8 |
| `goal.image.side_left` | `(1, 3, 300, 480)` | uint8 |
| `goal.image.side_right` | `(1, 3, 300, 480)` | uint8 |
| `camera_cond` | `(1, 3, 13)` | float32 |
| `state.pose` | `(1, 6, 2, 46)` | float32 |
| `side_valid` | `(1, 2)` | bool |

`6` is the history length `T` (30 Hz, so 200 ms of context). `2` is `[left, right]`.

**Verified: these 9 are all the forward pass needs.** `action.future_state`,
`action.commanded`, `goal.xyz` and `align_residual_ms` appear in a training batch but are labels
and diagnostics — a serving app must not supply them.

## Output

| key | shape | meaning |
|---|---|---|
| `policy.action` | `(1, 2, 6, 60)` | `(batch, side, horizon, action_dim)` |

Horizon 6 at 30 Hz = a 200 ms action chunk. Values are **standardised** — de-standardise with the
stats shipped beside the tokenizer checkpoint.

## Four things that will bite you

**1. Inference is DETERMINISTIC by default** (`sample_codes=False`, i.e. `argmax` over action
codes). Verified: repeated calls on identical inputs are bit-identical.

Set `model.sample_codes = True` for stochastic sampling — VQ-BeT can sample from its categorical
over codes, giving a multimodal action distribution. That is a real capability, but it is opt-in,
because a serving app that assumes determinism should not be surprised. Measured: sampling makes
repeated calls on identical inputs differ by ~1.1 in standardised units.

No loss depends on this flag while `teacher_force_offset` is true (the default): the code losses
are cross-entropy against tokenizer-encoded targets and the offset loss is teacher-forced from
those same codes. Verified — all five losses bit-identical under both settings. It changes only
the reported reconstruction metrics, which are now computed the way serving decodes.

**2. State goes in as 46 dims, actions come out as 60.** Different spaces, not a typo. Input is
canonical-quaternion storage (`3 + 4` per pose); output is the model-facing form
(`3 + 6D rotation`). Convert with `rmind.data.nero.state_quat_to_9d`. Quaternions are
discontinuous as a regression target, which is why the output uses 6D.

**3. The three cameras have different heights.** `base` is 270×480, the sides are 300×480 — the
real sensors differ (OAK-D W vs two OAK-D SR). Do not assume a common size, and do not resize
without propagating the change into `camera_cond`, which carries resolution-normalised
intrinsics.

**4. Images are `uint8`, not normalised floats.** Normalisation happens inside the model.

## `camera_cond`, and the current caveat

`(3, 13)` per sample: for each camera, resolution-normalised `fx/W, fy/H, cx/W, cy/H`, then
translation (3) and rotation as 6D (6) in the world frame. It is what lets the policy generalise
across camera-setup changes.

⚠️ **Extrinsics are still placeholders (identity).** Intrinsics are real, read from device EEPROM;
the camera→world poses are not yet calibrated. Until they are, only the first 4 of 13 dims carry
information. Plumb the full vector now; the values will improve without a shape change.

## What will change

- **Action space.** Iteration-1 teleop is robot-native, so the real action becomes NERO joint
  commands (7 DOF/arm + 6 actuated hand DOF ≈ **26 bimanual**), not the 120 here. This is a
  config seam — verified by a real forward/backward at `action_features=12` — so expect the last
  dimension to change and nothing else about the interface.
- **Depth** may be added as a fourth stream (in progress).
- Weights become real. Nothing above changes shape when they do.

## Reference

Full data contract: `nero-arms/DATA_CONTRACT.md`. Model design:
`docs/nero_arms_causal_patch_policy.md`. Machine-readable I/O spec: `io_contract.json`.
