# Providing intention to the palletjack policy: pickup/dropoff goals

Design notes for goal-conditioning `PatchPolicyContinuous` in the warehouse,
where the car model's waypoint intention does not transfer.

## Problem

For the car, intention is 10 map-matched waypoints of the road ~100 m ahead.
Indoors there is no road graph to map-match against, so waypoints have no source.

What we *do* have in the collected data is, per frame, the pose (position +
heading) of the **pallet to pick up** and the position of **where to place it**.
That is enough to condition the policy on a goal directly, rather than on a path.

## Framing

This is **goal-conditioned imitation learning**. Waypoints are only one intention
encoding; with no road graph the standard alternative is a **relative target
pose**. Our task is essentially **PointGoal navigation with a two-phase goal**
(pickup, then dropoff) and a pose-accurate terminal condition (fork engagement).

Taxonomy of intention encodings and fit:

| Encoding                              | Example                           | Fit                                      |
| ------------------------------------- | --------------------------------- | ---------------------------------------- |
| Discrete command (turn left/straight) | Codevilla CIL 2018                | too coarse for free-space maneuvering    |
| Dense waypoint path                   | car model, TransFuser             | needs a free-space planner we don't want |
| **Relative target pose**              | PointGoal nav (Habitat/DD-PPO)    | **chosen**                               |
| Goal image                            | Goal-Conditioned Transporter Nets | heavy; needs a goal render               |

Decision: **endpoint-only**, no waypoints. The policy learns obstacle avoidance
and implicit path planning from vision, exactly as PointGoal nav does.

## Goal representation

Two ego-relative pose goals, recomputed **every frame** from the SLAM pose (so
the policy is translation/rotation-invariant; absolute map coords would memorize
one layout). Heading as `(sin, cos)` to avoid the wrap discontinuity.

```
relative_pickup  = [dx, dy, sin_h, cos_h]   # pallet-to-pick, ego-relative
relative_dropoff = [dx, dy, sin_h, cos_h]   # placement pose,  ego-relative
```

Two **separate** tokens (not one concat), each with its own `Linear(4 -> d)`
embedding — they mean different things, so no weight sharing. Independent tokens
also allow per-token jitter/dropout augmentation.

## Phase, without a fork sensor

There is **no fork-height / load sensor**, so an explicit `fork_loaded` flag is
out — we could not supply it at inference, and feeding a wrong value points the
policy at the wrong goal. Instead the phase falls out of `relative_pickup`
geometry, *if* it is measured against the right reference:

- **Measured against the pallet's tracked pose** (available in training data):
  once engaged, the pallet rides on the forks, so `relative_pickup -> 0` and
  **stays** ~0. Unambiguous: large = "go pick", ~0 = "picked, go drop".
- **Measured against a fixed world location**: after engaging, the jack drives
  away and `relative_pickup` grows *again* — "approaching" and "leaving" look
  identical, so the policy can loop back. **Avoid.**

So: **train against the pallet-tracked pose.** Reproduce the same signal at
inference with a trivial geometric **latch**, no sensor needed:

> when the jack pose comes within epsilon of the clicked pickup pose ->
> `pickup_reached = True` -> pin `relative_pickup := 0` from then on.

The latch is pure SLAM-pose-vs-clicked-point geometry, fully available live. It
reconstructs `fork_loaded` from the one thing we *can* observe ("have I arrived
at the pickup point"), keeping train and inference consistent. Without the latch,
two always-on goal tokens give the policy no sensor-free way to know it has not
picked yet.

## Injection into `PatchPolicyContinuous`

Goals are **inputs, not targets** — no new head, no new loss. Mirror the existing
`speed` token path (scalar -> tokenizer -> embedding -> one `(b,t,1,d)` token
prepended to patches), but continuous (a pose vector, so `Linear`, no binner).

Frame block becomes (goal + speed prepended so it still **ends on a patch
token**, which is the readout at `[:, :, -1]`):

```
[pickup_tok, dropoff_tok, speed_tok, patches...]
```

Config (`config/model/palletjack/patch_policy/policy.yaml`):

```yaml
# Remapper: add the two ego-relative goal vectors (precomputed per frame)
continuous:
  ...
  relative_pickup:  [data, relative_pickup]
  relative_dropoff: [data, relative_dropoff]

# analogue of speed_tokenizer + speed_embedding, continuous -> Linear (no binner)
pickup_embedding:
  _target_: rmind.components.nn.Linear
  in_features: 4
  out_features: ${policy_embedding_dim}
dropoff_embedding:
  _target_: rmind.components.nn.Linear
  in_features: 4
  out_features: ${policy_embedding_dim}

# encoder: two extra context tokens per frame
tokens_per_frame: "${eval:'${num_cameras} * ${num_patches} + 3'}"   # was + 1
```

Model (`_frame_tokens`):

```python
pickup_tok  = self.pickup_embedding(self._get(inputs, self.relative_pickup))   # (b,t,1,d)
dropoff_tok = self.dropoff_embedding(self._get(inputs, self.relative_dropoff)) # (b,t,1,d)
speed_tok   = self.speed_embedding(self.speed_tokenizer(speed))                # (b,t,1,d)
# goal + speed first so the block ends on a patch token (the readout position)
return torch.cat([pickup_tok, dropoff_tok, speed_tok, patches], dim=-2)
```

Bump `tokens_per_frame` by exactly the number of context tokens added. Export
path is unaffected: the goal fields resolve like any other input; live they come
from the SLAM pose plus the map click (with the pickup latch above).

Alternative considered — **channel fusion** (the paper's `D+G`, concat the goal
latent onto every patch before `patch_projection`): more params, goal broadcast
to every patch. The extra-token route is lighter and idiomatic to this trunk;
revisit fusion only if the tokens are under-used.

## Data: task segmentation and relabeling

- **Task unit.** Segment recordings into variable-length **tasks** (a.k.a.
  "moves" — operators call one pallet relocation a move): a task starts when
  pickup != dropoff and ends when pickup == dropoff (goal reached). Keep "clip"
  / "window" for the fixed 2 s slices the model consumes. Reserve "episode" — it
  is overloaded.
- **Hindsight relabeling.** Any trajectory segment is an expert demonstration of
  reaching wherever it actually ended. Relabel `goal := achieved terminal pose`
  to turn *all* driving (repositioning, aborted moves) into goal-conditioned
  training data, not just clean pickup->dropoff tasks.

## Pitfalls

- **Target-point shortcut** (Hidden Biases of E2E Driving Models): target-
  conditioned models can recover steering from the goal alone and under-use
  perception. Keep goals reachable; watch for it in tight aisles.
- **Long-horizon multi-goal conflict** (multi-expert forklift RL): coarse
  navigation vs cm-precise fork engagement have competing objectives. The
  engagement phase may need its own loss weighting or a separate fine-approach
  policy.

## References

- Codevilla et al., End-to-end Driving via Conditional Imitation Learning — arXiv:1710.02410
- Hidden Biases of End-to-End Driving Models — arXiv:2306.07957
- PointGoal navigation, egocentric relative goal (Habitat, DD-PPO) — arXiv:2009.03231
- Goal-Conditioned Transporter Networks — arXiv:2012.03385
- Goal-conditioned Imitation Learning (Ding et al.) — arXiv:1906.05838
- Hindsight goal relabeling as imitation — arXiv:2209.13046
- Heterogeneous Multi-Expert RL for Long-Horizon Multi-Goal Autonomous Forklifts — arXiv:2601.07304
