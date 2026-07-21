# Foresight readout & target experiments (spinoffs 1.1 / 2.4 / 1.2)

Three changes off the foresight-multimodality findings ("FD-objective quality is
high-leverage; foresight carries continuous state but the single softmax summary
readout loses onset structure"). All additive and config-gated; the baseline
path is byte-identical when the new knobs are unset.

Branch: `feat/foresight-readout-and-target` (off `feat/decode-strategy-eval` @ d80929e).

## What changed

| # | Idea | Level | Files |
|---|------|-------|-------|
| **1.1** | Widen the policy-head readout: learned multi-query attention pool over the 256 foresight slots (more bandwidth than the single `observation_summary` token) | **FT-only** | `components/foresight_readout.py`, `objectives/joint_policy.py`, `policy_finetune_foresight.yaml` |
| **2.4** | Event-preserving readout: channel-wise **max-pool** over the foresight slots (keeps onset peaks the softmax summary averages away) | **FT-only** | `objectives/joint_policy.py`, `policy_finetune_foresight.yaml` |
| **1.2** | Better FD prediction target: frozen `target_encoder` re-encodes the FD target (e.g. a V-JEPA *video* encoder carrying motion single-frame DINO lacks) | **PT-level** | `objectives/forward_dynamics.py`, `components/vjepa_backbone.py` |
| — | `TrainingQualityLogger` on every run (grad/weight norms, dead-grad, train/val gap, effective rank) | both | `callbacks/training_quality.py`, `trainer/callbacks/{finetune,pretrain}.yaml` |

`JointPolicyObjective._features` now assembles its width from explicit branches
and **validates it against the code head's `in_features`** (raises on mismatch —
the shape-mismatch class that bit the 84pc0t8n export). Legacy configs
(`read_waypoints=None`, no foresight branch) keep the exact auto-detect behavior.

## 1.1 + 2.4 — finetune on an existing PT checkpoint (no re-pretrain)

Both read the encoder-output foresight slots of an already-pretrained model, so
they need only a finetune. `policy_finetune_foresight.yaml` enables both; the
assembled head width is `3*384 (obs_summary+obs_history+waypoints) + 4*384
(attn, num_queries=4) + 384 (maxpool) = 3072`, wired into the code/offset heads.

```sh
just finetune \
  model=yaak/control_transformer/policy_finetune_foresight \
  model.artifact=<PT_ckpt>:v9 \
  action_tokenizer_artifact=<tokenizer>:v9 \
  teacher_force_offset=true
```

**Ablations** (isolate each branch): in `policy_finetune_foresight.yaml` drop the
`foresight_attn` block (1.1 off) and/or set `foresight_maxpool: false` (2.4 off),
and set the code/offset `in_channels` back accordingly (attn −1536, maxpool −384;
1152 == the stock `policy_finetune`).

**Calibrated expectation (low prior).** `observation_summary` is a *learned
attention* readout (not mean-pool) and already out-extracts mean-pooled foresight
on event onsets (0.777 vs 0.629 AUROC), and the c0-report transformer-over-
un-pooled-tokens probe scored 58.2% < 61% pooled. So the only headroom is peaks a
single softmax token smooths over — real but likely small, and it mostly pays off
once 1.2 enriches the foresight *content*. A clean null here is itself a result:
"no incremental readout value."

## 1.2 — richer FD target (V-JEPA video)

`ForwardDynamicsPredictionObjective` gained `target_encoder: Module | None`. When
set, each gathered FD target is passed through it (frozen, detached) before the
loss, so the foresight head is supervised against that representation instead of
the per-frame DINOv3 patches. The head's output width must match the encoder's
output dim. Config diff for `raw.yaml`'s `objectives.forward_dynamics`:

```yaml
forward_dynamics:
  _target_: rmind.components.objectives.ForwardDynamicsPredictionObjective
  # ... existing norm / patch_pos_embed / projections ...
  target_encoder:
    _target_: rmind.components.vjepa_backbone.VjepaVideoBackbone   # -> (…, N, 1024)
  heads: { … output_projection.out_features: 1024 }               # was image_embedding_dim (384)
  losses: { … GramAnchoringLoss patches: <N_vjepa_tokens> }
  targets:
    foresight:
      cam_front_left: [input, image, cam_front_left]   # a RAW multi-frame source
```

**Status: scaffolding, validated in isolation, NOT trained.** The
`target_encoder` hook + `VjepaVideoBackbone` are ported and unit-tested, but the
full V-JEPA-video target needs two things this base does not have:

1. **A raw multi-frame source.** The episode's image frames are DINO-preprocessed
   (256 px, ImageNet norm); V-JEPA wants 384 px + its own normalization over a
   frame *window*. That raw-frame plumbing (and the unfrozen backbone integration)
   already lives on **`feat/vjepa-unfrozen-lejepa`** — 1.2 should be finished there,
   layering this hook on top, not duplicated here.
2. **A PT retrain** (FD is a pretraining objective; GPU-gated).

Instantiating `VjepaVideoBackbone` triggers a `torch.hub` load of the ViT-L/16
weights (on disk per the vjepa-c0 memory) — do not call it in a smoke test.

A self-contained alternative if V-JEPA is deferred: a **multi-future-frame DINO
target** (predict t+1..t+k from step t, using frames already in clip11) — richer
target, no new backbone, no clip rebuild. Not the horizon lever (2.1), which
needs a clip12+ rbyte rebuild.

## Validation done

`PYTHONPATH=src python smoke_foresight_readout.py`: ForesightAttentionPool
fwd/bwd, `_features` width assembly across every branch combo + the mismatch
guard, and the FD `target_encoder` re-encoding (frozen/detached). Package import
+ config YAML parse verified. No training run has been executed.
