# rmind × vjepa2.1 integration — agent constraints

These constraints apply to every Claude Code invocation in this repo, interactive
or headless. They override any conflicting instruction in a prompt.

## Task scope

Two deliverables, in order:

1. **Pre-training encoder integration.** Add vjepa2.1 (ViT-L/16) as an encoder
   in rmind, mirroring the existing dinov3 image-encoder integration pattern.
   Determine whether it slots in as the episode encoder, the image encoder, or
   replaces both — and justify the choice in writing before changing any file.

2. **Fine-tuning config.** A separate YAML/template that loads the pre-trained
   vjepa2.1 encoder and trains the action-prediction policy on top of it.

## Integration rules

- **YAML/templates first.** New components are wired in via the existing
  template system (Hydra / OmegaConf / whatever rmind uses — discover it, don't
  assume). Python code is a last resort.
- **If Python is necessary**, justify it explicitly in the plan: what YAML
  feature is missing, why a new module class is required. One paragraph minimum.
- **Mirror dinov3.** Before writing anything, diff the proposed structure
  against the dinov3 image-encoder YAML. The new files should look like
  siblings, not cousins.
- **Read-only dependencies.** Do not modify source files inside `~/Code/rbyte`
  or `~/Code/vjepa2`. They are dependencies. If something is broken in them,
  report it — don't patch it.

## Architectural decision: freezing

- The **vjepa2.1 encoder itself is NOT frozen** during pre-training in rmind.
- The **dinov3 component *inside* vjepa2.1 IS frozen**. (V-JEPA 2.1 uses a
  pre-trained DINOv3 as part of its target/teacher pipeline; that part stays
  frozen.)
- This means the integration must expose two parameter groups, or set
  `requires_grad=False` on the dinov3 sub-module specifically. Verify this is
  achievable from YAML; if not, that's a legitimate reason to add Python.

## Checkpoint

- Use the **ViT-L/16 vjepa2.1** checkpoint. Do not substitute ViT-B, ViT-H,
  or any other variant without asking.
- The exact path/URL of the checkpoint is unknown to the agent at the start.
  Phase 1 must locate it (inside `~/Code/vjepa2`, in a release artifact, or via
  the paper). If it cannot be located, STOP and ask.

## Open questions the agent must resolve in Phase 1

These are unknowns flagged by the user — do not guess; report findings.

1. Whether "vjepa2.1" lives in the `~/Code/vjepa2` repo (branch, subdir, tag)
   or somewhere else.
2. Whether the second paper URL (arXiv 2603.14482) is reachable. The ID looks
   malformed (the YYMM prefix doesn't parse). If `WebFetch` fails, report it —
   do not invent the paper's contents.
3. Whether vjepa2.1 fits the episode-encoder slot, the image-encoder slot, or
   subsumes both. The user's hypothesis is "both in one go" — verify against
   the actual rmind interfaces.
4. Action-prediction policy details for fine-tuning: action space, dataset,
   loss. If rmind has an existing action-prediction config, reuse its defaults
   and call them out. If not, STOP and ask.

## Safety rails

- All work on a feature branch: `feat/vjepa2.1-encoder`. Create it before any
  edit. Never commit to main.
- Never `git push`. Never `git push --force`. Never `git reset --hard` against
  anything the user has touched.
- Never run training. Validation = config loads + one forward pass on dummy
  tensors of the documented input shape. That's it.
- Never download model weights without confirming. If a checkpoint isn't on
  disk, report the URL and stop.
- Do not modify files outside `~/Code/rmind`.

## Reporting

- Every claim about an existing file must cite `path:line`.
- The Phase 2 plan is a hard checkpoint. Print it, then stop. Wait for the
  orchestrator script (or the user) to advance to Phase 3.
- On completion, produce a summary listing: files added, files edited, Python
  additions (with justification), config-load validation result, forward-pass
  validation result.
