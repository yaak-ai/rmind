# Task brief: trajectory tokenizer codebook check (before the VQ-BeT trajectory head)

> **Status 2026-10-15: CLOSED after step 1.** The project owner froze
> `model-403b4rrh:v26`. Steps 2–6 and the §4 arms are SHELVED. See §10 for
> the step 1 findings. Next brief: `docs/trajectory_vqbet_head_plan.md`.

Self-contained task brief, written for an agent with **no prior context**.
Everything needed is here or behind the `file:line` references. Branch:
`exp/cond-t2`, tokenizer code at commit `1f487843`.

## 1. Why this work exists

`TrajectoryTokenizer` (`src/rmind/models/trajectory_tokenizer.py`) is a
residual-VQ autoencoder over the dead-reckoned future ego trajectory. It is the
first half of a plan to replace `PatchPolicy`'s direct-regression MLP
trajectory head with a VQ-BeT head: classify one code per RVQ level, predict a
continuous offset, then decode with the frozen tokenizer.

The first full run, W&B `yaak/rmind/403b4rrh`, reconstructs well. Its
codebooks look under-used, though: the level-0 perplexity was **6.03 of 16
from epoch 0 to epoch 39** and never moved. The VQ-BeT head *classifies*
level 0, so this decides how many coarse trajectory modes the policy can pick
between. Before anyone builds the head, find out:

- **Q1:** Does the trajectory data really have about 6 coarse modes (a skewed
  distribution that is reconstructed fine either way)?
- **Q2:** Or is the codebook stuck, so that forcing more even usage or using a
  different size improves reconstruction?

The output is one codebook configuration to freeze for the head, plus the
evidence for it (§7).

## 2. Decisions already made: do not revisit

- **The representation is `absolute`.** A 1500-step A/B tied delta vs absolute
  (val ADE 0.183 vs 0.174 m, FDE 0.299 vs 0.316 m), and the project owner
  picked absolute. Do not re-run delta.
- **The target is fixed.** Use `rmind.components.nn.TrajectoryTarget`: 5
  future poses, one every 10 frames (~1/3 s, ~1.67 s total), anchor-frame
  `(x, y, theta)`, x lateral and y forward. It is the same target
  `patch_policy`'s trajectory head uses. Do not change the horizon or step.
- **Codebook sizes stay uniform across levels.** The policy heads assume one
  `(num_quantizers, codebook_size)` geometry: `patch_policy.py:920` (`g, c = quantizer.num_quantizers, quantizer.codebook_size`) and
  `objectives/joint_policy.py:112`. The rmind `ResidualVQ` wrapper
  (`components/vq.py:71`) only takes an `int`. A per-level size (for example
  8 codes for level 0 only) is **out of scope**: it would need changes to
  `vq.py`, `perplexity`, and both heads. If §7's evidence points strongly that
  way, record it as a follow-up and do not build it.
- **Building the VQ-BeT trajectory head is out of scope.** This brief only
  picks the tokenizer.
- **Keep the training recipe** from
  `config/experiment/yaak/trajectory_tokenizer/pretrain.yaml`: 40 epochs
  (~1239 steps/epoch at batch 2048, so ~49.6k steps) against the 50k-step
  cosine schedule in `config/model/yaak/trajectory_tokenizer/raw.yaml`. If you
  change `batch_size`, rescale the epochs so the schedule still completes (see
  commit `1f487843`).

## 3. Established facts (cite these; don't re-derive them)

### 3.1 Baseline run `403b4rrh` (commit `1f487843`, codebook 16 × 4 levels)

|                     | val ADE (m) | val FDE (m) | heading err | final heading err | perplexity q0/q1/q2/q3 |
| ------------------- | ----------- | ----------- | ----------- | ----------------- | ---------------------- |
| epoch 0             | 0.292       | 0.567       | 0.72°       | 0.97°             | 5.5 / 6.0 / 6.9 / 6.9  |
| epoch 26 (best ADE) | 0.114       | 0.201       | 0.61°       | 0.65°             | 6.0 / 7.6 / 8.1 / 7.9  |
| epoch 39 (last)     | 0.129       | 0.228       | 0.60°       | 0.62°             | 6.0 / 7.6 / 7.5 / 6.6  |

- **Checkpoint versions.** Artifacts `model-403b4rrh:v0..v39` hold one
  checkpoint per epoch. Assume `v26` is epoch 26, but check it against the
  checkpoint's stored `epoch` before using it.
- **The `best` alias is meaningless.** It points at `v39`, because
  `config/trainer/callbacks/action_tokenizer.yaml` has a `ModelCheckpoint` with
  no `monitor`, so best is simply the last epoch. §5 step 2 fixes this.
- **Val ADE is noisy between epochs.** It moved 0.114–0.129 over epochs 26–39
  with the LR nearly annealed. The val set is only the 5 drives in
  `config/_templates/dataset/yaak/trajectory_val.yaml` (14 batches,
  ~27.7k clips). Treat differences under ~0.015 m as noise until §5 step 4
  measures the seed floor.

### 3.2 What the logged metric means

- **`perplexity/q{i}` is NOT the number of codes used.**
  `ResidualVQ.perplexity` (`components/vq.py:131`) is `exp(entropy)` of one
  batch's code histogram, i.e. the *effective* number of codes. A perplexity
  of 6 fits both "6 codes used evenly" and "all 16 used, heavily skewed".
  Driving is mostly straight at a few speeds, so skew is expected. Q1/Q2 can't
  be answered until the actual histogram is measured (§5 step 1).

### 3.3 Dead-code replacement is already active

- The `ResidualVQ` wrapper defaults to `threshold_ema_dead_code=2.0`
  (`components/vq.py:75`) and passes it to the `vector_quantize_pytorch`
  library.
- In the library
  (`.venv/lib/python3.12/site-packages/vector_quantize_pytorch/vector_quantize_pytorch.py:555-564`),
  a code expires and is re-seeded from a random batch vector when its EMA
  cluster size drops below the threshold.
- With batch 2048, that means any code used by fewer than ~2 samples per batch
  (\<~0.1%) keeps being reset.
- **So lowering the threshold would let more codes die.** Raising it (for
  example to ~20, ~1% of a batch) is the lever that forces more even usage.
  An earlier conversation suggested "lower"; that was wrong.
- **Caveat: raising the threshold also forbids rare modes.** A code whose
  true share is below the threshold (for example a sharp turn at ~0.5% of
  clips) is re-seeded every step and can never hold a stable mode. The
  baseline threshold of 2 already rules out modes under ~0.1%. A better
  usage histogram under D20 can therefore hide the loss of exactly the turn
  modes the head needs, so check D20's level-0 heading table (§4.2.4), not
  only its histogram.

### 3.4 Decoding from a subset of levels

- `RVQ.get_codes_from_indices` (`residual_vq.py:305`, `mask = indices == -1`
  at :322) zeroes the contribution of any level whose index is `-1`.
- So `tokenizer.decode(tokenizer.quantizer.lookup(codes_with_later_levels_set_to_-1))`
  should give a level-prefix reconstruction. **Verify this in a unit test
  before relying on it** (§5 step 1).

### 3.5 Tokenizer API (`src/rmind/models/trajectory_tokenizer.py`)

| call      | maps                                 | defined at |
| --------- | ------------------------------------ | ---------- |
| `forward` | poses `(*b, 5, 3)` → codes `(*b, 4)` | :185       |
| `decode`  | `z_q` → poses                        | :192       |
| `invert`  | codes → poses                        | :197       |

- Normalization stats are persistent buffers (`norm_count`, `norm_sum`,
  `norm_sum_sq`) and are loaded with the checkpoint.
- Load a checkpoint with `TrajectoryTokenizer.load_from_wandb_artifact(...)`
  (`LoadableFromArtifact`, `rmind/utils/_wandb.py`), as the other tokenizers do.

### 3.6 Environment on `aboutblank` (all confirmed 2026-10-15)

- **Unit tests run locally.** The local `.venv` imports torch only with a
  narrow lib path:

  ```bash
  S=<scratch>/libs; mkdir -p $S
  for f in libstdc++.so.6 libgcc_s.so.1 libz.so.1; do
    ln -sf /run/current-system/sw/share/nix-ld/lib/$f $S/$f
  done
  LD_LIBRARY_PATH=$S:/run/opengl-driver/lib .venv/bin/python -m pytest tests/test_trajectory_tokenizer.py -q
  ```

  Do not put the whole nix-ld dir on the path: its glibc doesn't match.

- **Data loading does not work locally.** The `.venv` lacks rbyte's `yaak`
  extra, so nothing that builds a dataset can run there. `nix develop` also
  fails (`flake.nix` nixpkgs URL `?shallow=1`). Run all training and
  data-reading work in the Docker image `rmind:508f5b0aca982ff0a25b03c538db92b04e587358`.
  Its dependencies are identical to `1f487843`; only `src/` and `config/`
  differ, so mount those over `/app`.

- **GPU flag:** use `--device nvidia.com/gpu=all`. `--gpus all` fails with
  "AMD CDI spec not found".

- **Run the container as root.** The image's Python lives under `/root`.
  `trap` a `chown` on EXIT so no root-owned files are left behind.

- **Sample stores are already built:**
  `.rbyte_cache/yaak/trajectory_{train,val}/t6d10s10/samples`. Pass
  `++datamodule.{train,val}.dataset.samples.resume=true`, or each launch
  rebuilds them (~7 min).

- **Never instantiate the whole `cfg.datamodule` in an ad-hoc script.**
  pipefunc wipes the train sample store when it is instantiated. Instantiate
  `cfg.datamodule.val` only. See `project-rmind-training-gotchas` in the
  owner's notes.

- **Committing:** `just` and the pre-commit hook's cached `prek` are not on
  PATH. Shim both: `prek` → `uvx prek "$@"`, `just` → `uvx --from rust-just just "$@"`. The `ty` and `deptry` hooks fail repo-wide on pre-existing
  issues, so commit with `SKIP=ty,deptry` only after confirming none of their
  output mentions your files.

- **Ruff:** scope it to the files you edit. The repo sets `unsafe-fixes = true`, so a repo-wide `ruff check` rewrites unrelated files.

- **Run speed:** ~10.7 it/s, so a 40-epoch run takes ~80 min.

## 4. Experiment design

### 4.1 Arms

Every arm uses 40 epochs, `representation=absolute`, and 4 levels.

| arm | override                                      | question it answers                                                                                                                                                |
| --- | --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| B0  | none (= `403b4rrh`, already done)             | baseline                                                                                                                                                           |
| B1  | `seed=1338`                                   | seed noise floor (needed to read every other arm)                                                                                                                  |
| C8  | `codebook_size=8`                             | does halving the codebook cost reconstruction? If not, about 6 modes is real (Q1)                                                                                  |
| C32 | `codebook_size=32`                            | does a larger codebook buy reconstruction or level-0 diversity? (Q2)                                                                                               |
| D20 | `+model.quantizer.threshold_ema_dead_code=20` | does forbidding codes under ~1% of a batch (they get re-seeded) raise level-0 usage, and at what cost to reconstruction and to rare turn modes (§3.3 caveat)? (Q2) |

That is 4 new runs, ~5.5 h back to back. Running two at once on the RTX 5090
is probably fine (the model has ~100k params). Check that it/s doesn't drop
below ~8 for either run before you rely on that. If it does, run them in
sequence.

### 4.2 Metrics per arm

Extract all of these from the arm's checkpoint at its best val ADE epoch,
using the §5 step 1 script.

1. **Reconstruction:** val ADE, FDE, mean and final heading error. Report both
   the best epoch and the mean of the last 5 epochs (from the W&B logs). §7
   decides on the **last-5 mean**: the best epoch is the minimum of 40 noisy
   values and is biased low.
1. **Actual usage per level:**
   - the full code histogram over the whole val set (not per batch)
   - the number of codes with share >0.1% and >1%
   - perplexity over the whole val set
1. **Level-0 class quality (primary, no decoder):** for each level-0 code,
   take the mean ground-truth trajectory of the clips assigned to it. Report
   ADE/FDE from every clip to its own code's mean. This measures how good the
   coarse classes are on their own, which is what the head's classifier will
   choose between.
1. **Prefix reconstruction (diagnostic only):** ADE/FDE decoding from the
   first k levels, k = 1, 2, 3, 4. The tokenizer was trained without
   `quantize_dropout`, so the decoder has only seen the sum of all 4 levels;
   k < 4 inputs are out of distribution. A bad k=1 number can mean "the
   decoder breaks on partial sums", not "level 0 is poor". Use it for the
   §9 `num_quantizers=3` follow-up, not for §7.
1. **What level 0 encodes:** for each level-0 code, its share of clips, mean
   speed (from the y-extent of the final pose / 1.67 s), mean final heading
   (signed, degrees), the mean ground-truth trajectory of its clips
   (primary), and the decoded codebook vector (secondary). This tells whether
   the ~6 effective modes are "speed bins" or "speed × turn".

## 5. Work steps (in dependency order; each leaves the tree green)

### Step 1: audit script plus prefix-decode test

- **Script:** add `src/rmind/scripts/trajectory_tokenizer_audit.py`.
  - Arguments: a W&B artifact (`yaak/rmind/model-<run>:v<N>`) and an output
    JSON path.
  - Loads the tokenizer, instantiates **only** `cfg.datamodule.val` from the
    experiment config, and runs over the whole val set in eval mode.
  - Writes the §4.2 metrics 2–5 plus ADE/FDE/heading as JSON.
  - For the prefix decode, set later levels' codes to `-1` before
    `quantizer.lookup` (§3.4).
- **Test:** add a test in `tests/test_trajectory_tokenizer.py`. With a tiny
  tokenizer, `decode(lookup(codes with levels k.. = -1))` must equal
  `decode(sum of the first k levels' codebook vectors)`. Use
  `quantizer.codebook(level)` (`vq.py`) to build the reference.
- **First real run:** point it at `model-403b4rrh:v26` (after checking it is
  epoch 26) and `:v39`. This alone may answer Q1: if 16 codes have >0.1%
  share and the level-0 ground-truth means are distinct, sensible modes, then
  the low perplexity is just skew.
- Run the script in the Docker image with `/app/src`, `/app/config` and the
  rbyte cache mounted, as in §3.6.

### Step 2: make the `best` checkpoint meaningful

- Add `config/trainer/callbacks/trajectory_tokenizer.yaml`. Copy
  `action_tokenizer.yaml`, but give `ModelCheckpoint` `monitor: val/ade`,
  `mode: min` and `save_last: true`.
- Point `config/experiment/yaak/trajectory_tokenizer/pretrain.yaml` at it.
- Check that W&B's `best` alias then follows val ADE.
- With a `monitor`, the default `save_top_k=1` keeps only the best checkpoint
  plus `last`, so new arms will not have per-epoch artifacts like
  `403b4rrh:v0..v39`. That is fine: last-5-epoch means come from the W&B logs.
- Do **not** touch `action_tokenizer.yaml`, which the action tokenizer still
  uses.

### Step 3: commit and push steps 1–2

The launch flow requires a committed, pushed tree (`just train` runs
`check-git`). All four arms must run from the same commit. Include this brief
(`docs/trajectory_tokenizer_codebook_plan.md`) in the commit.

### Step 4: launch the arms

Export the commit's code to `outputs/traj-cb-<sha8>/`:

```bash
git archive <sha> src config | tar -x -C <dir>
ytt --file <dir>/config/_templates --output-files <dir>/config/ \
    --output yaml --ignore-unknown-comments --strict
```

`outputs/` is gitignored. Then launch each arm detached:

```bash
docker run -d --name rmind-traj-<arm> --device nvidia.com/gpu=all --shm-size=16g \
  -e PYTHONUNBUFFERED=1 -e WANDB_GIT_COMMIT=<full sha> -e WANDB_DIR=wandb_logs \
  -v $HOME/.netrc:/root/.netrc:ro -v /nasa:/nasa:ro \
  -v <dir>/src:/app/src -v <dir>/config:/app/config -v <dir>:/app/run \
  -v $PWD/.rbyte_cache:/app/run/.rbyte_cache -w /app/run --entrypoint "" \
  rmind:508f5b0aca982ff0a25b03c538db92b04e587358 \
  sh -c "trap 'chown -R $(id -u):$(id -g) /app/run /app/run/.rbyte_cache/yaak/trajectory_train /app/run/.rbyte_cache/yaak/trajectory_val' EXIT; \
    /app/.venv/bin/rmind-train --config-path /app/config --config-name train.yaml \
      experiment=yaak/trajectory_tokenizer/pretrain <ARM OVERRIDE> \
      ++datamodule.train.dataset.samples.resume=true ++datamodule.val.dataset.samples.resume=true"
```

- Set `wandb.name`/tags if the wandb config supports it; otherwise record the
  run-id → arm mapping in the result doc.
- After a run exits, remove the empty root-owned `<dir>/.rbyte_cache`
  mountpoint with `rmdir`.

### Step 5: audit every arm

Run the step 1 script on each arm's `best` artifact (the alias now works after
step 2). For B0, use `model-403b4rrh:v26`.

### Step 6: write up and decide (§7)

Write the results to `docs/trajectory_tokenizer_codebook_results.md`: a table
of §4.2 metrics per arm, the level-0 mode table for the chosen configuration,
and the decision.

## 6. Tests and verification

- `tests/test_trajectory_tokenizer.py` must pass, including the new
  prefix-decode test. Use the §3.6 local command.
- Run Ruff on the new and edited files only. Check that the hook output for
  `ty` doesn't mention them.
- **Before trusting the audit script's ADE:** on `403b4rrh:v39`, its full
  (k=4) val ADE must match W&B's logged epoch-39 `val/ade` (0.129) to within
  ~0.005 m.
  - The logged value is a mean of per-batch means. The script's is a mean
    over all clips. The last batch is smaller (1052 clips), so expect a small
    gap, not an exact match.
  - A large gap means the script differs from training in normalization
    buffers, eval mode, or which data it reads.

## 7. Decision rules

All ADE comparisons below use the **last-5-epoch mean** val ADE (§4.2.1).
Let **σ** = |B1 − B0| in that metric. If σ < 0.005 m, use 0.005 m.

Full (k=4) reconstruction barely depends on level-0 size, because levels 1–3
can make up for a smaller level 0. So Q1 is judged mainly on the level-0
histogram and the level-0 class-quality ADE (§4.2.3), not on k=4 ADE alone.

| outcome                                                                                                                       | conclusion                                                | freeze                                                                                                                                                                             |
| ----------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Step 1 shows ~all 16 codes used with distinct ground-truth level-0 means, **and** C8 ADE ≤ B0 + 2σ, **and** C32 ADE ≥ B0 − 2σ | Q1: the data has few coarse modes and size barely matters | **C8** (smaller classification target for the head), unless C8's level-0 class-quality ADE (§4.2.3) is clearly worse than B0's or C8 loses the turn modes B0 has (§4.2.5); then B0 |
| C32 or D20 ADE < B0 − 2σ                                                                                                      | Q2: the codebook was the bottleneck                       | the better of C32/D20. If D20, first confirm its level-0 table still has distinct turn modes (§3.3 caveat), and record its usage histogram so the head can use class weighting     |
| D20 raises level-0 usage but ADE > B0 + 2σ                                                                                    | even usage costs reconstruction                           | B0; note that D20's spread-out level 0 might still help the head's classifier and is worth testing at head time                                                                    |
| nothing clears 2σ                                                                                                             | inconclusive at this noise level                          | B0 (`403b4rrh:v26`); state that the codebook isn't the limiting factor                                                                                                             |

Whatever is frozen, report its artifact (`model-<run>:v<N>`) and its §4.2
level-0 mode table. That table is what the head design will be built from.

## 8. Risk register

| risk                                                             | effect                                                                                             | mitigation                                                                            |
| ---------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| Instantiating the full datamodule in the audit script            | wipes the train sample store; the next launch spends ~7 min rebuilding, or a concurrent run breaks | instantiate `cfg.datamodule.val` only (§3.6)                                          |
| Running the audit while a training container uses the same cache | val store contention                                                                               | audit only after the arms exit, or with `resume=true` on val only                     |
| Mistaking perplexity for code count                              | wrong Q1/Q2 answer                                                                                 | use the step 1 histogram counts, never perplexity alone (§3.2)                        |
| Single val split of 5 drives                                     | ADE differences are seed and val-set noise                                                         | B1 seed arm; decide on 2σ; report last-5-epoch means too                              |
| `v26 ≠ epoch 26`                                                 | auditing the wrong checkpoint                                                                      | read `epoch` from the checkpoint before using it                                      |
| `-1` prefix masking behaves differently from §3.4                | wrong k-prefix ADE                                                                                 | the step 1 unit test must pass first                                                  |
| Reading the k\<4 prefix ADE as level-0 quality                   | the decoder never saw partial sums, so the number is out of distribution                           | decide on the ground-truth class-mean ADE (§4.2.3); the prefix ADE is diagnostic only |
| Best-epoch ADE is the minimum of 40 noisy values                 | biased low, so arms look better or worse by chance                                                 | decide on the last-5-epoch mean (§7)                                                  |
| D20 re-seeds rare codes                                          | rare turn modes get no stable code                                                                 | check D20's level-0 heading table, not only its histogram (§3.3)                      |
| Two concurrent runs slow each other                              | 2× wall time, no saving                                                                            | check it/s in the first minute; fall back to running in sequence                      |
| Root-owned files left after `docker stop`                        | later local tools hit permission errors                                                            | the `trap … EXIT` in step 4; `find <dir> .rbyte_cache ! -user $USER` after each run   |

## 9. Out of scope (recorded so nobody drops it silently)

- Per-level codebook sizes: they break the uniform `(g, c)` head geometry (§2).
- More or fewer RVQ levels: not tested here. If the k-prefix ADE shows level 3
  adds almost nothing (k=3 ≈ k=4 within σ), note it as a follow-up for
  `num_quantizers=3`.
- The stale `config/_templates/dataset/yaak/action_{train,val}.yaml`: they use
  the pre-rbyte-0.39 `fields:` + proto-path schema, and the action tokenizer
  datamodule cannot build on the current rbyte. Unrelated to this task. The
  trajectory templates were fixed in `8c1eed62`.
- The VQ-BeT trajectory head in `PatchPolicy`: the next brief, built on this
  one's frozen tokenizer.
- **Train-set level-0 histogram (open item).** All histograms here come from
  the 5 val drives. Class weighting for the head should use train
  statistics, but instantiating the train datamodule in an ad-hoc script
  risks wiping the train sample store (§3.6). The head brief must solve this,
  for example by reading the train store read-only or logging histograms
  during training.

## 10. Outcome (2026-10-15)

Step 1 is done. The audit script is
`src/rmind/scripts/trajectory_tokenizer_audit.py`, its JSON output is in
`outputs/traj-audit/`, and the prefix-decode test is in
`tests/test_trajectory_tokenizer.py`.

**Checks:**

- The audit's full ADE on `v39` is 0.126 m, against W&B's 0.129.
- `v26` is epoch 26, at ADE 0.114 m.

**Findings (`model-403b4rrh:v26`, 27,676 val clips):**

- **Level 0 is not stuck.** The logged `perplexity/q0` is a per-batch
  average, and the val batches are drive-contiguous (not shuffled). That is
  where the flat 6.0 comes from. Over the whole val set the perplexity is
  8.6, and 12 codes each hold more than 1% of clips.
- **What level 0 encodes:**
  - 8 straight speed-bin codes hold ~94% of clips.
  - 4 turn codes: 5/6 (~±30° at ~4 m/s; 2.3% / 1.4%) and 11/0 (~−12°/+17°
    at ~10 m/s; ~1% each).
  - 3 codes (12, 7, 10) hold ~180° heading-glitch outliers: 6 clips in total.
  - Code 2 is unused.
- **Class quality:**
  - Level-0 class-mean ADE is 0.82 m (one global mean: 5.85 m), and levels
    0+1 give 0.25 m.
  - Prefix decode for k = 1..4 levels gives 0.84 / 0.28 / 0.16 / 0.11 m.
    The decoder handles partial sums well.
  - Level 3 is still worth keeping (0.16 → 0.11 m), so the §9
    `num_quantizers=3` follow-up is dropped.

**Decision:** freeze `model-403b4rrh:v26` as it is. Don't drop the glitch or
unused codes from the tokenizer: the glitch codes act as a garbage bin that
keeps the glitch clips out of the turn modes. Mask those codes in the
policy head instead (head brief §5 step 1).
