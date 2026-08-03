# eval_v0: driving-quality regression (patch_policy_eval, argmax protocol)

Question: did adding the max-speed token hurt driving? Checkpoints:
parent `model-ifuusvwq:v8` (dinov2_dinowm winner) vs Arm M
`model-1n0ih44y:v0` (epoch-1) and Arm MV `model-0nr1ydjm:v0` (epoch-1),
both warm-started from the parent.

Protocol: `rmind.scripts.patch_policy_eval` metrics (argmax columns are the
trustworthy ones per the standing protocol; sampled decoding is entangled
with entropy calibration). All three checkpoints evaluated on the IDENTICAL
val subset: the seed-1337 shuffled val stream, first 24 batches = **768
samples** (`caches/eval_v0_val_batches.pt` on aboutblank), scored via
`evaluate()` from the standing script (ppe_cached.py wrapper). Arm M
additionally has the full standing 200-batch / **6400-sample** run
(`ppe_armM.log`) as a subset-consistency cross-check.

Why the 768-sample fallback: concurrent datamodule instantiations corrupted
a pipefunc file_array pickle in the aboutblank `.rbyte_cache`
(`EOFError: Ran out of input` -> `ppe_armMV.log`), the known
concurrent-writer failure mode. The cached-batch path bypasses rbyte
entirely; the cache itself was built single-writer before the corruption.
A follow-up full-subset retry also died on the damaged store with a second
symptom (`TypeError: Unsupported dataframe type, got NoneType` from
`Dataset.from_config`, `ppe_armMV_manual.log`) -- the
`.rbyte_cache/yaak/train/*/samples` store needs a single-writer rebuild
before any full-subset rerun on aboutblank.

## Last-frame summary (deployment position, 768 samples)

| metric | parent ifuusvwq:v8 | armM 1n0ih44y:v0 | armMV 0nr1ydjm:v0 |
|---|---|---|---|
| top1_acc (marginal, mean over 4 quantizers) | 0.4040 | 0.4160 | **0.4189** |
| joint_acc (exact behavior token) | 0.0560 | 0.0638 | **0.0729** |
| recon_argmax (L1, normalized) | 0.0464 | 0.0443 | **0.0435** |
| recon_sampled (reference only) | 0.0469 | 0.0469 | 0.0470 |
| code_focal | 2.7493 | 2.1122 | 2.0722 |
| p_gt | 0.3814 | 0.3742 | 0.3774 |
| entropy (nats; uniform-16 = 2.77) | 0.6545 | 0.7947 | 0.7976 |
| offset L1 | 0.0089 | 0.0086 | 0.0086 |

All-frames (wandb-comparable): joint_acc 0.0547 / 0.0527 / 0.0601;
recon_argmax 0.0461 / 0.0456 / 0.0443 (parent / armM / armMV).

Arm M cross-check on 6400 samples (last frame): top1 0.4123, joint 0.0602,
recon_argmax 0.0470 -- within ~0.004 of the 768-sample values, so the small
subset is representative at the headline level (cluster tails are thin,
n>=36).

**Parent 6400-sample run** (landed after the first commit; the one manual
full-subset retry that survived the cache corruption --
`ppe_parent_6400_results.txt`), last frame: **top1 0.4153, joint 0.0616,
recon_argmax 0.0449**. This revises the 768-sample read: at 6400 samples
the parent is TIED with Arm M (joint 0.0616 vs 0.0602, recon 0.0449 vs
0.0470); the parent's own 768-vs-6400 joint delta (0.0560 -> 0.0616) shows
subset noise ~ +-0.006, the same order as the between-model gaps. Arm MV's
joint 0.0729 at 768 exceeds the parent on BOTH subsets, so its edge is
suggestive but unconfirmed (its full-subset attempts died on the corrupted
cache). Parent 6400 per-cluster argmax (gas/brake/steer): cruise
0.0785/0.0019/0.0068, braking 0.0235/0.1143/0.0392, highway
0.1538/0.0020/0.0034, braking_turn 0.0189/0.1312/0.1707.

## Per-cluster argmax L1 @ last frame (gas / brake / steer, 768 samples)

| cluster | n | parent | armM | armMV |
|---|---|---|---|---|
| cruise | 312 | 0.0785 / 0.0011 / 0.0073 | 0.0788 / 0.0008 / 0.0079 | 0.0747 / 0.0018 / 0.0074 |
| idle_coast | 149 | 0.0398 / 0.0457 / 0.0230 | 0.0410 / 0.0520 / 0.0244 | 0.0420 / 0.0504 / 0.0213 |
| braking | 84 | 0.0156 / 0.1223 / 0.0499 | 0.0087 / 0.1040 / 0.0435 | 0.0294 / 0.1117 / 0.0265 |
| highway | 55 | 0.1662 / 0.0059 / 0.0036 | 0.1787 / 0.0059 / 0.0039 | 0.1730 / 0.0059 / 0.0034 |
| acceleration | 49 | 0.0606 / 0.0124 / 0.0129 | 0.0589 / 0.0108 / 0.0215 | 0.0677 / 0.0118 / 0.0093 |
| cruise_turn | 43 | 0.0792 / 0.0070 / 0.0824 | 0.0830 / 0.0061 / 0.0829 | 0.0746 / 0.0050 / 0.0806 |
| gas_release | 40 | 0.0776 / 0.0006 / 0.0091 | 0.0800 / 0.0007 / 0.0108 | 0.0727 / 0.0007 / 0.0101 |
| braking_turn | 36 | 0.0293 / 0.1534 / 0.1955 | 0.0245 / 0.1259 / 0.1548 | 0.0293 / 0.1378 / 0.2058 |

## Read

- **No driving regression -- and, with the parent's 6400-sample run in, no
  clear gain either.** On the shared 768 subset both arms beat the parent
  (joint +0.8pt armM / +1.7pt armMV), but at 6400 samples parent and armM
  are tied (0.0616 vs 0.0602 joint) and the 768 subset's parent value
  (0.0560) sits ~0.006 below its 6400 value -- subset noise spans the gaps.
  Verdict: adding the token did NOT hurt driving; claims of improvement are
  not supported at this eval size. (The token itself is behaviorally inert
  at this stage, see override_probe_v0.md.)
- Cluster level: armM improves braking (brake L1 0.1040 vs 0.1223) and
  braking_turn; its highway gas L1 is slightly worse (0.1787 vs 0.1662).
  armMV improves braking-turn steer the least; differences at n=36-84 are
  noise-level -- re-check on the full subset at final checkpoints.
- Arms' entropy is higher than the parent's (0.79 vs 0.65 nats), so
  sampled-decoding numbers flatter the parent; per the standing protocol the
  argmax columns carry the verdict.
- Full 200-batch three-way rerun on the final checkpoints (single-writer,
  sequential) is the follow-up once the rbyte-cache samples store is
  rebuilt.
