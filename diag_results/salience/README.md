# Max-speed token salience — is the conditioning signal audible in the trunk?

Measured on the FINAL checkpoints (`model-1n0ih44y:v4` Arm M, `model-0nr1ydjm:v4`
Arm MV, both epoch 5) over 6 real val batches. Script: `embed_norms.py`, raw output:
`embed_norms_v4.txt`.

Motivation: the override probes showed the token is plumbed correctly but behaviourally
inert (`../eval_v0/`), and CFG could not amplify it (`../eval_v0/cfg_sweep_mv_v1.md`).
The untested explanation was *salience* — the token being numerically negligible in the
residual stream rather than heard-and-ignored.

## Result

| quantity                            | Arm M v4   | Arm MV v4  |
| ----------------------------------- | ---------- | ---------- |
| max_speed embedding row norm (mean) | 0.159      | 0.239      |
| speed embedding row norm (mean)     | 0.759      | 0.760      |
| **live patch-token norm**           | **15.98**  | **15.99**  |
| live speed-token norm               | 0.791      | 0.792      |
| live max_speed-token norm           | 0.173      | 0.239      |
| **ratio max_speed / patch**         | **0.0108** | **0.0149** |
| ratio max_speed / speed             | 0.219      | 0.301      |
| share of frame-block norm mass      | 4.2e-5     | 5.8e-5     |
| pairwise class-row distance (mean)  | 0.196      | 0.319      |

## Reading

**The token is ~92x (M) / ~67x (MV) quieter than a single patch token, and there are 256
patch tokens per frame — so it carries ~0.004-0.006 % of the frame block's norm mass.**
It is not zero (the warm-start zero-init did move), and the 13 class rows ARE mutually
distinguishable (pairwise distance ~0.20/0.32, comparable to their own norms, so classes
are not collapsed) — consistent with the probe finding that overrides produce distinct but
tiny logit changes.

The aux head has a visible effect: Arm MV's rows are **1.5x louder** and **1.6x better
separated** than Arm M's, which is exactly the ordering of their measured override
sensitivity (MV ~100x Arm M). So the aux gradient *is* the mechanism that grows this
embedding — it is simply nowhere near enough on its own.

Note the speed token, which the policy demonstrably uses, sits at 0.79 — **4.6x louder**
than max_speed on Arm M. The conditioning input is quiet even relative to the one scalar
input we know works.

## Consequence for the next round

This is a salience problem *at the input*, not (only) a head-attention problem, so the
two candidate fixes are now separable and both are cheap:

1. **Scale / normalize the token into the residual stream** (e.g. LayerNorm the embedding
   or learn a gain, as `fusion_norm` does for the patch/goal concat — which does NOT apply
   here, since max_speed enters as its own sequence token, not through `patch_projection`).
1. **Inject at the action head** (concat/FiLM on the readout feature) so the signal does
   not have to survive attention routing among 1548 tokens at 1e-4 relative amplitude.

Neither addresses *why* BC never grew the embedding — the token has near-zero marginal
information given vision + ego speed (see `../eval_v0/synthesis_v0.md`). Expect to need
an objective change (informativeness-weighted loss / future-speed-profile aux) alongside
whichever injection change is chosen.
