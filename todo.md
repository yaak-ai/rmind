- [ ] Fix Waypoint logger issue (?)
- [ ] Add turn_signal prediction
- [ ] Experiments:
```
Tier 0 — Implementation sanity (no full training, minutes)

- E0.1 Oracle fit: run flow_oracle.py on the synthetic oracle; confirm flow_mse and sample_l1 converge. This validates the decoder/sampler before any GPU finetune.
- E0.2 Sampler/NFE sweep on oracle: vary flow_sampling_method × flow_sampling_steps (1,2,4,8,16,32) and plot sample_l1 vs NFE. Tells you the accuracy/compute frontier independent of the real data.

Tier 1 — Fair head-to-head vs baseline (the core question)

- E1.1 Matched open-loop accuracy: Train both on the same backbone/data/steps. Evaluate per-channel L1 + RMSE on the first predicted future step (the only timestep both produce). Primary metric: steering RMSE (usually the hardest/safety-critical channel).
- E1.2 Controlled ablation of confounds: to attribute any gain to flow matching itself rather than the horizon/conditioning changes, add a baseline variant that predicts the same 6-step horizon with Gaussian heads and the same (waypoint-free) conditioning. If flow only wins because of the multi-step horizon or the prediction shift, this exposes it.

Tier 2 — Ablate the new design choices

Run as single-factor sweeps off the finetune.yaml defaults:
- E2.1 Time sampling: uniform vs logit-normal.
- E2.2 Sampler at inference: euler/midpoint/heun × steps {1,2,4,8,16} → val L1 vs inference latency (real-data version of E0.2; matters for on-vehicle compute budget).
- E2.3 Time conditioning: AdaLN vs the plain CrossAttentionDecoderBlock (both exist in decoder.py); and time_logit_scale 0 vs 0.25.
- E2.4 Capacity: dim_model 256 vs 384, num_layers 2 vs 4 — is any flow gain just from extra params? (compare against a param-matched baseline).
- E2.5 Channel weights: WeightedMSELoss [1,1,1] vs up-weighting steering.

Tier 3 — Distributional quality (the actual reason to use flow matching)

A Gaussian head is unimodal; flow matching's theoretical edge is capturing multimodal action distributions (e.g. lane choice at intersections, pass-left-vs-right). Open-loop L1 penalizes multimodality (regression-to-the-mean wins on MSE), so Tier 1 can understate flow's value.
- E3.1 Best-of-N / minADE: sample N=16 trajectories per condition, report best-of-N L1 and sample variance vs the Gaussian's analytic std. Flow should show real gains here if multimodality is being captured.
- E3.2 Mode-coverage on curated multimodal scenes: slice the val set to intersections/turns and compare sample spread + mode recovery.
- E3.3 Seed sensitivity: vary inference noise seed, measure sample_l1 variance — quantifies determinism cost of stochastic sampling for deployment.

Tier 4 — Closed-loop / downstream (decisive)

Open-loop accuracy is a weak proxy for driving quality (no compounding-error / covariate-shift signal). If a rerun-replay or sim closed-loop harness exists:
- E4.1 Closed-loop rollout: intervention rate, lateral/heading error over horizon, comfort (jerk), collision/off-road rate. This is where the multi-step horizon and multimodality should pay off — or not.

---
Suggested decision criterion

Ship the action expert only if it (a) matches or beats the horizon-matched Gaussian baseline (E1.2) on first-step steering RMSE, (b) shows a measurable best-of-N / multimodality gain (E3.1), and (c) is non-inferior closed-loop (E4.1) at an inference NFE you can afford on-vehicle (E2.2). If it only wins on E1.1 but ties E1.2, the win is the horizon/shift, not flow matching — and you could keep the cheaper deterministic head.
```