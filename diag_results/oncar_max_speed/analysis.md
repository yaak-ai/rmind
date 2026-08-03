# On-car max-speed conditioning — did changing the input move gas/brake?

Drive 2026-08-03 (all times **UTC**), checkout `~/Code/drivr-max-speed-input` @ `2fd3760` on
`delta-emc1` (reachable only via `ssh -J max@172.30.0.62 max@172.30.0.40`).
Script: `analyze_oncar_max_speed.py`; raw data: `raw/`; console output: `analysis_output.txt`.

## Answer

**No effect measurable — and the drive as executed could not have shown one.** Two
independent, hard reasons:

1. **No before/after pair exists.** Exactly one max-speed change is recoverable in the whole
   day: `max_speed = 10.0` at **13:49:44.736**, landing in the process that wrote
   `20260803_134900`. That process's first prediction tick is **13:50:03.257 — 18.5 s after
   the change** (AI policy was off at 13:49:44, enabled at 13:49:50). So that session has
   **0 pre-change ticks vs 153 post-change ticks**. Every other session lies outside the
   change window and its own setting is unrecoverable.
2. **Even a perfect A/B here was ~63x underpowered.** Best available contrast resolves
   Δgas >= **0.0251** (α=.05, power=.80, unit = plan). The offline probe on the *same
   checkpoint* predicts Δgas = **+0.0004** for the `10` token.

The run confirms the *plumbing* (the UI POST reached the app; the value is read every step)
but neither confirms nor contradicts the offline finding.

## Why the setting was nearly unrecoverable

`max_speed` was in **no** persistent artifact: not in `*_predictions.jsonl` keys, not in the
`<ts>.json` event log, not in `*_vehicle.jsonl` (CAN feedback only), not in the `.rrd`
(`rerun_viz.py` logs camera/gnss/signals/timing/ego only), and structlog is stdout-only with
no file handler. Recovered from a live **zellij pane scrollback** (capped ~10k lines, flooded
by 8022 `MCM error status` lines) covering only **13:49:40–13:52:06**. Dumps are committed as
`raw/zellij_pane_t*.txt` — **they exist nowhere else.**

Caveats: drivr logs the POST unconditionally, so the line proves the value was *set to* 10.0,
not that it *differed* from the prior state. And `10` is a distinct vocabulary class — **not**
WALK/Schrittgeschwindigkeit (`5`).

→ Fixed for future drives: `max_speed` + `max_speed_binding` are now logged per prediction row
(drivr `feat/max-speed-input` @ `bf26fc4`).

## Plumbing verdict

**The binding was never verified for either analysed session** — `max_speed_binding=True` and
the mismatch warnings are emitted at engine *load*, before the scrollback window begins.

- Only one launch recovered verbatim (10:28, from a `ps aux` snapshot):
  `--model .../patch_policy_dinov2_dinowm_maxspeed_aux-0nr1ydjm-v1.trt --image-norm unit
  --max-speed 100`. That process was `kill -9`'d at 10:29 → **zero data**.
- `134521` and `134900` are provably different processes (plan configs `n6s0` vs `n5s1`, which
  is CLI-only), so their `--max-speed` may have differed too.
- **Circumstantial fingerprint:** all seven max-speed-checkout sessions cluster at median
  `inference_ms` **265–276 ms**, including the confirmed-maxspeed 10:29 relaunch (271.6) and
  `134900` (276.0). Known controls differ: `ilch4dyo-v2.fp16strict` **171 ms**,
  `dinowm_big-zt39kjn4-v1.fp16strict` **297–307 ms**. Consistent with all sessions running the
  same non-fp16strict `maxspeed_aux-0nr1ydjm-v1.trt` (binding probably active) — timing, not a
  load log.

## Timeline

`plans` = fresh-plan rows (`engine` ends `*`); ticks within a plan are steps of one forward
pass and are **not** independent. `speed` = achieved km/h from `_vehicle.jsonl` (<=0.25 s join).

| session | window UTC | ticks | plans | plancfg | gas mean | gas med | brake mean | brake med | speed mean | inf_ms | max_speed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 095728 | 09:58:04–10:07:57 | 944 | 161 | n6s0 | 0.0263 | 0.0041 | 0.0289 | 0.0004 | 7.35 | 267 | unrecoverable |
| 101342 | 10:16:07–10:23:00 | 669 | 114 | n6s0 | 0.0158 | 0.0012 | 0.0189 | 0.0004 | 4.42 | 265 | unrecoverable |
| 102818 | — (killed 10:29) | 0 | 0 | — | — | — | — | — | — | — | **`--max-speed 100` (verified, no data)** |
| 102952 | 10:30:35–10:34:22 | 410 | 70 | n6s0 | 0.0250 | 0.0054 | 0.0232 | 0.0003 | 7.88 | 272 | unrecoverable |
| 103557 | 10:36:20–10:57:59 | 1841 | 313 | n6s0 | 0.0394 | 0.0110 | 0.0327 | 0.0001 | 10.38 | 267 | unrecoverable |
| 105853 | 10:59:40–11:03:34 | 238 | 45 | n6s0 | 0.0868 | 0.0918 | 0.0121 | 0.0000 | 13.42 | 265 | unrecoverable |
| 134521 | 13:45:49–13:47:35 | 193 | 33 | n6s0 | 0.0239 | 0.0031 | 0.0433 | 0.0018 | 10.05 | 268 | unrecoverable |
| 134900 | 13:50:03–13:51:56 | 153 | 34 | n5s1 | 0.0292 | 0.0067 | 0.0158 | 0.0000 | 9.68 | 276 | **10 km/h (verified)** |

Speed envelope: mean 4–13 km/h, **max 43 km/h across all sessions** — the standing offline
go/no-go probe (30-vs-100 gas contrast) lives at **>50 km/h**, a regime this drive never entered.

## Speed-binned comparison (best available, heavily confounded)

`134521` (setting unknown) vs `134900` (10 km/h, verified), 2.5 min apart. `pl` = plans/cell;
MDE uses the plan as unit.

| speed bin | nA | nB | plA | plB | gasA | gasB | Δgas | MDE_gas | brakeA | brakeB | Δbrake | MDE_brake |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0–5 | 35 | 27 | 8 | 6 | 0.0162 | 0.0183 | +0.0021 | 0.0407 | 0.1030 | 0.0183 | −0.0847 | 0.1069 |
| 5–10 | 67 | 55 | 11 | 12 | 0.0205 | 0.0222 | +0.0017 | 0.0443 | 0.0356 | 0.0075 | −0.0281 | 0.0501 |
| 10–15 | 39 | 31 | 8 | 7 | 0.0474 | 0.0560 | +0.0085 | 0.0607 | 0.0370 | 0.0001 | −0.0369 | 0.0697 |
| 15–20 | 52 | 40 | 8 | 10 | 0.0159 | 0.0252 | +0.0093 | 0.0344 | 0.0176 | 0.0378 | +0.0202 | 0.0559 |
| 20–30 / 30+ | 0 | 0 | — | — | — | — | — | — | — | — | — | — |

**Every cell is underpowered (6–12 plans/arm); every delta — gas and brake — sits inside its
own cell's MDE.** The eye-catching −0.0847 does not clear its 0.1069 resolution.

Pooled, 95 % CI from a plan-block bootstrap (5000 resamples, seed 1337):

| session | gas | brake | plans |
|---|---|---|---|
| 134521 | 0.0239 [0.0125, 0.0382] | 0.0433 [0.0213, 0.0686] | 33 |
| 134900 | 0.0292 [0.0190, 0.0400] | 0.0158 [0.0079, 0.0256] | 34 |

Pooled MDE: Δgas **0.0251**, Δbrake **0.0446**.

Part of the pooled brake gap is **mechanical, not conditioning**: `134900` (`n5s1`) skips step
1/6 of every plan and `134521`'s brake is front-loaded within the plan (brake 0.051/0.061 at
steps 1–2 vs 0.022–0.027 at steps 5–6). And `134521`'s value is unknown, so no *sign* can be
attributed. Treat the table as a bound on effect size, not a measurement.

## Cross-check against the offline finding

Offline probes on the same checkpoint (`model-0nr1ydjm:v1` = armMV epoch-2 = the deployed
engine), 768 val samples, argmax (`../eval_v0/override_probe_mv_v1.md`):

| condition | offline Δgas | on-car resolution gap | resolvable? |
|---|---|---|---|
| WALK (5) | +0.0167 | 2x | no |
| 30 | +0.0008 | 31x | no |
| UNLIMITED | +0.0005 | 50x | no |
| **10 (value actually used)** | **+0.0004** | **63x** | **no** |
| 100 | +0.0001 | 251x | no |

The "WALK gives a large wrongly-signed +gas response" prediction does **not** apply — the value
used was `10`, a different token, essentially inert offline (KL 0.0004, 0.75 % code flips).
**The on-car data is consistent with the offline finding but cannot test it.**

## To make the next drive answer the question

1. **DONE** — `max_speed` + `max_speed_binding` now logged per prediction row (`bf26fc4`);
   pull that onto the car before the next drive. (Still worth adding: a `max_speed_changed`
   event + a `map/max_speed` rerun scalar, and `wants_max_speed` into the event log at load.)
2. **Toggle while AI is engaged, repeatedly** — A/B/A/B every 20–30 s on the same loop, >=10
   alternations. Converts a hopeless between-session comparison (different processes, plan
   configs, roads) into a within-episode crossover that differences out road context.
3. **Use the extreme pair and reach the relevant regime** — `5 (WALK)` vs `100`, not `10` vs
   `100`. WALK is the only pair with a nonzero offline prediction. Resolving +0.0167 needs
   ~10x more plans per arm (~10–15 min engaged per arm); +0.0004 is out of on-car reach at any
   realistic drive length.

## Runbook

```bash
# reproduce offline
cd diag_results/oncar_max_speed && python3 analyze_oncar_max_speed.py

# re-pull from the car (relay required; not directly routable)
CAR="ssh -J max@172.30.0.62 max@172.30.0.40"
$CAR 'ls ~/Code/drivr-max-speed-input/reports/'
# only source of the setting on pre-bf26fc4 drives — dump every pane and grep:
$CAR 'for s in $(zellij list-sessions -n); do for i in $(seq 1 25); do \
        zellij --session $s action dump-screen --pane-id terminal_$i --full \
          --path /tmp/zdump/${s}_t$i.txt; done; done; \
      grep -l "Zone max-speed" /tmp/zdump/*.txt'
```
