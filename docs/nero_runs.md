# nero bimanual runs: how to reproduce each one

Every bimanual NERO training run (tokenizers and patch policies, 2026-10-07 to
2026-10-10) has a committed experiment named after the run. One command
reproduces it, with no CLI overrides:

```sh
just train experiment=yaak/nero_robot/<run>
```

`just train` renders the ytt templates (`just generate-config`), refuses a dirty
tree (`just check-git`) and runs `rmind-train`. The experiment pins the corpus,
split, stats fit, frame cache and tokenizer checkpoint. Only the machine-local
data roots below may change, and their defaults are the NAS copies every host
mounts.

## Environment

| variable                | default                              | holds                                                           |
| ----------------------- | ------------------------------------ | --------------------------------------------------------------- |
| `NERO_DATA_ROOT`        | `/nasa/max/nero-cache/cube-bimanual` | takes, frame caches, stats (layout below)                       |
| `NERO_FRAME_CACHE_ROOT` | `$NERO_DATA_ROOT`                    | frame caches only, e.g. a `/dev/shm` copy                       |
| `NERO_STATS_ROOT`       | `$NERO_DATA_ROOT`                    | stats fits only                                                 |
| `NERO_CKPT_ROOT`        | `/nasa/max/nero-sweep-ckpts`         | tokenizer checkpoints (`<run>/.../checkpoints/*.ckpt`)          |
| `NERO_RUNS_DIR`         | `runs`                               | output: CSV logs + checkpoints (`<dir>/<run_name>/version_<n>`) |

The run experiments do NOT read `NERO_ROBOT_DIR`, `NERO_FRAME_CACHE`,
`NERO_STATS_DIR` or `NERO_TOKENIZER_CKPT`. The old launch scripts exported those,
so a stale shell could otherwise swap a run's corpus, stats or tokenizer without
any error. The generic bases (`bimanual_causal`, `bimanual_hand_off`,
`bimanual_w1*`, `causal`, ...) still read them, as before.

Layout under a data root. NAS is the default; renate's `~/paper/data/cube-bimanual`
has the same layout.

```
2026-10-07/<take>/{data.mcap,outcome.json,base,side_left,side_right.mp4}   v1 corpus, 85 takes
takes-v3/<take> -> /nasa/drives/nero-arms/cube-bimanual/<day>/<take>        v3 corpus, 130 takes
frame-cache-140x224/2026-10-07          frame-cache-224x224-stretch/{2026-10-07,v3}
frame-cache-416x640/2026-10-07 (*)      frame-cache-210x336/2026-10-07 (*)
stats_v1   stats_c10   stats_c10_v3
```

(\*) Not built yet. The runs that read these caches used `/dev/shm` copies that
no longer exist (see "Not reproducible as-is").

## How it is wired

- **Data profiles** (`config/_templates/nero_data/nero_data.lib.yml`, rendered
  to `config/nero_data/<profile>.yaml`). A profile is one corpus, its split, one
  stats fit and optionally one frame-cache grid. It sets `nero_robot_dir`,
  `nero_stats_dir`, `nero_frame_cache` and, for v3, `nero_split_file`, all under
  the roots above. A run selects one with `- /nero_data@_global_: <profile>`.
  To re-point a run from the CLI, use `nero_data@_global_=<profile>`.

  | profile            | takes        | split | stats          | frame cache                              |
  | ------------------ | ------------ | ----- | -------------- | ---------------------------------------- |
  | `cube_v1`          | `2026-10-07` | v1    | `stats_v1`     | -                                        |
  | `cube_v1_140x224`  | `2026-10-07` | v1    | `stats_v1`     | `frame-cache-140x224/2026-10-07`         |
  | `cube_v1_210x336`  | `2026-10-07` | v1    | `stats_v1`     | `frame-cache-210x336/2026-10-07` (\*)    |
  | `cube_v1_416x640`  | `2026-10-07` | v1    | `stats_v1`     | `frame-cache-416x640/2026-10-07` (\*)    |
  | `cube_v1_c10`      | `2026-10-07` | v1    | `stats_c10`    | -                                        |
  | `cube_v1_c10_224s` | `2026-10-07` | v1    | `stats_c10`    | `frame-cache-224x224-stretch/2026-10-07` |
  | `cube_v3_c10`      | `takes-v3`   | v3    | `stats_c10_v3` | -                                        |
  | `cube_v3_c10_224s` | `takes-v3`   | v3    | `stats_c10_v3` | `frame-cache-224x224-stretch/v3`         |

  The split itself is unchanged: the take lists still come from
  `robot_bimanual_split{,_v3}.lib.yml` through the datamodule. For v3, the
  profile also sets `nero_split_file: config/splits/nero_cube_bimanual_v3.json`.
  The path is relative to the repo root, which is where `just train` runs. Only
  the arm-selection callback reads it; v1 uses the callback's default, the v1
  JSON.

- **Tokenizer pins** (`config/_templates/nero_tokenizer/nero_tokenizer.lib.yml`,
  rendered to `config/nero_tokenizer/<pin>.yaml`). Each pin sets `tokenizer_ckpt`
  (under `NERO_CKPT_ROOT`) and `tokenizer_ckpt_sha256`. `rmind-train` hashes the
  file before it builds the model and refuses a mismatch.

  | pin                         | file under `NERO_CKPT_ROOT`                                                | sha256                                                             | produced by                                                                       |
  | --------------------------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------ | --------------------------------------------------------------------------------- |
  | `bimanual_tokenizer_v1`     | `bimanual_tokenizer/version_1/checkpoints/epoch=19-step=1320.ckpt`         | `9b5257364a3406f33daa742e27927e1272553e1535e6153e87aa3be7b7ad05b5` | `bimanual_tokenizer`                                                              |
  | `bimanual_tokenizer_v2`     | `bimanual_tokenizer_v2/version_0/checkpoints/epoch=149-step=9900.ckpt`     | `1ba2f3b2c10891c6bec3b77ab73a544257afc0eefd018da8baeeb897e32a93f2` | `bimanual_tokenizer_v2`                                                           |
  | `bimanual_tokenizer_v2_q32` | `bimanual_tokenizer_v2_q32/version_0/checkpoints/epoch=149-step=9900.ckpt` | `a1ec2a431a1f89a3bd9502e702a481a1e8051a8fa7a95dc288f2d824c7b8aa75` | `bimanual_tokenizer_v2_q32`                                                       |
  | `paper_tok_c10_e399`        | `paper_tok_c10/version_0/checkpoints/epoch=399-step=32400.ckpt`            | `b33dc8e51e4f9f84f1f1365395a23e9b206fcc1ef7d38feef087c1c9c5808e41` | `paper_tok_c10`, final epoch (launched as `~/paper/tok/paper_tok_c10_final.ckpt`) |
  | `paper_tok_c10_e89`         | `paper_tok_c10/version_0/checkpoints/epoch=89-step=7290.ckpt`              | `af48270187ba35dcb74b521316074cf8567b23cfabbea7f5e2cb0800ca309007` | `paper_tok_c10` (launched as `~/paper/tok_c10_for_l1.ckpt`)                       |
  | `paper2_tok_cb64_e999`      | `paper2_tok_cb64/checkpoints/epoch=999-step=150000.ckpt`                   | `60eb44496305740a5c72561aae1ea1b77f18c9a426b2573d7dd46def15888ccd` | `paper2_tok_cb64` (launched as `~/paper2/tok_cb64_e999.ckpt`)                     |
  | `paper2_tok_cb64_e9`        | `paper2_tok_cb64/checkpoints/epoch=9-step=1500.ckpt`                       | `24848d226ec8a6fb0771d6a0069565c150e79462ba11a0c406b835b11cdc8d45` | `paper2_tok_cb64` (launched as `~/paper2/tok_v3_for_l1.ckpt`)                     |
  | `paper2_tok_q4_e999`        | `paper2_tok_q4/checkpoints/epoch=999-step=150000.ckpt`                     | `3b7249d71fc753c0ccefc72a64cec40ee7d1650a6e0d26914265d2c616ab3aec` | `paper2_tok_q4` (launched as `~/paper2/tok_q4_e999.ckpt`)                         |

  The l1-head runs `paper_pp_w2_c10_l1` and `paper2_pp_l1_hand{on,off}` load an
  early epoch of a tokenizer that was still training at launch. The l1 head only
  uses the tokenizer's chunk-10 action standardizer and horizon; it never
  computes the codes.

- **Seed:** `train.yaml` lists `_self_` after `experiment`, so an experiment
  cannot set `seed` itself. It sets `experiment_seed` instead, which
  `seed: ${oc.select:experiment_seed,1337}` picks up (`l1_codes_w1_drop_seed2`).
  A CLI `seed=` still overrides both.

## Runs

The run name, the experiment name and the `just train` argument are the same:
`just train experiment=yaak/nero_robot/<run>`. "Code" is the tree the run
trained with. `just train` on this branch uses the head code, which carries
every later feature as opt-in config (see "Verification"). For a bit-exact code
replay, check out that commit and keep the experiment from this branch.
Checkpoints marked `NAS:` are under `/nasa/max/nero-sweep-ckpts/`.

| run                         | host / code                                                                        | data profile       | tokenizer (pin, sha256)                    | wandb (yaak/rmind)                                    | checkpoints                                                | status   |
| --------------------------- | ---------------------------------------------------------------------------------- | ------------------ | ------------------------------------------ | ----------------------------------------------------- | ---------------------------------------------------------- | -------- |
| `bimanual_tokenizer`        | sisyphos / uncommitted tree on bdaaa57a == 3ed679f8 (committed 2 min after launch) | `cube_v1`          | -                                          | CSV only                                              | `NAS:bimanual_tokenizer/version_1/checkpoints`             | finished |
| `bimanual_tokenizer_e300`   | sisyphos / 745c5853                                                                | `cube_v1`          | -                                          | CSV only                                              | `NAS:bimanual_tokenizer_e300/version_0/checkpoints`        | finished |
| `bimanual_tokenizer_v2`     | sisyphos / 745c5853 + the v2 yaml (byte-identical to 348c62b3)                     | `cube_v1`          | -                                          | CSV only                                              | `NAS:bimanual_tokenizer_v2/version_0/checkpoints`          | finished |
| `bimanual_tokenizer_v2_q32` | sisyphos / 348c62b3 + the q32 yaml (== 31adcc21)                                   | `cube_v1`          | -                                          | CSV only                                              | `NAS:bimanual_tokenizer_v2_q32/version_0/checkpoints`      | finished |
| `bimanual_tokenizer_v2_ew2` | sisyphos / 93a40a07                                                                | `cube_v1`          | -                                          | CSV only                                              | `NAS:bimanual_tokenizer_v2_ew2/version_0/checkpoints`      | finished |
| `bimanual_hand_off_v2`      | sisyphos / b069267e + save_top_k -1 (== c7333fc2's callbacks)                      | `cube_v1_140x224`  | bimanual_tokenizer_v1 (`9b5257364a34`)     | CSV only                                              | `NAS:bimanual_hand_off_v2/version_0/checkpoints`           | finished |
| `bimanual_hand_on_v2`       | sisyphos / b069267e + save_top_k -1 (== c7333fc2's callbacks)                      | `cube_v1_140x224`  | bimanual_tokenizer_v1 (`9b5257364a34`)     | CSV only                                              | `NAS:bimanual_hand_on_v2/version_0/checkpoints`            | finished |
| `bimanual_hand_on_tokv2`    | sisyphos / 63db0f6e                                                                | `cube_v1_140x224`  | bimanual_tokenizer_v2 (`1ba2f3b2c108`)     | [9rhho3dk](https://wandb.ai/yaak/rmind/runs/9rhho3dk) | `NAS:bimanual_hand_on_tokv2/version_0/checkpoints`         | finished |
| `bimanual_hand_on_tokq32`   | sisyphos / 63db0f6e                                                                | `cube_v1_140x224`  | bimanual_tokenizer_v2_q32 (`a1ec2a431a1f`) | [13u5znu3](https://wandb.ai/yaak/rmind/runs/13u5znu3) | `NAS:bimanual_hand_on_tokq32/version_0/checkpoints`        | finished |
| `bimanual_hand_off_tokv2`   | sisyphos / 63db0f6e                                                                | `cube_v1_140x224`  | bimanual_tokenizer_v2 (`1ba2f3b2c108`)     | [quzkrzn3](https://wandb.ai/yaak/rmind/runs/quzkrzn3) | `NAS:bimanual_hand_off_tokv2/version_0/checkpoints`        | finished |
| `bimanual_hand_off_tokq32`  | sisyphos / 63db0f6e                                                                | `cube_v1_140x224`  | bimanual_tokenizer_v2_q32 (`a1ec2a431a1f`) | [13947qag](https://wandb.ai/yaak/rmind/runs/13947qag) | `NAS:bimanual_hand_off_tokq32/version_0/checkpoints`       | finished |
| `sweep_w1_off`              | sisyphos / 9450f679                                                                | `cube_v1_140x224`  | bimanual_tokenizer_v2 (`1ba2f3b2c108`)     | [8rohcto5](https://wandb.ai/yaak/rmind/runs/8rohcto5) | `NAS:sweep_w1_off/checkpoints`                             | finished |
| `sweep_w1_off_drop`         | sisyphos / 9450f679                                                                | `cube_v1_140x224`  | bimanual_tokenizer_v2 (`1ba2f3b2c108`)     | [44pn9l5u](https://wandb.ai/yaak/rmind/runs/44pn9l5u) | `NAS:sweep_w1_off_drop/checkpoints`                        | finished |
| `sweep_w1_off_dinoft`       | sisyphos / a9f83e69                                                                | `cube_v1_140x224`  | bimanual_tokenizer_v2 (`1ba2f3b2c108`)     | [bt3ot95f](https://wandb.ai/yaak/rmind/runs/bt3ot95f) | `NAS:sweep_w1_off_dinoft/checkpoints`                      | finished |
| `sweep_w1_off_r18`          | sisyphos / a9f83e69                                                                | `cube_v1_416x640`  | bimanual_tokenizer_v2 (`1ba2f3b2c108`)     | [slrc7svt](https://wandb.ai/yaak/rmind/runs/slrc7svt) | `NAS:sweep_w1_off_r18/checkpoints`                         | finished |
| `sweep_w1_off_dinoft336`    | sisyphos / a9f83e69                                                                | `cube_v1_210x336`  | bimanual_tokenizer_v2 (`1ba2f3b2c108`)     | [llukfyrt](https://wandb.ai/yaak/rmind/runs/llukfyrt) | `NAS:sweep_w1_off_dinoft336/checkpoints`                   | finished |
| `sweep_w1_on_drop`          | sisyphos / a9f83e69                                                                | `cube_v1_140x224`  | bimanual_tokenizer_v2 (`1ba2f3b2c108`)     | [9lolrfwm](https://wandb.ai/yaak/rmind/runs/9lolrfwm) | `NAS:sweep_w1_on_drop/checkpoints`                         | finished |
| `l1_codes_w1_drop_seed2`    | sisyphos / a9f83e69                                                                | `cube_v1_140x224`  | bimanual_tokenizer_v2 (`1ba2f3b2c108`)     | [sz8emvxb](https://wandb.ai/yaak/rmind/runs/sz8emvxb) | `NAS:l1_codes_w1_drop_seed2/checkpoints`                   | finished |
| `l1_w1_drop`                | sisyphos / 554f7fde                                                                | `cube_v1_140x224`  | bimanual_tokenizer_v2 (`1ba2f3b2c108`)     | [jc1kpc0v](https://wandb.ai/yaak/rmind/runs/jc1kpc0v) | `NAS:l1_w1_drop/checkpoints`                               | finished |
| `l1_w1_drop_pad`            | sisyphos / 554f7fde                                                                | `cube_v1_140x224`  | bimanual_tokenizer_v2 (`1ba2f3b2c108`)     | [313jc4pn](https://wandb.ai/yaak/rmind/runs/313jc4pn) | `NAS:l1_w1_drop_pad/checkpoints`                           | finished |
| `codes_w1_drop_pad`         | sisyphos / 554f7fde                                                                | `cube_v1_140x224`  | bimanual_tokenizer_v2 (`1ba2f3b2c108`)     | [2yzacpah](https://wandb.ai/yaak/rmind/runs/2yzacpah) | `NAS:codes_w1_drop_pad/checkpoints`                        | finished |
| `l1_w1_r18_pad`             | tresor / 554f7fde (rsynced tree)                                                   | `cube_v1_416x640`  | bimanual_tokenizer_v2 (`1ba2f3b2c108`)     | [vtpxwv31](https://wandb.ai/yaak/rmind/runs/vtpxwv31) | `NAS:l1_w1_r18_pad/checkpoints`                            | finished |
| `paper_tok_c10`             | renate / snapshot 617c1de of 47d6a7a7                                              | `cube_v1_c10`      | -                                          | [hnaplcpg](https://wandb.ai/yaak/rmind/runs/hnaplcpg) | `NAS:paper_tok_c10/version_0/checkpoints`                  | finished |
| `paper_tok_c10_q4`          | renate / snapshot 617c1de of 47d6a7a7                                              | `cube_v1_c10`      | -                                          | [tw95pidb](https://wandb.ai/yaak/rmind/runs/tw95pidb) | `NAS:paper_tok_c10_q4/version_0/checkpoints`               | finished |
| `paper_pp_w2_c10_codes`     | renate / snapshot 1b2167e (47d6a7a7 + lr_total_steps 19964 == 037e21ad)            | `cube_v1_c10_224s` | paper_tok_c10_e399 (`b33dc8e51e4f`)        | [zf5smc1c](https://wandb.ai/yaak/rmind/runs/zf5smc1c) | `NAS:paper_pp_w2_c10_codes/checkpoints`                    | finished |
| `paper_pp_w2_c10_l1`        | renate / snapshot 1b2167e (47d6a7a7 + lr_total_steps 19964 == 037e21ad)            | `cube_v1_c10_224s` | paper_tok_c10_e89 (`af48270187ba`)         | [ehz4fyk9](https://wandb.ai/yaak/rmind/runs/ehz4fyk9) | `NAS:paper_pp_w2_c10_l1/checkpoints`                       | finished |
| `paper2_tok_cb64`           | sisyphos / snapshot ad3e5ee (== 257321e6)                                          | `cube_v3_c10`      | -                                          | [bgqnkaua](https://wandb.ai/yaak/rmind/runs/bgqnkaua) | `NAS:paper2_tok_cb64/checkpoints`                          | finished |
| `paper2_tok_cb256`          | sisyphos / snapshot ad3e5ee (== 257321e6)                                          | `cube_v3_c10`      | -                                          | [cp0bg0s6](https://wandb.ai/yaak/rmind/runs/cp0bg0s6) | `NAS:paper2_tok_cb256/checkpoints`                         | finished |
| `paper2_tok_split`          | sisyphos / snapshot ad3e5ee (== 257321e6)                                          | `cube_v3_c10`      | -                                          | [amrozbzs](https://wandb.ai/yaak/rmind/runs/amrozbzs) | `NAS:paper2_tok_split/checkpoints`                         | finished |
| `paper2_tok_q4`             | renate / snapshot a5ea41f (== a9790ebc)                                            | `cube_v3_c10`      | -                                          | [fecmqhwy](https://wandb.ai/yaak/rmind/runs/fecmqhwy) | `NAS:paper2_tok_q4/checkpoints`                            | finished |
| `paper2_pp_l1_handon`       | sisyphos / snapshot ad3e5ee (== 257321e6)                                          | `cube_v3_c10_224s` | paper2_tok_cb64_e9 (`24848d226ec8`)        | [d2p6bu78](https://wandb.ai/yaak/rmind/runs/d2p6bu78) | `/home/max/paper2/runs/paper2_pp_l1_handon (sisyphos)`     | running  |
| `paper2_pp_codes_cond`      | renate / snapshot a5ea41f (== a9790ebc)                                            | `cube_v3_c10_224s` | paper2_tok_cb64_e999 (`60eb44496305`)      | [tpn8a9c4](https://wandb.ai/yaak/rmind/runs/tpn8a9c4) | `/home/max/paper2/runs/paper2_pp_codes_cond (renate)`      | running  |
| `paper2_pp_codes_nocond`    | tresor / snapshot 715a629 (== a9790ebc)                                            | `cube_v3_c10_224s` | paper2_tok_cb64_e999 (`60eb44496305`)      | [qppislok](https://wandb.ai/yaak/rmind/runs/qppislok) | `/home/max/paper2/runs/paper2_pp_codes_nocond (tresor)`    | running  |
| `paper2_pp_l1_handoff`      | sisyphos / snapshot 4a7a79e (== 04fc11e1)                                          | `cube_v3_c10_224s` | paper2_tok_cb64_e9 (`24848d226ec8`)        | -                                                     | `/home/max/paper2/runs/paper2_pp_l1_handoff (sisyphos)`    | queued   |
| `paper2_pp_codes_q4_nocond` | renate / snapshot feadc1a (== 04fc11e1)                                            | `cube_v3_c10_224s` | paper2_tok_q4_e999 (`3b7249d71fc7`)        | -                                                     | `/home/max/paper2/runs/paper2_pp_codes_q4_nocond (renate)` | queued   |

ACT runs on the same corpora are reproduced in nutron-cli through the
`ops/kit.toml` presets (`ORCH_TRAIN_RUN=<preset> just train`; nutron-cli
`runtime/training/RUNBOOK.md`).

## Verification (2026-10-10)

For every run, three configs were composed (Hydra compose, no training) and
fully resolved:

- **recorded**: what the run actually used. This is the run's own
  `.hydra/config.yaml`, resolved under the launch script's environment. For the
  six runs whose hydra output dir is gone (`paper2_tok_{cb64,cb256,split,q4}`,
  `paper2_pp_l1_handon`, `paper2_pp_codes_cond`), it is the wandb run config,
  which `rmind-train` logs fully resolved. Where both exist (20 runs) they agree
  on every key. That check validates the reconstructed launch environments.
- **base**: `04fc11e1` (patch/paper before this change) plus the recorded
  experiment, CLI overrides and environment.
- **new**: `experiment=yaak/nero_robot/<run>` on this branch, no CLI overrides,
  only `HOME` set. The legacy `NERO_ROBOT_DIR` / `NERO_FRAME_CACHE` /
  `NERO_STATS_DIR` / `NERO_TOKENIZER_CKPT` were set to junk to prove they are
  ignored.

Results:

- **new vs base: 0 differences in any of the 35 runs**, other than the expected
  root and path keys.
- **new vs recorded**: only the following differences, and none of them changes
  numerics:
  - **Paths.** Some paths name different roots holding identical bytes:
    - v1 and v3 takes: sha256 of every file (v1), or of `data.mcap` and
      `outcome.json` plus the mp4 sizes (v3), on sisyphos, renate, tresor and NAS.
    - Stats dirs: sha256 of the files.
    - Frame caches: sha256 of the manifests plus the `.npy` sizes. The caches
      also check their own mp4 fingerprints at load.
    - Tokenizer checkpoints: sha256 of the file.
    - The v3 split file: byte copy.
  - **New keys**, cosmetic: `nero_data_root`, `nero_stats_root`,
    `nero_frame_cache_root`, `nero_ckpt_root`, `tokenizer_ckpt_sha256`,
    `experiment_seed`. The output dir (`nero_runs_dir`, the logger `save_dir`)
    also differs.
  - **Template defaults made explicit since the run** (runs before patch/paper):
    `max_pad_steps: null` is now spelled out in the reader, where it used to be
    absent. rbyte's `NeroRobotReader` default is `None`. Each frame-cache source
    now spells out `preprocessing: letterbox`, the default of
    `NeroFrameCacheSource` and `preprocess_module`.
  - **Logging only**: the seven CSV-only runs (`bimanual_tokenizer*`,
    `bimanual_hand_{off,on}_v2`) ran with `wandb.mode: disabled` and a single
    CSVLogger. A rerun logs CSV plus wandb. `bimanual_tokenizer` (v1) logged
    every 100 steps; it is 20 now.
- **Snapshots vs commits**: the remote and snapshot trees were checked against
  commits with `src/` and `config/` diffed.
  - renate `617c1de` is identical to `47d6a7a7`.
  - renate `1b2167e` is identical to `037e21ad`.
  - sisyphos `ad3e5ee` is identical to `257321e6`.
  - renate `a5ea41f` and tresor `715a629` are identical to `a9790ebc`.
  - tresor `~/l1e/rmind-l1` is identical to `554f7fde`.
  - `uv.lock` (torch 2.12.1, rbyte `c08920ba`) is the same from `745c5853` to
    `04fc11e1`.
- **Data smoke** (CPU, no GPU was free): the val datasets of
  `paper2_pp_codes_cond` (v3, 4872 samples) and `sweep_w1_off` (v1, 2644
  samples) built from the NAS defaults. Samples loaded through the NAS frame
  caches. Every pinned tokenizer passed its sha256 check.
- **Other experiments**: all other experiments in `config/experiment`, including
  the non-nero ones, compose exactly as before. That includes `seed`, which
  still resolves to 1337.

## Not reproducible as-is

- **The `/dev/shm` frame caches are gone.** This affects `sweep_w1_off_r18` and
  `l1_w1_r18_pad` (416x640) and `sweep_w1_off_dinoft336` (210x336). Rebuild them
  once before running; the build is deterministic and fingerprinted, and `--verify`
  checks the result:

  ```sh
  python -m rmind.scripts.nero_frame_cache --root /nasa/max/nero-cache/cube-bimanual/2026-10-07 \
      --out /nasa/max/nero-cache/cube-bimanual/frame-cache-416x640/2026-10-07 --input-hw 416 640 --workers 8 --verify 16
  python -m rmind.scripts.nero_frame_cache --root /nasa/max/nero-cache/cube-bimanual/2026-10-07 \
      --out /nasa/max/nero-cache/cube-bimanual/frame-cache-210x336/2026-10-07 --input-hw 210 336 --workers 8 --verify 16
  ```

  The sizes are about 65 GB and about 17 GB. A `/dev/shm` copy works too: set
  `NERO_FRAME_CACHE_ROOT`.

- **Tokenizer checkpoints are host-bound through their own hparams.**
  `NeroChunkTokenizer.load_from_checkpoint` re-reads the action standardizer and
  the event reference from the absolute paths stored at training time.

  - `paper_tok_c10_*` reads `~/paper/data/cube-bimanual/stats_c10`. It loads on
    sisyphos and renate, but not on tresor.
  - `paper2_tok_q4_e999` reads `~/paper/data/cube-bimanual/stats_c10_v3`. It
    loads on renate only.
  - `bimanual_tokenizer_*` and `paper2_tok_cb64_*` read
    `~/data/nero-arms/cube-bimanual/rmind/stats_{v1,c10_v3}`. They load on all
    three hosts.

  The numbers themselves were always in the checkpoint (`standardizer.mean/std`
  and `event_reference` are buffers); only `__init__` needed the files. Since
  rmind#283, a newly saved tokenizer checkpoint also stores both JSON payloads
  under `checkpoint["nero_tokenizer_stats"]` and loads on any host. It does not
  need the paths. If they exist, it compares them with the embedded stats, and
  if they differ it warns loudly and uses the embedded ones. Checkpoints without the key, which
  includes every one listed above, load exactly as before.

  To make an existing checkpoint portable, write a self-contained copy. The
  original is never written, and the script refuses stats files that do not
  match the checkpoint's buffers:

  ```sh
  python -m rmind.scripts.nero_tokenizer_embed_stats \
      "$NERO_CKPT_ROOT/paper2_tok_q4/checkpoints/epoch=999-step=150000.ckpt" \
      --stats-dir /nasa/max/nero-cache/cube-bimanual/stats_c10_v3
  # -> .../epoch=999-step=150000.selfcontained.ckpt, prints its sha256
  ```

  Without `--stats-dir`, it reads the hparams paths. The copy has a new sha256,
  so the pins above still name the originals. The script is deterministic: re-running it on
  the same input reproduces the copy byte for byte, so the hashes below are the
  ones to pin once someone writes a copy under `NERO_CKPT_ROOT`. Nobody has
  published one yet; the copies below were scratch builds. They give
  bit-identical standardization, codes, latents and decodes to the original
  on renate, on 192 real chunks:

  | copy of              | standardizer sha256 | self-contained sha256                                              |
  | -------------------- | ------------------- | ------------------------------------------------------------------ |
  | `paper_tok_c10_e399` | `18a2edb93934`      | `9ff3ec2fdcfacc9a51878a219dc623c6a9e5f3f7431ea19dc7f0bca82370f2d5` |
  | `paper_tok_c10_e89`  | `18a2edb93934`      | `b3649e285e54dc10eb80d39aeb949884e2b3cdf9a2b771049d2ec88ec79343da` |
  | `paper2_tok_q4_e999` | `57d535d08900`      | `1ff32f5beda82a8e034f7c0e2c70d52e8735e064e3c1090378ea54b42faa3bb3` |

  A policy checkpoint still re-loads its tokenizer from the `tokenizer_ckpt`
  path in its own hparams, so it needs that file to exist. The fix above only
  removes the second hop, from the tokenizer to the stats.

  For an original on another host, link the byte-identical NAS stats dir to
  the stored path,
  e.g. `mkdir -p ~/paper/data/cube-bimanual && ln -s /nasa/max/nero-cache/cube-bimanual/stats_c10_v3 ~/paper/data/cube-bimanual/`.

- **Code drift.** Runs from before `04fc11e1` trained on older code (the "code"
  column). The head adds features but keeps their defaults (window 1, the l1
  head, stretch preprocessing, the offset modes, the relabel reader for v3 only).
  The configs compose identically, but nobody has re-trained and compared the
  numbers. For a bit-exact code replay, use the recorded commit.

  - `bimanual_tokenizer` (v1) ran from an uncommitted tree on `bdaaa57a`. That
    tree was committed 2 min later as `c207bb4e`..`3ed679f8`.
  - `bimanual_hand_{off,on}_v2` ran `b069267e` with the `save_top_k: -1`
    callback edit that became `c7333fc2`.

- **Pending runs.** `paper2_pp_l1_handon`, `paper2_pp_codes_cond` and
  `paper2_pp_codes_nocond` were still training on 2026-10-10. Their checkpoints
  are on the hosts and not yet on NAS. `paper2_pp_l1_handoff` and
  `paper2_pp_codes_q4_nocond` are queued behind them. Their recorded config is
  the queued chain's launch arguments composed at `04fc11e1`.
