# Demonstrator speed profiles around speed-limit transitions

Sidecars: 638 drives -> 4548 debounced transitions (limit runs >= 2 s, unknown gap <= 3 s, full +-10 s motion coverage); 4141 with a moving approach (v_pre > 10 km/h). -1 = explicitly UNLIMITED.

t0 = first frame asserting the new limit. t_comply = first time speed <= new limit + 3 km/h (negative: before the sign). t_adapt = braking onset (speed sustainably below approach speed - 3 km/h), drops from above the new limit only. t_headroom = first time speed > old limit + 3 km/h after a raise.

## Counts per transition (old -> new km/h)

| old -> new | direction | n | n moving |
|---|---|---|---|
| 50 -> 30 | drop | 589 | 504 |
| 30 -> 50 | raise | 550 | 374 |
| 70 -> 50 | drop | 395 | 376 |
| 50 -> 70 | raise | 373 | 346 |
| 70 -> 100 | raise | 249 | 244 |
| 100 -> 70 | drop | 227 | 218 |
| 100 -> 50 | drop | 219 | 214 |
| 50 -> 100 | raise | 217 | 212 |
| 60 -> 50 | drop | 164 | 157 |
| 50 -> 60 | raise | 145 | 134 |
| 100 -> 80 | drop | 132 | 132 |
| 80 -> 100 | raise | 113 | 113 |
| 80 -> 50 | drop | 96 | 95 |
| 80 -> 60 | drop | 93 | 91 |
| 70 -> 80 | raise | 84 | 80 |
| 50 -> 80 | raise | 69 | 67 |
| 80 -> 70 | drop | 68 | 67 |
| 120 -> 100 | drop | 56 | 55 |
| 60 -> 100 | raise | 54 | 48 |
| 60 -> 80 | raise | 45 | 43 |
| 60 -> 70 | raise | 43 | 31 |
| 100 -> 60 | drop | 42 | 40 |
| 100 -> 120 | raise | 41 | 41 |
| 80 -> 30 | drop | 38 | 37 |
| 70 -> 60 | drop | 32 | 29 |
| 70 -> 30 | drop | 31 | 30 |
| 60 -> 120 | raise | 27 | 27 |
| 30 -> 80 | raise | 26 | 26 |
| 120 -> 60 | drop | 25 | 25 |
| 40 -> 50 | raise | 24 | 21 |
| 50 -> 20 | drop | 23 | 23 |
| 20 -> 50 | raise | 23 | 22 |
| 50 -> 40 | drop | 21 | 18 |
| 100 -> 30 | drop | 16 | 15 |
| 30 -> 70 | raise | 14 | 10 |
| 120 -> 70 | drop | 14 | 14 |
| 50 -> 120 | raise | 14 | 14 |
| 60 -> 30 | drop | 13 | 13 |
| 30 -> 100 | raise | 12 | 11 |
| 120 -> 50 | drop | 12 | 12 |
| 60 -> 40 | drop | 11 | 10 |
| 40 -> 30 | drop | 9 | 9 |
| 80 -> 120 | raise | 8 | 8 |
| 30 -> 60 | raise | 7 | 7 |
| 120 -> 40 | drop | 7 | 7 |
| 100 -> 40 | drop | 6 | 6 |
| 70 -> 120 | raise | 6 | 6 |
| 80 -> 40 | drop | 5 | 5 |
| 130 -> 100 | drop | 5 | 5 |
| 30 -> 120 | raise | 4 | 4 |
| 100 -> 130 | raise | 4 | 4 |
| 130 -> 70 | drop | 3 | 3 |
| 50 -> 10 | drop | 3 | 3 |
| 70 -> 130 | raise | 3 | 3 |
| 10 -> 50 | raise | 3 | 2 |
| 40 -> 20 | drop | 3 | 1 |
| 40 -> 60 | raise | 2 | 2 |
| 20 -> 30 | raise | 2 | 2 |
| 40 -> 120 | raise | 2 | 2 |
| 50 -> 5 | drop | 2 | 2 |
| 120 -> 80 | drop | 2 | 2 |
| 30 -> 20 | drop | 2 | 2 |
| 30 -> 40 | raise | 2 | 1 |
| 40 -> 80 | raise | 2 | 2 |
| 70 -> -1 | raise | 1 | 1 |
| 130 -> 50 | drop | 1 | 1 |
| 70 -> 10 | drop | 1 | 1 |
| 80 -> 130 | raise | 1 | 1 |
| -1 -> 60 | drop | 1 | 1 |
| 130 -> 110 | drop | 1 | 1 |
| 130 -> 80 | drop | 1 | 1 |
| -1 -> 80 | drop | 1 | 1 |
| 80 -> -1 | raise | 1 | 1 |
| 70 -> 40 | drop | 1 | 0 |
| 20 -> 40 | raise | 1 | 0 |
| 120 -> 30 | drop | 1 | 1 |
| 60 -> -1 | raise | 1 | 1 |
| 110 -> 130 | raise | 1 | 1 |
| 10 -> 30 | raise | 1 | 1 |
| -1 -> 70 | drop | 1 | 1 |

## Adaptation by group (all transitions)

| group | n | comply<=t0 | med t_comply | med t_adapt | med overshoot | mean frac>lim after | med t_headroom | median profile (km/h) |
|---|---|---|---|---|---|---|---|---|
| drop 21-40 | 448 | 88% | -7.0 | -6.0 | 0.0 | 21% | - | -10s:67 -5s:65 -2s:62 +0s:60 +2s:58 +5s:57 +10s:54 |
| drop <=20 | 1576 | 89% | -10.0 | -5.0 | 0.0 | 21% | - | -10s:47 -5s:46 -2s:44 +0s:43 +2s:42 +5s:42 +10s:42 |
| drop >40 | 346 | 82% | -5.0 | -6.5 | 2.0 | 36% | - | -10s:66 -5s:58 -2s:54 +0s:50 +2s:49 +5s:48 +10s:47 |
| drop from UNLIMITED | 3 | 100% | -10.0 | -0.5 | 29.1 | 76% | - | -10s:95 -5s:92 -2s:90 +0s:89 +2s:88 +5s:86 +10s:88 |
| raise 21-40 | 433 | - | - | - | - | - | 0.5 | -10s:56 -5s:60 -2s:63 +0s:64 +2s:66 +5s:68 +10s:72 |
| raise <=20 | 1427 | - | - | - | - | - | 1.0 | -10s:38 -5s:40 -2s:41 +0s:43 +2s:45 +5s:47 +10s:49 |
| raise >40 | 312 | - | - | - | - | - | -1.0 | -10s:46 -5s:47 -2s:49 +0s:51 +2s:54 +5s:60 +10s:63 |
| raise to UNLIMITED | 3 | - | - | - | - | - | -10.0 | -10s:92 -5s:89 -2s:88 +0s:86 +2s:89 +5s:96 +10s:104 |

## Adaptation by group (moving approaches only)

| group | n | comply<=t0 | med t_comply | med t_adapt | med overshoot | mean frac>lim after | med t_headroom | median profile (km/h) |
|---|---|---|---|---|---|---|---|---|
| drop 21-40 | 434 | 88% | -7.0 | -6.0 | 0.0 | 21% | - | -10s:68 -5s:65 -2s:63 +0s:61 +2s:59 +5s:57 +10s:54 |
| drop <=20 | 1452 | 88% | -10.0 | -5.0 | 0.0 | 22% | - | -10s:50 -5s:47 -2s:46 +0s:45 +2s:45 +5s:43 +10s:43 |
| drop >40 | 339 | 81% | -5.0 | -6.5 | 2.2 | 37% | - | -10s:67 -5s:59 -2s:54 +0s:51 +2s:49 +5s:48 +10s:47 |
| drop from UNLIMITED | 3 | 100% | -10.0 | -0.5 | 29.1 | 76% | - | -10s:95 -5s:92 -2s:90 +0s:89 +2s:88 +5s:86 +10s:88 |
| raise 21-40 | 414 | - | - | - | - | - | 0.5 | -10s:58 -5s:61 -2s:64 +0s:65 +2s:66 +5s:69 +10s:73 |
| raise <=20 | 1190 | - | - | - | - | - | 0.0 | -10s:45 -5s:46 -2s:46 +0s:48 +2s:49 +5s:52 +10s:53 |
| raise >40 | 306 | - | - | - | - | - | -1.0 | -10s:46 -5s:47 -2s:49 +0s:51 +2s:55 +5s:60 +10s:62 |
| raise to UNLIMITED | 3 | - | - | - | - | - | -10.0 | -10s:92 -5s:89 -2s:88 +0s:86 +2s:89 +5s:96 +10s:104 |

## Notes

- Per-transition rows (incl. 1 s speed profiles): `/home/max/Code/rmind-traffic-rules/diag_results/map_probe/transitions.parquet`.
- This windowing is the template for the headline compliance metric: the same +-10 s cuts around transitions, evaluated on POLICY rollouts vs these demonstrator profiles.
