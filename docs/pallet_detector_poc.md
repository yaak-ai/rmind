# Pallet detector PoC: what the D12 data says

PoC for `pallet_detection_plan.md` (monocular keypoints + floor-constrained PnP on
`cam_fork`), done 2026-10-19 on `feat/pallet-detector-poc` (branched from
`feat/d12-flow-matching-fork-height` @ `47ec996`). No AprilTag data and no camera
calibration exist yet, so everything here is either measured from the 38 D12 jobs
without labels or simulated.

## Code

| Path                                        | What                                                                                              |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `src/rmind/pallet/geometry.py`              | Frames, EUR keypoint model (14 points), Kannala-Brandt fisheye camera, `CAM_FORK_GUESS`           |
| `src/rmind/pallet/pose.py`                  | Closed-form floor pose, RANSAC over pairs, LM refine with free camera pitch; 6-DoF baseline       |
| `src/rmind/pallet/labels.py`                | AprilTag 36h11 detection, fisheye tag pose, tag -> pallet pose -> keypoint labels, tag inpainting |
| `src/rmind/pallet/keypoints.py`             | Crop handling, heatmap keypoint net (timm + FPN, stride 4), Gaussian targets, DARK-style decoding |
| `src/rmind/scripts/pallet_poc_synthetic.py` | Monte Carlo accuracy vs distance, calibration-error bias                                          |
| `src/rmind/scripts/pallet_poc_zeroshot.py`  | Grounding DINO proposals over every approach, checked against UWB                                 |
| `tests/test_pallet.py`                      | Projection, solver, tag labels, decoding                                                          |

Extra deps live in the `pallet` extra (`uv sync --extra pallet`). `transformers` is
capped below 5.15 so that the lock stays on `huggingface-hub` 1.x.

Later additions (2026-10-26/27): hand-labelled entry-face keypoints and their training
(`pallet/entry.py`, `scripts/pallet_train_entry.py`, `pallet_predict_entry.py`,
`pallet_select_hard.py`), labelling tools (`pallet_label_tools.py`), demo videos
(`pallet_demo_entry.py`) and the proposer head on the policy's DINOv3 features
(`pallet_proposer_head.py`).

## Data, weights and runs

- Workspace (labels, images, checkpoints, predictions, demos):
  `/nasa/team-space/nikita/pallet-detector`, described in its `README.md`.
- Runs, label datasets and weights: wandb `yaak/pallet-detector` (`pallet/tracking.py`).
  Training scripts log by default (`--no-wandb` to turn off); runs keep generated
  names and are described by tags and notes.

## Findings

**UWB cannot place the pallet.** While the pallet sits on the forks (lifted more
than 0.3 m, rigidly fixed to the truck), the UWB pallet vector in the robot frame
has a median std of 0.21 m forward and 0.26 m lateral (20 jobs, 2026-10-02/12).
The target is about ±2 cm.

**`cam_fork` is a fisheye, mounted high-ish and behind the fork tips.**
Straight edges bend strongly, so it is modelled as Kannala-Brandt, not pinhole.
From the frames: the 0.8 m entry face spans about 794 px when it reaches the fork
tips, and the horizon sits near the image centre. If f is about 850 px, the camera
is about 0.43 m up, level, and about 0.8 m behind the tips. This is a guess, not a
calibration. The fork tips look different on 2026-08-25 (indoor, another truck or
mount), so calibration has to be per vehicle/mount.

**Insertion is blind to the entry face.** The approach covers 4 m to the fork tips
in about 3 s. The entry face drops off the bottom of the image at about the fork
tips, not in the last 0.3–0.5 m as the plan assumed. The keypoint set therefore
includes the far deck corners, which stay in view. In the synthetic study, the
solver still works with only those few points (`< 0` in the plots), where generic
6-DoF PnP falls apart. After that, dead reckoning carries the remaining ~1 m.

**Free camera pitch is required** (`pallet_poc_synthetic`, p95, no pixel noise):

| Calibration error | Pitch fixed, 1 m     | Pitch free, 1 m |
| ----------------- | -------------------- | --------------- |
| pitch +0.5°       | 9.4 cm fwd, 5.5° yaw | 0               |
| height +1 cm      | 3.7 cm fwd, 2.3° yaw | 0.1 cm, 0.02°   |
| camera yaw +0.5°  | 1.6 cm lat, 0.5° yaw | same            |
| f +1 %            | 1.9 cm fwd           | same            |

A pitch error rescales depth differently for floor-level and deck-level keypoints,
and a 3-DoF fit absorbs that into yaw. The vertical structure of the pallet makes
pitch observable, so `pose.solve` fits a pitch offset by default. Pitch also moves
at runtime (braking, load, floor joints). What remains for calibration is **camera
yaw (needs ≤0.3°)** and f.

**Noise budget holds with a good keypoint net** (pitch free, CAM_FORK_GUESS):

| Distance ahead of tips | 1 px: lat / yaw / fwd   | 2 px: lat / yaw / fwd   |
| ---------------------- | ----------------------- | ----------------------- |
| 0 m                    | 0.1 cm / 0.13° / 0.2 cm | 0.1 cm / 0.22° / 0.4 cm |
| 1 m                    | 0.2 cm / 0.19° / 0.7 cm | 0.3 cm / 0.59° / 1.6 cm |
| 2 m                    | 0.2 cm / 0.38° / 1.2 cm | 0.6 cm / 0.69° / 3.4 cm |

With the far deck corners in the set, generic 6-DoF matches this while 6 or more
points are visible. The floor constraint's value is the closed-form start, the
few-point regime once the forks are in, and robustness. It is not better yaw at
range. Real keypoint errors are correlated and heavy-tailed, so treat these as
lower bounds.

**A zero-shot proposer works as a crop source** (`pallet_poc_zeroshot`, Grounding
DINO base, prompt "a wooden pallet.", 5474 approach frames over 38 jobs). It
returns a box in 95–99 % of frames at every UWB distance bin from 0 to 8 m. This
is recall of *a* box, not verified precision. Seen failure modes: a false positive
on an all-dark auto-exposure frame, a second pallet in the background (indoor
session), and fork tips (masked by a fixed ROI). 58 % of boxes touch the bottom
image edge, because near the forks the pallet fills the bottom of the frame.

**Box-bottom range does not validate the scale.** Back-projecting the box's
bottom-centre gives slopes of 0.23–0.81 against UWB forward, depending on session.
A box bottom is the nearest corner, not the entry face, and UWB is ±0.2 m. Only a
checkerboard calibration settles the camera model.

**The policy sees pockets at under one patch.** The D12 policy takes 256×144
`cam_fork` into DINOv3 ViT-S/16 (`d12.yaml`), so one patch covers about 120
full-res pixels. The plan's "DINOv2, 14 px, 224 px" is outdated. The fork-camera
resolution proposal (Notion, 2026-10-16) measures pockets at 13–27 × 7–10 px at
that resolution. At full resolution they are about 100–200 × 50–75 px, which is
ample for corner keypoints.

## Next

1. Checkerboard-calibrate `cam_fork` per truck (`cv2.fisheye.calibrate`) and
   measure the mount, camera yaw in particular. Re-run `pallet_poc_synthetic` with
   the result.
1. AprilTag collection runs (step 2 of the plan) and wire `labels.py` to the
   video. Use `TagPallet.off_floor` to check the floor and mount assumptions on
   real frames.
1. Train `PalletKeypointNet` on tag labels, using Grounding DINO crops at first.
   Measure real keypoint noise and feed it back into the synthetic study.
1. Fork occlusion: block-bottom keypoints behind the fork tips need a visibility
   mask in the labels (TODO in `labels.keypoint_labels`).
