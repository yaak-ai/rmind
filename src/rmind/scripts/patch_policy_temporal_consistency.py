"""Tick-to-tick stability of a PatchPolicy's SERVED action, on the val set.

Every metric `PatchPolicy` logs is a per-readout, single-tick measurement:
`code_q_last`, `offset_argmax_recon`, `predict/*/score_l1` all score one frame
against its own ground truth and average. None of them can see whether the
errors at frame `t` and frame `t+1` are the SAME error or DIFFERENT errors --
and that distinction is the whole difference between a policy that tracks
smoothly and one that oscillates. Two checkpoints with identical per-frame L1
can behave completely differently in closed loop.

This script measures it. A val episode already yields `t` per-frame readouts
from one forward pass, so nothing new has to be computed or collected:

    (1) code flip rate     -- P(argmax code_q differs between adjacent readouts),
                              against the ground-truth code sequence's OWN flip
                              rate as the floor. The RATIO is the number to read:
                              >1 means the policy switches behavior tokens more
                              often than the human did.
    (2) executed-action TV -- in closed loop only element 0 of each predicted
                              chunk is ever actuated, so the executed command is
                              `chunk[t, 0, :]` over t. Its total variation vs the
                              ground truth's, per channel. This is the direct
                              offline analogue of steering/pedal jitter.
    (3) TV decomposition   -- (2) recomputed with the offset held at zero (code
                              chatter alone) and with the codes held at ground
                              truth (offset-head noise alone), so a jittery arm
                              can be attributed to discrete code switching or to
                              a noisy continuous head. The fixes differ.
    (4) plan incoherence   -- the chunk at frame t predicts frames t..t+h-1, so
                              `chunk[t, k]` and `chunk[t+k, 0]` describe the SAME
                              instant. Disagreement means the plan rewrites
                              itself as the episode advances -- the model
                              disagreeing with its own past self, which per-frame
                              L1 cannot express.

A second, independent block measures the PEDALS, because everything above
(and every other offline metric this project has) scores steering, while the
closed-loop failure of the 3-camera arms is longitudinal -- they die by
collision having barely braked (`1cam_vs_3cam_offline_gap.md` §12.6):

    (5) pedal marginals    -- mean/sd/p10/p50/p90 of the executed gas and brake,
                              model and human, plus the model/human ratio. A low
                              mean is only a defect relative to the human floor
                              on the SAME frames.
    (6) brake events       -- P(brake > tau) and mean event duration over a
                              threshold sweep. A policy that brakes with the
                              right average in short chattery bursts is a
                              different defect from one that never brakes, and
                              the mean hides both.
    (7) conditional resp.  -- the decisive one. Recall P(model brake | human
                              brake), the false-positive rate, and P(model gas |
                              human brake) -- gas commanded where the human
                              braked. Closed loop, "brakes less" is inseparable
                              from "met fewer situations demanding a brake",
                              because each checkpoint drives itself into its own
                              state distribution; scoring every checkpoint on the
                              same human frames is what separates them.
    (8) pedal decomposition-- (7) recomputed on the `codes only` and `offset
                              only` arms, so a brake miss is attributable to code
                              selection or to the offset head the way (3) does it
                              for steering chatter. §5 found steering chatter was
                              entirely code selection; that is NOT assumed here.

All argmax, matching what `load_for_export` serves; a checkpoint's own
`sample_codes` is ignored (a sampled draw would swamp these deltas, exactly as
in `patch_policy_camera_probe.py`).

TWO CAVEATS, both load-bearing:

* Adjacent readouts are `episode_step` raw frames apart, NOT one deployment
  control tick. These numbers are a proxy at the episode's own rate, valid for
  COMPARING checkpoints (every causal arm shares `episode_step: 10`), not as an
  absolute jitter figure for the vehicle.
* Compare checkpoints only on the SAME val set. `val` and `val_3cam` are
  DISJOINT drive sets, so a 1-camera run's wandb numbers are not comparable with
  a 3-camera run's. Both can be run here: a 1-camera checkpoint's own
  `input_transform` reads only `data/cam_front_left`, which `val_3cam` also
  provides, so point BOTH at `--experiment yaak/patch_policy/
  dinov2_dinowm_causal_3cam` and the comparison is clean.

Usage (from a repo checkout with the val_3cam rbyte cache built):
    uv run python -m rmind.scripts.patch_policy_temporal_consistency \
        --artifact yaak/alex-tmp/model-kughoqfi:v4 \
        --config-dir /abs/path/to/config \
        --experiment yaak/patch_policy/dinov2_dinowm_causal_3cam \
        [--batches 50] [--device cuda] [--out results.json] \
        [--override ++datamodule.val.dataset.samples.resume=true]

`--override` is forwarded to hydra verbatim. Pass the `samples.resume=true`
override (with a `++` prefix -- the key is not in the base schema) on any
second-or-later invocation, or rbyte rebuilds the sample table from scratch
(~15-25 min) on every single run.
"""

from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path
from typing import IO, Any

import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch import Tensor, nn
from torch.utils._pytree import tree_map  # noqa: PLC2701

from rmind.models.patch_policy import PatchPolicy

# `_structure`/`predict_step` order the action features gas, brake, steer,
# turn_signal. The first three are continuous and get total variation; the
# fourth is a categorical indicator, so it gets a switch rate instead.
CONTINUOUS_CHANNELS = ("gas", "brake", "steer")
TURN_SIGNAL_CHANNEL = 3
# Pedal-probe indices into that same executed vector. `gas`/`brake` are
# CONTINUOUS_CHANNELS 0/1, and the action tokenizer normalizes all three
# continuous channels with `Identity` (only `turn_signal` is scaled -- see
# config/model/yaak/action_tokenizer/raw.yaml), so these thresholds are in the
# raw `*_pedal_normalized` [0, 1] units that rsim's `pred_brake_mean` reports
# and are directly comparable with it. `_pedal_units_are_raw` re-checks that on
# the loaded checkpoint rather than trusting the config.
GAS_CHANNEL = 0
BRAKE_CHANNEL = 1
PEDAL_THRESHOLDS = (0.05, 0.1, 0.2, 0.5)
MODEL_ARMS = ("served", "code_only", "offset_only")
# the ground-truth plan-incoherence floor (4) must be exact zero up to bf16
# accumulation; anything above this means the horizon index convention is wrong
ALIGNMENT_TOLERANCE = 1e-3


def _to_device(batch: object, device: torch.device) -> object:
    return tree_map(
        lambda x: x.to(device, non_blocking=True) if isinstance(x, Tensor) else x, batch
    )


def _load_model(*, artifact: str | None, ckpt: str | None) -> PatchPolicy:
    """Load a checkpoint, tolerating two eras of older ones.

    First `_modernize_hparams` rewrites the pre-`cameras` `image` hparam, which
    otherwise makes a 1-camera checkpoint like `do8m9ot8` unloadable outright.

    Then the same fallback as `patch_policy_camera_probe._load_model`, for the same
    reason: `CausalFrameTransformer.__init__` unconditionally creates
    `encoder.intra_position_gain`/`intra_position_norm` in the CURRENT code, so
    a strict load fails on a checkpoint trained before they existed (e.g.
    `03tuy3q9`). A bare `strict=False` would then run a never-trained-for
    `LayerNorm` over the position table and silently change the forward pass, so
    on that specific failure `intra_position_norm` is restored to `Identity`
    (`intra_position_gain` default-inits to 1.0, a no-op multiplier) -- making
    the load bit-identical to how the checkpoint actually trained and served.

    Raises:
        RuntimeError: re-raised for any load failure other than that one.
        ValueError: if neither `artifact` nor `ckpt` is given.
    """
    kwargs: dict[str, Any] = {"weights_only": False, "map_location": "cpu"}
    if artifact is not None:
        source = PatchPolicy.download_wandb_artifact(artifact)
    elif ckpt is not None:
        source = Path(ckpt)
    else:
        msg = "one of artifact/ckpt is required"
        raise ValueError(msg)
    modernized = _modernize_hparams(source)

    def loader(**kw: Any) -> PatchPolicy:
        if isinstance(modernized, io.BytesIO):
            modernized.seek(0)  # rewind: the strict=False retry re-reads it
        return PatchPolicy.load_from_checkpoint(modernized, **kw)

    try:
        return loader(**kwargs)
    except RuntimeError as e:
        if "intra_position_gain" not in str(e):
            raise
        print(  # noqa: T201
            f"note: {artifact or ckpt} predates b846a4f -- reloading strict=False "
            "and restoring the pre-fix raw-table behavior "
            "(intra_position_norm -> Identity)"
        )
        model = loader(**kwargs, strict=False)
        model.encoder.intra_position_norm = nn.Identity()
        return model


class _Accumulator:
    """Sample-weighted running means, so batches of unequal size average right."""

    def __init__(self) -> None:
        self._sums: dict[str, float] = {}
        self._weights: dict[str, float] = {}

    def add(self, key: str, value: Tensor, weight: float) -> None:
        self._sums[key] = self._sums.get(key, 0.0) + float(value) * weight
        self._weights[key] = self._weights.get(key, 0.0) + weight

    def mean(self) -> dict[str, float]:
        return {k: v / self._weights[k] for k, v in self._sums.items()}


def _modernize_hparams(ckpt_path: Path) -> Path | IO[bytes]:
    """Translate a pre-`cameras` checkpoint's hparams, or pass the path through.

    `PatchPolicy.__init__` used to take `image: Path` (the remapper path to the
    single camera, e.g. `("image", "cam_front_left")`); it now takes
    `cameras: tuple[str, ...]`. `@validate_call` rejects the stale key outright
    -- `unexpected_keyword_argument` for `image` -- so a 1-camera checkpoint from
    that era (e.g. `do8m9ot8`) cannot be loaded at all without this. The
    translation is exact: `("image", "cam_front_left") -> ("cam_front_left",)`,
    which is precisely what the old single-camera path meant.

    Returned as an in-memory buffer rather than a temp file so a ~450 MB
    checkpoint is not written to disk just to change one hparam.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hparams = ckpt.get("hyper_parameters")
    if not isinstance(hparams, dict) or "image" not in hparams:
        return ckpt_path
    image = hparams.pop("image")
    hparams["cameras"] = (image[-1],) if isinstance(image, tuple | list) else (image,)
    print(  # noqa: T201
        f"note: {ckpt_path.name} predates the `cameras` hparam -- translating "
        f"image={image!r} to cameras={hparams['cameras']!r}"
    )
    buffer = io.BytesIO()
    torch.save(ckpt, buffer)
    buffer.seek(0)
    return buffer


def _turn_signal_class(values: Tensor) -> Tensor:
    """Normalized turn_signal -> class index, exactly as `PatchPolicy._structure`
    and `predict_step` emit it, so the switch rate counts real indicator changes.
    """
    bounds = torch.tensor([0.5, 1.5], device=values.device, dtype=values.dtype)
    return torch.bucketize(values * 2, bounds)


def _pedal_units_are_raw(tokenizer: Any) -> bool:
    """Is the tokenizer's normalizer the identity on gas/brake/steer?

    `PEDAL_THRESHOLDS` are quoted in raw `*_pedal_normalized` units so they mean
    the same thing as rsim's `pred_brake_mean`. Everything decoded here lives in
    the tokenizer's NORMALIZED space, and the two coincide only because the
    action tokenizer normalizes the continuous channels with `Identity`. That is
    a config fact about a checkpoint trained months ago, so it is verified on the
    loaded checkpoint instead of assumed -- if it ever stops holding, every
    threshold in (6)/(7) silently changes meaning.
    """
    device = next(tokenizer.parameters()).device
    probe = torch.tensor([[0.13, 0.37, -0.61, 0.5]], device=device)
    normalized = tokenizer._normalize(probe)  # noqa: SLF001
    keep = len(CONTINUOUS_CHANNELS)
    return bool(torch.allclose(normalized[0, :keep], probe[0, :keep], atol=1e-6))


def _event_stats(mask: Tensor) -> tuple[float, float]:
    """Event rate and mean event duration in FRAMES for a `(seq, frames)` mask.

    Duration counts maximal runs of consecutive True along dim 1. A run still
    open at a sequence edge is counted at its truncated length, so the figure is
    a slight under-estimate; and, like everything temporal in this script, it is
    a proxy at the episode's readout rate (`episode_step` raw frames per step),
    not a duration in seconds.
    """
    total = int(mask.sum())
    if not total:
        return 0.0, 0.0
    previous = torch.cat([torch.zeros_like(mask[:, :1]), mask[:, :-1]], dim=1)
    starts = int((mask & ~previous).sum())
    return float(mask.float().mean()), total / starts


def _pedal_metrics(executed: dict[str, Tensor]) -> dict[str, float]:
    """(5)-(8): pedal marginals, brake events, and the conditional response.

    `executed[arm]` is `(sequences, frames, action_features)` -- element 0 of
    every chunk, the only element deployment ever actuates -- with one sequence's
    frames contiguous along dim 1, which is what makes the event-duration count
    in `_event_stats` meaningful.
    """
    out: dict[str, float] = {}
    human = executed["ground_truth"]

    # (5) marginals, every arm including the human floor
    for arm, view in executed.items():
        for c, channel in enumerate(CONTINUOUS_CHANNELS):
            values = view[..., c]
            out[f"pedal/marginal/{arm}/{channel}/mean"] = float(values.mean())
            out[f"pedal/marginal/{arm}/{channel}/sd"] = float(values.std())
            for q in (0.1, 0.5, 0.9):
                key = f"pedal/marginal/{arm}/{channel}/p{int(q * 100)}"
                out[key] = float(values.quantile(q))

    # (6) gas and brake treated as events over a threshold sweep
    for tau in PEDAL_THRESHOLDS:
        for arm, view in executed.items():
            for c, channel in ((GAS_CHANNEL, "gas"), (BRAKE_CHANNEL, "brake")):
                rate, frames = _event_stats(view[..., c] > tau)
                out[f"pedal/event/{arm}/{channel}/t{tau:g}/rate"] = rate
                out[f"pedal/event/{arm}/{channel}/t{tau:g}/frames"] = frames

    # (7)+(8) the conditional response: every model arm against the SAME human
    # frames, which is the whole point -- it is what a closed-loop log cannot do.
    for tau in PEDAL_THRESHOLDS:
        human_brakes = human[..., BRAKE_CHANNEL] > tau
        support = float(human_brakes.sum())
        quiet = ~human_brakes
        out[f"pedal/cond/support/t{tau:g}"] = support
        # the human's own gas-while-braking rate: the floor the model's
        # `gas_given_brake` has to be read against, not zero
        out[f"pedal/cond/ground_truth/t{tau:g}/gas_given_brake"] = (
            float((human[..., GAS_CHANNEL] > tau)[human_brakes].float().mean())
            if support
            else float("nan")
        )
        for arm in MODEL_ARMS:
            view = executed[arm]
            brakes = view[..., BRAKE_CHANNEL] > tau
            gasses = view[..., GAS_CHANNEL] > tau
            out[f"pedal/cond/{arm}/t{tau:g}/recall"] = (
                float(brakes[human_brakes].float().mean()) if support else float("nan")
            )
            out[f"pedal/cond/{arm}/t{tau:g}/fpr"] = (
                float(brakes[quiet].float().mean())
                if float(quiet.sum())
                else float("nan")
            )
            out[f"pedal/cond/{arm}/t{tau:g}/gas_given_brake"] = (
                float(gasses[human_brakes].float().mean()) if support else float("nan")
            )
    return out


def _total_variation(executed: Tensor) -> Tensor:
    """Mean |a_t - a_{t-1}| per channel over `(b, t, action_features)`."""
    return (executed[:, 1:] - executed[:, :-1]).abs().mean(dim=(0, 1))


@torch.no_grad()
def evaluate(  # noqa: C901, PLR0912, PLR0914, PLR0915
    model: PatchPolicy,
    loader: Any,
    *,
    device: torch.device,
    max_batches: int,
    autocast: bool,
) -> dict[str, float]:
    tokenizer = model.tokenizer
    num_quantizers = tokenizer.quantizer.num_quantizers
    action_features = tokenizer._action_features  # noqa: SLF001
    # positions >= window-1 see the FULL context the ring buffer serves; earlier
    # readouts run under a partial window and are not what deployment does.
    window = getattr(model.encoder, "window", None)
    acc = _Accumulator()
    alignment_error = 0.0
    batches = 0
    # the executed command itself, kept per arm for the pedal block: the
    # marginals need percentiles and the events need contiguity, neither of
    # which a running mean can carry. ~(40 x 32) x 17 x 4 floats -- nothing.
    executed_by_arm: dict[str, list[Tensor]] = {}

    for i, cpu_batch in enumerate(loader):
        if i >= max_batches:
            break

        batch = _to_device(cpu_batch, device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=autocast):
            features, chunk = model._features(batch)  # noqa: SLF001
            code_logits, offsets = model._heads(features)  # noqa: SLF001

        if chunk is None:  # `_features(require_chunk=True)` guarantees otherwise
            msg = "the batch carries no action chunk to score against"
            raise RuntimeError(msg)

        code_logits = code_logits.float()
        offsets = offsets.float()
        target_codes = tokenizer(chunk)  # (b, t, g)
        target = tokenizer._normalize(chunk.flatten(-2, -1))  # noqa: SLF001

        argmax_codes = code_logits.argmax(dim=-1)  # (b, t, g)
        # the three decodes whose TVs separate code chatter from offset noise
        decoded = tokenizer.invert(argmax_codes)
        served = decoded + model._offset(offsets, argmax_codes)  # noqa: SLF001
        teacher = tokenizer.invert(target_codes) + model._offset(  # noqa: SLF001
            offsets, target_codes
        )

        b, t = features.shape[:2]
        start = 0 if window is None else min(window - 1, t - 2)
        span = slice(start, None)
        # (b, t, horizon, action_features); element 0 is the actuated command
        views = {
            "served": served.unflatten(-1, (-1, action_features))[:, span],
            "code_only": decoded.unflatten(-1, (-1, action_features))[:, span],
            "offset_only": teacher.unflatten(-1, (-1, action_features))[:, span],
            "ground_truth": target.unflatten(-1, (-1, action_features))[:, span],
        }

        # (1) code flip rate, model vs the ground-truth sequence's own floor
        for name, codes in (("served", argmax_codes), ("ground_truth", target_codes)):
            windowed = codes[:, span]
            flips = (windowed[:, 1:] != windowed[:, :-1]).float()  # (b, t-1, g)
            for q in range(num_quantizers):
                acc.add(f"code_flip/{name}/q{q}", flips[..., q].mean(), b)
            acc.add(
                f"code_flip/{name}/any",
                (windowed[:, 1:] != windowed[:, :-1]).any(dim=-1).float().mean(),
                b,
            )

        # (2)+(3) executed-action total variation, and the L1 of that same
        # executed element so accuracy and smoothness are read side by side
        gt_executed = views["ground_truth"][:, :, 0, :]
        for name, view in views.items():
            executed = view[:, :, 0, :]
            tv = _total_variation(executed)
            for c, channel in enumerate(CONTINUOUS_CHANNELS):
                acc.add(f"exec_tv/{name}/{channel}", tv[c], b)
            # turn_signal is CATEGORICAL: compare the bucketized class the way
            # `_structure`/`predict_step` emit it, not the raw normalized float.
            # Comparing floats makes the switch rate ~1.0 for any continuous
            # head (every tick differs in the last decimal) and ~0 for the
            # ground truth, which is a measurement artifact, not chatter.
            indicator = _turn_signal_class(executed[..., TURN_SIGNAL_CHANNEL])
            acc.add(
                f"turn_switch/{name}",
                (indicator[:, 1:] != indicator[:, :-1]).float().mean(),
                b,
            )
            if name != "ground_truth":
                l1 = (executed - gt_executed).abs().mean(dim=(0, 1))
                for c, channel in enumerate(CONTINUOUS_CHANNELS):
                    acc.add(f"exec_l1/{name}/{channel}", l1[c], b)

        for name, view in views.items():
            executed_by_arm.setdefault(name, []).append(view[:, :, 0, :].float().cpu())

        # (4) plan incoherence: chunk[t, k] and chunk[t+k, 0] are the same
        # instant. The ground-truth arm is a SELF-CHECK on that claim -- it is
        # built by unfolding one action series, so its value must be ~0; a
        # non-zero floor means this index convention is wrong, not that the
        # model is incoherent.
        horizon = views["served"].shape[2]
        for name in ("served", "ground_truth"):
            view = views[name]
            frames = view.shape[1]
            for k in range(1, horizon):
                if frames - k < 1:
                    continue
                delta = (view[:, : frames - k, k, :] - view[:, k:, 0, :]).abs().mean()
                acc.add(f"plan_incoherence/{name}/k{k}", delta, b)
                if name == "ground_truth":
                    alignment_error = max(alignment_error, float(delta))

        batches += 1

    if batches == 0:
        msg = "the val loader yielded no batches"
        raise RuntimeError(msg)

    results = acc.mean()
    results.update(
        _pedal_metrics({k: torch.cat(v) for k, v in executed_by_arm.items()})
    )
    results["_pedal_units_selfcheck"] = float(_pedal_units_are_raw(tokenizer))
    results["_alignment_selfcheck"] = alignment_error
    results["_batches"] = float(batches)
    results["_first_full_window_frame"] = float(0 if window is None else window - 1)
    return results


def _ratio(results: dict[str, float], served_key: str, truth_key: str) -> str:
    """Model-over-human ratio. Both full keys are passed explicitly: the metric
    names interleave the arm into the MIDDLE of the key (`exec_tv/served/steer`),
    so building them by suffixing an arm onto a prefix silently misses.
    """
    served, truth = results.get(served_key), results.get(truth_key)
    if served is None or not truth:
        return "   --  "
    return f"{served / truth:6.2f}x"


def _report(results: dict[str, float], *, checkpoint: str, experiment: str) -> None:
    align = results["_alignment_selfcheck"]
    print(f"\ncheckpoint : {checkpoint}")  # noqa: T201
    print(f"val set    : {experiment}")  # noqa: T201
    print(  # noqa: T201
        f"batches    : {int(results['_batches'])}   "
        f"readouts from frame {int(results['_first_full_window_frame'])} "
        "(full window only)"
    )
    verdict = (
        "OK" if align < ALIGNMENT_TOLERANCE else "FAILED -- numbers below are NOT valid"
    )
    print(f"alignment self-check: ground-truth plan incoherence {align:.2e}  {verdict}")  # noqa: T201
    if align >= ALIGNMENT_TOLERANCE:
        print(  # noqa: T201
            "  chunk[t, k] should equal chunk[t+k, 0] on the ground truth. It "
            "does not, so the horizon index convention assumed here is wrong."
        )

    print("\n(1) argmax code flip rate between adjacent readouts")  # noqa: T201
    print(f"{'':12s} {'model':>9s} {'human':>9s} {'ratio':>8s}")  # noqa: T201
    for label in [f"q{q}" for q in range(8)] + ["any"]:
        served_key = f"code_flip/served/{label}"
        if served_key not in results:
            continue
        truth_key = f"code_flip/ground_truth/{label}"
        name = f"code_{label[1:]}" if label != "any" else "any level"
        ratio = _ratio(results, served_key, truth_key)
        s, g = results[served_key], results[truth_key]
        print(f"{name:12s} {s:9.4f} {g:9.4f} {ratio:>8s}")  # noqa: T201

    print("\n(2) executed-action total variation (element 0 of each chunk)")  # noqa: T201
    print(  # noqa: T201
        f"{'':12s} {'model TV':>9s} {'human TV':>9s} {'ratio':>8s} {'model L1':>9s}"
    )
    for channel in CONTINUOUS_CHANNELS:
        print(  # noqa: T201
            f"{channel:12s} {results[f'exec_tv/served/{channel}']:9.5f} "
            f"{results[f'exec_tv/ground_truth/{channel}']:9.5f} "
            f"{_ratio(results, f'exec_tv/served/{channel}', f'exec_tv/ground_truth/{channel}'):>8s} "
            f"{results[f'exec_l1/served/{channel}']:9.5f}"
        )
    print(  # noqa: T201
        f"{'turn_signal':12s} {results['turn_switch/served']:9.5f} "
        f"{results['turn_switch/ground_truth']:9.5f} "
        f"{_ratio(results, 'turn_switch/served', 'turn_switch/ground_truth'):>8s}"
    )

    print("\n(3) where the jitter comes from (TV of the same executed element)")  # noqa: T201
    print(f"{'':12s} {'codes only':>11s} {'offset only':>12s} {'served':>9s}")  # noqa: T201
    for channel in CONTINUOUS_CHANNELS:
        print(  # noqa: T201
            f"{channel:12s} {results[f'exec_tv/code_only/{channel}']:11.5f} "
            f"{results[f'exec_tv/offset_only/{channel}']:12.5f} "
            f"{results[f'exec_tv/served/{channel}']:9.5f}"
        )

    print("\n(4) plan incoherence: |chunk[t, k] - chunk[t+k, 0]|")  # noqa: T201
    for k in range(1, 16):
        key = f"plan_incoherence/served/k{k}"
        if key not in results:
            break
        print(f"  k={k}  {results[key]:.5f}")  # noqa: T201

    _report_pedal(results)


def _report_pedal(results: dict[str, float]) -> None:
    """(5)-(8). Kept in its own function, and reading only its own `pedal/*`
    keys, so it cannot perturb the §5 steering numbers above it -- those are the
    regression test this script is held to.
    """
    if "pedal/marginal/served/brake/mean" not in results:
        return

    units_ok = bool(results.get("_pedal_units_selfcheck"))
    print(  # noqa: T201
        "\npedal-units self-check: tokenizer normalizer is "
        + (
            "the identity on gas/brake/steer  OK"
            if units_ok
            else "NOT the identity -- the thresholds below are NOT raw pedal units"
        )
    )

    print("\n(5) pedal marginals of the executed command")  # noqa: T201
    print(  # noqa: T201
        "    (the ratio column is only interpretable for gas/brake -- steer is "
        "signed\n     and its mean is ~0, so its ratio divides two near-zero "
        "numbers)"
    )
    print(  # noqa: T201
        f"{'':20s} {'mean':>8s} {'sd':>8s} {'p10':>8s} {'p50':>8s} {'p90':>8s} "
        f"{'ratio':>8s}"
    )
    for channel in CONTINUOUS_CHANNELS:
        human_mean = results[f"pedal/marginal/ground_truth/{channel}/mean"]
        for arm in ("served", "ground_truth"):
            prefix = f"pedal/marginal/{arm}/{channel}"
            label = "model" if arm == "served" else "human"
            ratio = (
                f"{results[f'{prefix}/mean'] / human_mean:6.2f}x"
                if arm == "served" and human_mean
                else ""
            )
            print(  # noqa: T201
                f"{channel + ' ' + label:20s} {results[f'{prefix}/mean']:8.4f} "
                f"{results[f'{prefix}/sd']:8.4f} {results[f'{prefix}/p10']:8.4f} "
                f"{results[f'{prefix}/p50']:8.4f} {results[f'{prefix}/p90']:8.4f} "
                f"{ratio:>8s}"
            )

    print("\n(6) gas/brake as events: rate, and mean run length in readouts")  # noqa: T201
    print(  # noqa: T201
        f"{'':14s} {'model rate':>10s} {'human rate':>10s} {'ratio':>8s} "
        f"{'model len':>10s} {'human len':>10s}"
    )
    for channel in ("brake", "gas"):
        for tau in PEDAL_THRESHOLDS:
            served = f"pedal/event/served/{channel}/t{tau:g}"
            truth = f"pedal/event/ground_truth/{channel}/t{tau:g}"
            ratio = _ratio(results, f"{served}/rate", f"{truth}/rate")
            print(  # noqa: T201
                f"{channel + ' >' + f'{tau:g}':14s} {results[f'{served}/rate']:10.4f} "
                f"{results[f'{truth}/rate']:10.4f} {ratio:>8s} "
                f"{results[f'{served}/frames']:10.2f} "
                f"{results[f'{truth}/frames']:10.2f}"
            )

    print("\n(7) conditional response on the SAME frames (served)")  # noqa: T201
    print(  # noqa: T201
        f"{'':14s} {'support':>9s} {'recall':>8s} {'fpr':>8s} "
        f"{'gas|brake':>10s} {'human gas|brake':>16s}"
    )
    for tau in PEDAL_THRESHOLDS:
        arm = f"pedal/cond/served/t{tau:g}"
        print(  # noqa: T201
            f"{'tau ' + f'{tau:g}':14s} {results[f'pedal/cond/support/t{tau:g}']:9.0f} "
            f"{results[f'{arm}/recall']:8.4f} {results[f'{arm}/fpr']:8.4f} "
            f"{results[f'{arm}/gas_given_brake']:10.4f} "
            f"{results[f'pedal/cond/ground_truth/t{tau:g}/gas_given_brake']:16.4f}"
        )

    print("\n(8) is a brake miss code selection or the offset head?")  # noqa: T201
    print(  # noqa: T201
        f"{'':14s} {'recall':>26s} {'gas | human brake':>26s}\n"
        f"{'':14s} {'codes':>8s} {'offset':>8s} {'served':>8s} "
        f"{'codes':>8s} {'offset':>8s} {'served':>8s}"
    )
    for tau in PEDAL_THRESHOLDS:
        cells = [
            results[f"pedal/cond/{arm}/t{tau:g}/{metric}"]
            for metric in ("recall", "gas_given_brake")
            for arm in ("code_only", "offset_only", "served")
        ]
        print(  # noqa: T201
            f"{'tau ' + f'{tau:g}':14s} " + " ".join(f"{v:8.4f}" for v in cells)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--artifact", help="wandb model artifact, e.g. yaak/rmind/model-<id>:v4"
    )
    group.add_argument("--ckpt", help="local checkpoint path")
    parser.add_argument(
        "--config-dir", required=True, help="absolute path to the hydra config dir"
    )
    parser.add_argument(
        "--experiment",
        default="yaak/patch_policy/dinov2_dinowm_causal_3cam",
        help="experiment supplying the val datamodule; keep it IDENTICAL across "
        "the checkpoints being compared (val and val_3cam are disjoint drives)",
    )
    parser.add_argument("--batches", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="fixes the shuffled val subset across invocations",
    )
    parser.add_argument("--out", help="write the raw metrics to this JSON file")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="extra hydra override, repeatable (e.g. "
        "datamodule.val.dataset.samples.resume=true)",
    )
    args = parser.parse_args()

    pl.seed_everything(args.seed, workers=True)
    device = torch.device(args.device)

    with initialize_config_dir(config_dir=args.config_dir, version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[f"experiment={args.experiment}", *args.override],
        )
    # ONLY the val node. `instantiate(cfg.datamodule)` would build the TRAIN
    # dataset too, and rbyte/pipefunc WIPES a samples store at instantiation --
    # so merely starting a val-only script against a train cache destroys it
    # (confirmed 2026-08-31: it cost a ~20 min rebuild of train_3cam), and
    # `samples.resume=true` does not prevent that, it only reuses a COMPLETE
    # store. `cfg.datamodule.val` carries its own `dataset` node, so this builds
    # the val dataloader and never touches train.
    val_dataloader = instantiate(cfg.datamodule.val)

    model = _load_model(artifact=args.artifact, ckpt=args.ckpt).to(device).eval()

    results = evaluate(
        model,
        val_dataloader,
        device=device,
        max_batches=args.batches,
        autocast=device.type == "cuda",
    )

    _report(results, checkpoint=args.artifact or args.ckpt, experiment=args.experiment)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:  # noqa: PTH123
            # a threshold with no support in the val subset yields NaN, which
            # `json.dump` happily writes and every strict reader (jq included)
            # then rejects -- write the absence as null instead
            json.dump(
                {k: (v if math.isfinite(v) else None) for k, v in results.items()},
                f,
                indent=1,
            )
        print(f"\nwrote {args.out}")  # noqa: T201


if __name__ == "__main__":
    main()
