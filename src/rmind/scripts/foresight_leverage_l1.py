"""G-1/L1 policy-leverage ablation: how much do the finetuned policy's outputs
depend on the CONTENT of the foresight-slot activations?

Architecture-aware arms (the 256 foresight slots per timestep are the ONLY
spatial route from observations to obs_summary/obs_history, which are all the
policy head reads):

- baseline                 no intervention (bit-identity asserted vs unhooked model)
- cut_summary_foresight    mask surgery: slots 523/524 forbidden from attending
                           267:523 -> structural upper bound (policy blind to obs)
- permute_foresight_all_t  after every encoder block, permute foresight hidden
                           states across the batch (all timesteps)
- permute_foresight_last_t same, timestep 5 only (timestep-resolved leverage)
- reset_foresight_last_t   replace timestep-5 foresight hidden states with the
                           batch-mean activation at that layer
- control_permute_speed    permute the speed slot (256) across batch, all
                           timesteps -- small-conduit effect-size control

Usage:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    uv run python -m rmind.scripts.foresight_leverage_l1 \
        --arms all --max-samples 256 \
        --out foresight_mm/results/l1_leverage.json

Outputs a nested JSON (arm -> stratum -> metrics) and prints a summary table.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch.utils._pytree import tree_map  # noqa: PLC2701

from rmind.components.base import Modality, SummaryToken
from rmind.components.mask import FactorizedAttentionMask, TorchAttentionMaskLegend
from rmind.components.objectives.joint_policy import JointPolicyObjective
from rmind.components.transformer.encoder import (
    FactorizedTransformerEncoderBlock,
    TransformerEncoder,
)
from rmind.models.control_transformer import ControlTransformer

if TYPE_CHECKING:
    from collections.abc import Iterator

    from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "config"

POLICY_CKPT = REPO_ROOT / "artifacts" / "model-zcglgth4_v9" / "model.ckpt"
TOKENIZER_CKPT = REPO_ROOT / "artifacts" / "model-y74asdtd_v9" / "model.ckpt"

# per-timestep slot layout (raw.yaml L9-20; verified against episode.index at runtime)
N_SLOTS = 530
N_TIMESTEPS = 6
SLOT_SPEED = 256
FORESIGHT = slice(267, 523)
SLOT_OBS_SUMMARY = 523
SLOT_OBS_HISTORY = 524

GAS_THRESH = 1.0 / 255.0 + 0.001
BRAKE_THRESH = 1.0 / 164.0 + 0.001
SPEED_STOPPED = 0.5
STEER_TURN_QUANTILE = 0.80
PERM_SEED = 42

FIELDS = ("gas", "brake", "steer", "turn")

HOOK_ARMS = (
    "permute_foresight_all_t",
    "permute_foresight_last_t",
    "reset_foresight_last_t",
    "control_permute_speed",
)
ALL_ARMS = ("baseline", "cut_summary_foresight", *HOOK_ARMS)

OFFSET_LOSS_RANGE = (0.005, 0.03)  # expected ~0.0077-0.02


# --------------------------------------------------------------------------- #
# dataloader (adapted from rmind-rqv offset_diag.build_dataloader)
# --------------------------------------------------------------------------- #


def build_dataloader(
    split: str = "val", batch_size: int = 16, num_workers: int = 3
) -> Any:
    overrides = [
        "experiment=yaak/control_transformer/finetune",
        # only cfg.datamodule is consumed; these just satisfy mandatory keys
        "model.artifact=placeholder/placeholder:v0",
        "action_tokenizer_artifact=placeholder/placeholder:v0",
    ]
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="train", overrides=overrides)

    cfg.paths.rbyte.cache = str(REPO_ROOT / ".rbyte_cache")

    node = cfg.datamodule.val if split == "val" else cfg.datamodule.train
    node.batch_size = batch_size
    node.num_workers = num_workers
    node.shuffle = False
    return instantiate(node)


def shutdown_dataloader(dataloader: Any) -> None:
    """Best-effort shutdown of TorchDataNodeDataLoader worker threads."""
    root = getattr(getattr(dataloader, "_loader", None), "root", None)
    stack: list[Any] = [root]
    seen: set[int] = set()
    while stack:
        node = stack.pop()
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        shutdown = getattr(node, "_shutdown", None)
        if callable(shutdown):
            shutdown()
        stack.extend(getattr(node, attr, None) for attr in ("_it", "source", "root"))


def _to_device(batch: Any, device: torch.device) -> Any:
    return tree_map(
        lambda x: x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x,
        batch,
    )


# --------------------------------------------------------------------------- #
# model loading
# --------------------------------------------------------------------------- #


def _features_waypoint_autodetect(
    self: JointPolicyObjective, episode: Any, embedding: Tensor
) -> Tensor:
    """Port of rmind-decode-opt d80929e: feature width follows the code head.

    The zcglgth4 checkpoint was trained with waypoints in the policy head
    (code_head in_channels=1152 = 3*d), but the current branch's _features
    returns only 2*d (768). Selects 2 vs 3 summary tokens by inspecting the
    code head's first Linear input width.
    """
    if self.norm is not None:
        embedding = self.norm(embedding)

    embeddings = (
        episode.index[-1]
        .select(
            (Modality.SUMMARY, SummaryToken.OBSERVATION_HISTORY),
            (Modality.SUMMARY, SummaryToken.OBSERVATION_SUMMARY),
            (Modality.CONTEXT, "waypoints"),
        )
        .parse(embedding)
    )

    observation_history = embeddings.get((
        Modality.SUMMARY,
        SummaryToken.OBSERVATION_HISTORY,
    ))
    observation_summary = embeddings.get((
        Modality.SUMMARY,
        SummaryToken.OBSERVATION_SUMMARY,
    ))

    tokens = [observation_summary, observation_history]
    d = observation_summary.shape[-1]
    head_in = next(
        m.in_features
        for m in self.code_head.modules()
        if isinstance(m, torch.nn.Linear)
    )
    if head_in == 3 * d:
        tokens.append(
            embeddings.get((Modality.CONTEXT, "waypoints")).mean(dim=1, keepdim=True)
        )

    from einops import rearrange  # noqa: PLC0415

    return rearrange(tokens, "i b 1 d -> b (i d)")


def load_policy(device: str | torch.device = "cuda") -> ControlTransformer:
    """Load the finetuned policy checkpoint with explicit compat handling.

    hparams_jq rewrites:
    - tokenizer: wandb-artifact reference -> local checkpoint (no network)
    - teacher_force_offset: removed (param no longer exists on this branch)
    - encoder.disable: true (already true in this ckpt; forced so forward
      hooks fire eagerly, never through torch.compile)
    """
    jq_prog = (
        '.objectives.modules.policy.tokenizer = {"_target_": '
        '"rmind.models.action_tokenizer.ActionTokenizer.load_from_checkpoint", '
        f'"checkpoint_path": "{TOKENIZER_CKPT}", '
        '"map_location": "cpu", "weights_only": false} '
        "| del(.objectives.modules.policy.teacher_force_offset) "
        "| .encoder.disable = true"
    )

    try:
        model = ControlTransformer.load_from_checkpoint(
            POLICY_CKPT,
            map_location="cpu",
            weights_only=False,
            hparams_jq=jq_prog,
            strict=True,
        )
        print("[load] strict=True load OK (no missing/unexpected keys)")
    except RuntimeError as err:
        print(f"[load] strict=True failed: {err}\n[load] retrying strict=False")
        model = ControlTransformer.load_from_checkpoint(
            POLICY_CKPT,
            map_location="cpu",
            weights_only=False,
            hparams_jq=jq_prog,
            strict=False,
        )
        ckpt = torch.load(POLICY_CKPT, map_location="meta", weights_only=False)
        ckpt_keys = set(ckpt["state_dict"])
        model_keys = set(model.state_dict())
        print(f"[load] missing (in model, not ckpt): {sorted(model_keys - ckpt_keys)}")
        print(f"[load] unexpected (in ckpt, not model): {sorted(ckpt_keys - model_keys)}")

    model = model.to(torch.device(device)).eval().requires_grad_(requires_grad=False)

    # make sure the encoder is eager (hooks must fire); undo any compile wrap
    encoder = model.encoder
    if getattr(encoder, "_compiled_call_impl", None) is not None:
        encoder._compiled_call_impl = None  # noqa: SLF001
        print("[load] cleared encoder._compiled_call_impl (was compiled)")
    if not isinstance(encoder, TransformerEncoder):
        msg = f"encoder is {type(encoder).__name__}, expected TransformerEncoder"
        raise TypeError(msg)

    policy = cast("JointPolicyObjective", model.objectives["policy"])
    if not isinstance(policy, JointPolicyObjective):
        msg = f"objectives['policy'] is {type(policy).__name__}"
        raise TypeError(msg)

    head_in = next(
        m.in_features for m in policy.code_head.modules()
        if isinstance(m, torch.nn.Linear)
    )
    print(f"[load] code_head in_channels = {head_in}")
    if head_in == 1152:
        # monkeypatch on the INSTANCE only; tracked source stays untouched
        import types  # noqa: PLC0415

        policy._features = types.MethodType(_features_waypoint_autodetect, policy)  # noqa: SLF001
        print("[load] patched policy._features (waypoint auto-detect, d80929e port)")

    policy.sample_codes = False  # argmax decode, matches export path

    quantizer = policy.tokenizer.quantizer
    if (quantizer.num_quantizers, quantizer.codebook_size) != (4, 16):
        msg = (
            f"unexpected quantizer geometry G={quantizer.num_quantizers} "
            f"C={quantizer.codebook_size}"
        )
        raise ValueError(msg)

    return model


# --------------------------------------------------------------------------- #
# encoder-block hooks
# --------------------------------------------------------------------------- #


class InterventionState:
    """Shared mutable state read by the per-block forward hooks."""

    def __init__(self) -> None:
        self.mode: str | None = None
        self.perm: Tensor | None = None
        self.fires: int = 0


def _make_block_hook(state: InterventionState):  # noqa: ANN202
    def hook(_module: Any, _args: Any, output: Tensor) -> Tensor | None:
        state.fires += 1
        if output.ndim != 4:  # noqa: PLR2004
            msg = f"expected (b,t,s,d) block output, got {tuple(output.shape)}"
            raise AssertionError(msg)
        b, t, s, _d = output.shape
        if s != N_SLOTS or t != N_TIMESTEPS:
            msg = f"expected s={N_SLOTS}, t={N_TIMESTEPS}; got s={s}, t={t}"
            raise AssertionError(msg)

        mode = state.mode
        if mode is None:
            return None

        out = output.clone()
        if mode == "permute_foresight_all_t":
            out[:, :, FORESIGHT] = output[state.perm][:, :, FORESIGHT]
        elif mode == "permute_foresight_last_t":
            out[:, -1, FORESIGHT] = output[state.perm][:, -1, FORESIGHT]
        elif mode == "reset_foresight_last_t":
            out[:, -1, FORESIGHT] = output[:, -1, FORESIGHT].mean(dim=0, keepdim=True)
        elif mode == "control_permute_speed":
            out[:, :, SLOT_SPEED] = output[state.perm][:, :, SLOT_SPEED]
        else:
            msg = f"unknown intervention mode {mode!r}"
            raise AssertionError(msg)
        return out

    return hook


def _fixed_perm(b: int, device: torch.device) -> Tensor:
    g = torch.Generator().manual_seed(PERM_SEED)
    perm = torch.randperm(b, generator=g)
    if b > 1 and bool((perm == torch.arange(b)).all()):
        perm = torch.roll(torch.arange(b), 1)
    return perm.to(device)


# --------------------------------------------------------------------------- #
# per-batch forward paths
# --------------------------------------------------------------------------- #


def _cut_mask(episode: Any) -> FactorizedAttentionMask:
    """Spatial-mask surgery: obs_summary/obs_history may not attend foresight."""
    mask = episode.attention_mask
    legend = mask.spatial.legend
    if legend.DO_NOT_ATTEND is not True:
        msg = f"unexpected mask legend {legend}"
        raise AssertionError(msg)
    spatial = mask.spatial.mask_tensor.clone()
    spatial[SLOT_OBS_SUMMARY : SLOT_OBS_HISTORY + 1, FORESIGHT] = True  # DO_NOT_ATTEND
    return FactorizedAttentionMask.from_tensors(
        spatial_mask_tensor=spatial,
        temporal_mask_tensor=mask.temporal.mask_tensor,
        legend=TorchAttentionMaskLegend,
    )


def _decode(
    policy: JointPolicyObjective, episode: Any, embedding: Tensor
) -> tuple[Tensor, Tensor]:
    """(codes (b,4) int64, decoded chunk (b,6,4)) with argmax codes."""
    features = policy._features(episode, embedding)  # noqa: SLF001
    _, codes, offset = policy._predict(features)  # noqa: SLF001
    chunk = (policy.tokenizer.invert(codes) + offset).unflatten(
        -1, (-1, policy.tokenizer._action_features)  # noqa: SLF001
    )
    return codes, chunk


# --------------------------------------------------------------------------- #
# validations (first batch)
# --------------------------------------------------------------------------- #


def _assert_slot_layout(episode: Any) -> None:
    i0 = episode.index[0]
    fs = i0.foresight["cam_front_left"].flatten().cpu()
    if not torch.equal(fs, torch.arange(FORESIGHT.start, FORESIGHT.stop)):
        msg = f"foresight slots {fs.min()}..{fs.max()} != 267..522"
        raise AssertionError(msg)
    obs_sum = int(i0.summary["observation_summary"].flatten()[0])
    obs_hist = int(i0.summary["observation_history"].flatten()[0])
    speed = int(i0.continuous["speed"].flatten()[0])
    if (obs_sum, obs_hist, speed) != (SLOT_OBS_SUMMARY, SLOT_OBS_HISTORY, SLOT_SPEED):
        msg = f"slot layout mismatch: {(obs_sum, obs_hist, speed)}"
        raise AssertionError(msg)
    i1 = episode.index[1]
    if int(i1.continuous["speed"].flatten()[0]) != SLOT_SPEED + N_SLOTS:
        msg = "timestep stride != 530"
        raise AssertionError(msg)
    print("[validate] slot layout OK (foresight=267:523, speed=256, sum=523/524)")


def _validate_baseline_bit_identity(
    model: ControlTransformer,
    episode: Any,
    state: InterventionState,
    handles: list[Any],
    hook_fn: Any,
) -> None:
    state.mode = None
    state.fires = 0
    emb_hooked = model.encoder(
        src=episode.embeddings_flattened, mask=episode.attention_mask
    )
    n_layers = len(model.encoder.layers)
    if state.fires != n_layers:
        msg = f"hooks fired {state.fires} times, expected {n_layers}"
        raise AssertionError(msg)

    for h in handles:
        h.remove()
    emb_unhooked = model.encoder(
        src=episode.embeddings_flattened, mask=episode.attention_mask
    )
    handles.clear()
    handles.extend(
        blk.register_forward_hook(hook_fn) for blk in model.encoder.layers
    )

    if not torch.equal(emb_hooked, emb_unhooked):
        msg = "baseline with no-op hooks is NOT bit-identical to unhooked model"
        raise AssertionError(msg)
    print(f"[validate] baseline bit-identity OK; hooks fired {n_layers}/{n_layers}")


def _validate_cut_blinds(
    model: ControlTransformer, batch: Any, episode: Any
) -> None:
    """Under cut mask, permuting images must leave summary outputs unchanged."""
    mask = _cut_mask(episode)
    emb_a = model.encoder(src=episode.embeddings_flattened, mask=mask)

    b = episode.embeddings_flattened.shape[0]
    perm = torch.roll(torch.arange(b), 1)
    batch_perm = tree_map(lambda x: x, batch)  # shallow-ish copy of the tree
    cam = batch_perm["data"]["cam_front_left"]
    batch_perm["data"]["cam_front_left"] = cam[perm.to(cam.device)]
    episode_perm = model.episode_builder(batch_perm)
    emb_b = model.encoder(src=episode_perm.embeddings_flattened, mask=mask)

    sel = episode.index.select(
        (Modality.SUMMARY, SummaryToken.OBSERVATION_SUMMARY),
        (Modality.SUMMARY, SummaryToken.OBSERVATION_HISTORY),
    )
    td_a, td_b = sel.parse(emb_a), sel.parse(emb_b)
    max_diff = 0.0
    for key in (
        (Modality.SUMMARY, SummaryToken.OBSERVATION_SUMMARY),
        (Modality.SUMMARY, SummaryToken.OBSERVATION_HISTORY),
    ):
        max_diff = max(
            max_diff, float((td_a.get(key) - td_b.get(key)).abs().max())
        )
    if max_diff >= 1e-5:  # noqa: PLR2004
        msg = f"cut arm leaks: summary outputs differ by {max_diff:.3e} >= 1e-5"
        raise AssertionError(msg)
    print(f"[validate] cut_summary_foresight blinds policy OK (max diff {max_diff:.3e})")


def _validate_policy_metrics(
    model: ControlTransformer, episode: Any, embedding: Tensor
) -> dict[str, float]:
    policy = cast("JointPolicyObjective", model.objectives["policy"])
    metrics = policy.compute_metrics(episode=episode, embedding=embedding)
    losses = {k: float(v) for k, v in metrics["loss"].items()}
    for name, val in losses.items():
        if not math.isfinite(val):
            msg = f"non-finite loss {name}={val}"
            raise AssertionError(msg)
    return losses


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #


def _conflict(chunk: Tensor) -> tuple[float, float]:
    """(rate at chunk step 0, rate over all b*6 steps)."""
    conf = (chunk[..., 0] > GAS_THRESH) & (chunk[..., 1] > BRAKE_THRESH)  # (n, 6)
    return float(conf[:, 0].float().mean()), float(conf.float().mean())


def _arm_metrics(
    codes_base: Tensor, chunk_base: Tensor, codes_arm: Tensor, chunk_arm: Tensor
) -> dict[str, Any]:
    flips = codes_base != codes_arm  # (n, 4)
    delta = (chunk_arm - chunk_base).abs()  # (n, 6, 4)
    conf0, conf_all = _conflict(chunk_arm)
    return {
        "n": int(codes_base.shape[0]),
        "code_flip_rate": {
            **{f"q{q}": float(flips[:, q].float().mean()) for q in range(4)},
            "full_tuple": float(flips.any(dim=-1).float().mean()),
        },
        "action_abs_delta_step0": {
            f: float(delta[:, 0, i].mean()) for i, f in enumerate(FIELDS)
        },
        "action_abs_delta_mean_steps": {
            f: float(delta[:, :, i].mean()) for i, f in enumerate(FIELDS)
        },
        "conflict_rate_step0": conf0,
        "conflict_rate_all_steps": conf_all,
    }


def _strata_masks(
    speed5: Tensor, brake5: Tensor, steer5: Tensor
) -> tuple[dict[str, Tensor], float]:
    steer_p80 = float(steer5.abs().quantile(STEER_TURN_QUANTILE))
    stopped = speed5 < SPEED_STOPPED
    braking = ~stopped & (brake5 > BRAKE_THRESH)
    turning = ~stopped & ~braking & (steer5.abs() > steer_p80)
    cruising = ~stopped & ~braking & ~turning
    masks = {
        "all": torch.ones_like(stopped),
        "stopped": stopped,
        "braking": braking,
        "turning": turning,
        "cruising": cruising,
    }
    return masks, steer_p80


# --------------------------------------------------------------------------- #
# main loop
# --------------------------------------------------------------------------- #


def _iter_batches(dataloader: Any, max_samples: int) -> Iterator[Any]:
    n = 0
    for batch in dataloader:
        yield batch
        n += batch["data"]["cam_front_left"].shape[0]
        if n >= max_samples:
            return


def run(args: argparse.Namespace) -> dict[str, Any]:  # noqa: PLR0915, PLR0914, C901
    device = torch.device(args.device)
    arms = (
        list(ALL_ARMS)
        if args.arms == "all"
        else [a.strip() for a in args.arms.split(",") if a.strip()]
    )
    for arm in arms:
        if arm not in ALL_ARMS:
            msg = f"unknown arm {arm!r}; choose from {ALL_ARMS}"
            raise ValueError(msg)
    if "baseline" not in arms:
        arms.insert(0, "baseline")

    model = load_policy(device)
    policy = cast("JointPolicyObjective", model.objectives["policy"])

    state = InterventionState()
    hook_fn = _make_block_hook(state)
    handles = [blk.register_forward_hook(hook_fn) for blk in model.encoder.layers]
    for blk in model.encoder.layers:
        if not isinstance(blk, FactorizedTransformerEncoderBlock):
            msg = f"unexpected block type {type(blk).__name__}"
            raise TypeError(msg)

    dataloader = build_dataloader(
        "val", batch_size=args.batch_size, num_workers=args.num_workers
    )

    collected: dict[str, dict[str, list[Tensor]]] = {
        arm: {"codes": [], "chunk": []} for arm in arms
    }
    sig: dict[str, list[Tensor]] = {"speed": [], "brake": [], "steer": []}
    input_ids: list[str] = []
    baseline_losses: list[dict[str, float]] = []

    n_done = 0
    n_batches = 0
    t_start = time.perf_counter()

    try:
        with torch.inference_mode():
            for batch_idx, batch_cpu in enumerate(_iter_batches(dataloader, args.max_samples)):
                batch = _to_device(batch_cpu, device)
                episode = model.episode_builder(batch)
                b = episode.embeddings_flattened.shape[0]
                state.perm = _fixed_perm(b, device)

                if batch_idx == 0:
                    _assert_slot_layout(episode)
                    _validate_baseline_bit_identity(
                        model, episode, state, handles, hook_fn
                    )
                    _validate_cut_blinds(model, batch, episode)

                per_arm_emb: dict[str, Tensor] = {}
                for arm in arms:
                    if arm == "cut_summary_foresight":
                        state.mode = None
                        emb = model.encoder(
                            src=episode.embeddings_flattened, mask=_cut_mask(episode)
                        )
                    else:
                        state.mode = None if arm == "baseline" else arm
                        emb = model.encoder(
                            src=episode.embeddings_flattened,
                            mask=episode.attention_mask,
                        )
                        state.mode = None
                    per_arm_emb[arm] = emb

                if batch_idx < 3:  # noqa: PLR2004
                    baseline_losses.append(
                        _validate_policy_metrics(model, episode, per_arm_emb["baseline"])
                    )
                    if batch_idx == 2:  # noqa: PLR2004
                        mean_off = sum(d["offset"] for d in baseline_losses) / 3
                        if not (
                            OFFSET_LOSS_RANGE[0] <= mean_off <= OFFSET_LOSS_RANGE[1]
                        ):
                            msg = (
                                f"baseline offset loss {mean_off:.4f} outside "
                                f"expected {OFFSET_LOSS_RANGE}"
                            )
                            raise AssertionError(msg)
                        print(
                            f"[validate] baseline offset loss {mean_off:.4f} "
                            f"in {OFFSET_LOSS_RANGE} OK; code losses "
                            f"{ {k: round(v, 3) for k, v in baseline_losses[-1].items()} }",
                            flush=True,
                        )

                for arm in arms:
                    codes, chunk = _decode(policy, episode, per_arm_emb[arm])
                    collected[arm]["codes"].append(codes.cpu())
                    collected[arm]["chunk"].append(chunk.float().cpu())

                data = batch["data"]
                sig["speed"].append(data["meta/VehicleMotion/speed"][:, 5].float().cpu())
                sig["brake"].append(
                    data["meta/VehicleMotion/brake_pedal_normalized"][:, 5].float().cpu()
                )
                sig["steer"].append(
                    data["meta/VehicleMotion/steering_angle_normalized"][:, 5]
                    .float()
                    .cpu()
                )
                meta = batch_cpu.get("meta", {}) if isinstance(batch_cpu, dict) else {}
                ids = meta.get("input_id")
                if ids is not None:
                    input_ids.extend(str(i) for i in ids)

                n_done += b
                n_batches += 1
                if n_batches % 20 == 0:
                    dt = time.perf_counter() - t_start
                    print(
                        f"[run] {n_done} samples in {dt:.1f}s "
                        f"({dt / n_batches:.2f}s/batch)",
                        flush=True,
                    )
    finally:
        shutdown_dataloader(dataloader)

    elapsed = time.perf_counter() - t_start
    n = min(n_done, args.max_samples)

    cat = {
        arm: {
            "codes": torch.cat(collected[arm]["codes"])[:n],
            "chunk": torch.cat(collected[arm]["chunk"])[:n],
        }
        for arm in arms
    }
    speed5 = torch.cat(sig["speed"])[:n]
    brake5 = torch.cat(sig["brake"])[:n]
    steer5 = torch.cat(sig["steer"])[:n]
    masks, steer_p80 = _strata_masks(speed5, brake5, steer5)

    # offset-loss sanity check (validation, not assertion-fatal thresholds are wide)
    mean_offset = sum(d["offset"] for d in baseline_losses) / len(baseline_losses)
    if not (OFFSET_LOSS_RANGE[0] <= mean_offset <= OFFSET_LOSS_RANGE[1]):
        msg = (
            f"baseline offset loss {mean_offset:.4f} outside expected "
            f"{OFFSET_LOSS_RANGE} -- model/dataloader mismatch?"
        )
        raise AssertionError(msg)
    print(f"[validate] baseline offset loss {mean_offset:.4f} in {OFFSET_LOSS_RANGE} OK")

    results: dict[str, Any] = {
        "meta": {
            "checkpoint": str(POLICY_CKPT),
            "tokenizer": str(TOKENIZER_CKPT),
            "n_samples": n,
            "n_batches": n_batches,
            "batch_size": args.batch_size,
            "perm_seed": PERM_SEED,
            "gas_thresh": GAS_THRESH,
            "brake_thresh": BRAKE_THRESH,
            "steer_p80_abs": steer_p80,
            "strata_sizes": {k: int(v.sum()) for k, v in masks.items()},
            "baseline_offset_loss_first3": mean_offset,
            "baseline_losses_first3": baseline_losses,
            "elapsed_s": elapsed,
            "sec_per_batch": elapsed / max(n_batches, 1),
        },
        "arms": {},
    }

    base = cat["baseline"]
    for arm in arms:
        results["arms"][arm] = {}
        for stratum, mask in masks.items():
            if int(mask.sum()) == 0:
                results["arms"][arm][stratum] = {"n": 0}
                continue
            results["arms"][arm][stratum] = _arm_metrics(
                base["codes"][mask],
                base["chunk"][mask],
                cat[arm]["codes"][mask],
                cat[arm]["chunk"][mask],
            )

    _print_table(results, arms)
    return results


def _print_table(results: dict[str, Any], arms: list[str]) -> None:
    print("\n=== L1 policy-leverage summary (stratum: all) ===")
    hdr = (
        f"{'arm':<26} {'flip_q0..q3':<24} {'full':>6} "
        f"{'|dGas|':>8} {'|dBrk|':>8} {'|dStr|':>8} {'|dTrn|':>8} {'conf0':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for arm in arms:
        m = results["arms"][arm]["all"]
        f = m["code_flip_rate"]
        d = m["action_abs_delta_step0"]
        qs = "/".join(f"{f[f'q{q}']:.3f}" for q in range(4))
        print(
            f"{arm:<26} {qs:<24} {f['full_tuple']:>6.3f} "
            f"{d['gas']:>8.4f} {d['brake']:>8.4f} {d['steer']:>8.4f} "
            f"{d['turn']:>8.4f} {m['conflict_rate_step0']:>7.4f}"
        )
    print("\nstrata sizes:", results["meta"]["strata_sizes"])
    print(f"steer |.| p80: {results['meta']['steer_p80_abs']:.4f}")
    print(f"sec/batch: {results['meta']['sec_per_batch']:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", default="all")
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument(
        "--out", default=str(REPO_ROOT / "foresight_mm" / "results" / "l1_leverage.json")
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    try:
        results = run(args)
    except torch.cuda.OutOfMemoryError:
        print("[oom] retrying once with half batch size", flush=True)
        torch.cuda.empty_cache()
        args.batch_size = max(args.batch_size // 2, 1)
        results = run(args)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_forkserver_preload(["rbyte", "polars"])

    import os
    import sys

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    main()

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)  # noqa: SLF001
