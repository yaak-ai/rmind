"""Standalone ``PatchPolicy`` -> ONNX exporter for the deployment/serving contract.

``rmind.models.patch_policy.PatchPolicy`` has no hydra export config
(``config/export/yaak/control_transformer/finetuned.yaml`` only targets
``ControlTransformer`` -- a different class with a different constructor
signature and input layout). This bypasses rmind's hydra ``export_onnx``
entrypoint entirely, mirroring ``rmind.scripts.decoder_only_export``'s
``--mode baseline`` path but shaped for serving rather than architecture/
latency comparison: one image input per camera plus
``speed``/``waypoints_xy_normalized``, and per-channel outputs
(``policy.continuous.{gas_pedal,brake_pedal,steering_angle}``,
``policy.discrete.turn_signal`` bucketized to ``{0,1,2}``) for the newest
frame's immediate action -- matching an ONNX-serving agent's existing
per-channel output branch, so no client-side unpacking is needed.

Handles TWO checkpoint schemas, auto-detected off the loaded model:

- current HEAD (``cameras: tuple[str, ...]`` hparam, ``PatchPolicy.
  load_for_export``, ``forward(..., require_chunk=False)``) -- loads and
  exports directly.
- pre-``ead564b4`` (2026-08-12) checkpoints (``image: Path`` hparam, no
  ``load_for_export``, ``forward`` always needs a `chunk` the transform
  builds from raw pedal/turn-signal telemetry) -- these need PatchPolicy AS
  IT WAS at the checkpoint's own training commit, which current HEAD's class
  can no longer construct (rejects the old hparams). Pass ``--commit`` (the
  wandb run's recorded git commit -- ``wandb.Api().run(...).commit``) and the
  script re-execs itself under an auto-managed `git worktree` pinned to that
  commit, with PYTHONPATH pointing there so `import rmind` resolves to the
  OLD source tree -- but still using the CURRENT venv (torch/hydra/pydantic
  APIs are unchanged; only rmind's own model schema drifted, and a fresh `uv
  sync` on an old lockfile is liable to fail rebuilding native deps like
  `ptars` against today's toolchain).

    python -m rmind.scripts.export_patch_policy_onnx \\
        --artifact yaak/rmind/model-<run_id>:best \\
        --out /path/to/out.onnx

    # pre-ead564b4 checkpoint: pin to its training commit
    python -m rmind.scripts.export_patch_policy_onnx \\
        --artifact yaak/alex-tmp/model-9xfz6ify:v6 \\
        --out /path/to/out.onnx \\
        --commit 216a6ed73e91be35f7be45e3e4e56603b2cc82d0
"""

import argparse
import os
import subprocess  # noqa: S404
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKTREE_ROOT = Path.home() / ".cache" / "rmind-export-worktrees"
_PINNED_COMMIT_ENV = "_RMIND_EXPORT_PINNED_COMMIT"


def _reexec_under_commit(commit: str) -> None:
    """Re-exec this exact script file with PYTHONPATH pointed at `commit`'s tree.

    A worktree, not a second venv: reusing the CURRENT venv (just remapping
    which `rmind` source `import rmind` finds first) sidesteps `uv sync`
    trying to rebuild native deps (e.g. `ptars`) against an old lockfile,
    which has failed here against today's toolchain/glibc.
    """
    worktree = WORKTREE_ROOT / commit[:12]
    if not worktree.exists():
        WORKTREE_ROOT.mkdir(parents=True, exist_ok=True)
        subprocess.run(  # noqa: S603
            ["git", "worktree", "add", "--detach", str(worktree), commit],  # noqa: S607
            cwd=REPO_ROOT,
            check=True,
        )

    env = os.environ.copy()
    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{worktree / 'src'}{os.pathsep}{old_pythonpath}"
        if old_pythonpath
        else str(worktree / "src")
    )
    env[_PINNED_COMMIT_ENV] = commit
    os.execve(sys.executable, [sys.executable, __file__, *sys.argv[1:]], env)  # noqa: S606


# `--commit` must be resolved and (if needed) re-exec'd BEFORE any `rmind`
# import below, so that import binds to the pinned tree's source instead.
if (
    __name__ == "__main__"
    and "--commit" in sys.argv
    and _PINNED_COMMIT_ENV not in os.environ
):
    _commit = sys.argv[sys.argv.index("--commit") + 1]
    _reexec_under_commit(_commit)


import torch  # noqa: E402
import torch.fx.experimental._config as _fx_config  # noqa: E402, PLC2701
from structlog import get_logger  # noqa: E402
from tensordict import TensorDict  # noqa: E402
from torch import Tensor, nn  # noqa: E402
from torch.utils._pytree import tree_flatten_with_path  # noqa: E402, PLC2701

from rmind.models.patch_policy import PatchPolicy  # noqa: E402
from rmind.utils.patch import monkeypatched  # noqa: E402

# tensordict's global dicts mutated during export tracing cause a spurious
# "pending unbacked symbol u0" error even though the exported graph is valid
# (same workaround as rmind.scripts.decoder_only_export).
_fx_config.soft_pending_unbacked_not_found_error = True  # ty:ignore[invalid-assignment]

logger = get_logger(__name__)

NUM_WAYPOINTS = 10
DEFAULT_CAMERAS = ("cam_front_left",)


class PatchPolicyOnnxWrapper(nn.Module):
    """Adapts a current-schema ``PatchPolicy`` to a per-channel ONNX contract.

    Camera order comes from the checkpoint's own ``model.cameras`` hparam --
    ``PatchPolicy._features`` stacks ``image_by_camera[camera] for camera in
    self.cameras``, so this must stay in that exact order (it does, by
    construction, since both read the same attribute).
    """

    def __init__(self, model: PatchPolicy) -> None:
        super().__init__()
        self.model = model
        self.cameras = model.cameras

    def forward(self, *args: Tensor) -> TensorDict:
        n = len(self.cameras)  # ty:ignore[invalid-argument-type]
        camera_images, (speed, waypoints) = args[:n], args[n:]
        batch = {
            "data": {
                **dict(  # ty:ignore[no-matching-overload]
                    zip(self.cameras, camera_images, strict=True)  # ty:ignore[invalid-argument-type]
                ),
                "meta/VehicleMotion/speed": speed,
                "waypoints/xy_normalized": waypoints,
            }
        }
        out = self.model(batch)
        # `forward` already predicts from the newest frame only (`features[:,
        # -1]`); this last index picks the immediate action out of the
        # `action_horizon` chunk.
        joint_actions = out["policy", "joint_actions"][:, 0]  # (b, 4)
        return _structure(joint_actions)


class LegacyPatchPolicyOnnxWrapper(nn.Module):
    """Adapts a pre-``ead564b4`` ``PatchPolicy`` (single ``image: Path`` camera,
    no ``require_chunk``) to the same per-channel ONNX contract.

    That vintage's ``forward`` unconditionally fetches (and discards) an
    action chunk that ``input_transform`` builds from raw pedal/turn-signal
    telemetry via ``ChunkFields`` (episode_length/action_horizon-windowed) --
    there is no way to opt out. So this pads the caller's ``episode_length``
    raw ticks up to what ``ChunkFields`` needs by repeating the last tick
    (inert: the built chunk is discarded downstream, only its shape matters),
    and bypasses the checkpoint's own crop/resize/normalize image pipeline
    with `nn.Identity` -- the caller is expected to already supply
    cropped/resized `[0, 1]` frames, same convention as current HEAD's
    ``load_for_export``.
    """

    def __init__(self, model: PatchPolicy, cameras: tuple[str, ...]) -> None:
        super().__init__()
        model.input_transform[2]["image"] = nn.Identity()
        model.sample_codes = False
        self.model = model
        self.cameras = cameras
        chunk_fields = next(
            m for m in model.input_transform.children() if hasattr(m, "action_horizon")
        )
        self.pad = chunk_fields.action_horizon - 1

    def _pad(self, x: Tensor) -> Tensor:
        return (
            torch.cat([x, x[:, -1:].expand(-1, self.pad, -1)], dim=1) if self.pad else x
        )

    def forward(self, *args: Tensor) -> TensorDict:
        n = len(self.cameras)
        camera_images, rest = args[:n], args[n:]
        brake_pedal, gas_pedal, steering_angle, speed, turn_signal, waypoints = rest
        batch = {
            "data": {
                **dict(zip(self.cameras, camera_images, strict=True)),
                "meta/VehicleMotion/speed": speed,
                "meta/VehicleMotion/gas_pedal_normalized": self._pad(gas_pedal),
                "meta/VehicleMotion/brake_pedal_normalized": self._pad(brake_pedal),
                "meta/VehicleMotion/steering_angle_normalized": self._pad(
                    steering_angle
                ),
                "meta/VehicleState/turn_signal": self._pad(turn_signal),
                "waypoints/xy_normalized": waypoints,
            }
        }
        out = self.model(batch)
        joint_actions = out["policy", "joint_actions"][:, 0]  # (b, 4): immediate action
        return _structure(joint_actions)


def _structure(joint_actions: Tensor) -> TensorDict:
    """`(b, 4)` normalized chunk -> per-channel output, `PatchPolicy._structure`'s
    on-graph twin. `tokenizer.invert()` lands gas/brake/steer directly in
    physical units; turn_signal comes out `{0, 0.5, 1}` and needs the same
    `*2` + bucketize `PatchPolicy._structure`/`OnnxOutputUnpacker` use to
    recover the categorical `{0,1,2}` encoding.
    """
    turn_signal_bucket = torch.bucketize(
        joint_actions[..., 3] * 2, torch.tensor([0.5, 1.5], device=joint_actions.device)
    )
    return TensorDict({
        "policy": {
            "continuous": TensorDict({
                "gas_pedal": joint_actions[..., 0],
                "brake_pedal": joint_actions[..., 1],
                "steering_angle": joint_actions[..., 2],
            }),
            "discrete": TensorDict({"turn_signal": turn_signal_bucket}),
        }
    })  # ty:ignore[invalid-argument-type]


def _guard_kmeans_init() -> None:
    """No-op `vector_quantize_pytorch.Codebook.init_embed_` for export.

    Only needed on the LEGACY path: current HEAD's `rmind.components.vq.
    ResidualVQ` already guards this itself (gates on `codebook.training`, a
    Python bool that's a constant under `torch.export`, instead of the
    tensor-typed `if self.initted:` upstream uses -- a data-dependent branch
    `torch.export` refuses to trace). The goal/action tokenizers are frozen
    and already initialized, so no-opping this changes nothing about the
    exported weights.
    """
    from vector_quantize_pytorch.vector_quantize_pytorch import (  # noqa: PLC0415
        Codebook,
    )

    Codebook.init_embed_ = lambda self, *args, **kwargs: None  # noqa: ARG005


def _image_size(model: PatchPolicy, *, default: int = 224) -> int:
    try:
        size = model.hparams["image_encoder"]["_args_"][0]["img_size"]
    except (KeyError, IndexError, TypeError):
        return default
    return size[0] if isinstance(size, list | tuple) else int(size)


def _episode_length(model: PatchPolicy, *, default: int = 6) -> int:
    """The raw per-field tick count `input_transform`'s `ChunkFields` step needs.

    NOT the trunk's attention `window` -- `ChunkFields` unconditionally narrows
    every field (images included) to its own `episode_length`, baked in from
    the training config, *before* `require_chunk` is ever consulted (current
    schema) or at all (legacy schema, which has no such flag). Narrower dummy
    input than this raises inside `ChunkFields`, not a shape mismatch where
    you'd expect it.
    """
    for module in model.input_transform.children():
        length = getattr(module, "episode_length", None)
        if length is not None:
            return length
    return default


def _input_names(cameras: tuple[str, ...], *, legacy: bool) -> list[str]:
    if not legacy:
        return [*cameras, "speed", "waypoints_xy_normalized"]
    return [
        *cameras,
        "brake_pedal_normalized",
        "gas_pedal_normalized",
        "steering_angle_normalized",
        "speed",
        "turn_signal",
        "waypoints_xy_normalized",
    ]


def _build_dummy_args(
    cameras: tuple[str, ...], *, image_size: int, episode_length: int, legacy: bool
) -> tuple[Tensor, ...]:
    b, t = 1, episode_length
    images = tuple(torch.rand(b, t, 3, image_size, image_size) for _ in cameras)
    waypoints = torch.rand(b, t, NUM_WAYPOINTS, 2) * 2 - 1
    speed = torch.rand(b, t, 1) * 130
    if not legacy:
        return (*images, speed, waypoints)
    return (
        *images,
        torch.rand(b, t, 1),  # brake_pedal_normalized
        torch.rand(b, t, 1),  # gas_pedal_normalized
        torch.rand(b, t, 1) * 2 - 1,  # steering_angle_normalized
        speed,
        torch.randint(0, 3, (b, t, 1)).float() * 0.5,  # turn_signal
        waypoints,
    )


def export(  # noqa: PLR0913
    model: nn.Module,
    args: tuple[Any, ...],
    out: Path,
    *,
    cameras: tuple[str, ...],
    legacy: bool,
    verify: bool,
) -> None:
    eager_out: Any = None
    for patch in (False, True):
        with monkeypatched(obj=torch.compiler, name="_is_exporting_flag", patch=patch):
            eager_out = model(*args)
    paths_and_leaves, _ = tree_flatten_with_path(eager_out)
    output_names = [
        ".".join(mk.key for mk in path)  # ty:ignore[unresolved-attribute]
        for path, _ in paths_and_leaves
    ]
    logger.debug("inferred output_names", output_names=output_names)

    exported = torch.export.export(mod=model, args=tuple(args), strict=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model=exported,
        f=out,
        input_names=_input_names(cameras, legacy=legacy),
        output_names=output_names,
        dynamo=True,
        external_data=False,
        optimize=True,
        verify=verify,
        report=False,
        artifacts_dir=out.parent,
    )
    logger.info("exported", path=out.as_posix(), mb=round(out.stat().st_size / 1e6, 1))


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        required=True,
        help="wandb model artifact, e.g. yaak/rmind/model-<run_id>:vN",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--commit",
        default=None,
        help="pin rmind's PatchPolicy source to this commit for a pre-"
        "ead564b4 checkpoint (the wandb run's recorded git commit). Handled "
        "by a module-level re-exec before any `rmind` import -- see module "
        "docstring.",
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=None,
        help="camera names, in the checkpoint's `model.cameras`/`model.image` "
        "order. Required only for a multi-camera LEGACY (pre-cameras-rename) "
        "checkpoint, whose hparams don't self-describe this; ignored (with a "
        "warning) for a current-schema checkpoint, which always does.",
    )
    parser.add_argument(
        "--episode-length",
        type=int,
        default=None,
        help="raw per-field tick count. Defaults to `input_transform`'s own "
        "`ChunkFields.episode_length` (baked in from the training config) -- "
        "NOT the trunk's attention `window`, which only bounds how many of "
        "those ticks get attended.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="ORT-vs-eager check on the dummy inputs (weights are real here, "
        "but the inputs are random, so treat any flagged relative error as a "
        "cue to check with real data, not as a hard failure).",
    )
    args = parser.parse_args()

    if args.out.exists():
        logger.info("exists, skipping", path=args.out.as_posix())
        return

    torch.manual_seed(1337)
    legacy = not hasattr(PatchPolicy, "load_for_export")
    if legacy:
        _guard_kmeans_init()
        model = PatchPolicy.load_from_wandb_artifact(
            args.artifact, filename="model.ckpt", map_location="cpu", weights_only=False
        ).eval()
        cameras = tuple(args.cameras) if args.cameras else DEFAULT_CAMERAS
    else:
        model = PatchPolicy.load_for_export(args.artifact)
        cameras = model.cameras
        if args.cameras and tuple(args.cameras) != cameras:
            logger.warning(
                "ignoring --cameras: current-schema checkpoints self-describe "
                "this via model.cameras",
                requested=list(args.cameras),
                checkpoint=list(cameras),
            )

    episode_length = args.episode_length or _episode_length(model)
    image_size = _image_size(model)
    logger.info(
        "loaded",
        legacy=legacy,
        pinned_commit=os.environ.get(_PINNED_COMMIT_ENV),
        cameras=cameras,
        encoder=type(model.encoder).__name__,
        episode_length=episode_length,
        image_size=image_size,
        parameters_m=round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
    )

    wrapper: nn.Module = (
        LegacyPatchPolicyOnnxWrapper(model, cameras)
        if legacy
        else PatchPolicyOnnxWrapper(model)
    ).eval()
    dummy_args = _build_dummy_args(
        cameras, image_size=image_size, episode_length=episode_length, legacy=legacy
    )
    export(
        wrapper,
        dummy_args,
        args.out,
        cameras=cameras,
        legacy=legacy,
        verify=args.verify,
    )


if __name__ == "__main__":
    main()
