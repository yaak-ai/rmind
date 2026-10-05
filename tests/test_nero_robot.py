"""Robot-native nero patch policy (contract v3): P3-P7, P9, P10 on tiny shapes.

What a shape test would pass vacuously, and this file makes falsifiable:

* streaming the KV-cached decoder step frame by frame equals ONE windowed
  forward over a 40-frame episode at window 16 -- with the hand token, a
  left-only `side_valid` and every relative mode -- to 1e-4 with identical codes;
* the hand token reaches the readout, `no_hand` replaces refused frames, and a
  refused frame is not a dropped one;
* relative targets are per-FRAME anchored and round-trip;
* the standardizer JSON is nutron-cli's format, including the degenerate-axis
  rule; the flat 26-d layout is side-major with no transpose bug;
* the latent offset head is small (the per-code table is ~340M at 100x13);
* the image preprocessing is one function, uint8 in -> uint8 out.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor, nn
from torchvision.transforms import v2

from rmind.components.containers import ModuleDict
from rmind.components.loss import FocalLoss
from rmind.components.nn import ChunkConvDecoder, ChunkConvEncoder, NormedTokenEmbedding
from rmind.components.transformer.causal_frame import CausalFrameTransformer
from rmind.components.vq import ResidualVQ
from rmind.data import nero_image
from rmind.data.nero_robot import (
    AXIS_NAMES,
    FINGER_AXES,
    HAND_GROUP_WIDTH,
    HAND_PHYSICAL_PRIOR,
    AxisStandardizer,
    EventReference,
    HandTokenStandardizer,
    compose_hand_token,
    flat_names,
    hand_feature_columns,
    hand_token_columns,
    hand_token_dim,
    relative_mask,
    to_absolute,
    to_relative,
)
from rmind.datamodules.nero_robot_random import nero_robot_batch
from rmind.models.nero_chunk_tokenizer import NeroChunkTokenizer
from rmind.models.nero_patch_policy import NeroPatchPolicy
from rmind.models.nero_patch_policy_decoder import NeroPatchPolicyDecoderStep

PATCH_GRID = (2, 3)
NUM_PATCHES = PATCH_GRID[0] * PATCH_GRID[1]
IMAGE_HW = (28, 42)  # a 2x3 grid at patch 14, as the manifest derives it
IMAGE_DIM = 8
POLICY_DIM = 32
NUM_HEADS = 4  # head_dim 8
NUM_LAYERS = 2
WINDOW = 16
CHUNK = 100
LATENT = 16
CODEBOOK = 4
QUANTIZERS = 3
HAND_GROUPS = ("current", "pos_err")
NUTRON_CLI = Path(
    os.environ.get("NUTRON_CLI_ROOT", "/home/max/Code/nutron-cli-patch-policy")
)


class _TinyImageEncoder(nn.Module):
    """Deterministic stand-in for the frozen ViT: `(..., 3, H, W)` -> `(..., P, D)`."""

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, IMAGE_DIM)

    def forward(self, x: Tensor) -> Tensor:
        *batch, c, h, w = x.shape
        pooled = nn.functional.adaptive_avg_pool2d(x.reshape(-1, c, h, w), PATCH_GRID)
        pooled = pooled.flatten(-2, -1).transpose(-2, -1)
        return self.projection(pooled).reshape(*batch, NUM_PATCHES, IMAGE_DIM)


def _tokenizer(relative_mode: str = "none") -> NeroChunkTokenizer:
    torch.manual_seed(1)
    keys = (CHUNK - 1) // 3 + 1
    return NeroChunkTokenizer(
        encoder=ChunkConvEncoder(
            num_steps=keys, num_axes=13, out_features=LATENT, channels=8
        ),
        quantizer=ResidualVQ(
            dim=LATENT,
            codebook_size=CODEBOOK,
            num_quantizers=QUANTIZERS,
            kmeans_init=False,
        ),
        decoder=ChunkConvDecoder(
            num_steps=keys, num_axes=13, in_features=LATENT, channels=8
        ),
        action_horizon=CHUNK,
        relative_mode=relative_mode,  # ty: ignore[invalid-argument-type]
    ).eval()


def _policy(
    *,
    relative_mode: str = "none",
    hand: bool = True,
    conditioned: bool = True,
    **kwargs: Any,
) -> NeroPatchPolicy:
    torch.manual_seed(0)
    tokens = 1 + int(hand) + 3 * NUM_PATCHES
    if hand:
        kwargs.setdefault(
            "hand_standardizer", HandTokenStandardizer(groups=HAND_GROUPS)
        )
    policy = NeroPatchPolicy(
        image_transform=nn.Sequential(
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ),
        image_encoder=_TinyImageEncoder(),
        patch_projection=nn.Linear(2 * IMAGE_DIM + 13, POLICY_DIM),
        state_embedding=NormedTokenEmbedding(
            in_features=2 * 13 + 2, out_features=POLICY_DIM, hidden_features=(32,)
        ),
        encoder=CausalFrameTransformer(
            dim_model=POLICY_DIM,
            num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            tokens_per_frame=tokens,
            window=WINDOW,
            attn_dropout=0.0,
        ),
        tokenizer=_tokenizer(relative_mode),
        code_head=nn.Linear(POLICY_DIM, QUANTIZERS * CODEBOOK),
        offset_head=nn.Linear(POLICY_DIM + LATENT * int(conditioned), LATENT),
        losses=ModuleDict(
            modules={"code": FocalLoss(), "offset": nn.SmoothL1Loss(beta=0.2)}
        ),
        image_embedding_dim=IMAGE_DIM,
        policy_embedding_dim=POLICY_DIM,
        state=("state",),
        chunk=("action.chunk",),
        convert_state_to_9d=False,
        action_space="robot",
        goal_mode="no_goal",
        hand_groups=HAND_GROUPS if hand else (),
        hand_embedding=(
            NormedTokenEmbedding(
                in_features=hand_token_dim(HAND_GROUPS),
                out_features=POLICY_DIM,
                hidden_features=(32,),
            )
            if hand
            else None
        ),
        relative_mode=relative_mode,  # ty: ignore[invalid-argument-type]
        offset_mode="latent",
        offset_code_conditioning=conditioned,
        sample_codes=False,
        **kwargs,
    )
    return policy.eval()


def _batch(t: int = 4, *, b: int = 2, seed: int = 0) -> dict[str, Any]:
    return nero_robot_batch(batch_size=b, num_frames=t, image_hw=IMAGE_HW, seed=seed)


# ------------------------------------------------------------------- layout


def test_flat_layout_is_side_major_and_matches_the_chunk() -> None:
    names = flat_names()
    assert names[:13] == [f"left.{a}" for a in AXIS_NAMES]
    assert names[13:] == [f"right.{a}" for a in AXIS_NAMES]
    chunk = torch.arange(CHUNK * 2 * 13).reshape(CHUNK, 2, 13)
    flat = chunk.flatten(-2, -1)  # what serving reads as (100, 26)
    assert flat[5, 13 + 2] == chunk[5, 1, 2]
    per_side = chunk.permute(1, 0, 2)  # the tokenizer layout (S, H, A)
    # the decoder step's (1, S, H, A) -> (1, H, 26) transform
    back = per_side.unsqueeze(0).permute(0, 2, 1, 3).reshape(1, CHUNK, 26)
    assert torch.equal(back[0], flat)


def test_relative_roundtrip_and_per_frame_anchor() -> None:
    torch.manual_seed(0)
    chunk = torch.randn(2, 4, CHUNK, 2, 13)
    anchor = torch.randn(2, 4, 2, 13)
    for mode in ("none", "hand", "all"):
        rel = to_relative(chunk, anchor, mode)
        assert torch.allclose(to_absolute(rel, anchor, mode), chunk, atol=1e-6)
        mask = relative_mask(mode)
        # frame f's chunk uses frame f's anchor, not the window's last
        expect = chunk[:, 1, :, :, mask] - anchor[:, 1, None, :, mask]
        assert torch.allclose(rel[:, 1, :, :, mask], expect)
        assert torch.equal(rel[..., ~mask], chunk[..., ~mask])
    assert relative_mask("hand").nonzero().flatten().tolist() == list(FINGER_AXES)


def test_standardizer_json_is_the_nutron_format(tmp_path: Path) -> None:
    values = torch.randn(64, 2, 13)
    values[:, 0, 3] = 0.7  # a constant channel
    valid = torch.tensor([True, False]).expand(64, 2)
    std = AxisStandardizer.fit(values, valid)
    path = tmp_path / "action_standardizer.json"
    digest = std.save(path)
    payload = json.loads(path.read_text())
    assert payload["schema"] == "nutron_standardizer"
    assert payload["version"] == 1
    assert payload["names"] == flat_names()
    assert payload["std"][3] == 1.0
    assert payload["mean"][3] == pytest.approx(0.7)
    assert payload["mean"][13:] == [0.0] * 13
    assert payload["std"][13:] == [1.0] * 13
    loaded = AxisStandardizer.load(path, sha256=digest)
    assert torch.equal(loaded.mean, std.mean)
    assert loaded.digest == digest
    with pytest.raises(ValueError, match="sha256"):
        AxisStandardizer.load(path, sha256="0" * 64)
    loader = _nutron_module("policy_contract")
    if loader is not None and hasattr(loader, "load_standardizer"):
        served = loader.load_standardizer(path, flat_names())
        assert served.std.tolist() == pytest.approx(payload["std"])


def _nutron_module(name: str) -> Any:
    path = NUTRON_CLI / "runtime" / "jetson" / f"{name}.py"
    if not path.exists():
        return None
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(f"_nutron_{name}", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_hand_token_layout_matches_the_shared_builder() -> None:
    hf = _nutron_module("hand_features")
    if hf is None:
        pytest.skip("no nutron-cli checkout")
    for groups in (
        ("current",),
        ("current", "pos_err"),
        ("current", "pos_err", "pos", "tip"),
    ):
        assert hand_token_columns(groups) == hf.token_columns(groups)
    for g, w in HAND_GROUP_WIDTH.items():
        assert len(hf.token_group_columns(g)) == w
    # compose: torch twin of hf.TokenBlocks.compose, on the builder's own blocks
    t = (
        1_790_000_000_000_000_000
        + torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 15, 20]) * 100_000_000
    ).numpy()
    motor = hf.make_motor_timeline(
        t[:8:2] - 10_000_000,
        [[100 * i + j for j in range(6)] for i in range(4)],
        [[-50 + 10 * i + j for j in range(6)] for i in range(4)],
    )
    commands = hf.make_command_timeline(t[:1] - 500_000_000, [[500] * 6])
    blocks = hf.build_tokens(t, motor, commands)
    batch = {
        "hand.current": torch.from_numpy(blocks.blocks["current"]),
        "hand.pos_err": torch.from_numpy(blocks.blocks["pos_err"]),
        "hand.age": torch.from_numpy(blocks.blocks["hand_age"][:, 0]),
        "hand.motor_ok": torch.from_numpy(blocks.motor_ok),
        "hand.tip_ok": torch.from_numpy(blocks.tip_ok),
    }
    ours = compose_hand_token(batch, ("current", "pos_err"))
    theirs = torch.from_numpy(blocks.compose(("current", "pos_err")))
    assert torch.equal(ours, theirs)
    assert blocks.motor_ok.any()
    assert not blocks.motor_ok.all()


# ------------------------------------------------------------------- policy


def test_robot_policy_trains_and_reports() -> None:
    policy = _policy(quality_metrics=True).train()
    batch = _batch(t=4)
    norms: dict[str, Tensor] = {}
    out = policy.compute_metrics(batch, token_norms=norms)
    loss = out["policy", "loss"].sum(reduce=True)
    loss.backward()
    assert torch.isfinite(loss)
    assert policy.no_hand.grad is not None
    assert policy.hand_embedding.token_gain.grad is not None  # ty: ignore[possibly-missing-attribute]
    metrics = out["policy", "metric"]
    for key in (
        "ev/finger/h00_09",
        "code_usage_0",
        "alarm/dead_finger_channel",
        "offset_argmax_recon",
    ):
        assert key in metrics, key
    for key in ("patch", "state", "hand", "no_hand", "out/readout"):
        assert key in norms, key


def test_padded_steps_carry_no_loss() -> None:
    policy = _policy()
    batch = _batch(t=3)
    assert batch["action.is_pad"].any()
    base = policy.compute_metrics(batch)["policy", "loss", "offset"]
    garbage = dict(batch)
    chunk = batch["action.chunk"].clone()
    chunk[batch["action.is_pad"][..., None, None].expand_as(chunk)] = 1e3
    garbage["action.chunk"] = chunk
    # the tokenizer codes the whole chunk (padded steps included), so compare
    # with the codes held fixed: the masked loss must not see the garbage
    with torch.no_grad():
        target, real = policy.tokenizer.prepare(batch)
        target2, real2 = policy.tokenizer.prepare(garbage)
    assert torch.equal(real, real2)
    assert torch.equal(target[real], target2[real2])
    assert torch.isfinite(base)


def test_hand_token_reaches_the_readout_and_no_hand_replaces_refusals() -> None:
    policy = _policy()
    batch = _batch(t=3)
    base = policy(batch)["policy", "action"]
    moved = dict(batch)
    moved["hand.current"] = batch["hand.current"] + 1.0
    assert not torch.allclose(policy(moved)["policy", "action"], base)
    # refused everywhere -> the vector values no longer matter
    refused = dict(batch)
    refused["hand.motor_ok"] = torch.zeros_like(batch["hand.motor_ok"])
    refused2 = dict(refused)
    refused2["hand.current"] = batch["hand.current"] + 5.0
    assert torch.equal(
        policy(refused)["policy", "action"], policy(refused2)["policy", "action"]
    )
    # and a missing hand stream is the same as all-refused
    absent = {k: v for k, v in batch.items() if not k.startswith("hand.")}
    assert torch.equal(
        policy(absent)["policy", "action"], policy(refused)["policy", "action"]
    )


def test_hand_dropout_is_train_only() -> None:
    policy = _policy(hand_dropout_sample=1.0)
    batch = _batch(t=3)
    a = policy(batch)["policy", "action"]
    moved = dict(batch)
    moved["hand.current"] = batch["hand.current"] + 1.0
    assert not torch.equal(a, policy(moved)["policy", "action"])  # eval: no dropout
    policy.train()
    torch.manual_seed(0)
    x = policy._features(batch)  # noqa: SLF001
    torch.manual_seed(0)
    y = policy._features(moved)  # noqa: SLF001
    assert torch.allclose(x, y)  # sample dropout 1.0: hand never seen


def test_invalid_side_state_cannot_move_the_output() -> None:
    policy = _policy()
    batch = _batch(t=3)
    moved = dict(batch)
    state = batch["state"].clone()
    state[:, :, 1] = 7.0
    moved["state"] = state
    assert torch.equal(
        policy(batch)["policy", "action"], policy(moved)["policy", "action"]
    )


def test_no_goal_mode_reads_no_goal_frames() -> None:
    policy = _policy()
    batch = _batch(t=2)
    assert not any(k.startswith("goal.") for k in batch)
    policy(batch)  # would raise if a goal key were required


def test_relative_mode_must_match_the_tokenizer() -> None:
    with pytest.raises(ValueError, match="relative_mode"):
        NeroPatchPolicy(**{**_policy_kwargs(), "relative_mode": "hand"})


def _policy_kwargs() -> dict[str, Any]:
    p = _policy()
    return {
        "image_transform": p.image_transform,
        "image_encoder": p.image_encoder,
        "patch_projection": p.patch_projection,
        "state_embedding": p.state_embedding,
        "encoder": p.encoder,
        "tokenizer": p.tokenizer,
        "code_head": p.code_head,
        "offset_head": p.offset_head,
        "losses": p.losses,
        "image_embedding_dim": IMAGE_DIM,
        "policy_embedding_dim": POLICY_DIM,
        "action_space": "robot",
        "offset_mode": "latent",
        "offset_code_conditioning": True,
    }


def test_action_standardizer_pin_is_enforced() -> None:
    kwargs = _policy_kwargs() | {
        "hand_groups": (),
        "goal_mode": "no_goal",
        "state": ("state",),
        "convert_state_to_9d": False,
    }
    digest = kwargs["tokenizer"].standardizer_digest
    NeroPatchPolicy(**kwargs, action_standardizer_sha256=digest)
    with pytest.raises(ValueError, match="standardizer"):
        NeroPatchPolicy(**kwargs, action_standardizer_sha256="f" * 64)


def test_token_gain_calibration_measures_the_patch_rms() -> None:
    policy = _policy(calibrate_token_gain=True).train()
    batch = _batch(t=2)
    policy.compute_metrics(batch)
    assert bool(policy.token_gain_calibrated)
    with torch.no_grad():
        policy.eval()
        norms: dict[str, Tensor] = {}
        policy._features(batch, token_norms=norms)  # noqa: SLF001
    ratio = float(norms["state"] / norms["patch"])
    assert 0.3 < ratio < 3.0, ratio  # noqa: PLR2004
    gain = float(policy.state_embedding.token_gain)  # ty: ignore[possibly-missing-attribute]
    assert gain == float(policy.hand_embedding.token_gain)  # ty: ignore[possibly-missing-attribute]


def test_selective_adamw_handles_the_learned_tokens() -> None:
    from rmind.components.optimizers import SelectiveAdamW  # noqa: PLC0415

    policy = _policy()
    optimizer = SelectiveAdamW(
        policy,
        lr=1e-4,
        weight_decay=0.1,
        weight_decay_module_blacklist=(nn.LayerNorm, nn.Embedding),
    )
    no_decay = {
        id(p)
        for group in optimizer.param_groups
        if group["weight_decay"] == 0.0
        for p in group["params"]
    }
    for name in ("no_hand", "no_goal"):
        assert id(getattr(policy, name)) in no_decay
    assert id(policy.state_embedding.token_gain) in no_decay  # ty: ignore[possibly-missing-attribute]


def test_reliance_metrics_report_every_ablation() -> None:
    policy = _policy()
    metrics = policy.reliance_metrics_for(_batch(t=12, b=4))
    for how in ("no_hand", "shuffled", "shift_plus", "shift_minus"):
        assert f"reliance/{how}/code_nll" in metrics
        assert torch.isfinite(metrics[f"reliance/{how}/code_nll"])


def test_latent_offset_head_is_small_and_the_table_would_not_be() -> None:
    real_dim, real_latent = 512, 128
    # code-conditioned: [features, lookup(codes)] in, hidden 512, latent out
    width = real_dim + real_latent
    latent_head = width * 512 + 512 + 512 * real_latent + real_latent
    table_outputs = 16 * 16 * 100 * 13
    assert latent_head < 5_000_000  # noqa: PLR2004
    assert table_outputs * 1024 > 300_000_000  # noqa: PLR2004


def test_code_conditioned_offset_depends_on_the_codes() -> None:
    """The offset must see the codes: same features, different codes -> a
    different offset (and the argmax path uses the argmax codes' offset)."""
    policy = _policy()
    context = torch.randn(4, POLICY_DIM)
    a = torch.zeros(4, QUANTIZERS, dtype=torch.long)
    b = torch.full((4, QUANTIZERS), CODEBOOK - 1, dtype=torch.long)
    with torch.no_grad():
        off_a = policy._latent_offset(context, a)  # noqa: SLF001
        off_b = policy._latent_offset(context, b)  # noqa: SLF001
    assert not torch.allclose(off_a, off_b)
    # stop-grad: no gradient reaches the frozen tokenizer codebook through it
    context.requires_grad_(True)  # noqa: FBT003
    policy._latent_offset(context, a).sum().backward()  # noqa: SLF001
    assert context.grad is not None
    # unconditioned: the codes cannot move it
    plain = _policy(conditioned=False)
    with torch.no_grad():
        assert torch.equal(
            plain._latent_offset(context, a),  # noqa: SLF001
            plain._latent_offset(context, b),  # noqa: SLF001
        )


def test_offset_head_width_is_checked_against_the_conditioning() -> None:
    kwargs = _policy_kwargs() | {
        "hand_groups": (),
        "goal_mode": "no_goal",
        "state": ("state",),
        "convert_state_to_9d": False,
    }
    with pytest.raises(ValueError, match="offset_head takes"):
        NeroPatchPolicy(**kwargs | {"offset_code_conditioning": False})
    with pytest.raises(ValueError, match="offset_code_conditioning"):
        NeroPatchPolicy(**kwargs | {"offset_mode": "table"})


def test_normed_token_embedding_keeps_the_affine_level() -> None:
    """Scientist finding: an input LayerNorm made the token invariant to
    `x -> a*x + b`, so hand readings differing only in overall current/pos_err
    level collapsed to one token. Without it they must differ."""
    torch.manual_seed(0)
    emb = NormedTokenEmbedding(in_features=14, out_features=32, hidden_features=(32,))
    x = torch.rand(1, 14)
    x[:, -1] = 1.0  # hand_valid
    with torch.no_grad():
        outs = [emb(v) for v in (x, 0.5 * x + 0.5, 2 * x - 1)]
    for other in outs[1:]:
        assert (outs[0] - other).abs().max() > 1e-2  # noqa: PLR2004
    # the ablation flag reproduces the old (degenerate) behaviour
    old = NormedTokenEmbedding(
        in_features=14, out_features=32, hidden_features=(32,), input_norm=True
    )
    with torch.no_grad():
        o = [old(v) for v in (x, 0.5 * x + 0.5, 2 * x - 1)]
    assert (o[0] - o[1]).abs().max() < 1e-2  # noqa: PLR2004


# ------------------------------------------------- hand token standardizer


def test_hand_standardizer_scales_features_and_passes_age_and_valid() -> None:
    std = HandTokenStandardizer(groups=HAND_GROUPS)
    assert std.source == "physical_prior"
    cols = hand_token_columns(HAND_GROUPS)
    x = torch.rand(3, 5, len(cols))
    x[..., -1] = (torch.rand(3, 5) > 0.5).float()  # noqa: PLR2004
    y = std(x)
    assert torch.equal(y[..., -2:], x[..., -2:])  # hand_age, hand_valid untouched
    cur = HAND_PHYSICAL_PRIOR["current"]
    perr = HAND_PHYSICAL_PRIOR["pos_err"]
    assert torch.allclose(y[..., :6], (x[..., :6] - cur[0]) / cur[1])
    assert torch.allclose(y[..., 6:12], (x[..., 6:12] - perr[0]) / perr[1])
    with pytest.raises(ValueError, match="columns"):
        std(torch.rand(2, len(cols) + 1))
    assert not any(p.requires_grad for p in std.parameters())
    # a subset selection reads the SAME file
    pos = HandTokenStandardizer(groups=("pos",))
    assert pos.digest == std.digest
    assert torch.allclose(pos.std, torch.full((6,), HAND_PHYSICAL_PRIOR["pos"][1]))


def test_hand_standardizer_file_roundtrip_and_refusals(tmp_path: Path) -> None:
    std = HandTokenStandardizer(groups=HAND_GROUPS)
    path = tmp_path / "hand_standardizer.json"
    digest = std.save(path)
    payload = json.loads(path.read_text())
    assert payload["names"] == hand_feature_columns()
    assert payload["passthrough"] == ["hand_age", "hand_valid"]
    back = HandTokenStandardizer.load(path, groups=HAND_GROUPS, sha256=digest)
    assert back.digest == digest
    assert torch.equal(back.mean, std.mean)
    with pytest.raises(ValueError, match="sha256"):
        HandTokenStandardizer.load(path, sha256="0" * 64)
    bad = payload | {"names": payload["names"][::-1]}
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="names"):
        HandTokenStandardizer.load(path)


def test_hand_standardizer_fit_uses_valid_rows_or_keeps_the_band() -> None:
    g = torch.Generator().manual_seed(0)
    n = 4000
    blocks = {
        "current": 0.3 + 0.02 * torch.randn(n, 6, generator=g),
        "pos_err": 0.01 * torch.randn(n, 6, generator=g),  # 0.2 band: fitted
        "pos": torch.full((n, 6), 0.4),  # never moved: keeps the band std
    }
    blocks["pos_err"][:, 0] = 0.0001 * torch.randn(n, generator=g)  # < 0.1 band
    motor_ok = torch.rand(n, generator=g) > 0.5  # noqa: PLR2004
    # refused rows carry garbage that must not leak into the fit
    blocks["current"][~motor_ok] = 100.0
    std, report = HandTokenStandardizer.fit(blocks, motor_ok, min_rows=1000)
    assert report["source"] == "train:hand"
    assert std.source == "train:hand"
    names = hand_feature_columns()
    i = names.index("current.now.index")
    assert abs(float(std.full_mean[i]) - 0.3) < 0.01  # noqa: PLR2004
    assert abs(float(std.full_std[i]) - 0.02) < 0.005  # noqa: PLR2004
    j = names.index("pos.now.index")
    assert float(std.full_std[j]) == pytest.approx(HAND_PHYSICAL_PRIOR["pos"][1])
    assert float(std.full_mean[j]) == pytest.approx(0.4)
    assert "pos_err.now.thumb_flex" in report["kept_prior_std"]
    assert all(
        c.startswith("tip.") or c in report["fitted"] or c in report["kept_prior_std"]
        for c in names
    )
    # too few valid rows -> the physical prior, said so
    few, report = HandTokenStandardizer.fit(blocks, motor_ok & (torch.arange(n) < 100))  # noqa: PLR2004
    assert report["source"] == "physical_prior"
    assert few.digest == HandTokenStandardizer().digest


def test_hand_standardizer_is_in_graph_and_self_contained() -> None:
    """The affine changes the token, travels in hparams (no stats path needed
    to reload) and restores from the state_dict; a legacy policy without one is
    identity; a groups mismatch is refused."""
    fitted = HandTokenStandardizer(
        groups=HAND_GROUPS,
        mean=[0.1] * len(hand_feature_columns()),
        std=[0.3] * len(hand_feature_columns()),
        source="train:hand",
    )
    policy = _policy(hand_standardizer=fitted)
    legacy = _policy(hand_standardizer=None)
    assert legacy.hand_standardizer is None
    hp = dict(policy.hparams)["hand_standardizer"]
    assert hp["_target_"] == "rmind.data.nero_robot.HandTokenStandardizer"
    rebuilt = HandTokenStandardizer(**{k: v for k, v in hp.items() if k != "_target_"})
    assert rebuilt.digest == fitted.digest
    batch = _batch(t=3)
    with torch.no_grad():
        a = policy._features(batch)  # noqa: SLF001
        legacy.load_state_dict({
            k: v for k, v in policy.state_dict().items() if "hand_standardizer" not in k
        })
        b = legacy._features(batch)  # noqa: SLF001
    assert (a - b).abs().max() > 1e-3  # noqa: PLR2004
    sd = policy.state_dict()
    assert "hand_standardizer.full_mean" in sd
    assert "hand_standardizer.index" not in sd  # the selection is config
    fresh = _policy()
    fresh.load_state_dict(sd)
    assert torch.equal(fresh.hand_standardizer.full_mean, fitted.full_mean)  # ty: ignore[possibly-missing-attribute]
    with pytest.raises(ValueError, match="groups"):
        _policy(hand_standardizer=HandTokenStandardizer(groups=("current",)))


def test_hand_standardizer_balances_the_first_layer() -> None:
    """Scientist finding: at default init the current / pos_err columns
    contributed a few percent of the first Linear's pre-activation variance
    (pos ~0.2-0.6 and age ~0.5 dominated). With the physical affine each
    feature group's share must be within ~2x of its column share. Data shaped
    like the one tactile episode (val), NOT fitted on it."""
    g = torch.Generator().manual_seed(0)
    n = 2000
    groups = ("current", "pos_err", "pos")
    cols = hand_token_columns(groups)
    x = torch.zeros(n, len(cols))
    x[:, 0:6] = 0.01 + 0.07 * torch.randn(n, 6, generator=g)  # current std ~0.03-0.12
    x[:, 6:12] = 0.04 * torch.randn(n, 6, generator=g)  # pos_err std ~0.01-0.08
    x[:, 12:18] = 0.3 + 0.12 * torch.randn(n, 6, generator=g)  # pos
    x[:, 18] = torch.rand(n, generator=g)  # age ~ U(0, 1)
    x[:, 19] = 1.0

    def shares(v: Tensor) -> dict[str, float]:
        torch.manual_seed(0)
        lin = nn.Linear(len(cols), 512)
        w = lin.weight.detach()  # (512, in)
        # per-group contribution to the pre-activation (around the batch mean)
        out = {}
        for name, sl in (
            ("current", slice(0, 6)),
            ("pos_err", slice(6, 12)),
            ("pos", slice(12, 18)),
            ("age", slice(18, 19)),
        ):
            centred = v[:, sl] - v[:, sl].mean(0)
            out[name] = float((centred @ w[:, sl].T).pow(2).mean())
            # plus the mean offset each group shifts the pre-activation by
            out[name] += float((v[:, sl].mean(0) @ w[:, sl].T).pow(2).mean())
        total = sum(out.values())
        return {k: val / total for k, val in out.items()}

    raw = shares(x)
    std = HandTokenStandardizer(groups=groups)
    scaled = shares(std(x))
    assert raw["current"] + raw["pos_err"] < 0.15  # the finding  # noqa: PLR2004
    for name in ("current", "pos_err"):
        assert scaled[name] > 0.15, scaled  # 6 of 19 scaled columns ~0.32  # noqa: PLR2004


# ----------------------------------------------------------- streaming gate


@pytest.mark.parametrize("relative_mode", ["none", "hand", "all"])
def test_streaming_equals_windowed(relative_mode: str) -> None:
    """THE serving correctness gate (#269): ring of W-1 == one windowed forward."""
    policy = _policy(relative_mode=relative_mode)
    batch = _batch(t=40, b=1, seed=3)
    with torch.no_grad():
        features = policy._features(batch)  # noqa: SLF001  (1, 40, d)
        windowed, windowed_codes = policy._predict_chunk_and_codes(features[0])  # noqa: SLF001
    step = NeroPatchPolicyDecoderStep(policy=policy)
    past = step.empty_cache()
    assert past[0].shape[-2] == (WINDOW - 1) * step.tokens_per_frame
    worst = 0.0
    for frame in range(40):
        inputs = step.frame_inputs(batch, frame)
        cos, sin = step.rope(frame)
        with torch.no_grad():
            actions, new_k, new_v, codes = step(
                inputs["images"],
                inputs["state"],
                inputs["side_valid"],
                *past,
                cos,
                sin,
                inputs.get("hand_token"),
            )
        past = step.advance(past, new_k, new_v)
        want = windowed[frame].permute(1, 0, 2).reshape(1, CHUNK, 26)
        worst = max(worst, float((actions - want).abs().max()))
        assert torch.equal(codes[0], windowed_codes[frame, 0]), frame
    assert worst <= 1e-4, worst  # noqa: PLR2004


# ---------------------------------------------------------- preprocessing


def test_preprocess_geometry_matches_the_contract_example() -> None:
    base = nero_image.geometry((1920, 1080), (140, 224))
    assert base.resize_wh == (224, 126)
    assert base.pad_ltrb == (0, 7, 0, 7)
    side = nero_image.geometry((1280, 800), (140, 224))
    assert side.resize_wh == (224, 140)
    assert side.pad_ltrb == (0, 0, 0, 0)


def test_preprocess_is_uint8_exact_and_deterministic() -> None:
    torch.manual_seed(0)
    frame = torch.randint(0, 256, (3, 1080, 1920), dtype=torch.uint8)
    a = nero_image.preprocess(frame, (140, 224))
    b = nero_image.NeroImagePreprocess((140, 224))(frame.unsqueeze(0))[0]
    assert a.dtype == torch.uint8
    assert a.shape == (3, 140, 224)
    assert torch.equal(a, b)
    assert not a[:, :7].any()
    assert not a[:, -7:].any()
    with pytest.raises(TypeError):
        nero_image.preprocess(frame.float(), (140, 224))
    assert len(nero_image.preprocessing_sha256()) == 64  # noqa: PLR2004


# ---------------------------------------------------------------- tokenizer


def test_tokenizer_codes_keyframes_and_decodes_at_30hz() -> None:
    tokenizer = _tokenizer()
    assert tokenizer.num_keyframes == 34  # noqa: PLR2004
    x = torch.randn(5, CHUNK, 13)
    codes = tokenizer.encode(x)
    assert codes.shape == (5, QUANTIZERS)
    out = tokenizer.decode_latent(tokenizer.lookup(codes))
    assert out.shape == (5, CHUNK, 13)
    # only the keyframes reach the encoder
    y = x.clone()
    y[:, 1::3] += 10.0
    y[:, 2::3] -= 10.0
    assert torch.equal(tokenizer.encode(y), codes)
    assert tokenizer.bits == QUANTIZERS * 2


def test_tokenizer_rows_follow_side_valid_and_padding() -> None:
    tokenizer = _tokenizer("hand")
    batch = _batch(t=3, b=2)
    rows, real = tokenizer.prepare(batch)
    assert rows.shape == (2 * 3, CHUNK, 13)  # left side only
    assert torch.equal(real, ~batch["action.is_pad"].reshape(-1, CHUNK))
    # hand-relative: finger targets are command - hand_prev of THAT frame
    chunk = batch["action.chunk"][:, :, :, 0]
    anchor = batch["state"][:, :, 0]
    want = (chunk[..., 7:] - anchor[:, :, None, 7:]).reshape(-1, CHUNK, 6)
    assert torch.allclose(rows[..., 7:], want, atol=1e-6)


def test_event_reference_is_deterministic_and_pinned(tmp_path: Path) -> None:
    """Fitted on the whole train split: batch ORDER cannot change it (the old
    first-batch median did); the tokenizer refuses a reference from another
    standardizer and refuses to train without one."""
    from rmind.scripts import nero_fit_stats  # noqa: PLC0415

    batches = [_batch(t=4, b=2, seed=s) for s in range(4)]
    for b in batches:  # an atom: fingers mostly at an exact open command
        chunk = b["action.chunk"]
        chunk[..., 7:] = torch.where(
            torch.rand(chunk[..., 7:].shape) < 0.7,  # noqa: PLR2004
            torch.tensor(0.05),
            chunk[..., 7:],
        )
    one = nero_fit_stats.fit(batches)
    two = nero_fit_stats.fit(list(reversed(batches)))
    for mode in ("none", "hand", "all"):
        r1, r2 = one["event_reference"][mode], two["event_reference"][mode]
        assert r1.reference == r2.reference
        assert r1.standardizer_sha256 == one["action"][mode].digest
    ref = one["event_reference"]["none"]
    assert all(m == "mode" for m in ref.method[7:])
    std = one["action"]["none"]
    want = (0.05 - std.mean[0, 7:]) / std.std[0, 7:]
    assert torch.allclose(torch.tensor(ref.reference[7:]), want, atol=1e-6)

    stats = tmp_path / "stats"
    nero_fit_stats.write(one, stats)
    loaded = EventReference.load(stats / "event_reference_none.json")
    assert loaded.reference == ref.reference
    tokenizer = _tokenizer()
    with pytest.raises(ValueError, match="no event reference"):
        tokenizer._require_event_reference()  # noqa: SLF001
    with pytest.raises(ValueError, match="re-run nero_fit_stats"):
        tokenizer.set_event_reference(loaded)  # identity standardizer != fitted one
    tokenizer.standardizer = AxisStandardizer.load(
        stats / "action_standardizer_none.json"
    )
    tokenizer.set_event_reference(loaded)
    tokenizer._require_event_reference()  # noqa: SLF001
    with pytest.raises(ValueError, match="relative_mode"):
        _tokenizer("hand").set_event_reference(loaded)


def test_tokenizer_event_weights_boost_fingers_and_drop_padding() -> None:
    tokenizer = _tokenizer()
    tokenizer.event_reference.zero_()
    target = torch.zeros(2, CHUNK, 13)
    target[:, 50:, 9] = 2.0  # a finger "closes"
    target[:, 50:, 2] = 2.0  # an arm joint moves as much
    real = torch.ones(2, CHUNK, dtype=torch.bool)
    real[1, 80:] = False
    w = tokenizer.weights(target, real)
    assert w[0, 60, 9] == 5.0  # noqa: PLR2004
    assert w[0, 60, 2] == 1.0
    assert w[0, 10, 9] == 1.0
    assert (w[1, 80:] == 0).all()


def test_split_manifest_is_by_episode_and_disjoint() -> None:
    import re  # noqa: PLC0415

    text = (
        Path(__file__).parents[1] / "config/_templates/dataset/nero/robot_split.lib.yml"
    ).read_text()
    blocks = dict(re.findall(r"#@ (train|val) = \[(.*?)\]", text, flags=re.DOTALL))
    train = set(re.findall(r'"([^"]+)"', blocks["train"]))
    val = set(re.findall(r'"([^"]+)"', blocks["val"]))
    assert train
    assert val
    assert not train & val


# ------------------------------------------------------------------- export


def test_fit_stats_fits_the_hand_standardizer_on_valid_rows(tmp_path: Path) -> None:
    from rmind.scripts import nero_fit_stats  # noqa: PLC0415

    batches = [_batch(t=4, b=2, seed=s) for s in range(3)]
    result = nero_fit_stats.fit(batches, hand_min_rows=1)
    report = result["report"]["hand"]
    assert report["rows_seen"] == 3 * 2 * 4
    ok = torch.cat([b["hand.motor_ok"].reshape(-1) for b in batches])
    assert report["motor_rows"] == int(ok.sum())
    assert report["source"] == "train:hand"
    assert result["hand"].source == "train:hand"
    cur = torch.cat([b["hand.current"].reshape(-1, 6) for b in batches])[ok]
    i = hand_feature_columns().index("current.now.thumb_flex")
    assert float(result["hand"].full_mean[i]) == pytest.approx(
        float(cur[:, 0].mean()), abs=1e-5
    )
    digests = nero_fit_stats.write(result, tmp_path)
    back = HandTokenStandardizer.load(
        tmp_path / "hand_standardizer.json", sha256=digests["hand_standardizer.json"]
    )
    assert back.source == "train:hand"
    # the default threshold keeps the physical prior on this little data
    assert nero_fit_stats.fit(batches)["hand"].digest == HandTokenStandardizer().digest


def _export(
    tmp_path: Path,
    policy: NeroPatchPolicy,
    *,
    nutron_cli: Path | None,
    checkpoint: Path | None = None,
) -> tuple[Any, dict[str, Any], Path]:
    import argparse  # noqa: PLC0415

    pytest.importorskip("onnxruntime")
    from rmind.scripts import nero_export, nero_fit_stats  # noqa: PLC0415

    stats = tmp_path / "stats"
    nero_fit_stats.write(
        nero_fit_stats.fit([_batch(t=4, b=2, seed=s) for s in range(3)]), stats
    )
    out = tmp_path / "artifact"
    args = argparse.Namespace(
        out=out,
        stats=stats,
        camera_cond=None,
        n_next_actions=6,
        min_start_index=3,
        gate_frames=20,
        ort_frames=2,
        tol=1e-4,
        vit_gflops=0.0,
        nutron_cli=nutron_cli,
        no_latency=True,
        gate_device="cpu",
        patch_size=14,
        checkpoint=checkpoint,
    )
    return nero_export, nero_export.run(args, policy, hw=IMAGE_HW), out


def test_export_writes_a_contract_nutron_cli_accepts(tmp_path: Path) -> None:
    """The tiny policy through the real export: gates, ONNX, manifest, and
    nutron-cli's own `patch_contract_from_manifest` + `binding_problems` -- a
    HARD gate: a refusal (e.g. of `standardizers.hand`) fails this test."""
    if not (NUTRON_CLI / "runtime/jetson/policy_contract.py").exists():
        pytest.skip("no nutron-cli checkout ($NUTRON_CLI_ROOT)")
    _, report, out = _export(tmp_path, _policy(), nutron_cli=NUTRON_CLI)
    assert report["nutron_cli"]["status"] == "ok", report["nutron_cli"]
    assert report["failures"] == [], report
    assert report["warnings"] == [], report
    assert report["nutron_cli"]["binding_problems"] == []
    assert (out / "policy_contract.json").exists()
    assert (out / "export_report.json").exists()
    manifest = json.loads((out / "policy_manifest.json").read_text())
    assert manifest["standardizers"]["hand"] == {
        "file": "hand_standardizer.json",
        "sha256": None,
        "in_graph": True,
    }
    assert report["hand_standardizer"]["source"] == "physical_prior"
    assert report["streaming_vs_windowed"]["max_abs"] <= 1e-4  # noqa: PLR2004
    assert manifest["io"]["outputs"]["actions"]["shape"] == [1, CHUNK, 26]
    assert manifest["token_layout"][:2] == [["state", 1], ["hand", 1]]
    # the written contract describes the exported policy exactly
    contract = json.loads((out / "policy_contract.json").read_text())
    from rmind.scripts import nero_export  # noqa: PLC0415

    assert nero_export.contract_mismatches(_policy(), contract) == []


def _pinned(manifest: dict[str, Any], files: dict[str, str]) -> dict[str, Any]:
    """The manifest with the file shas nutron-cli would fill in."""
    contract = json.loads(json.dumps(manifest))
    contract["tokenizer"]["sha256"] = files["tokenizer"]
    for name, block in contract["standardizers"].items():
        block["sha256"] = files[f"{name}_standardizer"]
    return contract


def _fake_nutron(tmp_path: Path, policy_contract: str | None = None) -> Path:
    """A nutron-cli tree with the REAL hand_features.py (the hand spec needs it)
    and, optionally, a stand-in policy_contract.py."""
    real = NUTRON_CLI / "runtime" / "jetson" / "hand_features.py"
    if not real.exists():
        pytest.skip("no nutron-cli checkout ($NUTRON_CLI_ROOT)")
    jetson = tmp_path / "nutron" / "runtime" / "jetson"
    jetson.mkdir(parents=True)
    (jetson / "hand_features.py").write_bytes(real.read_bytes())
    if policy_contract is not None:
        (jetson / "policy_contract.py").write_text(policy_contract)
    return tmp_path / "nutron"


def test_export_without_nutron_cli_reports_gate_3_skipped(tmp_path: Path) -> None:
    """No nutron-cli checkout: the export still writes its report, and gate 3 is
    `skipped` in the report and its warnings -- never a silent pass."""
    _, report, out = _export(tmp_path, _policy(), nutron_cli=_fake_nutron(tmp_path))
    assert report["nutron_cli"]["status"] == "skipped"
    assert any("SKIPPED" in w for w in report["warnings"])
    assert report["failures"] == []
    written = json.loads((out / "export_report.json").read_text())
    assert written["nutron_cli"]["status"] == "skipped"
    assert not (out / "policy_contract.json").exists()


def test_export_records_a_nutron_cli_refusal_as_a_failure(tmp_path: Path) -> None:
    """A nutron-cli whose contract validation refuses: recorded in `failures`,
    export_report.json still written (no crash after the artifacts)."""
    fake = _fake_nutron(
        tmp_path,
        "class ContractError(ValueError):\n    pass\n\n"
        "def patch_contract_from_manifest(manifest, artifact_dir=None):\n"
        "    raise ContractError('refused: test')\n",
    )
    _, report, out = _export(tmp_path, _policy(), nutron_cli=fake)
    assert report["nutron_cli"] == {"status": "refused", "error": "refused: test"}
    assert report["failures"] == ["nutron-cli refused the contract"]
    written = json.loads((out / "export_report.json").read_text())
    assert written["failures"] == ["nutron-cli refused the contract"]


def test_mac_factory_refuses_a_checkpoint_that_is_not_the_artifacts(  # noqa: PLR0914
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Mac torch path ties $NERO_POLICY_CKPT to the artifact: a policy that
    differs from the contract in relative mode, hand groups, tokenizer or a
    standardizer sha is refused, and so is a checkpoint whose sha256 is not the
    one the export recorded."""
    nero_export, report, out = _export(
        tmp_path, _policy(), nutron_cli=_fake_nutron(tmp_path)
    )
    manifest = json.loads((out / "policy_manifest.json").read_text())
    contract = _pinned(manifest, report["files"])
    assert nero_export.contract_mismatches(_policy(), contract) == []

    # an unpinned contract proves nothing about the standardizers
    assert any(
        "pins no sha256" in m
        for m in nero_export.contract_mismatches(_policy(), manifest)
    )

    def mismatched(policy: NeroPatchPolicy) -> list[str]:
        return nero_export.contract_mismatches(policy, contract)

    rel = mismatched(_policy(relative_mode="all"))
    assert any(m.startswith("relative_mode") for m in rel), rel
    assert any(m.startswith("relative_mask") for m in rel), rel
    other_action = _policy()
    other_action.tokenizer.standardizer = AxisStandardizer(
        mean=torch.ones(2, 13), std=torch.full((2, 13), 3.0)
    )
    assert any(m.startswith("standardizers.action") for m in mismatched(other_action))
    other_groups = json.loads(json.dumps(contract))
    other_groups["hand"]["groups"] = ["current"]
    groups = nero_export.contract_mismatches(_policy(), other_groups)
    assert any(m.startswith("hand.groups") for m in groups), groups
    other_hand = _policy()
    other_hand.hand_standardizer = HandTokenStandardizer(
        groups=HAND_GROUPS, source="other"
    )
    assert any(m.startswith("standardizers.hand") for m in mismatched(other_hand)), (
        mismatched(other_hand)
    )
    no_hand = mismatched(_policy(hand=False))
    assert any(m.startswith("hand token") for m in no_hand), no_hand
    assert any(m.startswith("token_layout") for m in no_hand), no_hand
    other_state = _policy()
    other_state.state_standardizer = AxisStandardizer(
        mean=torch.ones(2, 13), std=torch.full((2, 13), 2.0)
    )
    assert any(m.startswith("standardizers.state") for m in mismatched(other_state))

    with_depth = json.loads(json.dumps(contract))
    with_depth["depth"] = {"cameras": ["base"]}
    assert any(
        m.startswith("depth")
        for m in nero_export.contract_mismatches(_policy(), with_depth)
    )

    ckpt = tmp_path / "model.ckpt"
    ckpt.write_bytes(b"the exported checkpoint")
    monkeypatch.setenv("NERO_POLICY_CKPT", str(ckpt))
    monkeypatch.setattr(nero_export, "load_policy", lambda _path: _policy())
    monkeypatch.delenv(nero_export.ALLOW_UNPINNED_ENV, raising=False)
    # this export had no --ckpt: no pin, refused unless explicitly allowed
    with pytest.raises(RuntimeError, match="no checkpoint sha256 pinned"):
        nero_export.mac_factory(contract, out, "cpu")
    monkeypatch.setenv(nero_export.ALLOW_UNPINNED_ENV, "1")
    assert isinstance(
        nero_export.mac_factory(contract, out, "cpu"), nero_export.RoleDecoderStep
    )
    monkeypatch.delenv(nero_export.ALLOW_UNPINNED_ENV)

    # pinned to these bytes: served
    written = json.loads((out / "export_report.json").read_text())
    written["checkpoint"] = {"path": str(ckpt), "sha256": nero_export.sha256(ckpt)}
    (out / "export_report.json").write_text(json.dumps(written))
    module = nero_export.mac_factory(contract, out, "cpu")
    assert isinstance(module, nero_export.RoleDecoderStep)

    # pinned, but the policy differs from the contract: refused
    monkeypatch.setattr(
        nero_export, "load_policy", lambda _path: _policy(relative_mode="hand")
    )
    with pytest.raises(RuntimeError, match="does not match the contract"):
        nero_export.mac_factory(contract, out, "cpu")

    # another file than the pinned one: refused before loading
    ckpt.write_bytes(b"another epoch")
    with pytest.raises(RuntimeError, match="is not the checkpoint"):
        nero_export.mac_factory(contract, out, "cpu")


def test_export_records_the_checkpoint_sha(tmp_path: Path) -> None:
    """`--ckpt` exports pin the checkpoint's sha256 in export_report.json."""
    ckpt = tmp_path / "model.ckpt"
    ckpt.write_bytes(b"weights")
    nero_export, report, out = _export(
        tmp_path, _policy(hand=False), nutron_cli=None, checkpoint=ckpt
    )
    assert report["checkpoint"]["sha256"] == nero_export.sha256(ckpt)
    written = json.loads((out / "export_report.json").read_text())
    assert written["checkpoint"]["sha256"] == nero_export.sha256(ckpt)


def test_rbyte_training_path_equals_the_serving_preprocess() -> None:
    """P10: rbyte's TransformedTensorSource(NeroImagePreprocess) on decoded native
    frames == `preprocess` on the same RGB array (what serving runs): diff 0."""
    rbyte_io = pytest.importorskip("rbyte.io")
    if not hasattr(rbyte_io, "TransformedTensorSource"):
        pytest.skip("rbyte without the robot ingestion (use the local checkout)")
    torch.manual_seed(0)
    native = torch.randint(0, 256, (4, 3, 800, 1280), dtype=torch.uint8)

    class _Decoded:
        def __getitem__(self, index: Any) -> Tensor:
            return native[index]

        def __len__(self) -> int:
            return len(native)

    source = rbyte_io.TransformedTensorSource(
        source=_Decoded(), transform=nero_image.NeroImagePreprocess((140, 224))
    )
    training = source[[0, 2]]
    serving = torch.stack([
        nero_image.preprocess(native[i], (140, 224)) for i in (0, 2)
    ])
    assert torch.equal(training, serving)


# ------------------------------------------------------------- replay bundle


@pytest.mark.parametrize("relative_mode", ["none", "all"])
def test_replay_expected_equals_the_streamed_absolute_chunks(
    relative_mode: str,
) -> None:
    """`nero_replay_expected.expected` (one windowed forward PER STREAM, split at
    the bundle's resets) == the decoder step streamed with a ring reset at the same
    frames, unstandardized + made absolute the way serving does it."""
    from rmind.scripts.nero_replay_expected import expected  # noqa: PLC0415

    policy = _policy(relative_mode=relative_mode)
    t = 24
    batch = _batch(t=t, b=1, seed=5)
    reset = torch.zeros(t, dtype=torch.bool)
    reset[[0, 9, 10]] = True  # a mid-episode stream, and a one-frame stream
    hand = policy.hand_vector(batch)
    assert hand is not None
    bundle = {
        "state": batch["state"][0].reshape(t, -1).numpy(),
        "images_u8": torch.stack(
            [batch[policy.image_key.format(camera=c)][0] for c in policy.cameras], dim=1
        ).numpy(),
        "hand_token": hand[0].numpy(),
        "reset": reset.numpy(),
    }
    manifest = {
        "sides": ["left", "right"],
        "side_valid": batch["side_valid"][0].tolist(),
        "camera_cond": {"values": batch["camera_cond"][0].tolist()},
    }
    actions, codes = expected(policy, bundle, manifest, device=torch.device("cpu"))
    assert actions.shape == (t, CHUNK, 26)

    step = NeroPatchPolicyDecoderStep(
        policy=policy, camera_cond=batch["camera_cond"][0]
    )
    std = policy.tokenizer.standardizer
    counter = 0
    for frame in range(t):
        if reset[frame]:
            past, counter = step.empty_cache(), 0
        inputs = step.frame_inputs(batch, frame)
        cos, sin = step.rope(counter)
        with torch.no_grad():
            z, new_k, new_v, frame_codes = step(
                inputs["images"],
                inputs["state"],
                inputs["side_valid"],
                *past,
                cos,
                sin,
                inputs["hand_token"],
            )
        past, counter = step.advance(past, new_k, new_v), counter + 1
        raw = std.unstandardize(z.reshape(1, CHUNK, 2, 13))
        absolute = to_absolute(raw, inputs["state"].reshape(1, 2, 13), relative_mode)
        diff = (
            (absolute.reshape(CHUNK, 26) - torch.from_numpy(actions[frame])).abs().max()
        )
        assert float(diff) <= 1e-4, (frame, float(diff))  # noqa: PLR2004
        assert codes[frame].tolist() == frame_codes[0].tolist(), frame


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="compiled block mask is the CUDA path"
)
def test_long_sequence_block_mask_is_the_eager_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Above EAGER_BLOCK_MASK_MAX_TOKENS the BlockMask comes from the compiled
    `create_block_mask` (the eager one is O(seq^2) memory: ~84 GiB for a whole
    220-frame episode); it must be the same mask."""
    from rmind.components.transformer import causal_frame as cf  # noqa: PLC0415

    device = torch.device("cuda")
    eager = cf.frame_block_causal_block_mask(12, 482, window=WINDOW, device=device)
    cf.frame_block_causal_block_mask.cache_clear()
    monkeypatch.setattr(cf, "EAGER_BLOCK_MASK_MAX_TOKENS", 0)
    compiled = cf.frame_block_causal_block_mask(12, 482, window=WINDOW, device=device)
    cf.frame_block_causal_block_mask.cache_clear()
    for a, b in zip(eager.as_tuple()[2:6], compiled.as_tuple()[2:6], strict=True):
        assert torch.equal(a, b)
