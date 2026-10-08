"""The per-side (bimanual) hand token, WP5: `hand_sides = ("left", "right")`.

What this file makes falsifiable, on the tiny shapes of `test_nero_robot.py`:

* a bimanual batch builds `1 + 2 + 3P` tokens per frame (483 at the real
  geometry) and its loss covers BOTH sides' rows;
* each side's `no_hand` substitution is driven by THAT side's `hand_valid`
  only: perturbing a refused side's vector cannot move the output, perturbing
  the other side's does; per-side dropout drops the sides independently;
* the single-token (and hand-off) model is unchanged: no new parameters, same
  init, the untagged `hand.*` path;
* streaming == windowed with every side's codes compared, the replay bundle's
  `(N, 2, 14)` hand tokens, and the tiny bimanual export through nutron-cli's
  own `patch_contract_from_manifest` (WP0: `hand_sides`, `hand_token (1, 2, 14)`).
"""

from __future__ import annotations

import json
from typing import Any

import pytest
import torch

from rmind.data.nero_robot import (
    compose_hand_token,
    compose_hand_tokens,
    hand_token_dim,
    normalize_hand_sides,
    to_absolute,
)
from rmind.datamodules.nero_robot_random import nero_robot_batch
from rmind.models.nero_patch_policy_decoder import NeroPatchPolicyDecoderStep
from tests.test_nero_robot import (
    CHUNK,
    HAND_GROUPS,
    IMAGE_HW,
    NUM_PATCHES,
    NUTRON_CLI,
    _batch,
    _export,
    _policy,
)

SIDES = ("left", "right")
DIM = hand_token_dim(HAND_GROUPS)


def _bi_batch(t: int = 4, *, b: int = 2, seed: int = 0) -> dict[str, Any]:
    return nero_robot_batch(
        batch_size=b, num_frames=t, image_hw=IMAGE_HW, seed=seed, bimanual=True
    )


def _sided(**kwargs: Any) -> Any:
    return _policy(hand_sides=SIDES, **kwargs)


def _with_token(batch: dict[str, Any], token: torch.Tensor) -> dict[str, Any]:
    out = {k: v for k, v in batch.items() if not k.startswith("hand.")}
    out["hand_token"] = token
    return out


# ------------------------------------------------------------------ data


def test_normalize_hand_sides() -> None:
    assert normalize_hand_sides(None) == ()
    assert normalize_hand_sides(["left", "right"]) == SIDES
    assert normalize_hand_sides(["right"]) == ("right",)
    for bad in (["right", "left"], ["left", "left"], ["up"]):
        with pytest.raises(ValueError, match="hand sides"):
            normalize_hand_sides(bad)


def test_bimanual_batch_keeps_the_left_side_and_adds_an_independent_right() -> None:
    single = _batch(t=4, seed=3)
    bi = _bi_batch(t=4, seed=3)
    assert bi["side_valid"].all()
    assert torch.equal(bi["state"][:, :, 0], single["state"][:, :, 0])
    assert torch.equal(bi["action.chunk"][..., 0, :], single["action.chunk"][..., 0, :])
    for camera in ("base", "side_left", "side_right"):
        assert torch.equal(bi[f"image.{camera}"], single[f"image.{camera}"])
    assert not any(k.startswith("hand.") and k.count(".") == 1 for k in bi)
    for key in ("current", "pos_err", "age", "motor_ok"):
        assert torch.equal(bi[f"hand.left.{key}"], single[f"hand.{key}"])
        assert not torch.equal(bi[f"hand.right.{key}"], bi[f"hand.left.{key}"])
    assert bi["state"][:, :, 1].abs().sum() > 0
    assert bi["action.chunk"][..., 1, :].abs().sum() > 0


def test_compose_hand_tokens_is_compose_hand_token_per_side() -> None:
    bi = _bi_batch(t=5, seed=1)
    tokens = compose_hand_tokens(bi, HAND_GROUPS, SIDES)
    assert tokens.shape == (2, 5, 2, DIM)
    for i, side in enumerate(SIDES):
        assert torch.equal(
            tokens[:, :, i], compose_hand_token(bi, HAND_GROUPS, prefix=f"hand.{side}.")
        )
        assert torch.equal(tokens[:, :, i, -1], bi[f"hand.{side}.motor_ok"].float())


# ----------------------------------------------------------------- model


def test_single_token_model_is_unchanged() -> None:
    """Back-compat: no per-side parameters unless hand_sides is set (the sided
    model adds exactly one). The single-token model's init/outputs are pinned
    bit-identical to the pre-WP5 code by digest in the WP5 report."""
    one = _policy()
    assert one.hand_sides == ()
    assert one.n_hand_tokens == 1
    assert one.hand_side_embedding is None
    assert not any("hand_side" in k for k in one.state_dict())
    assert _policy(hand=False).n_hand_tokens == 0
    two = _sided()
    assert two.n_hand_tokens == len(SIDES)
    assert two.hparams["hand_sides"] == SIDES
    assert set(two.state_dict()) - set(one.state_dict()) == {
        "hand_side_embedding.weight"
    }
    assert set(one.state_dict()) <= set(two.state_dict())


def test_both_sides_forward_and_loss_cover_both_sides() -> None:
    policy = _sided()
    batch = _bi_batch(t=3)
    tokens = policy._frame_tokens(batch)  # noqa: SLF001
    assert tokens.shape[-2] == 1 + 2 + 3 * NUM_PATCHES == policy.tokens_per_frame()
    assert policy._num_patch_tokens() == 3 * NUM_PATCHES  # noqa: SLF001
    policy.train()
    norms: dict[str, torch.Tensor] = {}
    metrics = policy.compute_metrics(batch, token_norms=norms)
    loss = metrics["policy", "loss"].sum(reduce=True)
    assert torch.isfinite(loss)
    # every (batch, frame, side) row with both sides valid
    rows = policy._robot_rows(batch, policy._features(batch))  # noqa: SLF001
    assert int(rows["row_valid"].sum()) == 2 * 3 * 2
    assert set(rows["side"].tolist()) == {0, 1}
    loss.backward()
    grad = policy.hand_side_embedding.weight.grad
    assert grad is not None
    assert (grad.abs().sum(dim=-1) > 0).all()
    assert {"hand_valid_frac/left", "hand_valid_frac/right"} <= set(norms)


@pytest.mark.parametrize("refused", [0, 1])
def test_per_side_no_hand_is_independent(refused: int) -> None:
    """Side `refused`'s hand_valid 0: its vector no longer matters (no_hand),
    the other side's still does."""
    policy = _sided()
    batch = _bi_batch(t=3, seed=2)
    token = compose_hand_tokens(batch, HAND_GROUPS, SIDES)
    token[..., :, -1] = 1.0
    token[..., refused, -1] = 0.0
    base = policy(_with_token(batch, token))["policy", "action"]

    moved_refused = token.clone()
    moved_refused[..., refused, :-1] += 3.0
    assert torch.equal(
        policy(_with_token(batch, moved_refused))["policy", "action"], base
    )
    other = 1 - refused
    moved_other = token.clone()
    moved_other[..., other, :-1] += 3.0
    assert not torch.allclose(
        policy(_with_token(batch, moved_other))["policy", "action"], base
    )
    # the refused side's slot is exactly that side's no_hand, the other's is not
    tok = policy._frame_tokens(_with_token(batch, token))  # noqa: SLF001
    no_hand = policy._no_hand_token()  # noqa: SLF001
    assert torch.allclose(
        tok[:, :, 1 + refused], no_hand[refused].expand_as(tok[:, :, 1])
    )
    assert not torch.allclose(
        tok[:, :, 1 + other], no_hand[other].expand_as(tok[:, :, 1])
    )


def test_no_hand_is_side_tagged() -> None:
    policy = _sided()
    no_hand = policy._no_hand_token()  # noqa: SLF001
    assert no_hand.shape == (2, policy.policy_embedding_dim)
    assert not torch.allclose(no_hand[0], no_hand[1])
    # a missing hand stream is every side refused
    batch = _bi_batch(t=3)
    absent = {k: v for k, v in batch.items() if not k.startswith("hand.")}
    refused = dict(batch)
    for side in SIDES:
        refused[f"hand.{side}.motor_ok"] = torch.zeros_like(
            batch[f"hand.{side}.motor_ok"]
        )
    assert torch.equal(
        policy(absent)["policy", "action"], policy(refused)["policy", "action"]
    )


def test_per_side_dropout_draws_each_side() -> None:
    policy = _sided(hand_dropout_sample=0.0, hand_dropout_frame=0.5)
    batch = _bi_batch(t=16, b=4)
    token = compose_hand_tokens(batch, HAND_GROUPS, SIDES)
    token[..., -1] = 1.0
    batch = _with_token(batch, token)
    no_hand = policy._no_hand_token()  # noqa: SLF001

    def dropped() -> torch.Tensor:
        tok = policy._frame_tokens(batch)[:, :, 1:3]  # noqa: SLF001
        return torch.isclose(tok, no_hand.expand_as(tok)).all(dim=-1)  # (b, T, 2)

    with torch.no_grad():
        assert not dropped().any()  # eval: no dropout
        policy.train()
        torch.manual_seed(0)
        mask = dropped()
    assert mask.any()
    assert (mask[..., 0] != mask[..., 1]).any()  # one side dropped, the other kept


def test_layout_mismatches_are_refused() -> None:
    sided, single = _sided(), _policy()
    bi, one = _bi_batch(t=2), _batch(t=2)
    with pytest.raises(ValueError, match="no hand.left"):
        sided.hand_vector(one)
    with pytest.raises(ValueError, match="hand_sides"):
        single.hand_vector(bi)
    with pytest.raises(ValueError, match=r"\(b, T, 2, dim\)"):
        sided.hand_vector({"hand_token": torch.zeros(1, 2, DIM)})
    with pytest.raises(ValueError, match="one untagged"):
        single.hand_vector({"hand_token": torch.zeros(1, 2, 2, DIM)})
    with pytest.raises(ValueError, match="without hand_groups"):
        _policy(hand=False, hand_sides=SIDES)


def test_stale_tokens_per_frame_is_refused() -> None:
    """A sided model whose trunk still counts ONE hand token must raise, not
    tile the slot embedding wrong."""
    policy = _policy(hand_sides=SIDES)
    policy.encoder.tokens_per_frame -= 1
    with pytest.raises(ValueError, match="tokens per frame"):
        policy._frame_tokens(_bi_batch(t=2))  # noqa: SLF001


def test_reliance_metrics_ablate_each_side() -> None:
    policy = _sided()
    out = policy.reliance_metrics_for(_bi_batch(t=3))
    for how in (
        "no_hand",
        "shuffled",
        "shift_plus",
        "shift_minus",
        "no_hand_left",
        "no_hand_right",
    ):
        assert f"reliance/{how}/code_nll" in out, how
        assert torch.isfinite(out[f"reliance/{how}/offset"])


# ---------------------------------------------------- decoder / export


def test_streaming_equals_windowed_with_every_sides_codes() -> None:
    from rmind.scripts.nero_export import streaming_gate  # noqa: PLC0415

    policy = _sided(relative_mode="all")
    batch = _bi_batch(t=40, b=1, seed=3)
    step = NeroPatchPolicyDecoderStep(policy=policy)
    inputs = step.frame_inputs(batch, 0)
    assert inputs["hand_token"].shape == (1, 2, DIM)
    gate = streaming_gate(policy, step, batch, frames=40)
    assert gate["max_abs"] <= 1e-4, gate  # noqa: PLR2004
    assert gate["code_agreement"] == pytest.approx(1.0), gate
    assert gate["bound_code_agreement"] == pytest.approx(1.0), gate
    assert gate["sides_checked"] == 2  # noqa: PLR2004
    # the bound output stays (1, Q), the first valid side's
    cos, sin = step.rope(0)
    args = (
        inputs["images"],
        inputs["state"],
        inputs["side_valid"],
        *step.empty_cache(),
        cos,
        sin,
        inputs["hand_token"],
    )
    with torch.no_grad():
        bound = step(*args)[3]
        every = step.forward_all_codes(*args)[3]
    assert bound.shape == (1, every.shape[-1])
    assert every.shape[:2] == (1, 2)
    assert torch.equal(bound[0], every[0, 0])


def test_single_token_frame_inputs_keep_the_old_binding() -> None:
    step = NeroPatchPolicyDecoderStep(policy=_policy())
    assert step.frame_inputs(_batch(t=2), 0)["hand_token"].shape == (1, DIM)
    absent = {k: v for k, v in _batch(t=2).items() if not k.startswith("hand.")}
    assert step.frame_inputs(absent, 1)["hand_token"].shape == (1, DIM)
    sided = NeroPatchPolicyDecoderStep(policy=_sided())
    absent = {k: v for k, v in _bi_batch(t=2).items() if not k.startswith("hand.")}
    token = sided.frame_inputs(absent, 1)["hand_token"]
    assert token.shape == (1, 2, DIM)
    assert not token.any()


def test_replay_expected_takes_per_side_hand_tokens() -> None:  # noqa: PLR0914
    from rmind.scripts.nero_replay_expected import expected  # noqa: PLC0415

    policy = _sided(relative_mode="all")
    t = 12
    batch = _bi_batch(t=t, b=1, seed=5)
    reset = torch.zeros(t, dtype=torch.bool)
    reset[[0, 7]] = True
    hand = policy.hand_vector(batch)
    assert hand is not None
    bundle = {
        "state": batch["state"][0].reshape(t, -1).numpy(),
        "images_u8": torch.stack(
            [batch[policy.image_key.format(camera=c)][0] for c in policy.cameras], dim=1
        ).numpy(),
        "hand_token": hand[0].numpy(),  # (N, 2, 14), as serving's bundle has it
        "reset": reset.numpy(),
    }
    assert bundle["hand_token"].shape == (t, 2, DIM)
    manifest = {
        "sides": list(SIDES),
        "side_valid": [True, True],
        "camera_cond": {"values": batch["camera_cond"][0].tolist()},
    }
    actions, codes = expected(policy, bundle, manifest, device=torch.device("cpu"))
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
                torch.from_numpy(bundle["hand_token"][frame]).unsqueeze(0),
            )
        past, counter = step.advance(past, new_k, new_v), counter + 1
        raw = std.unstandardize(z.reshape(1, CHUNK, 2, 13))
        absolute = to_absolute(raw, inputs["state"].reshape(1, 2, 13), "all")
        diff = (
            (absolute.reshape(CHUNK, 26) - torch.from_numpy(actions[frame])).abs().max()
        )
        assert float(diff) <= 1e-4, (frame, float(diff))  # noqa: PLR2004
        assert codes[frame].tolist() == frame_codes[0].tolist(), frame


def _bi_export(
    tmp_path: Any, policy: Any, nutron_cli: Any
) -> tuple[Any, dict[str, Any], Any]:
    """`_export` with the stats fitted on BIMANUAL batches (side_valid [T, T])."""
    import argparse  # noqa: PLC0415

    pytest.importorskip("onnxruntime")
    from rmind.scripts import nero_export, nero_fit_stats  # noqa: PLC0415

    stats = tmp_path / "stats"
    nero_fit_stats.write(
        nero_fit_stats.fit([_bi_batch(t=4, b=2, seed=s) for s in range(3)]), stats
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
        checkpoint=None,
    )
    return nero_export, nero_export.run(args, policy, hw=IMAGE_HW), out


def _wp0_nutron_cli() -> Any:
    pc = NUTRON_CLI / "runtime/jetson/policy_contract.py"
    if not pc.exists() or "hand_sides" not in pc.read_text():
        pytest.skip(
            f"no WP0 (hand_sides) nutron-cli at {NUTRON_CLI} ($NUTRON_CLI_ROOT)"
        )
    return NUTRON_CLI


def test_bimanual_export_passes_every_gate_and_nutron_cli(tmp_path: Any) -> None:
    nutron = _wp0_nutron_cli()
    policy = _sided()
    nero_export, report, out = _bi_export(tmp_path, policy, nutron)
    assert report["failures"] == [], report
    assert report["nutron_cli"]["status"] == "ok", report["nutron_cli"]
    assert report["nutron_cli"]["binding_problems"] == []
    gate = report["streaming_vs_windowed"]
    assert gate["sides_checked"] == len(SIDES), gate
    assert gate["code_agreement"] == pytest.approx(1.0), gate
    assert report["ort_vs_eager"]["codes_equal"]
    manifest = json.loads((out / "policy_manifest.json").read_text())
    assert manifest["hand_sides"] == ["left", "right"]
    assert manifest["side_valid"] == [True, True]
    assert manifest["token_layout"][:2] == [["state", 1], ["hand", 2]]
    assert manifest["tokens_per_frame"] == 1 + 2 + 3 * NUM_PATCHES
    assert manifest["hand"]["dim"] == DIM  # the PER-SIDE spec
    assert manifest["io"]["inputs"]["hand_token"]["shape"] == [1, 2, DIM]
    assert manifest["io"]["outputs"]["codes"]["shape"][0] == 1
    assert len(manifest["io"]["outputs"]["codes"]["shape"]) == 2  # noqa: PLR2004
    contract = json.loads((out / "policy_contract.json").read_text())
    assert nero_export.contract_mismatches(_sided(), contract) == []
    # the one-token checkpoint is not this artifact
    assert any(
        "hand_sides" in m for m in nero_export.contract_mismatches(_policy(), contract)
    )
    # and nutron-cli's loader agrees on the token shape
    pc = nero_export.import_file(nutron / "runtime/jetson/policy_contract.py")
    loaded = pc.patch_contract_from_manifest(out / "policy_manifest.json", out)
    assert loaded.n_hand_tokens == 2  # noqa: PLR2004
    assert tuple(loaded.hand_token_shape) == (2, DIM)


def test_bimanual_export_refuses_single_arm_stats(tmp_path: Any) -> None:
    """hand_sides must equal the stats' valid sides (contract v3): refused
    BEFORE the gates, not by nutron-cli after the ONNX is written."""
    pytest.importorskip("onnxruntime")
    with pytest.raises(RuntimeError, match="side_valid"):
        _export(tmp_path, _sided(), nutron_cli=None)
    assert not (tmp_path / "artifact" / "policy.onnx").exists()


def test_bimanual_causal_composes_two_hand_tokens() -> None:
    from rmind.scripts.nero_steps import compose  # noqa: PLC0415

    cfg = compose("yaak/nero_robot/bimanual_causal", None)
    assert list(cfg.model.hand_sides) == list(SIDES)
    assert cfg.model.encoder.tokens_per_frame == 1 + 2 + 3 * 160
    off = compose("yaak/nero_robot/bimanual_hand_off", None)
    assert off.model.encoder.tokens_per_frame == 1 + 3 * 160
    assert list(off.model.hand_sides) == []
    single = compose("yaak/nero_robot/causal", None)
    assert single.model.encoder.tokens_per_frame == 1 + 1 + 3 * 160
    assert list(single.model.hand_sides) == []
