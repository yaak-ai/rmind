from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from rmind.components.fm_action_head import ActionHead
from rmind.components.rtc_flow_head import TokenTimeFlowHead
from rmind.models.patch_policy_continuous import PatchPolicyContinuous


def small_head(horizon=8):
    return TokenTimeFlowHead(
        8,
        3,
        horizon,
        channel_dim=16,
        channel_hidden_dim=32,
        token_hidden_dim=8,
        num_layers=2,
    )


@pytest.mark.parametrize("length", [0, 1, 2, 4])
def test_explicit_sampling_prefix_and_policy_api(length) -> None:
    torch.manual_seed(3)
    head = small_head()
    features = torch.randn(2, 3, 8)
    noise = torch.randn(2, 3, 24)
    prefix = torch.randn(2, 3, 8, 3)
    mask = (torch.arange(8) < length).expand(2, 3, -1)
    saved = [value.clone() for value in (features, noise, prefix, mask)]
    policy = SimpleNamespace(
        fm_head=head,
        action_horizon=8,
        action_names=list(range(3)),
        fm_inference_steps=5,
        fm_x0="sample",
    )
    expected = head.sample(
        features.flatten(0, 1),
        noise.flatten(0, 1),
        prefix.flatten(0, 1),
        mask.flatten(0, 1),
    )
    actual = PatchPolicyContinuous.fm_sample(
        policy, features, noise, clean_prefix=prefix, prefix_mask=mask
    )
    torch.testing.assert_close(actual.flatten(0, 1), expected, rtol=0, atol=0)
    torch.testing.assert_close(actual[mask], prefix[mask], rtol=0, atol=0)
    for value, original in zip((features, noise, prefix, mask), saved, strict=True):
        torch.testing.assert_close(value, original, rtol=0, atol=0)
    with pytest.raises(ValueError, match="together"):
        PatchPolicyContinuous.fm_sample(policy, features, noise, clean_prefix=prefix)


def test_legacy_head_cannot_be_conditioned() -> None:
    policy = SimpleNamespace(
        fm_head=ActionHead("fm", 8, 1, 24, hidden=16),
        action_horizon=8,
        action_names=list(range(3)),
        fm_inference_steps=5,
        fm_x0="zero",
    )
    with pytest.raises(ValueError, match="trained token-time"):
        PatchPolicyContinuous.fm_sample(
            policy, torch.zeros(1, 8), clean_prefix=torch.zeros(1, 8, 3)
        )


@pytest.mark.parametrize(
    "failure", ["fractional", "hole", "long", "nan_mask", "nan_prefix"]
)
def test_sampler_rejects_invalid_eager_prefix(failure) -> None:
    head = small_head()
    prefix = torch.zeros(1, 8, 3)
    mask = torch.zeros(1, 8)
    if failure == "fractional":
        mask[0, 0] = 0.5
    elif failure == "hole":
        mask[0, 1] = 1
    elif failure == "long":
        mask[0, :5] = 1
    elif failure == "nan_mask":
        mask[0, 0] = float("nan")
    else:
        prefix[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="RTC prefix"):
        head.sample(torch.zeros(1, 8), torch.zeros(1, 8, 3), prefix, mask)


def test_token_times_change_the_vector_field() -> None:
    head = small_head()
    features = torch.randn(2, 8)
    state = torch.randn(2, 8, 3)
    nn.init.normal_(head.final_adaln.weight, std=0.02)
    global_time = torch.full((2,), 0.2)
    token_time = global_time[:, None].expand(-1, 8).clone()
    token_time[:, :2] = 1
    assert not torch.equal(
        head.velocity(features, state, global_time),
        head.velocity(features, state, token_time),
    )


def test_suffix_element_normalization_and_padding() -> None:
    head = small_head()
    features = torch.randn(3, 8)
    target = torch.randn(3, 8, 3)
    noise = torch.randn_like(target)
    time = torch.tensor([0.1, 0.4, 0.8])
    lengths = torch.tensor([0, 2, 4])
    weights = torch.ones_like(target)
    weights[0, -2:] = 0
    weights[..., 1] *= 2
    prefix = torch.arange(8)[None] < lengths[:, None]
    token_time = torch.where(prefix, 1.0, time[:, None])
    state = (1 - token_time[..., None]) * noise + token_time[..., None] * target
    error = (head.velocity(features, state, token_time) - (target - noise)).square()
    weights *= (~prefix)[..., None]
    expected = (weights * error).sum() / weights.sum()
    actual = head.loss_explicit(features, target, noise, time, lengths, weights)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    zero_loss = head.loss_explicit(
        features, target, noise, time, lengths, torch.zeros_like(weights)
    )
    assert zero_loss.item() == 0
    zero_loss.backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in head.parameters())


@pytest.mark.parametrize("weighted", [False, True])
def test_actual_patch_policy_training_updates_head_and_trunk(weighted) -> None:
    policy = PatchPolicyContinuous(
        input_transform=nn.Identity(),
        image_encoder=nn.Identity(),
        patch_projection=nn.Linear(3, 8),
        speed_tokenizer=nn.Identity(),
        speed_embedding=nn.Linear(1, 8),
        encoder=nn.Identity(),
        heads=None,
        losses=None,
        targets={
            "continuous": {name: (name,) for name in ("traction", "steering", "fork1")}
        },
        fm_head=small_head(),
        action_horizon=8,
        action_mean=(0.1, -0.2, 0.3),
        action_std=(0.2, 0.3, 0.4),
        fm_action_weights=(1.0, 2.0, 1.0) if weighted else None,
    )
    trunk = nn.Linear(5, 8)
    policy._features = lambda batch: trunk(torch.ones(2, 3, 5))
    policy.log_dict = lambda *args, **kwargs: None
    result = policy.training_step(
        {name: torch.randn(2, 3, 8) for name in ("traction", "steering", "fork1")}, 0
    )
    result["loss"].backward()
    assert torch.isfinite(result["loss"])
    assert trunk.weight.grad.abs().sum() > 0
    assert policy.fm_head.in_proj.weight.grad.abs().sum() > 0


def test_exported_head_onnx_preserves_raw_prefix(tmp_path) -> None:
    ort = pytest.importorskip("onnxruntime")

    class RawHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.head = small_head()
            self.register_buffer("mean", torch.tensor([0.1, -0.2, 0.3]))
            self.register_buffer("std", torch.tensor([0.2, 0.3, 0.4]))

        def forward(self, features, noise, clean_prefix, prefix_mask):
            return (
                self.head.sample(
                    features, noise, (clean_prefix - self.mean) / self.std, prefix_mask
                )
                * self.std
                + self.mean
            )

    model = RawHead().eval()
    inputs = (
        torch.randn(1, 8),
        torch.randn(1, 8, 3),
        torch.randn(1, 8, 3),
        torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
    )
    names = ["features", "fm_x0", "clean_prefix", "prefix_mask"]
    path = tmp_path / "rtc.onnx"
    torch.onnx.export(
        model,
        inputs,
        path,
        input_names=names,
        output_names=["actions"],
        dynamo=True,
        opset_version=18,
    )
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual = session.run(
        None, {name: value.numpy() for name, value in zip(names, inputs, strict=True)}
    )[0]
    np.testing.assert_allclose(
        actual, model(*inputs).detach().numpy(), atol=1e-5, rtol=1e-5
    )
    np.testing.assert_allclose(
        actual[:, :3], inputs[2].numpy()[:, :3], atol=1e-6, rtol=1e-6
    )
