import io
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from rmind.components.rtc_flow_head import TokenTimeFlowHead


def make_head(
    frequency: list[float] | None = None, mode: str | None = None
) -> TokenTimeFlowHead:
    return TokenTimeFlowHead(
        8,
        3,
        8,
        channel_dim=16,
        channel_hidden_dim=32,
        token_hidden_dim=8,
        num_layers=2,
        time_frequency=frequency,
        time_embedding_mode=mode,
    )


def test_explicit_legacy_frequency_preserves_original_arithmetic_and_state() -> None:
    head = make_head(mode="legacy_dynamic")
    assert head.time_frequency is None
    assert "time_frequency" not in head.state_dict()
    times = torch.tensor([[0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 0.2]])
    fraction = torch.linspace(0, 1, 8, dtype=times.dtype)
    period = 4e-3 * 1000.0**fraction
    angle = times[..., None] * (1.0 / period * (2 * math.pi))
    expected = torch.cat((angle.sin(), angle.cos()), dim=-1)
    captured = []
    handle = head.time_mlp.register_forward_pre_hook(
        lambda _module, arguments: captured.append(arguments[0].detach().clone())
    )
    head.velocity(torch.randn(1, 8), torch.randn(1, 8, 3), times)
    handle.remove()
    torch.testing.assert_close(captured[0], expected, atol=0, rtol=0)
    make_head(mode="legacy_dynamic").load_state_dict(head.state_dict(), strict=True)


@pytest.mark.parametrize("frequency", [[1.0] * 7, [float("nan")] * 8, [-1.0] * 8])
def test_invalid_source_frequencies_rejected(frequency: list[float]) -> None:
    with pytest.raises(ValueError, match="finite positive"):
        make_head(frequency)


def test_source_frequency_is_serialized_nontrainable_state() -> None:
    frequency = [float(value) for value in range(1, 9)]
    head = make_head(frequency)
    assert head.time_frequency is not None
    assert not head.time_frequency.requires_grad
    assert "time_frequency" not in dict(head.named_parameters())
    stream = io.BytesIO()
    torch.save(head.state_dict(), stream)
    stream.seek(0)
    restored = make_head(frequency)
    restored.load_state_dict(torch.load(stream, weights_only=True), strict=True)
    assert restored.time_frequency is not None
    torch.testing.assert_close(
        restored.time_frequency, torch.tensor(frequency), atol=0, rtol=0
    )
    features = torch.randn(2, 8, requires_grad=True)
    target = torch.randn(2, 8, 3)
    loss = head.loss_explicit(
        features,
        target,
        torch.randn_like(target),
        torch.tensor([0.2, 0.7]),
        torch.tensor([2, 4]),
    )
    loss.backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in head.parameters()
    )
    assert head.time_frequency.grad is None


def test_source_frequency_onnx_preserves_normalization_and_prefix(
    tmp_path: Path,
) -> None:
    ort = pytest.importorskip("onnxruntime")

    class RawHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.head = make_head([1.5, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0])
            with torch.no_grad():
                for parameter in self.head.parameters():
                    parameter.normal_(std=0.1)
            self.register_buffer("mean", torch.tensor([0.1, -0.2, 0.3]))
            self.register_buffer("std", torch.tensor([0.2, 0.3, 0.4]))
            self.mean: torch.Tensor
            self.std: torch.Tensor

        def forward(
            self,
            features: torch.Tensor,
            noise: torch.Tensor,
            prefix: torch.Tensor,
            mask: torch.Tensor,
        ) -> torch.Tensor:
            return (
                self.head.sample(features, noise, (prefix - self.mean) / self.std, mask)
                * self.std
                + self.mean
            )

    torch.manual_seed(493)
    module = RawHead().eval()
    inputs = (
        torch.randn(2, 8),
        torch.randn(2, 8, 3),
        torch.randn(2, 8, 3),
        (torch.arange(8)[None] < torch.tensor([[2], [4]])).float(),
    )
    path = tmp_path / "source_frequency.onnx"
    torch.onnx.export(
        module,
        inputs,
        path,
        input_names=["features", "noise", "prefix", "mask"],
        output_names=["actions"],
        opset_version=17,
        dynamo=False,
    )
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual = session.run(
        None,
        {
            name: value.numpy()
            for name, value in zip(
                ("features", "noise", "prefix", "mask"), inputs, strict=True
            )
        },
    )[0]
    expected = module(*inputs).detach().numpy()
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
    prefix_mask = inputs[3].bool().numpy()
    np.testing.assert_allclose(
        actual[prefix_mask], inputs[2].numpy()[prefix_mask], atol=1e-6
    )
