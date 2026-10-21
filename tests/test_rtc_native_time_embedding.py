import io
import math
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch

from rmind.components.rtc_flow_head import TokenTimeFlowHead
from rmind.components.rtc_time_embedding import (
    NativeTimeEmbedding,
    native_time_frequencies,
)


def test_actual_head_defaults_to_native_without_external_vector() -> None:
    head = TokenTimeFlowHead(8, 3, 8, channel_dim=16)
    assert head.time_embedding_mode == "native"
    torch.testing.assert_close(
        head.time_frequency, native_time_frequencies(16), atol=0, rtol=0
    )
    restored = TokenTimeFlowHead(8, 3, 8, channel_dim=16)
    restored.load_state_dict(head.state_dict(), strict=True)
    changed = {name: value.clone() for name, value in head.state_dict().items()}
    changed["time_frequency"][0] += 0.1
    with pytest.raises(RuntimeError, match="native analytical recipe"):
        restored.load_state_dict(changed, strict=True)


def test_reference_compatibility_and_native_modes_are_not_interchangeable() -> None:
    supplied = [float(index + 1) for index in range(8)]
    legacy = TokenTimeFlowHead(8, 3, 8, channel_dim=16, time_frequency=supplied)
    assert legacy.time_embedding_mode == "reference_compatibility"
    with pytest.raises(ValueError, match="only reference_compatibility"):
        TokenTimeFlowHead(
            8,
            3,
            8,
            channel_dim=16,
            time_frequency=supplied,
            time_embedding_mode="native",
        )
    with pytest.raises(ValueError, match="only reference_compatibility"):
        TokenTimeFlowHead(
            8, 3, 8, channel_dim=16, time_embedding_mode="reference_compatibility"
        )


@pytest.mark.parametrize("channels", [2, 16, 64, 128, 256, 512])
@pytest.mark.parametrize("periods", [(0.004, 4.0), (0.125, 0.125), (0.01, 100.0)])
def test_native_frequencies_match_independent_analytical_formula(
    channels: int, periods: tuple[float, float]
) -> None:
    minimum, maximum = periods
    count = channels // 2
    expected = torch.tensor(
        [
            2
            * math.pi
            / (minimum * math.pow(maximum / minimum, index / max(count - 1, 1)))
            for index in range(count)
        ],
        dtype=torch.float32,
    )
    actual = native_time_frequencies(channels, minimum, maximum)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual.device.type == "cpu"
    assert actual.dtype == torch.float32
    assert not actual.requires_grad
    assert (actual[:-1] >= actual[1:]).all()


@pytest.mark.parametrize("channels", [True, 0, -2, 3, 16.0])
def test_invalid_channel_dimensions_rejected(channels: float) -> None:
    error_type = (
        TypeError
        if isinstance(channels, bool) or not isinstance(channels, int)
        else ValueError
    )
    with pytest.raises(error_type, match="positive even integer"):
        native_time_frequencies(cast("int", channels))


@pytest.mark.parametrize(
    "periods", [(0, 4), (-1, 4), (4, 1), (math.nan, 4), (1, math.inf)]
)
def test_invalid_periods_rejected(periods: tuple[float, float]) -> None:
    with pytest.raises(ValueError, match="periods must"):
        native_time_frequencies(16, *periods)


@pytest.mark.parametrize("shape", [(), (5,), (2, 8), (2, 3, 50), (1, 100)])
def test_arbitrary_time_axes_and_gradients(shape: tuple[int, ...]) -> None:
    module = NativeTimeEmbedding(64)
    time = torch.rand(shape, requires_grad=True)
    actual = module(time)
    assert actual.shape == (*shape, 64)
    assert torch.isfinite(actual).all()
    actual.square().mean().backward()
    assert time.grad is not None
    assert torch.isfinite(time.grad).all()
    flattened = module(time.detach().reshape(-1)).reshape(*shape, 64)
    torch.testing.assert_close(actual, flattened, atol=0, rtol=0)


def test_serialization_and_horizon_independence() -> None:
    module = NativeTimeEmbedding(256)
    stream = io.BytesIO()
    torch.save(module.state_dict(), stream)
    stream.seek(0)
    restored = NativeTimeEmbedding(256)
    restored.load_state_dict(torch.load(stream, weights_only=True), strict=True)
    times = torch.rand(2, 100)
    torch.testing.assert_close(restored(times), module(times), atol=0, rtol=0)
    torch.testing.assert_close(
        module(times[:, :8]), module(times)[:, :8], atol=0, rtol=0
    )
    assert not dict(module.named_parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("channels", [16, 64, 128, 256, 512])
def test_cuda_uses_identical_native_frequency_state(channels: int) -> None:
    module = NativeTimeEmbedding(channels)
    frequency = module.frequency.clone()
    time = torch.linspace(0, 1, 150).reshape(3, 50)
    expected = module(time)
    module.cuda()
    torch.testing.assert_close(module.frequency.cpu(), frequency, atol=0, rtol=0)
    torch.testing.assert_close(
        module(time.cuda()).cpu(), expected, atol=2e-7, rtol=2e-7
    )


@pytest.mark.parametrize("channels", [16, 64, 128, 256, 512])
def test_onnx_dynamic_time_axes(channels: int, tmp_path: Path) -> None:
    ort = pytest.importorskip("onnxruntime")
    module = NativeTimeEmbedding(channels).eval()
    path = tmp_path / "native_time.onnx"
    torch.onnx.export(
        module,
        (torch.rand(2, 8),),
        path,
        input_names=["time"],
        output_names=["embedding"],
        dynamic_axes={
            "time": {0: "batch", 1: "horizon"},
            "embedding": {0: "batch", 1: "horizon"},
        },
        opset_version=17,
        dynamo=False,
    )
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for horizon in (8, 12, 50, 100):
        time = torch.linspace(0, 1, 3 * horizon).reshape(3, horizon)
        actual = session.run(None, {"time": time.numpy()})[0]
        np.testing.assert_allclose(actual, module(time).numpy(), atol=2e-7, rtol=2e-7)
