import numpy as np
from types import SimpleNamespace

from rmind.scripts import replay_d12_rollouts

from rmind.scripts.replay_d12_rollouts import (
    candidate_execution_indices,
    noise_index,
    resize_bilinear_u8,
)
from rmind.scripts.summarize_d12_replay import traction_statistics


def test_bilinear_identity_and_constant() -> None:
    image = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
    np.testing.assert_array_equal(resize_bilinear_u8(image, 3, 4), image)
    constant = np.full((9, 16, 3), 137, dtype=np.uint8)
    np.testing.assert_array_equal(resize_bilinear_u8(constant, 3, 5), 137)


def test_bilinear_half_pixel_rounding() -> None:
    image = np.array([[[0], [10]], [[20], [30]]], dtype=np.uint8)
    np.testing.assert_array_equal(resize_bilinear_u8(image, 1, 1), [[[15]]])


def test_execution_indices_reference_camera_not_submission() -> None:
    row = {"frame_wall_ns": 1_000_000_000, "wall_ns": 1_050_000_000}
    predictions = [{"wall_ns": 1_100_000_000}, {"wall_ns": 1_133_333_334}]
    np.testing.assert_array_equal(candidate_execution_indices(predictions, row), [3, 4])


def test_noise_alignment_is_explicit_and_piecewise() -> None:
    assert noise_index(353, 0, [(356, -1)]) == 353
    assert noise_index(356, 0, [(356, -1)]) == 355
    assert noise_index(999, 0, [(356, -1)]) == 998


def test_suffix_requires_sustained_direction_not_alternating_magnitude() -> None:
    chunks = np.zeros((2, 50, 3))
    chunks[0, 8:23, 0] = 0.2
    chunks[1, 8:, 0] = np.tile([0.2, -0.2], 21)
    statistics = traction_statistics(chunks, threshold=0.15, duration=15)
    assert statistics["early_mean_absolute"] == 0
    assert statistics["fraction_chunks_sustained_suffix"] == 0.5


def test_frame_identity_joins_on_pts_not_independent_counters(
    tmp_path, monkeypatch
) -> None:
    (tmp_path / "sensor--0.mcap").write_bytes(b"fixture")
    reference = SimpleNamespace(camera="cam_fork", frame_index=104, pts_ns=1000)
    camera = SimpleNamespace(frame_index=68, pts_ns=1000)
    inputs = SimpleNamespace(
        sent_to_model=True,
        step_tov_ns=2000,
        reset_generation=1,
        frames=[reference],
        aux=[],
    )
    messages = [
        (
            None,
            SimpleNamespace(topic="cam_fork/frame"),
            SimpleNamespace(log_time=3000),
            camera,
        ),
        (
            None,
            SimpleNamespace(topic="policy/inputs"),
            SimpleNamespace(log_time=4000),
            inputs,
        ),
    ]
    reader = SimpleNamespace(
        get_summary=lambda: True, iter_decoded_messages=lambda **kwargs: iter(messages)
    )
    monkeypatch.setattr(
        replay_d12_rollouts, "make_reader", lambda *args, **kwargs: reader
    )
    rows, _, skipped = replay_d12_rollouts.read_recording(tmp_path)
    assert rows[0]["input_frame"] == 104
    assert rows[0]["frame"] == 68
    assert rows[0]["frame_wall_ns"] == 3000
    assert not skipped
