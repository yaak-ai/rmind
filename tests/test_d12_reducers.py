import numpy as np
from numpy.testing import assert_allclose

from rmind.data.d12 import TRACTION_SOURCES, box_mean

SRC_HZ = 83.0  # the traction publish rate on the Linde CAN bus
WINDOW_S = 0.1  # one decimated model step: episode_step=3 at ~33 ms per camera frame


def testbox_mean_averages_within_the_window() -> None:
    # Each 100 ms bin covers ~8.3 source samples of a linear ramp.
    src_t = np.arange(0.0, 1.0, 1 / SRC_HZ)
    src_v = src_t * 100.0
    ref_t = np.arange(0.0, 0.9, WINDOW_S)

    got = box_mean(src_t, src_v, ref_t, WINDOW_S)

    # A ramp averages to the window midpoint, up to the source sampling offset.
    assert_allclose(got, (ref_t + WINDOW_S / 2) * 100.0, atol=1.0)


def testbox_mean_splits_a_transition_that_nearest_would_miss() -> None:
    # A step halfway through the bin starting at t=5.0.
    src_t = np.arange(0.0, 10.0, 1 / SRC_HZ)
    src_v = np.where(src_t < 5.05, 0.0, 100.0)  # noqa: PLR2004

    got = box_mean(src_t, src_v, np.array([4.9, 5.0, 5.1]), WINDOW_S)

    # Nearest-asof reports 0.0 at t=5.0 and loses the first half of the step;
    # the box mean carries it across as a partial value.
    assert_allclose(got[[0, 2]], [0.0, 100.0])
    assert 30.0 < got[1] < 70.0  # noqa: PLR2004


def testbox_mean_falls_back_to_nearest_on_an_empty_window() -> None:
    # Reference rows can outpace a sparse source; an empty window must still
    # yield the last known value rather than a nan or a zero.
    src_t = np.array([0.0, 9.0])
    src_v = np.array([1.0, 2.0])

    got = box_mean(src_t, src_v, np.array([5.0]), WINDOW_S)

    assert_allclose(got, [2.0])


def testbox_mean_returns_float32() -> None:
    got = box_mean(np.arange(0.0, 1.0, 0.01), np.ones(100), np.array([0.5]), WINDOW_S)

    assert got.dtype == np.float32


def test_deprecated_traction_aliases_resolve_to_their_replacements() -> None:
    assert TRACTION_SOURCES["reported"] == TRACTION_SOURCES["operator"]
    assert TRACTION_SOURCES["command"] == TRACTION_SOURCES["uds_diag"]
