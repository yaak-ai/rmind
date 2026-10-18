import numpy as np

from rmind.scripts.analyze_d12_rollouts import (
    intervals,
    journal_events,
    pallet_goal_distance,
    past,
)


def test_override_intervals_do_not_bridge_missing_status():
    result = intervals(
        np.array([0.0, 0.1, 0.2, 2.0, 2.1]), [False, True, True, True, False]
    )
    assert result == [(0.1, 0.2, "gap"), (2.0, 2.1, "off")]


def test_past_join_never_uses_future_data():
    source = np.array([[1.0, 10.0], [3.0, 30.0]])
    result = past(source, [0.5, 1.0, 2.0, 3.0])
    assert np.isnan(result[0]).all()
    np.testing.assert_equal(result[1:, 1], [10.0, 10.0, 30.0])


def test_pallet_goal_error_is_not_ego_goal_distance():
    inputs = np.zeros((2, 13))
    inputs[0, 6:10] = [0.3, 0.4, 0.3, 0.35]
    inputs[1, 6:10] = [1.0, 0.4, 0.3, 0.35]
    distance = pallet_goal_distance(inputs)
    np.testing.assert_allclose(distance[0], 0.5)
    assert np.isnan(distance[1])


def test_journal_preserves_console_subsecond_time(tmp_path):
    journal = tmp_path / "journal.log"
    journal.write_text(
        "2026-10-15T09:40:08+00:00 host app: 2026-10-15T09:40:08.673298Z WARN console requested RELEASE\n"
    )
    events = journal_events(journal)
    assert events[0][1] == "RELEASE"
    assert abs(events[0][0] % 1 - 0.673298) < 1e-6
