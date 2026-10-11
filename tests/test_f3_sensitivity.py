"""Post-hoc F3-log sensitivity helper (D119)."""

import numpy as np

from vaultic.fusion.f3_sensitivity import HIST, bounded_history


def test_bounded_history_logs_and_caps_only_the_count():
    ctx = np.array(
        [[1.0, 0.0, 1.0, 5.0, 3.0], [2.0, 366.0, 0.0, 5.0, 4.0], [3.0, 1413.0, 1.0, 6.0, 5.0]]
    )
    out = bounded_history(ctx, cap=366.0)
    assert out[0, HIST] == 0.0 and np.isclose(out[1, HIST], np.log1p(366.0))
    assert out[2, HIST] == out[1, HIST]  # capped at the training maximum
    assert np.array_equal(np.delete(out, HIST, axis=1), np.delete(ctx, HIST, axis=1))
    assert ctx[2, HIST] == 1413.0  # input not modified
