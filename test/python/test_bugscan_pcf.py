import numpy as np
import pytest

import stablebear as sb

Pcf = sb.Pcf


# ---------------------------------------------------------------------------
# Bug #11: Pcf construction validated t0 on the *unsorted* input row 0, then
# std::sort could move a negative time to the front -- yielding a PCF with
# t0 < 0 that happily evaluated at negative times. The t=0 invariant is now
# checked after sorting, and evaluation guards against 0 directly.
# ---------------------------------------------------------------------------


def test_negative_time_hidden_by_sort_rejected():
    # Row 0 is t=0 (so the old check passed), but a later row has t=-1.
    with pytest.raises(ValueError, match="t=0"):
        Pcf(np.array([[0.0, 1.0], [2.0, 2.0], [-1.0, 99.0]], dtype=np.float64))


def test_min_time_above_zero_rejected():
    with pytest.raises(ValueError, match="t=0"):
        Pcf(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64))


def test_multiple_negative_times_rejected():
    with pytest.raises(ValueError, match="t=0"):
        Pcf(np.array([[0.0, 1.0], [-2.0, 5.0], [-1.0, 9.0]], dtype=np.float64))


def test_unsorted_but_valid_is_accepted_and_sorted():
    # The minimum time is 0 -> a valid PCF given out of order; it must be
    # accepted and stored sorted (order of the input rows is irrelevant).
    f = Pcf(np.array([[2.0, 2.0], [0.0, 1.0]], dtype=np.float64))
    assert f.to_numpy().tolist() == [[0.0, 1.0], [2.0, 2.0]]


def test_intpcf_negative_time_hidden_by_sort_rejected():
    with pytest.raises(ValueError, match="t=0"):
        Pcf(np.array([[0, 1], [2, 2], [-1, 9]], dtype=np.int64))


def test_eval_negative_time_scalar_raises():
    f = Pcf(np.array([[0.0, 1.0], [2.0, 2.0]], dtype=np.float64))
    with pytest.raises(Exception, match="time 0"):
        f(-0.5)


def test_eval_negative_time_array_raises():
    f = Pcf(np.array([[0.0, 1.0], [2.0, 2.0]], dtype=np.float64))
    with pytest.raises(Exception, match="time 0"):
        f(np.array([-0.5, 0.5]))


def test_eval_valid_times_ok():
    f = Pcf(np.array([[0.0, 1.0], [2.0, 2.0]], dtype=np.float64))
    assert f(0.0) == 1.0
    assert f(1.0) == 1.0
    assert f(2.5) == 2.0
    np.testing.assert_array_equal(np.asarray(f(np.array([0.0, 2.5]))), [1.0, 2.0])
