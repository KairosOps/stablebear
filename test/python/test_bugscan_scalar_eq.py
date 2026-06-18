import numpy as np
import numpy.testing as npt
import pytest

import stablebear as sb


# ---------------------------------------------------------------------------
# Bug #47: == / != against a Python/NumPy scalar returned a plain Python bool
# (via Python's identity fallback) instead of an elementwise BoolTensor,
# silently breaking the documented masking idiom t[t == scalar]. (#57 fixed
# the ordering operators < <= > >=; == and != were still broken.)
# ---------------------------------------------------------------------------

_NUMERIC = [
    pytest.param(sb.FloatTensor, np.float64, id="float64"),
    pytest.param(sb.FloatTensor, np.float32, id="float32"),
    pytest.param(sb.IntTensor, np.int64, id="int64"),
    pytest.param(sb.IntTensor, np.int32, id="int32"),
]


@pytest.mark.parametrize("TensorType, dt", _NUMERIC)
def test_eq_scalar_returns_bool_tensor(TensorType, dt):
    np_a = np.array([1, 2, 3, 2], dtype=dt)
    r = TensorType(np_a) == 2
    assert isinstance(r, sb.BoolTensor)
    npt.assert_array_equal(np.asarray(r), np_a == 2)


@pytest.mark.parametrize("TensorType, dt", _NUMERIC)
def test_ne_scalar_returns_bool_tensor(TensorType, dt):
    np_a = np.array([1, 2, 3, 2], dtype=dt)
    r = TensorType(np_a) != 2
    assert isinstance(r, sb.BoolTensor)
    npt.assert_array_equal(np.asarray(r), np_a != 2)


def test_eq_scalar_mask_select_idiom():
    """The headline idiom: t[t == scalar] selects the matching elements."""
    np_a = np.array([1.0, 2.0, 3.0, 2.0])
    t = sb.FloatTensor(np_a)
    npt.assert_array_equal(np.asarray(t[t == 2.0]), np_a[np_a == 2.0])


def test_eq_numpy_scalar_and_0d_array():
    np_a = np.array([1.0, 2.0, 3.0])
    t = sb.FloatTensor(np_a)
    for rhs in (np.float64(2.0), np.array(2.0)):
        r = t == rhs
        assert isinstance(r, sb.BoolTensor)
        npt.assert_array_equal(np.asarray(r), np_a == rhs)


def test_eq_ndarray_broadcasts():
    np_a = np.array([[1.0, 2.0], [3.0, 4.0]])
    np_b = np.array([1.0, 4.0])
    r = sb.FloatTensor(np_a) == np_b
    assert isinstance(r, sb.BoolTensor)
    npt.assert_array_equal(np.asarray(r), np_a == np_b)


def test_eq_tensor_rhs_still_works():
    np_a = np.array([1.0, 2.0, 3.0])
    np_b = np.array([1.0, 9.0, 3.0])
    r = sb.FloatTensor(np_a) == sb.FloatTensor(np_b)
    assert isinstance(r, sb.BoolTensor)
    npt.assert_array_equal(np.asarray(r), np_a == np_b)


def test_eq_2d_scalar_broadcast():
    np_a = np.arange(6, dtype=np.float64).reshape(2, 3)
    r = sb.FloatTensor(np_a) == 3.0
    assert isinstance(r, sb.BoolTensor)
    assert r.shape == np_a.shape
    npt.assert_array_equal(np.asarray(r), np_a == 3.0)
