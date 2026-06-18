import numpy as np
import pytest

import stablebear as sb


def test_mean_dim_equal_ndim_raises():
    """mean with dim == ndim must raise, not silently return a wrong shape."""
    # BUG: Out-of-range dim in reductions is unvalidated (dim == ndim)
    # Expected: an out-of-range axis raises a clean Python exception; never a
    # silent wrong shape, and never heap corruption / interpreter abort.
    A = sb.zeros((3, 4))
    with pytest.raises((IndexError, ValueError)):
        sb.mean(A, dim=2)


def test_max_time_dim_equal_ndim_raises():
    """max_time with dim == ndim must raise, not silently return a wrong shape."""
    A = sb.zeros((3, 4))
    with pytest.raises((IndexError, ValueError)):
        sb.max_time(A, dim=2)


def test_mean_dim_greater_than_ndim_raises():
    """mean with dim > ndim must raise cleanly (previously corrupted the heap)."""
    A = sb.zeros((3, 4))
    for bad_dim in (3, 5, 6, 100):
        with pytest.raises((IndexError, ValueError)):
            sb.mean(A, dim=bad_dim)


def test_max_time_dim_greater_than_ndim_raises():
    """max_time with dim > ndim must raise cleanly (previously corrupted the heap)."""
    A = sb.zeros((3, 4))
    for bad_dim in (3, 5, 6, 100):
        with pytest.raises((IndexError, ValueError)):
            sb.max_time(A, dim=bad_dim)


def test_mean_1d_out_of_range_raises():
    """1-D reduction with dim >= ndim must raise rather than silently return."""
    A = sb.zeros((3,))
    for bad_dim in (1, 2):
        with pytest.raises((IndexError, ValueError)):
            sb.mean(A, dim=bad_dim)


def test_mean_in_range_dim_still_works():
    """Sanity check: valid dims keep producing the documented reduced shape."""
    A = sb.zeros((3, 4))
    assert sb.mean(A, dim=0).shape == (4,)
    assert sb.mean(A, dim=1).shape == (3,)
    assert sb.max_time(A, dim=0).shape == (4,)
    assert sb.max_time(A, dim=1).shape == (3,)


def test_max_time_empty_reduced_axis_raises_not_segfault():
    """max_time over a size-0 reduced axis must raise cleanly, never SIGSEGV.

    Issue #25 (bug scan): ``max_time`` segfaulted on an empty reduction
    dimension while ``mean`` handled it. Resolved together with #46 —
    ``max_element`` now rejects an empty reduction range with ``ValueError``
    (``max`` has no identity over an empty range). ``mean`` still returns a
    valid reduced tensor over the same empty axes, so the two stay consistent.
    """
    # 1-D empty tensor reduced along its only (empty) axis.
    with pytest.raises(ValueError):
        sb.max_time(sb.zeros((0,)), dim=0)
    # Empty inner dimension of a higher-rank tensor.
    with pytest.raises(ValueError):
        sb.max_time(sb.zeros((3, 0)), dim=1)
    # Contrast: mean returns a valid reduced tensor over the same empty axes.
    assert sb.mean(sb.zeros((0,)), dim=0).shape == (1,)
    assert sb.mean(sb.zeros((3, 0)), dim=1).shape == (3,)
