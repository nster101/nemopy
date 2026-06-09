"""Tests for NumPy compatibility fixes (issues #68, #69).

Each test has a stated goal and traces to a specific issue.
"""

import numpy as np
import pytest

from nemopy import Mat, ColVec, eye, mat
from nemopy._core import ShapeError


# ---------------------------------------------------------------------------
# Issue #68: eye() should accept dtype kwarg for NumPy compatibility
# ---------------------------------------------------------------------------


class TestEyeKwargs:
    """Goal: eye() accepts **kwargs without crashing (issue #68).
    Source: DESIGN.md §5.8, extended per issue #68 decision.
    """

    def test_eye_dtype_bool(self):
        """Goal: eye(3, dtype=bool) does not raise TypeError.
        Source: issue #68 — TestVdot::test_basic calls np.eye(3, dtype=bool).
        Expected: returns Mat of shape (3,3) with float64 dtype.
        """
        result = eye(3, dtype=bool)
        assert isinstance(result, Mat)
        assert result.shape == (3, 3)
        assert result.dtype == np.float64

    def test_eye_dtype_float32(self):
        """Goal: eye(3, dtype=np.float32) accepted, output always float64.
        Source: issue #68 — nemopy promotes all types to float64 per §3.
        Expected: returns Mat(3,3) with float64, not float32.
        """
        result = eye(3, dtype=np.float32)
        assert isinstance(result, Mat)
        assert result.dtype == np.float64

    def test_eye_dtype_complex(self):
        """Goal: eye(2, dtype=complex) accepted.
        Source: issue #68.
        Expected: returns Mat(2,2) with float64 (complex part dropped by Mat).
        """
        result = eye(2, dtype=complex)
        assert isinstance(result, Mat)
        assert result.shape == (2, 2)

    def test_eye_M_kwarg_rectangular(self):
        """Goal: eye(3, M=4) produces a rectangular identity.
        Source: issue #68 — NumPy's eye() accepts M for non-square.
        Expected: Mat of shape (3, 4).
        """
        result = eye(3, M=4)
        assert isinstance(result, Mat)
        assert result.shape == (3, 4)
        for i in range(3):
            assert result[i, i] == 1.0

    def test_eye_k_kwarg_diagonal_offset(self):
        """Goal: eye(3, k=1) produces an identity shifted by 1.
        Source: issue #68 — NumPy's eye() accepts k for diagonal offset.
        Expected: Mat of shape (3,3) with ones on the superdiagonal.
        """
        result = eye(3, k=1)
        assert isinstance(result, Mat)
        assert result.shape == (3, 3)
        assert result[0, 1] == 1.0
        assert result[0, 0] == 0.0

    def test_eye_n_only_unchanged(self):
        """Goal: eye(n) with no kwargs still works as before.
        Source: DESIGN.md §5.8 — existing behavior must be preserved.
        Expected: same as Mat(np.eye(4)).
        """
        result = eye(4)
        expected = Mat(np.eye(4))
        assert isinstance(result, Mat)
        assert result.shape == (4, 4)
        assert np.array_equal(result, expected)


# ---------------------------------------------------------------------------
# Issue #69: Mat.__getitem__ boolean indexing conflicts with numpy.testing
# ---------------------------------------------------------------------------


class TestMatBooleanIndexing:
    """Goal: flat boolean indexing on Mat returns plain ndarray (issue #69).
    Source: DESIGN.md §6.3 (column extraction contract) and issue #69.
    """

    def test_flat_boolean_mask_returns_plain_ndarray(self):
        """Goal: A[bool_mask] with a flat mask returns plain ndarray, not ColVec.
        Source: issue #69 — numpy.testing does a[mask] with flat boolean masks.
        Expected: plain 1D ndarray.
        """
        A = mat([1, 2, 3], [4, 5, 6])
        mask = np.array([[True, False], [False, True], [True, False]])
        result = A[mask]
        assert not isinstance(result, (ColVec, Mat))
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1

    def test_assert_equal_on_mat(self):
        """Goal: numpy.testing.assert_equal works on Mat without IndexError.
        Source: issue #69 — this is the exact failure scenario.
        Expected: no error raised.
        """
        from numpy.testing import assert_equal
        from numpy.linalg import matrix_power

        M = eye(4)
        mz = matrix_power(M, 0)
        assert_equal(mz, np.eye(4))

    def test_column_boolean_mask_still_works(self):
        """Goal: A[:, bool_mask] column mask still returns Mat (regression guard).
        Source: DESIGN.md §6.3 — column boolean mask → Mat.
        Expected: Mat of shape (3, k).
        """
        A = mat([1, 2, 3], [4, 5, 6], [7, 8, 9])
        col_mask = np.array([True, False, True])
        result = A[:, col_mask]
        assert isinstance(result, Mat)
        assert result.shape == (3, 2)

    def test_single_column_extraction_unchanged(self):
        """Goal: A[:, j] still returns ColVec (regression guard).
        Source: DESIGN.md §6.3 — single column → ColVec.
        Expected: ColVec of shape (3, 1).
        """
        A = mat([1, 2, 3], [4, 5, 6])
        result = A[:, 0]
        assert isinstance(result, ColVec)
        assert result.shape == (3, 1)

    def test_scalar_extraction_unchanged(self):
        """Goal: A[i, j] still returns float (regression guard).
        Source: DESIGN.md §6.3 — element extraction.
        Expected: float.
        """
        A = mat([1, 2, 3], [4, 5, 6])
        result = A[1, 0]
        assert isinstance(result, float)
        assert result == 2.0
