"""Tests for NumPy namespace passthrough (Issue #51).

## Test: test_numpy_functions_accessible
- Goal: Verify that standard NumPy top-level functions (zeros, ones, arange,
        linspace, array) are callable via `import nemopy` and return results.
- Source: Issue #51 — "standard NumPy code works unchanged after swapping imports."
- Expected: Each function is callable and returns an ndarray with correct shape.

## Test: test_numpy_linalg_accessible
- Goal: Verify that numpy.linalg functions are accessible as nemopy.linalg
        attributes (solve, inv, eig).
- Source: Issue #51 — "np.linalg.solve(A, b) works — passthrough to numpy."
- Expected: nemopy.linalg.solve, .inv, .eig are callable.

## Test: test_numpy_random_accessible
- Goal: Verify that numpy.random functions are accessible as nemopy.random
        attributes (rand, randn).
- Source: Issue #51 — submodule proxying for np.random.
- Expected: nemopy.random.rand, .randn are callable and return arrays.

## Test: test_nemopy_names_take_precedence
- Goal: Verify that nemopy's own mat and eye shadow numpy.mat and numpy.eye,
        so nemopy's versions are used when accessed via the nemopy namespace.
- Source: Issue #51 — "nemopy's names take precedence"; DESIGN.md §5.8.
- Expected: nemopy.mat is nemopy's mat (not numpy.matrix); nemopy.eye(3)
            returns a Mat, not a plain ndarray.

## Test: test_nemopy_specific_exports
- Goal: Verify that nemopy-specific names (_c, _m, ColVec, Mat, ShapeError,
        ConventionWarning, as_col, as_mat) remain accessible after the
        namespace passthrough is added.
- Source: Issue #51; DESIGN.md §2.3.
- Expected: All 10 nemopy-specific names are accessible as nemopy attributes.

## Test: test_existing_code_pattern
- Goal: Integration test verifying that a standard NumPy workflow (create
        arrays, solve a linear system, use results) works with nemopy as a
        drop-in replacement alongside nemopy-specific constructors.
- Source: Issue #51 — "standard NumPy code pattern works unchanged."
- Expected: Mixed NumPy/nemopy code produces correct numerical results.
"""

import nemopy
from nemopy._constructors import mat as _nemopy_mat, eye as _nemopy_eye
from nemopy._core import ColVec, Mat


def test_numpy_functions_accessible():
    assert callable(nemopy.zeros)
    assert callable(nemopy.ones)
    assert callable(nemopy.arange)
    assert callable(nemopy.linspace)
    assert callable(nemopy.array)

    result = nemopy.zeros((3, 3))
    assert result.shape == (3, 3)

    result = nemopy.ones((2, 1))
    assert result.shape == (2, 1)

    result = nemopy.arange(5)
    assert result.shape == (5,)

    result = nemopy.linspace(0, 1, 10)
    assert result.shape == (10,)

    result = nemopy.array([1, 2, 3])
    assert result.shape == (3,)


def test_numpy_linalg_accessible():
    import numpy as np

    assert hasattr(nemopy, "linalg")
    assert callable(nemopy.linalg.solve)
    assert callable(nemopy.linalg.inv)
    assert callable(nemopy.linalg.eig)

    A = np.array([[2.0, 1.0], [1.0, 3.0]])
    b = np.array([[5.0], [7.0]])
    x = nemopy.linalg.solve(A, b)
    assert x.shape == (2, 1)


def test_numpy_random_accessible():
    assert hasattr(nemopy, "random")
    assert callable(nemopy.random.rand)
    assert callable(nemopy.random.randn)

    result = nemopy.random.rand(3, 2)
    assert result.shape == (3, 2)


def test_nemopy_names_take_precedence():
    assert nemopy.mat is _nemopy_mat
    assert nemopy.eye is _nemopy_eye
    assert isinstance(nemopy.eye(3), Mat)


def test_nemopy_specific_exports():
    assert hasattr(nemopy, "_c")
    assert hasattr(nemopy, "_m")
    assert hasattr(nemopy, "ColVec")
    assert hasattr(nemopy, "Mat")
    assert hasattr(nemopy, "ShapeError")
    assert hasattr(nemopy, "ConventionWarning")
    assert hasattr(nemopy, "as_col")
    assert hasattr(nemopy, "as_mat")
    assert hasattr(nemopy, "mat")
    assert hasattr(nemopy, "eye")


def test_existing_code_pattern():
    A = nemopy.mat([2, 1], [1, 3])
    b = nemopy._c[5, 7]

    x = nemopy.linalg.solve(A, b)
    assert x.shape == (2, 1)

    result = nemopy.zeros((3, 3))
    assert result.shape == (3, 3)

    identity = nemopy.eye(2)
    assert isinstance(identity, Mat)
    assert identity.shape == (2, 2)
