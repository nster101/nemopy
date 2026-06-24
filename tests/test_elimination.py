"""Tests for Gaussian elimination and row echelon forms (issue #85; §20.4).

## Test: test_ref_structure
- Goal: A.ref() returns a Mat in row echelon form: each row's leading
        nonzero sits strictly right of the previous row's.
- Source: issue #85 methods table ("upper triangular form via forward
          elimination with partial pivoting").
- Expected: staircase structure holds and row space is preserved (rank
            of the stack equals rank of A).

## Test: test_ref_pivot_none
- Goal: pivot="none" performs classic unpivoted forward elimination
        (pedagogical mode) and an unknown strategy is rejected.
- Source: issue #85 pivoting strategy section.
- Expected: known hand-computed REF for a matrix needing no swaps;
            ValueError for pivot="bogus"; ValueError when a zero pivot
            forces a swap that pivot="none" forbids.

## Test: test_rref_known
- Goal: A.rref() returns the canonical reduced row echelon form plus
        pivot column indices.
- Source: issue #85 methods table ("fully reduced form + list of pivot
          column indices").
- Expected: exact known RREF and pivots (0, 2) for a rank-2 matrix with
            a dependent second column.

## Test: test_rank
- Goal: A.rank() counts pivot columns.
- Source: issue #85 methods table.
- Expected: 2 for the rank-2 matrix, n for the identity.

## Test: test_nullspace
- Goal: A.nullspace() returns a basis for the null space as Mat columns;
        full-rank matrices give an empty (k, 0) basis.
- Source: issue #85 methods table.
- Expected: A @ N ≈ 0 with dim = k - rank(A); empty basis on identity.

## Test: test_gaussian_eliminate
- Goal: A.gaussian_eliminate(b) solves Ax = b by forward elimination +
        back substitution; singular systems raise ValueError.
- Source: issue #85 methods table.
- Expected: solution matches np.linalg.solve; ValueError mentioning
            "singular" for a singular A.

## Test: test_gauss_jordan
- Goal: A.gauss_jordan() returns the RREF of A; A.gauss_jordan(b)
        returns the solution ColVec.
- Source: issue #85 methods table ("if b provided, returns solution").
- Expected: RREF equals A.rref()[0]; solution matches np.linalg.solve.

## Test: test_augment
- Goal: A.augment(b) builds the [A | b] augmented Mat and rejects row
        mismatches.
- Source: issue #85 methods table and augmented-systems note.
- Expected: equals A | b; ShapeError on mismatched rows.

## Test: test_ref_steps
- Goal: A.ref_steps() yields (Mat, description) elimination steps ending
        at the REF (pedagogical mode, Python-side by design).
- Source: issue #85 step-by-step mode section.
- Expected: every item is (Mat, str); final Mat equals A.ref().

## Test: test_gaussian_square_guard
- Goal: gaussian_eliminate rejects non-square systems with ShapeError.
- Source: issue #85 (solving Ax = b is defined for square A here;
          rectangular systems are out of the method's contract).
- Expected: ShapeError on a 3x2 Mat.

## Test: test_tier3_requires_rust
- Goal: every Tier-3 elimination method (ref, rref, rank, nullspace,
        gaussian_eliminate, gauss_jordan, augment) raises a clear
        ImportError naming the method when _rust_core is absent, rather
        than computing a NumPy/SciPy answer.
- Source: DESIGN_APPENDICES.md §20.1 (Rust-primary: no Python fallback)
          and §20.4 (these methods are tier "3 — Rust-primary"); issue
          #105.
- Expected: ImportError matching the method name with _core._RUST forced
            to None.
"""

import numpy as np
import pytest

from nemopy import ColVec, Mat, ShapeError, _c, _core, mat

# Tier-3 elimination (issue #85) is Rust-primary per DESIGN_APPENDICES.md
# §20.1/§20.4: there is no NumPy fallback, so the production computation
# only runs when the compiled extension is present. Correctness tests are
# gated on that; ImportError-when-absent is asserted separately.
requires_rust = pytest.mark.skipif(
    _core._RUST is None,
    reason="_rust_core extension not built (Tier-3 requires Rust)",
)

# Public Tier-3 elimination methods (#85) and a representative call for
# each, used to assert the ImportError contract when _rust_core is absent.
_TIER3_CALLS = {
    "ref": (),
    "rref": (),
    "rank": (),
    "nullspace": (),
    "gaussian_eliminate": (_c[5, 10],),
    "gauss_jordan": (_c[5, 10],),
    "augment": (_c[5, 10],),
}


def _np(x):
    return np.asarray(x)


def _staircase_ok(r, tol=1e-12):
    lead = -1
    for i in range(r.shape[0]):
        nz = np.flatnonzero(np.abs(r[i]) > tol)
        if nz.size == 0:
            lead = r.shape[1]
            continue
        if nz[0] <= lead:
            return False
        lead = nz[0]
    return True


@requires_rust
def test_ref_structure():
    A = mat([2, -3, -2], [1, -1, 1], [-1, 2, 2])
    R = A.ref()
    assert isinstance(R, Mat)
    assert _staircase_ok(_np(R))
    stacked = np.vstack([_np(A), _np(R)])
    assert np.linalg.matrix_rank(stacked) == np.linalg.matrix_rank(_np(A))


@requires_rust
def test_ref_pivot_none():
    A = mat([2, 1], [4, 3])  # rows [2,4],[1,3]: no swap needed
    R = A.ref(pivot="none")
    np.testing.assert_allclose(_np(R), [[2.0, 4.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="pivot"):
        A.ref(pivot="bogus")
    with pytest.raises(ValueError, match="(?i)zero pivot"):
        mat([0, 1], [1, 0]).ref(pivot="none")


@requires_rust
def test_rref_known():
    A = mat([1, 2, 3], [2, 4, 6], [1, 1, 1])  # col2 = 2*col1 -> rank 2
    R, pivots = A.rref()
    assert isinstance(R, Mat)
    assert tuple(pivots) == (0, 2)
    np.testing.assert_allclose(
        _np(R), [[1.0, 2.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
        atol=1e-12,
    )


@requires_rust
def test_rank():
    A = mat([1, 2, 3], [2, 4, 6], [1, 1, 1])
    assert A.rank() == 2
    assert isinstance(A.rank(), int)
    B = mat([1, 0], [0, 1])
    assert B.rank() == 2


@requires_rust
def test_nullspace():
    A = mat([1, 2, 3], [2, 4, 6], [1, 1, 1])
    N = A.nullspace()
    assert isinstance(N, Mat)
    assert N.shape == (3, 1)
    np.testing.assert_allclose(_np(A) @ _np(N), np.zeros((3, 1)), atol=1e-10)
    full = mat([1, 0], [0, 1]).nullspace()
    assert full.shape == (2, 0)


@requires_rust
def test_gaussian_eliminate():
    A = mat([2, 1], [1, 3])
    b = _c[5, 10]
    x = A.gaussian_eliminate(b)
    assert isinstance(x, ColVec)
    np.testing.assert_allclose(
        x.to_flat(), np.linalg.solve(_np(A), _np(b)).ravel(), atol=1e-12
    )
    singular = mat([1, 2], [2, 4])
    with pytest.raises(ValueError, match="(?i)singular"):
        singular.gaussian_eliminate(_c[1, 2])


@requires_rust
def test_gauss_jordan():
    A = mat([1, 2, 3], [2, 4, 6], [1, 1, 1])
    R = A.gauss_jordan()
    assert isinstance(R, Mat)
    np.testing.assert_allclose(_np(R), _np(A.rref()[0]), atol=1e-12)
    S = mat([2, 1], [1, 3])
    b = _c[5, 10]
    x = S.gauss_jordan(b)
    assert isinstance(x, ColVec)
    np.testing.assert_allclose(
        x.to_flat(), np.linalg.solve(_np(S), _np(b)).ravel(), atol=1e-12
    )


def test_augment():
    A = mat([1, 2], [3, 4])
    with pytest.raises(ShapeError):
        A.augment(_c[1, 2, 3])
    if _core._RUST is None:
        return
    b = _c[5, 6]
    Ab = A.augment(b)
    assert isinstance(Ab, Mat)
    np.testing.assert_array_equal(_np(Ab), _np(A | b))


def test_ref_steps():
    A = mat([2, -3, -2], [1, -1, 1], [-1, 2, 2])
    steps = list(A.ref_steps())
    assert steps
    for step, desc in steps:
        assert isinstance(step, Mat)
        assert isinstance(desc, str)
    if _core._RUST is not None:
        np.testing.assert_allclose(_np(steps[-1][0]), _np(A.ref()), atol=1e-12)


def test_gaussian_square_guard():
    rect = mat([1, 2], [3, 4], [5, 6])
    with pytest.raises(ShapeError):
        rect.gaussian_eliminate(_c[1, 2])


@pytest.mark.parametrize("method", sorted(_TIER3_CALLS))
def test_tier3_requires_rust(method, monkeypatch):
    monkeypatch.setattr(_core, "_RUST", None)
    A = mat([2, 1], [1, 3])
    with pytest.raises(ImportError, match=method):
        getattr(A, method)(*_TIER3_CALLS[method])
