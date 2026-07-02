"""Tests for advanced matrix decompositions (issue #84; §20.4).

## Test: test_ldu
- Goal: A.ldu() factors A = L @ D @ U with unit-triangular L and U and
        diagonal D; a matrix needing row exchanges (zero pivot) raises
        ValueError pointing at the pivot.
- Source: issue #84 core decompositions table ("A = L @ D @ U").
- Expected: reconstruction, unit diagonals, diagonal D; ValueError on the
            zero-pivot matrix.

## Test: test_qdr
- Goal: A.qdr() factors A = Q @ D @ R with orthogonal Q, diagonal D and
        unit-diagonal upper-triangular R (scaling separated from
        rotation).
- Source: issue #84 QDR table.
- Expected: reconstruction and structural properties hold.

## Test: test_schur
- Goal: A.schur() returns (T, Z) with A ≈ Z @ T @ Z.T and orthogonal Z.
- Source: issue #84 Schur decomposition row.
- Expected: reconstruction holds, Z orthogonal (LAPACK-delegated path).

## Test: test_polar
- Goal: A.polar() returns (U, P) with A ≈ U @ P, orthonormal-column U
        and symmetric positive-semidefinite P.
- Source: issue #84 polar decomposition row.
- Expected: reconstruction and structural properties hold on both
            backends (derived from the SVD kernel).

## Test: test_diagonalize
- Goal: A.diagonalize() returns (P, D) with A ≈ P @ D @ P^-1 for a
        diagonalizable A and raises ValueError for a defective one.
- Source: issue #84 similarity decomposition row ("raise if not
          diagonalizable").
- Expected: reconstruction for the diagonalizable case; ValueError for
            the Jordan-block case.

## Test: test_jordan
- Goal: A.jordan() returns (J, P) with A ≈ P @ J @ P^-1; J is diagonal
        for diagonalizable A and contains a Jordan block (superdiagonal
        1) for a defective A.
- Source: issue #84 Jordan normal form row.
- Expected: reconstruction; superdiagonal 1 present in the defective
            case.

## Test: test_advanced_square_guards
- Goal: Square-only advanced decompositions reject rectangular input
        with ShapeError.
- Source: issue #84 design principles ("Raise ShapeError for non-square
          matrices where squareness is required").
- Expected: ShapeError for ldu/schur/jordan/diagonalize on a 3x2 Mat.

## Test: test_tier3_requires_rust
- Goal: every Tier-3 advanced decomposition (ldu, qdr, schur, polar,
        diagonalize, jordan) raises a clear ImportError naming the method
        when _rust_core is absent, instead of assembling a NumPy/SciPy
        answer from Tier-2 primitives.
- Source: DESIGN_APPENDICES.md §20.1 (Rust-primary: no Python fallback)
          and §20.4 (these methods are tier "3 — Rust-primary"); issue
          #105.
- Expected: ImportError matching the method name with _core._RUST forced
            to None.
"""

import numpy as np
import pytest

from nemopy import Mat, ShapeError, _core, mat

# Advanced decompositions (issue #84) are Rust-primary per
# DESIGN_APPENDICES.md §20.1/§20.4: no NumPy/SciPy fallback exists, so the
# production computation only runs when the compiled extension is present.
# Correctness tests are gated on that; ImportError-when-absent is asserted
# separately in test_tier3_requires_rust.
requires_rust = pytest.mark.skipif(
    _core._RUST is None,
    reason="_rust_core extension not built (Tier-3 requires Rust)",
)

# Public Tier-3 advanced-decomposition methods (#84), asserted to raise
# ImportError when _rust_core is absent.
_TIER3_METHODS = (
    "ldu", "qdr", "schur", "polar", "diagonalize", "jordan",
)


def _np(x):
    return np.asarray(x)


@requires_rust
def test_ldu():
    A = mat([4, 2], [2, 5])
    L, D, U = A.ldu()
    np.testing.assert_allclose(_np(L) @ _np(D) @ _np(U), _np(A), atol=1e-10)
    np.testing.assert_allclose(np.diag(_np(L)), np.ones(2))
    np.testing.assert_allclose(np.diag(_np(U)), np.ones(2))
    np.testing.assert_allclose(_np(D), np.diag(np.diag(_np(D))), atol=1e-12)
    with pytest.raises(ValueError, match="(?i)pivot"):
        mat([0, 1], [1, 0]).ldu()


@requires_rust
def test_qdr():
    A = mat([3, 4, 0], [-4, 3, 1], [1, 0, 2])
    Q, D, R = A.qdr()
    np.testing.assert_allclose(_np(Q) @ _np(D) @ _np(R), _np(A), atol=1e-10)
    np.testing.assert_allclose(_np(Q).T @ _np(Q), np.eye(3), atol=1e-10)
    np.testing.assert_allclose(_np(D), np.diag(np.diag(_np(D))), atol=1e-12)
    np.testing.assert_allclose(np.diag(_np(R)), np.ones(3), atol=1e-12)


@requires_rust
def test_schur():
    pytest.importorskip("scipy")
    A = mat([4, 1, 0], [2, 3, 1], [0, 1, 2])
    T, Z = A.schur()
    assert isinstance(T, Mat) and isinstance(Z, Mat)
    np.testing.assert_allclose(_np(Z) @ _np(T) @ _np(Z).T, _np(A), atol=1e-9)
    np.testing.assert_allclose(_np(Z).T @ _np(Z), np.eye(3), atol=1e-10)


@requires_rust
def test_polar():
    A = mat([3, 1, 0], [1, 2, 1])  # tall (3, 2)
    U, P = A.polar()
    assert isinstance(U, Mat) and isinstance(P, Mat)
    np.testing.assert_allclose(_np(U) @ _np(P), _np(A), atol=1e-9)
    np.testing.assert_allclose(_np(U).T @ _np(U), np.eye(2), atol=1e-9)
    np.testing.assert_allclose(_np(P), _np(P).T, atol=1e-10)
    assert np.linalg.eigvalsh(_np(P)).min() >= -1e-10


@requires_rust
def test_diagonalize():
    A = mat([2, 0], [1, 3])
    P, D = A.diagonalize()
    np.testing.assert_allclose(
        _np(P) @ _np(D) @ np.linalg.inv(_np(P)), _np(A), atol=1e-9
    )
    np.testing.assert_allclose(_np(D), np.diag(np.diag(_np(D))), atol=1e-12)
    defective = mat([2, 0], [1, 2])  # single Jordan block, not diagonalizable
    with pytest.raises(ValueError, match="(?i)diagonaliz"):
        defective.diagonalize()


@requires_rust
def test_jordan():
    A = mat([2, 0], [1, 3])  # diagonalizable
    J, P = A.jordan()
    np.testing.assert_allclose(
        _np(P) @ _np(J) @ np.linalg.inv(_np(P)), _np(A), atol=1e-8
    )
    np.testing.assert_allclose(_np(J), np.diag(np.diag(_np(J))), atol=1e-8)

    B = mat([2, 0], [1, 2])  # defective: J = [[2,1],[0,2]]
    J, P = B.jordan()
    np.testing.assert_allclose(
        _np(P) @ _np(J) @ np.linalg.inv(_np(P)), _np(B), atol=1e-8
    )
    assert abs(_np(J)[0, 1] - 1.0) < 1e-8


def test_advanced_square_guards():
    rect = mat([1, 2], [3, 4], [5, 6])
    for method in ("ldu", "schur", "jordan", "diagonalize"):
        with pytest.raises(ShapeError):
            getattr(rect, method)()


@pytest.mark.parametrize("method", _TIER3_METHODS)
def test_tier3_requires_rust(method, monkeypatch):
    monkeypatch.setattr(_core, "_RUST", None)
    A = mat([4, 2], [2, 5])
    with pytest.raises(ImportError, match=method):
        getattr(A, method)()
