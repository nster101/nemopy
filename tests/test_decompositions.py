"""Tests for core matrix decompositions as Mat methods (issue #76; §20.4).

## Test: test_svd
- Goal: A.svd() returns a named SVDResult of nemopy types whose product
        reconstructs A (thin SVD, singular values descending).
- Source: issue #76 proposed API; §20.3 type persistence.
- Expected: U Mat, s ColVec, Vt Mat; U @ diag(s) @ Vt ≈ A; named field
            access works.

## Test: test_svd_wide
- Goal: SVD of a wide matrix (n < k) keeps thin shapes and reconstructs.
- Source: issue #76 proposed API (decompositions on general Mat).
- Expected: U (n,n), s (n,1), Vt (n,k); reconstruction holds.

## Test: test_qr
- Goal: A.qr() returns Q with orthonormal columns and upper-triangular R
        reconstructing A.
- Source: issue #76 proposed API.
- Expected: Q.T @ Q ≈ I, R upper triangular, Q @ R ≈ A.

## Test: test_lu
- Goal: A.lu() returns permutation P, unit-lower L, upper U with
        A ≈ P @ L @ U.
- Source: issue #76 proposed API.
- Expected: reconstruction holds; L unit lower; U upper; P a permutation.

## Test: test_cholesky
- Goal: A.cholesky() returns lower-triangular L with L @ L.T ≈ A for
        symmetric positive-definite A.
- Source: issue #76 proposed API.
- Expected: Mat, lower triangular, reconstructs A.

## Test: test_cholesky_not_positive_definite
- Goal: Cholesky of a symmetric non-PD matrix fails with ValueError
        (np.linalg.LinAlgError subclasses ValueError) naming positive
        definiteness, on both backends.
- Source: issue #76 — "cholesky() requires square symmetric PD".
- Expected: ValueError matching "positive definite" (case-insensitive).

## Test: test_shape_guards
- Goal: cholesky()/eigh() reject non-symmetric input; square-only
        decompositions reject rectangular matrices with ShapeError.
- Source: issue #76 design principle 5 (shape guards).
- Expected: ShapeError raised in each case.

## Test: test_eig_real
- Goal: A.eig() returns eigenpairs satisfying A @ v = λ v with nemopy
        types when the spectrum is real.
- Source: issue #76 proposed API (column eigenvectors).
- Expected: values ColVec, vectors Mat, defining relation per column.

## Test: test_eig_complex_spectrum
- Goal: When the spectrum is complex, eig() returns plain complex arrays
        (nemopy types are float64-only per DESIGN.md §4) rather than
        corrupting data.
- Source: issue #76 API; DESIGN.md §4 dtype contract.
- Expected: complex dtype results satisfying the defining relation.

## Test: test_eigh
- Goal: A.eigh() returns ascending real eigenvalues and orthonormal
        eigenvectors for symmetric A, agreeing with np.linalg.eigh.
- Source: issue #76 proposed API (symmetric eigendecomposition).
- Expected: values match, V.T @ V ≈ I, A @ V ≈ V @ diag(w).
"""

import numpy as np
import pytest

from nemopy import ColVec, Mat, ShapeError, mat


def _np(x):
    return np.asarray(x)


def test_svd(backend):
    A = mat([1, 2, 3, 4], [5, 4, 2, 1], [0, 1, 1, 0])
    U, s, Vt = A.svd()
    assert isinstance(U, Mat) and isinstance(s, ColVec) and isinstance(Vt, Mat)
    assert list(s.to_flat()) == sorted(s.to_flat(), reverse=True)
    np.testing.assert_allclose(
        _np(U) @ np.diag(s.to_flat()) @ _np(Vt), _np(A), atol=1e-10
    )
    res = A.svd()
    np.testing.assert_allclose(_np(res.U), _np(U))


def test_svd_wide(backend):
    A = mat([1, 2], [3, 4], [5, 6], [7, 9])  # shape (2, 4)
    U, s, Vt = A.svd()
    assert U.shape == (2, 2) and s.shape == (2, 1) and Vt.shape == (2, 4)
    np.testing.assert_allclose(
        _np(U) @ np.diag(s.to_flat()) @ _np(Vt), _np(A), atol=1e-10
    )


def test_qr(backend):
    A = mat([1, 2, 3, 4], [5, 4, 2, 1], [0, 1, 1, 0])
    Q, R = A.qr()
    assert isinstance(Q, Mat) and isinstance(R, Mat)
    np.testing.assert_allclose(_np(Q).T @ _np(Q), np.eye(3), atol=1e-10)
    np.testing.assert_allclose(_np(R), np.triu(_np(R)), atol=1e-12)
    np.testing.assert_allclose(_np(Q) @ _np(R), _np(A), atol=1e-10)


def test_lu(backend):
    A = mat([2, 4, 8], [1, 3, 7], [1, 3, 9])
    P, L, U = A.lu()
    assert all(isinstance(x, Mat) for x in (P, L, U))
    np.testing.assert_allclose(_np(P) @ _np(L) @ _np(U), _np(A), atol=1e-10)
    np.testing.assert_allclose(np.diag(_np(L)), np.ones(3))
    np.testing.assert_allclose(_np(L), np.tril(_np(L)), atol=1e-12)
    np.testing.assert_allclose(_np(U), np.triu(_np(U)), atol=1e-12)
    p = _np(P)
    assert ((p == 0) | (p == 1)).all()
    np.testing.assert_array_equal(p.sum(axis=0), np.ones(3))


def test_cholesky(backend):
    A = mat([4, 2], [2, 3])
    L = A.cholesky()
    assert isinstance(L, Mat)
    np.testing.assert_allclose(_np(L), np.tril(_np(L)), atol=1e-12)
    np.testing.assert_allclose(_np(L) @ _np(L).T, _np(A), atol=1e-10)


def test_cholesky_not_positive_definite(backend):
    A = mat([1, 2], [2, 1])  # symmetric, eigenvalues 3 and -1
    with pytest.raises(ValueError, match="(?i)positive definite"):
        A.cholesky()


def test_shape_guards():
    nonsym = mat([1, 0], [5, 1])
    with pytest.raises(ShapeError):
        nonsym.cholesky()
    with pytest.raises(ShapeError):
        nonsym.eigh()
    rect = mat([1, 2], [3, 4], [5, 6])
    for method in ("lu", "cholesky", "eig", "eigh"):
        with pytest.raises(ShapeError):
            getattr(rect, method)()


def test_eig_real():
    A = mat([2, 0], [1, 3])
    vals, vecs = A.eig()
    assert isinstance(vals, ColVec) and isinstance(vecs, Mat)
    for j in range(2):
        np.testing.assert_allclose(
            _np(A) @ _np(vecs)[:, j], vals[j] * _np(vecs)[:, j], atol=1e-10
        )


def test_eig_complex_spectrum():
    A = mat([0, 1], [-1, 0])  # rotation: eigenvalues ±i
    vals, vecs = A.eig()
    assert np.iscomplexobj(vals) and np.iscomplexobj(vecs)
    np.testing.assert_allclose(
        _np(A).astype(complex) @ vecs, vecs @ np.diag(np.ravel(vals)), atol=1e-10
    )


def test_eigh(backend):
    A = mat([2, 1, 0], [1, 3, 1], [0, 1, 4])
    w, V = A.eigh()
    assert isinstance(w, ColVec) and isinstance(V, Mat)
    np.testing.assert_allclose(w.to_flat(), np.linalg.eigh(_np(A))[0], atol=1e-10)
    np.testing.assert_allclose(_np(V).T @ _np(V), np.eye(3), atol=1e-10)
    np.testing.assert_allclose(
        _np(A) @ _np(V), _np(V) @ np.diag(w.to_flat()), atol=1e-9
    )
