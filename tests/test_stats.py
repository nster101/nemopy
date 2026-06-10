"""Tests for statistical methods on ColVec and Mat (issue #77; §20.4 stats domain).

Twelve tests are justified per CLAUDE.md §6.5: each covers one distinct
documented behaviour of the issue #77 API surface.

## Test: test_colvec_scalar_stats
- Goal: ColVec.mean/std/var/sum return Python floats with NumPy values
        (ddof=0 NumPy semantics for std/var).
- Source: issue #77 ColVec methods table.
- Expected: floats equal to np.mean/np.std/np.var/np.sum of the data.

## Test: test_colvec_norm
- Goal: ColVec.norm() is the L2 norm by default and honours p=1 / p=inf.
- Source: issue #77 — "support L1, L2, Linf via p parameter".
- Expected: values match np.linalg.norm with ord p.

## Test: test_colvec_normalize
- Goal: ColVec.normalize() returns a unit-L2 ColVec.
- Source: issue #77 — "normalize() — unit vector (u / ||u||)".
- Expected: ColVec type, norm 1, direction preserved.

## Test: test_colvec_normalize_zero_raises
- Goal: Normalizing a zero vector raises ValueError (undefined direction).
- Source: issue #77 normalize semantics (division by zero norm is
          mathematically undefined; spec silent on NaN propagation, so an
          explicit error is the safe surface).
- Expected: ValueError mentioning zero norm.

## Test: test_colvec_cumsum
- Goal: ColVec.cumsum() returns the running sum as a ColVec.
- Source: issue #77 ColVec methods table.
- Expected: ColVec([1, 3, 6]) for input [1, 2, 3].

## Test: test_mat_mean_colwise_and_rowwise
- Goal: Mat.mean() defaults to column-wise (k,1); axis=1 gives row-wise
        (n,1); both are ColVec.
- Source: issue #77 — column-first convention, Mat methods table.
- Expected: values match np.mean(axis=0/1), shapes (k,1)/(n,1).

## Test: test_mat_mean_invalid_axis
- Goal: Axes other than 0/1 are rejected; the column-first API does not
        silently fall back to flat reductions.
- Source: issue #77 Mat methods table (only axis 0 and 1 defined).
- Expected: ValueError.

## Test: test_mat_std_colwise
- Goal: Mat.std() is the column-wise standard deviation (ddof=0).
- Source: issue #77 Mat methods table.
- Expected: matches np.std(a, axis=0) as a (k,1) ColVec.

## Test: test_mat_cov
- Goal: Mat.cov() treats columns as variables (rows as observations).
- Source: issue #77 — "same as np.cov with rowvar=False".
- Expected: Mat of shape (k,k) equal to np.cov(a, rowvar=False).

## Test: test_mat_corr
- Goal: Mat.corr() is the correlation matrix of columns.
- Source: issue #77 Mat methods table.
- Expected: Mat equal to np.corrcoef(a, rowvar=False), unit diagonal.

## Test: test_mat_normalize
- Goal: Mat.normalize() rescales every column to unit L2 norm; a zero
        column raises ValueError naming the column.
- Source: issue #77 — "column-wise L2 normalization"; fused kernel note.
- Expected: unit column norms, Mat type; ValueError on zero column.

## Test: test_np_mean_protocol_unchanged
- Goal: np.mean/np.sum on nemopy types keep returning plain scalars via
        __array_function__ — the method overrides must not leak into the
        NumPy namespace path.
- Source: DESIGN.md §11 NumPy interoperability (protocol behaviour).
- Expected: plain Python/NumPy scalar equal to the flat mean.
"""

import numpy as np
import pytest

from nemopy import ColVec, Mat, mat, _c


def test_colvec_scalar_stats():
    u = _c[1, 2, 3, 4]
    data = np.array([1.0, 2.0, 3.0, 4.0])
    assert isinstance(u.mean(), float) and u.mean() == data.mean()
    assert isinstance(u.std(), float) and u.std() == data.std()
    assert isinstance(u.var(), float) and u.var() == data.var()
    assert isinstance(u.sum(), float) and u.sum() == data.sum()


def test_colvec_norm():
    u = _c[3, -4]
    assert u.norm() == 5.0
    assert u.norm(p=1) == 7.0
    assert u.norm(p=np.inf) == 4.0


def test_colvec_normalize():
    u = _c[3, 4]
    n = u.normalize()
    assert isinstance(n, ColVec)
    np.testing.assert_allclose(n.to_flat(), [0.6, 0.8])


def test_colvec_normalize_zero_raises():
    with pytest.raises(ValueError, match="zero norm"):
        _c[0, 0, 0].normalize()


def test_colvec_cumsum():
    c = _c[1, 2, 3].cumsum()
    assert isinstance(c, ColVec)
    assert c.to_list() == [1.0, 3.0, 6.0]


def test_mat_mean_colwise_and_rowwise(backend):
    A = mat([1, 2, 3], [4, 5, 6])
    a = np.asarray(A)
    m = A.mean()
    assert isinstance(m, ColVec) and m.shape == (2, 1)
    np.testing.assert_allclose(m.to_flat(), a.mean(axis=0))
    np.testing.assert_allclose(A.mean(axis=0).to_flat(), a.mean(axis=0))
    r = A.mean(axis=1)
    assert isinstance(r, ColVec) and r.shape == (3, 1)
    np.testing.assert_allclose(r.to_flat(), a.mean(axis=1))


def test_mat_mean_invalid_axis():
    with pytest.raises(ValueError, match="axis"):
        mat([1, 2], [3, 4]).mean(axis=2)


def test_mat_std_colwise(backend):
    A = mat([1, 2, 3, 4], [10, 30, 20, 40])
    a = np.asarray(A)
    s = A.std()
    assert isinstance(s, ColVec) and s.shape == (2, 1)
    np.testing.assert_allclose(s.to_flat(), a.std(axis=0))


def test_mat_cov(backend):
    rng = np.random.default_rng(7)
    A = Mat(rng.standard_normal((40, 3)))
    C = A.cov()
    assert isinstance(C, Mat) and C.shape == (3, 3)
    np.testing.assert_allclose(np.asarray(C), np.cov(np.asarray(A), rowvar=False))


def test_mat_corr(backend):
    rng = np.random.default_rng(11)
    A = Mat(rng.standard_normal((50, 4)))
    R = A.corr()
    assert isinstance(R, Mat) and R.shape == (4, 4)
    np.testing.assert_allclose(
        np.asarray(R), np.corrcoef(np.asarray(A), rowvar=False)
    )
    np.testing.assert_allclose(np.diag(np.asarray(R)), np.ones(4))


def test_mat_normalize(backend):
    A = mat([3, 0], [0, 4], [1, 1])
    N = A.normalize()
    assert isinstance(N, Mat)
    np.testing.assert_allclose(np.linalg.norm(np.asarray(N), axis=0), [1, 1, 1])
    with pytest.raises(ValueError, match="column 1"):
        mat([1, 2], [0, 0]).normalize()


def test_np_mean_protocol_unchanged():
    A = mat([1, 2], [3, 4])
    assert float(np.mean(A)) == 2.5
    u = _c[1, 2, 3]
    assert not isinstance(np.sum(u), ColVec)
    assert float(np.sum(u)) == 6.0
