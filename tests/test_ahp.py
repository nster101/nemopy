"""Tests for AHP/ANP multi-criteria decision methods (issue #87; §20.4).

Eleven tests are justified per CLAUDE.md §6.5: each covers one distinct
documented behaviour of the issue #87 API surface.

## Test: test_ahp_matrix
- Goal: ahp_matrix(*comparisons) builds the reciprocal pairwise matrix
        from row-wise upper-triangle values and validates input.
- Source: issue #87 pairwise comparison table.
- Expected: known 3x3 matrix from (3, 5, 2); ValueError for a
            non-triangular count or nonpositive judgement.

## Test: test_is_reciprocal
- Goal: Mat.is_reciprocal() checks A[i,j] == 1/A[j,i] with unit diagonal.
- Source: issue #87 pairwise comparison table.
- Expected: True for an ahp_matrix product, False for a non-reciprocal
            matrix.

## Test: test_ahp_priorities_eigenvector
- Goal: the principal-eigenvector priorities of a perfectly consistent
        matrix recover the underlying weights (sum 1, ColVec), on both
        the Rust power-iteration path and the NumPy eig fallback.
- Source: issue #87 priority extraction table; §20.1.
- Expected: priorities ≈ (0.6, 0.3, 0.1).

## Test: test_ahp_priorities_geometric_mean
- Goal: method="geometric_mean" matches the eigenvector method on a
        consistent matrix; unknown methods are rejected.
- Source: issue #87 design notes (geometric mean alternative).
- Expected: same priorities within tolerance; ValueError for unknown
            method.

## Test: test_ahp_priorities_requires_reciprocal
- Goal: priority extraction is gated by reciprocity validation.
- Source: issue #87 pairwise-comparison semantics.
- Expected: ValueError on a non-reciprocal matrix.

## Test: test_consistency_consistent
- Goal: a consistent matrix has lambda_max ≈ n, CI ≈ 0, CR ≈ 0 and
        is_consistent() True.
- Source: issue #87 priority extraction table ("CI = (λmax - n)/(n-1)").
- Expected: values within 1e-6; CR equals CI / RI(3) with RI(3) = 0.58.

## Test: test_consistency_inconsistent
- Goal: a wildly cyclic judgement matrix fails the 0.1 CR threshold.
- Source: issue #87 ("CR < threshold").
- Expected: is_consistent() False, consistency_ratio() > 0.1.

## Test: test_ahp_synthesize
- Goal: hierarchy synthesis combines local priorities with criteria
        weights into the global priority vector.
- Source: issue #87 hierarchy synthesis table.
- Expected: result equals the weighted sum of local priorities, sums to
            1, ColVec.

## Test: test_ahp_aggregate
- Goal: group aggregation takes the element-wise geometric mean across
        judges, preserving reciprocity.
- Source: issue #87 design notes (group decision support).
- Expected: known sqrt(ab) off-diagonal; ValueError for unknown method.

## Test: test_anp_supermatrix
- Goal: anp_supermatrix(blocks) assembles the block supermatrix with
        zero blocks for None entries.
- Source: issue #87 ANP table.
- Expected: assembled 4x4 from two 2x2 blocks placed off-diagonal.

## Test: test_limit_supermatrix
- Goal: limit_supermatrix() raises the column-stochastic supermatrix to
        its limit (Rust-primary engine: clear error without the
        extension; ValueError for non-column-stochastic input).
- Source: issue #87 ANP table ("W^k as k -> infinity"); amended §20.1.
- Expected: limit columns equal the principal eigenvector of W;
            RuntimeError mentioning _rust_core when the extension is
            forced absent; ValueError for invalid W.
"""

import numpy as np
import pytest

from nemopy import (
    ColVec,
    Mat,
    ahp_aggregate,
    ahp_matrix,
    ahp_synthesize,
    anp_supermatrix,
    mat,
)
from nemopy import _core

requires_rust = pytest.mark.skipif(
    _core._RUST is None, reason="_rust_core extension not built"
)


def _np(x):
    return np.asarray(x)


def _consistent(weights):
    w = np.asarray(weights, dtype=float)
    return Mat(w[:, None] / w[None, :])


def test_ahp_matrix():
    A = ahp_matrix(3, 5, 2)
    assert isinstance(A, Mat)
    np.testing.assert_allclose(
        _np(A),
        [[1, 3, 5], [1 / 3, 1, 2], [1 / 5, 1 / 2, 1]],
    )
    with pytest.raises(ValueError, match="(?i)triangular"):
        ahp_matrix(1, 2)
    with pytest.raises(ValueError, match="(?i)positive"):
        ahp_matrix(1, -2, 3)


def test_is_reciprocal():
    assert ahp_matrix(3, 5, 2).is_reciprocal()
    assert not mat([1, 2], [3, 1]).is_reciprocal()


def test_ahp_priorities_eigenvector(backend):
    A = _consistent([0.6, 0.3, 0.1])
    p = A.ahp_priorities()
    assert isinstance(p, ColVec)
    np.testing.assert_allclose(p.to_flat(), [0.6, 0.3, 0.1], atol=1e-8)
    assert abs(p.sum() - 1.0) < 1e-10


def test_ahp_priorities_geometric_mean():
    A = _consistent([0.5, 0.3, 0.2])
    p_eig = A.ahp_priorities()
    p_gm = A.ahp_priorities(method="geometric_mean")
    np.testing.assert_allclose(p_gm.to_flat(), p_eig.to_flat(), atol=1e-8)
    with pytest.raises(ValueError, match="(?i)method"):
        A.ahp_priorities(method="median")


def test_ahp_priorities_requires_reciprocal():
    with pytest.raises(ValueError, match="(?i)reciprocal"):
        mat([1, 2], [3, 1]).ahp_priorities()


def test_consistency_consistent(backend):
    A = _consistent([0.6, 0.3, 0.1])
    assert abs(A.ahp_eigenvalue() - 3.0) < 1e-6
    assert abs(A.consistency_index()) < 1e-6
    np.testing.assert_allclose(
        A.consistency_ratio(), A.consistency_index() / 0.58, atol=1e-12
    )
    assert A.is_consistent()


def test_consistency_inconsistent():
    A = ahp_matrix(9, 1 / 9, 9)  # cyclic preferences
    assert A.consistency_ratio() > 0.1
    assert not A.is_consistent()


def test_ahp_synthesize():
    weights = ColVec(np.array([[0.7], [0.3]]))
    A1 = _consistent([0.5, 0.5])
    A2 = _consistent([0.9, 0.1])
    g = ahp_synthesize(weights, [A1, A2])
    assert isinstance(g, ColVec)
    np.testing.assert_allclose(
        g.to_flat(), 0.7 * np.array([0.5, 0.5]) + 0.3 * np.array([0.9, 0.1]),
        atol=1e-8,
    )
    assert abs(g.sum() - 1.0) < 1e-10
    with pytest.raises(ValueError, match="(?i)criteria"):
        ahp_synthesize(weights, [A1])


def test_ahp_aggregate():
    a, b = 4.0, 9.0
    j1 = ahp_matrix(a)
    j2 = ahp_matrix(b)
    G = ahp_aggregate([j1, j2])
    assert isinstance(G, Mat)
    np.testing.assert_allclose(_np(G)[0, 1], np.sqrt(a * b))
    assert G.is_reciprocal()
    with pytest.raises(ValueError, match="(?i)method"):
        ahp_aggregate([j1, j2], method="mean")


def test_anp_supermatrix():
    M1 = mat([0.6, 0.4], [0.3, 0.7])
    M2 = mat([1.0, 0.0], [0.0, 1.0])
    W = anp_supermatrix([[None, M1], [M2, None]])
    assert isinstance(W, Mat)
    assert W.shape == (4, 4)
    np.testing.assert_allclose(_np(W)[:2, :2], np.zeros((2, 2)))
    np.testing.assert_allclose(_np(W)[:2, 2:], _np(M1))
    np.testing.assert_allclose(_np(W)[2:, :2], _np(M2))


@requires_rust
def test_limit_supermatrix(monkeypatch):
    # column-stochastic irreducible aperiodic supermatrix
    W = Mat(np.array([[0.6, 0.2], [0.4, 0.8]]))
    L = W.limit_supermatrix()
    assert isinstance(L, Mat)
    pi = np.array([1 / 3, 2 / 3])  # principal eigenvector of W
    np.testing.assert_allclose(_np(L)[:, 0], pi, atol=1e-8)
    np.testing.assert_allclose(_np(L)[:, 1], pi, atol=1e-8)
    with pytest.raises(ValueError, match="(?i)column-stochastic"):
        mat([1, 2], [3, 4]).limit_supermatrix()
    monkeypatch.setattr(_core, "_RUST", None)
    with pytest.raises(RuntimeError, match="_rust_core"):
        W.limit_supermatrix()
