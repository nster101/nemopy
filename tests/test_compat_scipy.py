"""Tests for scipy interoperability with ColVec and Mat.

Verifies that nemopy types (ColVec, Mat) work seamlessly with scipy's
most-used modules: scipy.linalg, scipy.optimize, scipy.stats, and
scipy.interpolate. Per DESIGN_APPENDICES.md §12, ColVec and Mat are
ndarray subclasses and should be accepted directly by scipy functions.
Return values are plain ndarray.

## Test: test_scipy_linalg_solve
- Goal: scipy.linalg.solve(Mat, ColVec) returns a usable result that
        satisfies Ax = b.
- Source: DESIGN_APPENDICES.md §12.1 — "la.solve(A, b) works".
- Expected: x has correct shape; A @ x ≈ b.

## Test: test_scipy_linalg_inv
- Goal: scipy.linalg.inv(Mat) returns an inverse such that A @ A_inv ≈ I.
- Source: DESIGN_APPENDICES.md §12.1 — "la.inv(A) works".
- Expected: product is approximately eye(2).

## Test: test_scipy_linalg_det
- Goal: scipy.linalg.det(Mat) returns a scalar determinant.
- Source: DESIGN_APPENDICES.md §12.1 — "All scipy.linalg functions accept
        ColVec and Mat directly".
- Expected: scalar float, value ≈ -2.0.

## Test: test_scipy_linalg_eig
- Goal: scipy.linalg.eig(Mat) returns eigenvalues and eigenvectors with
        correct shapes.
- Source: DESIGN_APPENDICES.md §12.1.
- Expected: vals shape (2,), vecs shape (2, 2).

## Test: test_scipy_linalg_lu
- Goal: scipy.linalg.lu(Mat) returns P, L, U such that P @ L @ U ≈ A.
- Source: DESIGN_APPENDICES.md §12.1.
- Expected: reconstruction matches original matrix.

## Test: test_scipy_linalg_qr
- Goal: scipy.linalg.qr(Mat, mode='economic') returns Q, R such that
        Q @ R ≈ A.
- Source: DESIGN_APPENDICES.md §12.1.
- Expected: reconstruction matches original matrix.

## Test: test_scipy_linalg_svd
- Goal: scipy.linalg.svd(Mat, full_matrices=False) returns U, s, Vt such
        that U @ diag(s) @ Vt ≈ A.
- Source: DESIGN_APPENDICES.md §12.1.
- Expected: reconstruction matches original matrix.

## Test: test_scipy_linalg_cholesky
- Goal: scipy.linalg.cholesky on a positive-definite Mat returns L such
        that L @ L.T ≈ A.
- Source: DESIGN_APPENDICES.md §12.1 — "la.cholesky(A) works".
- Expected: reconstruction matches original matrix.

## Test: test_scipy_linalg_expm
- Goal: scipy.linalg.expm(Mat) returns a matrix exponential with correct
        shape.
- Source: DESIGN_APPENDICES.md §12.1 — "la.expm(A) — matrix exponential — works".
- Expected: result shape (2, 2).

## Test: test_scipy_linalg_lstsq
- Goal: scipy.linalg.lstsq for an overdetermined system returns x with
        correct shape.
- Source: DESIGN_APPENDICES.md §12.1.
- Expected: x has 2 rows.

## Test: test_scipy_optimize_minimize_nelder_mead
- Goal: scipy.optimize.minimize works with a flattened ColVec as x0.
- Source: DESIGN_APPENDICES.md §12.3 — "Pass .flatten() when interfacing".
- Expected: minimizer converges to (1, 2).

## Test: test_scipy_optimize_minimize_from_colvec_with_to_flat
- Goal: Verify that ColVec.to_flat() produces a valid x0 for
        scipy.optimize.minimize.
- Source: DESIGN_APPENDICES.md §12.3, §13.2 — to_flat() returns (n,).
- Expected: minimizer converges.

## Test: test_scipy_optimize_root
- Goal: scipy.optimize.root works with a flattened ColVec as x0.
- Source: DESIGN_APPENDICES.md §12.3.
- Expected: root found at (2, 1).

## Test: test_scipy_optimize_least_squares
- Goal: scipy.optimize.least_squares works with a flattened ColVec as x0.
- Source: DESIGN_APPENDICES.md §12.3.
- Expected: solution at (1, 2).

## Test: test_scipy_stats_describe
- Goal: scipy.stats.describe accepts a flattened ColVec.
- Source: DESIGN_APPENDICES.md §12.4 — "Statistical functions generally accept
        2D arrays".
- Expected: nobs == 5.

## Test: test_scipy_stats_pearsonr
- Goal: scipy.stats.pearsonr accepts flattened ColVecs.
- Source: DESIGN_APPENDICES.md §12.4.
- Expected: r ≈ 1.0 for perfectly correlated data.

## Test: test_scipy_stats_linregress
- Goal: scipy.stats.linregress accepts flattened ColVecs.
- Source: DESIGN_APPENDICES.md §12.4.
- Expected: slope ≈ 2.0.

## Test: test_scipy_stats_norm_pdf
- Goal: scipy.stats.norm.pdf accepts ColVec and returns an array of
        correct length.
- Source: DESIGN_APPENDICES.md §12.4.
- Expected: result has 3 elements.

## Test: test_scipy_interpolate_interp1d
- Goal: scipy.interpolate.interp1d accepts flattened ColVecs and produces
        correct interpolation.
- Source: DESIGN_APPENDICES.md §12 (scipy interop generally).
- Expected: f(1.5) ≈ 2.5 for linear interpolation.
"""

import numpy as np
import pytest
import scipy.interpolate
import scipy.linalg
import scipy.optimize
import scipy.stats

from nemopy import ColVec, Mat, _c, eye, mat


# ---------------------------------------------------------------------------
# scipy.linalg
# ---------------------------------------------------------------------------


class TestScipyLinalg:

    def test_scipy_linalg_solve(self):
        A = mat([2, 1], [1, 3])
        b = _c[5, 7]
        x = scipy.linalg.solve(A, b)
        assert x.shape[0] == 2
        np.testing.assert_allclose(
            A @ np.asarray(x).reshape(-1, 1), b, atol=1e-12
        )

    def test_scipy_linalg_inv(self):
        A = mat([1, 2], [3, 4])
        A_inv = scipy.linalg.inv(A)
        product = np.asarray(A) @ A_inv
        np.testing.assert_allclose(product, np.eye(2), atol=1e-12)

    def test_scipy_linalg_det(self):
        A = mat([1, 2], [3, 4])
        d = scipy.linalg.det(A)
        assert isinstance(d, (float, np.floating))
        np.testing.assert_allclose(d, -2.0, atol=1e-12)

    def test_scipy_linalg_eig(self):
        A = mat([2, 1], [1, 2])
        vals, vecs = scipy.linalg.eig(A)
        assert vals.shape == (2,)
        assert vecs.shape == (2, 2)

    def test_scipy_linalg_lu(self):
        A = mat([2, 1], [1, 3])
        P, L, U = scipy.linalg.lu(A)
        np.testing.assert_allclose(P @ L @ U, np.asarray(A), atol=1e-12)

    def test_scipy_linalg_qr(self):
        A = mat([1, 3, 5], [2, 4, 6])
        Q, R = scipy.linalg.qr(A, mode='economic')
        np.testing.assert_allclose(Q @ R, np.asarray(A), atol=1e-12)

    def test_scipy_linalg_svd(self):
        A = mat([1, 3, 5], [2, 4, 6])
        U, s, Vt = scipy.linalg.svd(A, full_matrices=False)
        np.testing.assert_allclose(U @ np.diag(s) @ Vt, np.asarray(A), atol=1e-12)

    def test_scipy_linalg_cholesky(self):
        A = mat([4, 2], [2, 3])
        L = scipy.linalg.cholesky(A, lower=True)
        np.testing.assert_allclose(L @ L.T, np.asarray(A), atol=1e-12)

    def test_scipy_linalg_expm(self):
        A = mat([0, 1], [-1, 0])
        expA = scipy.linalg.expm(A)
        assert expA.shape == (2, 2)

    def test_scipy_linalg_lstsq(self):
        A = mat([1, 1, 1], [1, 2, 3])
        b = _c[1, 2, 2]
        x, residuals, rank, sv = scipy.linalg.lstsq(A, b)
        assert x.shape[0] == 2


# ---------------------------------------------------------------------------
# scipy.optimize
# ---------------------------------------------------------------------------


class TestScipyOptimize:

    def test_scipy_optimize_minimize_nelder_mead(self):
        def objective(x):
            return (x[0] - 1) ** 2 + (x[1] - 2) ** 2

        x0 = _c[0, 0].to_flat()
        result = scipy.optimize.minimize(objective, x0, method='Nelder-Mead')
        assert result.success
        np.testing.assert_allclose(result.x, [1, 2], atol=1e-4)

    def test_scipy_optimize_minimize_from_colvec_with_to_flat(self):
        x0 = _c[0, 0]
        result = scipy.optimize.minimize(
            lambda x: x[0] ** 2 + x[1] ** 2, x0.to_flat()
        )
        assert result.success

    def test_scipy_optimize_root(self):
        def equations(x):
            return [x[0] + x[1] - 3, x[0] - x[1] - 1]

        x0 = _c[1, 1].to_flat()
        result = scipy.optimize.root(equations, x0)
        assert result.success
        np.testing.assert_allclose(result.x, [2, 1], atol=1e-8)

    def test_scipy_optimize_least_squares(self):
        def residuals(x):
            return [x[0] - 1, x[1] - 2]

        x0 = _c[0, 0].to_flat()
        result = scipy.optimize.least_squares(residuals, x0)
        np.testing.assert_allclose(result.x, [1, 2], atol=1e-8)


# ---------------------------------------------------------------------------
# scipy.stats
# ---------------------------------------------------------------------------


class TestScipyStats:

    def test_scipy_stats_describe(self):
        v = _c[1, 2, 3, 4, 5]
        desc = scipy.stats.describe(v.to_flat())
        assert desc.nobs == 5

    def test_scipy_stats_pearsonr(self):
        x = _c[1, 2, 3, 4, 5]
        y = _c[2, 4, 6, 8, 10]
        r, p = scipy.stats.pearsonr(x.to_flat(), y.to_flat())
        np.testing.assert_allclose(r, 1.0, atol=1e-12)

    def test_scipy_stats_linregress(self):
        x = _c[1, 2, 3, 4, 5]
        y = _c[2, 4, 6, 8, 10]
        result = scipy.stats.linregress(x.to_flat(), y.to_flat())
        np.testing.assert_allclose(result.slope, 2.0, atol=1e-12)

    def test_scipy_stats_norm_pdf(self):
        x = _c[-1, 0, 1]
        pdf_vals = scipy.stats.norm.pdf(x.to_flat())
        assert pdf_vals.shape == (3,)


# ---------------------------------------------------------------------------
# scipy.interpolate
# ---------------------------------------------------------------------------


class TestScipyInterpolate:

    def test_scipy_interpolate_interp1d(self):
        x = _c[0, 1, 2, 3]
        y = _c[0, 1, 4, 9]
        f = scipy.interpolate.interp1d(x.to_flat(), y.to_flat())
        assert abs(f(1.5) - 2.5) < 1e-12
