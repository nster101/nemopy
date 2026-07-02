"""Statistical methods for ColVec and Mat (issue #77; §20.4 stats domain).

Column-first convention: ``Mat`` statistics treat columns as variables and
rows as observations. Methods dispatch to the ``_rust_core`` fused kernels
when the extension is available and fall back to NumPy with identical
semantics (DESIGN_APPENDICES.md §20.1).
"""

import numpy as np

from nemopy import _core
from nemopy._core import ColVec, Mat


def _colvec_mean(self):
    """Arithmetic mean as a Python float."""
    return float(np.asarray(self).mean())


def _colvec_std(self):
    """Standard deviation (ddof=0, NumPy semantics) as a Python float."""
    return float(np.asarray(self).std())


def _colvec_var(self):
    """Variance (ddof=0, NumPy semantics) as a Python float."""
    return float(np.asarray(self).var())


def _colvec_sum(self):
    """Element sum as a Python float."""
    return float(np.asarray(self).sum())


def _colvec_norm(self, p=2):
    """Vector p-norm as a Python float.

    Parameters
    ----------
    p : int or float, optional
        Norm order passed to ``numpy.linalg.norm`` (default 2). Use 1 for
        the L1 norm and ``numpy.inf`` for the max norm.
    """
    return float(np.linalg.norm(np.asarray(self).ravel(), ord=p))


def _colvec_normalize(self):
    """Unit vector ``u / ||u||`` (L2).

    Raises
    ------
    ValueError
        If the vector has zero norm (direction undefined).
    """
    nrm = np.linalg.norm(np.asarray(self))
    if nrm == 0.0:
        raise ValueError("cannot normalize a vector with zero norm")
    return ColVec(np.asarray(self) / nrm)


def _colvec_cumsum(self):
    """Cumulative sum as a ColVec."""
    return ColVec(np.cumsum(np.asarray(self), axis=0))


def _axis_view(self, axis):
    if axis == 0:
        return np.asarray(self)
    if axis == 1:
        return np.asarray(self).T
    raise ValueError(f"axis must be 0 (column-wise) or 1 (row-wise), got {axis}")


def _mat_mean_var(self, axis, ddof):
    a = _axis_view(self, axis)
    rust = _core._RUST
    if rust is not None:
        means, variances = rust.colwise_mean_var(a, ddof)
    else:
        means = a.mean(axis=0)
        variances = a.var(axis=0, ddof=ddof)
    return means, variances


def _mat_mean(self, axis=0):
    """Column-wise means by default (axis=0) as a (k,1) ColVec.

    Parameters
    ----------
    axis : int, optional
        0 (default) for column-wise means, 1 for row-wise means.
    """
    means, _ = _mat_mean_var(self, axis, 0)
    return ColVec(np.asarray(means).reshape(-1, 1))


def _mat_std(self, axis=0):
    """Column-wise standard deviation (ddof=0) as a ColVec.

    Parameters
    ----------
    axis : int, optional
        0 (default) for column-wise, 1 for row-wise.
    """
    _, variances = _mat_mean_var(self, axis, 0)
    return ColVec(np.sqrt(np.asarray(variances)).reshape(-1, 1))


def _mat_cov(self):
    """Covariance matrix with columns as variables (np.cov rowvar=False)."""
    a = np.asarray(self)
    rust = _core._RUST
    if rust is not None:
        return Mat(rust.cov(a, 1))
    return Mat(np.atleast_2d(np.cov(a, rowvar=False)))


def _mat_corr(self):
    """Correlation matrix with columns as variables."""
    a = np.asarray(self)
    rust = _core._RUST
    if rust is not None:
        return Mat(rust.corr(a))
    return Mat(np.atleast_2d(np.corrcoef(a, rowvar=False)))


def _mat_normalize(self):
    """Column-wise L2 normalization: every column rescaled to unit norm.

    Raises
    ------
    ValueError
        If any column has zero norm; the message names the column.
    """
    a = np.asarray(self)
    rust = _core._RUST
    if rust is not None:
        return Mat(rust.colwise_normalize(a))
    norms = np.linalg.norm(a, axis=0)
    zeros = np.flatnonzero(norms == 0.0)
    if zeros.size:
        raise ValueError(
            f"cannot normalize: column {int(zeros[0])} has zero norm"
        )
    return Mat(a / norms)


ColVec.mean = _colvec_mean
ColVec.std = _colvec_std
ColVec.var = _colvec_var
ColVec.sum = _colvec_sum
ColVec.norm = _colvec_norm
ColVec.normalize = _colvec_normalize
ColVec.cumsum = _colvec_cumsum

Mat.mean = _mat_mean
Mat.std = _mat_std
Mat.cov = _mat_cov
Mat.corr = _mat_corr
Mat.normalize = _mat_normalize
