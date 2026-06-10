"""AHP/ANP multi-criteria decision methods (issue #87; §20.4).

Pairwise comparison matrices are plain ``Mat`` objects; priority
extraction uses the Rust power-iteration kernel (``decomp.rs``, shared
with #84) with the NumPy eigensolver as the fallback — eigenvector
extraction is NumPy-replacement surface per §20.1. The ANP limit
supermatrix is a Rust-primary engine (repeated squaring with convergence
detection) with no Python port. Aggregation and synthesis are plain
NumPy expressions.
"""

import numpy as np

from nemopy import _core
from nemopy._core import ColVec, Mat, ShapeError

#: Saaty's random index table for n = 1..15.
RANDOM_INDEX = {
    1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32,
    8: 1.41, 9: 1.45, 10: 1.49, 11: 1.51, 12: 1.48, 13: 1.56,
    14: 1.57, 15: 1.59,
}


def ahp_matrix(*comparisons):
    """Build a reciprocal pairwise comparison matrix.

    Parameters
    ----------
    *comparisons : float
        Row-wise upper-triangle judgements ``a01, a02, ..., a12, ...``;
        the count must be a triangular number n(n-1)/2 and every value
        must be positive.

    Returns
    -------
    Mat
        The n x n reciprocal matrix with unit diagonal.

    Raises
    ------
    ValueError
        For a non-triangular count or a nonpositive judgement.
    """
    k = len(comparisons)
    n = int(round((1 + np.sqrt(1 + 8 * k)) / 2))
    if n * (n - 1) // 2 != k or k == 0:
        raise ValueError(
            f"ahp_matrix() takes a triangular number of upper-triangle "
            f"judgements (1, 3, 6, 10, ...), got {k}."
        )
    vals = [float(c) for c in comparisons]
    if any(v <= 0 for v in vals):
        raise ValueError("ahp_matrix() judgements must be positive.")
    a = np.eye(n)
    it = iter(vals)
    for i in range(n):
        for j in range(i + 1, n):
            v = next(it)
            a[i, j] = v
            a[j, i] = 1.0 / v
    return Mat(a)


def _mat_is_reciprocal(self):
    """Whether ``A[i, j] == 1 / A[j, i]`` with a unit diagonal."""
    a = np.asarray(self)
    if a.shape[0] != a.shape[1] or np.any(a <= 0):
        return False
    return bool(
        np.allclose(np.diag(a), 1.0)
        and np.allclose(a * a.T, 1.0, atol=1e-8)
    )


def _check_pairwise(self, name):
    if self.shape[0] != self.shape[1]:
        raise ShapeError(
            f"{name}() requires a square matrix, got {self.shape}."
        )
    if not _mat_is_reciprocal(self):
        raise ValueError(
            f"{name}() requires a positive reciprocal pairwise "
            f"comparison matrix."
        )
    return np.asarray(self)


def _principal_eigenpair(a):
    rust = _core._RUST
    if rust is not None:
        lam, vec = rust.power_iteration(a, 1e-12, 100_000)
        return float(lam), np.asarray(vec)
    vals, vecs = np.linalg.eig(a)
    i = int(np.argmax(vals.real))
    vec = np.abs(vecs[:, i].real)
    return float(vals[i].real), vec / vec.sum()


def _mat_ahp_priorities(self, method="eigenvector"):
    """Priority vector of a pairwise comparison matrix (sums to 1).

    Parameters
    ----------
    method : str, optional
        ``"eigenvector"`` (default) for the principal eigenvector
        (Saaty), or ``"geometric_mean"`` for normalized row geometric
        means.
    """
    a = _check_pairwise(self, "ahp_priorities")
    if method == "eigenvector":
        _, vec = _principal_eigenpair(a)
        return ColVec(vec.reshape(-1, 1))
    if method == "geometric_mean":
        g = np.exp(np.mean(np.log(a), axis=1))
        return ColVec((g / g.sum()).reshape(-1, 1))
    raise ValueError(
        f'method must be "eigenvector" or "geometric_mean", got {method!r}'
    )


def _mat_ahp_eigenvalue(self):
    """Principal eigenvalue lambda_max of the comparison matrix."""
    a = _check_pairwise(self, "ahp_eigenvalue")
    lam, _ = _principal_eigenpair(a)
    return lam


def _mat_consistency_index(self):
    """Consistency index ``CI = (lambda_max - n) / (n - 1)``."""
    n = self.shape[0]
    if n <= 1:
        return 0.0
    return (_mat_ahp_eigenvalue(self) - n) / (n - 1)


def _mat_consistency_ratio(self):
    """Consistency ratio ``CR = CI / RI(n)`` against Saaty's table.

    Raises
    ------
    ValueError
        If n exceeds the tabulated random indices (n > 15).
    """
    n = self.shape[0]
    if n not in RANDOM_INDEX:
        raise ValueError(f"random index table covers n <= 15, got {n}")
    ri = RANDOM_INDEX[n]
    if ri == 0.0:
        return 0.0
    return _mat_consistency_index(self) / ri


def _mat_is_consistent(self, threshold=0.1):
    """Whether ``CR < threshold`` (Saaty's 0.1 rule by default)."""
    return bool(_mat_consistency_ratio(self) < threshold)


def ahp_synthesize(criteria_weights, alternative_matrices):
    """Global priorities from criteria weights and per-criterion
    pairwise matrices over the alternatives.

    Parameters
    ----------
    criteria_weights : ColVec
        (m, 1) criteria priority vector.
    alternative_matrices : sequence of Mat
        One pairwise comparison matrix per criterion, each over the same
        alternatives.

    Returns
    -------
    ColVec
        The global priority vector (sums to 1).
    """
    w = np.asarray(criteria_weights, dtype=float).ravel()
    if len(alternative_matrices) != w.size:
        raise ValueError(
            f"expected one alternative matrix per criteria weight "
            f"({w.size}), got {len(alternative_matrices)}."
        )
    locals_ = np.column_stack(
        [np.asarray(m.ahp_priorities()).ravel() for m in alternative_matrices]
    )
    return ColVec((locals_ @ w).reshape(-1, 1))


def ahp_aggregate(matrices, method="geometric_mean"):
    """Aggregate judge matrices by element-wise geometric mean
    (preserves reciprocity).

    Raises
    ------
    ValueError
        For an unknown method or mismatched shapes.
    """
    if method != "geometric_mean":
        raise ValueError(
            f'method must be "geometric_mean", got {method!r}'
        )
    arrays = [np.asarray(m, dtype=float) for m in matrices]
    shape = arrays[0].shape
    if any(a.shape != shape for a in arrays):
        raise ValueError("all judge matrices must have the same shape")
    return Mat(np.exp(np.mean(np.log(np.stack(arrays)), axis=0)))


def anp_supermatrix(blocks):
    """Assemble an ANP supermatrix from block priority matrices.

    Parameters
    ----------
    blocks : sequence of sequence of (Mat or None)
        Block grid; ``None`` entries become zero blocks. Block sizes are
        inferred from the rows/columns of the given matrices.

    Returns
    -------
    Mat
    """
    nrows = len(blocks)
    ncols = len(blocks[0])
    row_sizes = [None] * nrows
    col_sizes = [None] * ncols
    for i, row in enumerate(blocks):
        if len(row) != ncols:
            raise ValueError("blocks must form a rectangular grid")
        for j, b in enumerate(row):
            if b is None:
                continue
            r, c = np.shape(b)
            if row_sizes[i] is None:
                row_sizes[i] = r
            elif row_sizes[i] != r:
                raise ValueError(f"inconsistent row size in block row {i}")
            if col_sizes[j] is None:
                col_sizes[j] = c
            elif col_sizes[j] != c:
                raise ValueError(f"inconsistent column size in block col {j}")
    if any(s is None for s in row_sizes) or any(s is None for s in col_sizes):
        raise ValueError("every block row and column needs at least one Mat")
    grid = [
        [
            np.zeros((row_sizes[i], col_sizes[j])) if b is None
            else np.asarray(b, dtype=float)
            for j, b in enumerate(row)
        ]
        for i, row in enumerate(blocks)
    ]
    return Mat(np.block(grid))


def _mat_limit_supermatrix(self):
    """Limit supermatrix ``lim W^k`` by repeated squaring.

    Rust-primary engine (amended §20.1): requires the ``_rust_core``
    extension.

    Raises
    ------
    ValueError
        If W is not column-stochastic, or the powers do not converge
        (cyclic supermatrix).
    """
    a = np.asarray(self)
    if (
        self.shape[0] != self.shape[1]
        or np.any(a < 0)
        or not np.allclose(a.sum(axis=0), 1.0, atol=1e-8)
    ):
        raise ValueError(
            "limit_supermatrix() requires a column-stochastic supermatrix"
        )
    rust = _core._require_rust("Mat.limit_supermatrix()")
    return Mat(np.asarray(rust.limit_supermatrix(a, 1e-12, 200)))


Mat.is_reciprocal = _mat_is_reciprocal
Mat.ahp_priorities = _mat_ahp_priorities
Mat.ahp_eigenvalue = _mat_ahp_eigenvalue
Mat.consistency_index = _mat_consistency_index
Mat.consistency_ratio = _mat_consistency_ratio
Mat.is_consistent = _mat_is_consistent
Mat.limit_supermatrix = _mat_limit_supermatrix
