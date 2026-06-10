"""Gaussian elimination and row echelon forms as Mat methods (issue #85).

The production path runs in ``_rust_core.linalg`` (REF/RREF, fused
Gaussian solve); the NumPy fallback mirrors its semantics per §20.1. The
step-by-step ``ref_steps()`` pedagogy mode stays in Python by design.
RREF is primarily valuable for exact/small systems and teaching; for
large systems prefer the LU-backed solvers (numerical-stability note in
issue #85).
"""

import numpy as np

from nemopy import _core
from nemopy._core import ColVec, Mat, ShapeError


def _default_tol(a):
    scale = max(1.0, float(np.abs(a).max())) if a.size else 1.0
    return np.finfo(float).eps * max(a.shape) * scale


def _ref_fallback(a, partial):
    m, n = a.shape
    r = a.copy()
    tol = _default_tol(a)
    row = 0
    for col in range(n):
        if row >= m:
            break
        if partial:
            p = row + int(np.argmax(np.abs(r[row:, col])))
        else:
            p = row
        if abs(r[p, col]) <= tol:
            if not partial and np.any(np.abs(r[row + 1:, col]) > tol):
                raise ValueError(
                    f'zero pivot in column {col} requires a row exchange; '
                    f'use pivot="partial"'
                )
            continue
        if p != row:
            r[[row, p], :] = r[[p, row], :]
        f = r[row + 1:, col] / r[row, col]
        r[row + 1:, col:] -= np.outer(f, r[row, col:])
        r[row + 1:, col] = 0.0
        row += 1
    return r


def _rref_fallback(a):
    m, n = a.shape
    r = a.copy()
    tol = _default_tol(a)
    pivots = []
    row = 0
    for col in range(n):
        if row >= m:
            break
        p = row + int(np.argmax(np.abs(r[row:, col])))
        if abs(r[p, col]) <= tol:
            continue
        if p != row:
            r[[row, p], :] = r[[p, row], :]
        r[row, :] /= r[row, col]
        r[row, col] = 1.0
        others = [i for i in range(m) if i != row]
        f = r[others, col]
        r[others, :] -= np.outer(f, r[row, :])
        r[others, col] = 0.0
        pivots.append(col)
        row += 1
    r[r == 0.0] = 0.0
    return r, pivots


def _mat_ref(self, pivot="partial"):
    """Row echelon form via forward elimination.

    Parameters
    ----------
    pivot : str, optional
        ``"partial"`` (default) for partial pivoting (row swaps),
        ``"none"`` for classic unpivoted elimination (pedagogical use).

    Raises
    ------
    ValueError
        For an unknown pivot strategy, or when ``pivot="none"`` hits a
        zero pivot that would require a row exchange.
    """
    if pivot not in ("partial", "none"):
        raise ValueError(
            f'pivot must be "partial" or "none", got {pivot!r}'
        )
    a = np.asarray(self)
    rust = _core._RUST
    if rust is not None:
        return Mat(rust.ref_(a, pivot == "partial"))
    return Mat(_ref_fallback(a, pivot == "partial"))


def _mat_rref(self):
    """Reduced row echelon form and pivot column indices.

    Returns
    -------
    (Mat, tuple of int)
        The canonical RREF and the pivot column indices.
    """
    a = np.asarray(self)
    rust = _core._RUST
    if rust is not None:
        r, pivots = rust.rref(a)
    else:
        r, pivots = _rref_fallback(a)
    return Mat(np.asarray(r)), tuple(int(p) for p in pivots)


def _mat_rank(self):
    """Rank as the number of RREF pivot columns (tolerance-aware)."""
    return len(_mat_rref(self)[1])


def _mat_nullspace(self):
    """Null space basis vectors as Mat columns (from the RREF).

    A full-rank matrix returns an empty ``(k, 0)`` Mat.
    """
    k = self.shape[1]
    r, pivots = _mat_rref(self)
    r = np.asarray(r)
    free = [j for j in range(k) if j not in pivots]
    basis = np.zeros((k, len(free)))
    for idx, f in enumerate(free):
        basis[f, idx] = 1.0
        for i, p in enumerate(pivots):
            basis[p, idx] = -r[i, f]
    return Mat(basis)


def _check_rhs(self, b, name):
    b = np.asarray(b, dtype=float)
    if b.ndim == 1:
        b = b.reshape(-1, 1)
    if b.ndim != 2 or b.shape[1] != 1 or b.shape[0] != self.shape[0]:
        raise ShapeError(
            f"{name}() requires b of shape ({self.shape[0]}, 1), "
            f"got {b.shape}."
        )
    return b


def _mat_gaussian_eliminate(self, b):
    """Solve ``Ax = b`` by forward elimination + back substitution.

    Raises
    ------
    ShapeError
        If A is not square or b has the wrong shape.
    ValueError
        If A is singular to working precision.
    """
    if self.shape[0] != self.shape[1]:
        raise ShapeError(
            f"gaussian_eliminate() requires a square matrix, got shape "
            f"{self.shape}."
        )
    a = np.asarray(self)
    b = _check_rhs(self, b, "gaussian_eliminate")
    rust = _core._RUST
    if rust is not None:
        x = rust.gauss_solve(a, b)
        return ColVec(np.asarray(x).reshape(-1, 1))
    w = np.hstack([a, b]).astype(float)
    n = a.shape[0]
    tol = _default_tol(a)
    for k in range(n):
        p = k + int(np.argmax(np.abs(w[k:, k])))
        if abs(w[p, k]) <= tol:
            raise ValueError("matrix is singular to working precision")
        if p != k:
            w[[k, p], :] = w[[p, k], :]
        f = w[k + 1:, k] / w[k, k]
        w[k + 1:, k:] -= np.outer(f, w[k, k:])
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (w[i, n] - w[i, i + 1:n] @ x[i + 1:]) / w[i, i]
    return ColVec(x.reshape(-1, 1))


def _mat_gauss_jordan(self, b=None):
    """Full Gauss-Jordan elimination to RREF.

    With no ``b``, returns the RREF of A as a Mat. With ``b``, solves
    ``Ax = b`` via the RREF of the augmented system and returns the
    solution ColVec.

    Raises
    ------
    ShapeError
        With ``b``: if A is not square or b has the wrong shape.
    ValueError
        With ``b``: if A is singular to working precision.
    """
    if b is None:
        return _mat_rref(self)[0]
    if self.shape[0] != self.shape[1]:
        raise ShapeError(
            f"gauss_jordan() requires a square matrix to solve, got shape "
            f"{self.shape}."
        )
    b = _check_rhs(self, b, "gauss_jordan")
    n = self.shape[0]
    aug = Mat(np.hstack([np.asarray(self), b]))
    r, pivots = _mat_rref(aug)
    if tuple(p for p in pivots if p < n) != tuple(range(n)):
        raise ValueError("matrix is singular to working precision")
    return ColVec(np.asarray(r)[:, n:].copy())


def _mat_augment(self, b):
    """Augmented matrix ``[A | b]`` for manual elimination.

    Raises
    ------
    ShapeError
        If b's row count differs from A's.
    """
    b = np.asarray(b, dtype=float)
    if b.ndim == 1:
        b = b.reshape(-1, 1)
    if b.shape[0] != self.shape[0]:
        raise ShapeError(
            f"augment() requires equal row counts, got {self.shape} "
            f"and {b.shape}."
        )
    return Mat(np.hstack([np.asarray(self), b]))


def _mat_ref_steps(self):
    """Yield ``(Mat, description)`` pairs for each elimination step.

    Pedagogical mode (issue #85): runs the partial-pivoting forward
    elimination in Python, materializing a snapshot after every row
    exchange and row update. The final snapshot equals ``self.ref()``.
    """
    a = np.asarray(self).copy()
    m, n = a.shape
    tol = _default_tol(a)
    yield Mat(a.copy()), "start"
    row = 0
    for col in range(n):
        if row >= m:
            break
        p = row + int(np.argmax(np.abs(a[row:, col])))
        if abs(a[p, col]) <= tol:
            continue
        if p != row:
            a[[row, p], :] = a[[p, row], :]
            yield Mat(a.copy()), f"swap R{row + 1} and R{p + 1}"
        for i in range(row + 1, m):
            f = a[i, col] / a[row, col]
            if f != 0.0:
                a[i, col:] -= f * a[row, col:]
                a[i, col] = 0.0
                yield (
                    Mat(a.copy()),
                    f"R{i + 1} -> R{i + 1} - ({f:.6g})*R{row + 1}",
                )
        row += 1


Mat.ref = _mat_ref
Mat.rref = _mat_rref
Mat.rank = _mat_rank
Mat.nullspace = _mat_nullspace
Mat.gaussian_eliminate = _mat_gaussian_eliminate
Mat.gauss_jordan = _mat_gauss_jordan
Mat.augment = _mat_augment
Mat.ref_steps = _mat_ref_steps
