"""Core matrix decompositions as Mat methods (issue #76; §20.4).

Each decomposition returns nemopy types inside a named tuple. Kernels
dispatch to ``_rust_core`` (LU, Cholesky, QR, eigh, SVD) when the
extension is available, with NumPy/LAPACK fallbacks of identical
semantics per §20.1. General ``eig`` is LAPACK-delegated on both paths
("NumPy is the BLAS") pending a faer-backed kernel.
"""

from collections import namedtuple

import numpy as np

from nemopy import _core
from nemopy._core import ColVec, Mat, ShapeError

SVDResult = namedtuple("SVDResult", ["U", "s", "Vt"])
QRResult = namedtuple("QRResult", ["Q", "R"])
LUResult = namedtuple("LUResult", ["P", "L", "U"])
EigResult = namedtuple("EigResult", ["values", "vectors"])


def _require_square(self, name):
    if self.shape[0] != self.shape[1]:
        raise ShapeError(
            f"{name}() requires a square matrix, got shape {self.shape}."
        )
    return np.asarray(self)


def _require_symmetric(a, name):
    scale = max(1.0, float(np.abs(a).max()))
    if not np.allclose(a, a.T, atol=1e-10 * scale):
        raise ShapeError(f"{name}() requires a symmetric matrix.")


def _mat_svd(self):
    """Thin singular value decomposition ``A = U @ diag(s) @ Vt``.

    Returns
    -------
    SVDResult
        ``U`` (Mat), ``s`` (ColVec, descending), ``Vt`` (Mat).
    """
    a = np.asarray(self)
    rust = _core._RUST
    if rust is not None:
        if a.shape[0] >= a.shape[1]:
            u, s, vt = rust.svd(a)
        else:
            u1, s, v1t = rust.svd(np.ascontiguousarray(a.T))
            u, vt = np.asarray(v1t).T, np.asarray(u1).T
    else:
        u, s, vt = np.linalg.svd(a, full_matrices=False)
    return SVDResult(
        Mat(u), ColVec(np.asarray(s).reshape(-1, 1)), Mat(vt)
    )


def _mat_qr(self):
    """Thin QR decomposition ``A = Q @ R``.

    Returns
    -------
    QRResult
        ``Q`` (Mat, orthonormal columns), ``R`` (Mat, upper triangular).
    """
    a = np.asarray(self)
    rust = _core._RUST
    if rust is not None:
        q, r = rust.qr(a)
    else:
        q, r = np.linalg.qr(a, mode="reduced")
    return QRResult(Mat(q), Mat(r))


def _lu_fallback(a):
    n = a.shape[0]
    u = a.copy()
    low = np.eye(n)
    perm = list(range(n))
    for k in range(n):
        p = k + int(np.argmax(np.abs(u[k:, k])))
        if u[p, k] == 0.0:
            continue
        if p != k:
            perm[k], perm[p] = perm[p], perm[k]
            u[[k, p], :] = u[[p, k], :]
            low[[k, p], :k] = low[[p, k], :k]
        f = u[k + 1:, k] / u[k, k]
        low[k + 1:, k] = f
        u[k + 1:, k:] -= np.outer(f, u[k, k:])
        u[k + 1:, k] = 0.0
    return perm, low, u


def _mat_lu(self):
    """LU decomposition with partial pivoting ``A = P @ L @ U``.

    Returns
    -------
    LUResult
        ``P`` (Mat, permutation), ``L`` (Mat, unit lower triangular),
        ``U`` (Mat, upper triangular).
    """
    a = _require_square(self, "lu")
    rust = _core._RUST
    if rust is not None:
        perm, low, u = rust.lu(a)
    else:
        perm, low, u = _lu_fallback(a)
    n = a.shape[0]
    p = np.zeros((n, n))
    p[list(perm), np.arange(n)] = 1.0
    return LUResult(Mat(p), Mat(np.asarray(low)), Mat(np.asarray(u)))


def _mat_cholesky(self):
    """Cholesky factor ``L`` (lower triangular) with ``A = L @ L.T``.

    Raises
    ------
    ShapeError
        If the matrix is not square or not symmetric.
    ValueError
        If the matrix is not positive definite
        (``numpy.linalg.LinAlgError`` on the fallback path).
    """
    a = _require_square(self, "cholesky")
    _require_symmetric(a, "cholesky")
    rust = _core._RUST
    if rust is not None:
        return Mat(rust.cholesky(a))
    return Mat(np.linalg.cholesky(a))


def _mat_eig(self):
    """Eigendecomposition with column eigenvectors.

    Returns
    -------
    EigResult
        ``values`` (ColVec) and ``vectors`` (Mat) when the spectrum is
        real; plain complex ndarrays (``values`` of shape ``(n, 1)``)
        when it is complex, since nemopy types are float64-only per §4.
    """
    a = _require_square(self, "eig")
    vals, vecs = np.linalg.eig(a)
    if np.iscomplexobj(vals):
        scale = max(1.0, float(np.abs(vals).max()))
        if np.allclose(vals.imag, 0.0, atol=1e-12 * scale):
            vals, vecs = vals.real, vecs.real
        else:
            return EigResult(vals.reshape(-1, 1), vecs)
    return EigResult(ColVec(vals.reshape(-1, 1)), Mat(vecs))


def _mat_eigh(self):
    """Symmetric eigendecomposition (ascending eigenvalues).

    Returns
    -------
    EigResult
        ``values`` (ColVec, ascending) and orthonormal ``vectors`` (Mat).

    Raises
    ------
    ShapeError
        If the matrix is not square or not symmetric.
    """
    a = _require_square(self, "eigh")
    _require_symmetric(a, "eigh")
    rust = _core._RUST
    if rust is not None:
        w, v = rust.eigh(a)
    else:
        w, v = np.linalg.eigh(a)
    return EigResult(ColVec(np.asarray(w).reshape(-1, 1)), Mat(v))


Mat.svd = _mat_svd
Mat.qr = _mat_qr
Mat.lu = _mat_lu
Mat.cholesky = _mat_cholesky
Mat.eig = _mat_eig
Mat.eigh = _mat_eigh
