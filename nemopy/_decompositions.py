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

LDUResult = namedtuple("LDUResult", ["L", "D", "U"])
QDRResult = namedtuple("QDRResult", ["Q", "D", "R"])
SchurResult = namedtuple("SchurResult", ["T", "Z"])
PolarResult = namedtuple("PolarResult", ["U", "P"])
DiagonalizeResult = namedtuple("DiagonalizeResult", ["P", "D"])
JordanResult = namedtuple("JordanResult", ["J", "P"])


def _mat_ldu(self):
    """LDU factorization ``A = L @ D @ U`` without row exchanges.

    ``L`` and ``U`` are unit triangular; ``D`` is diagonal.

    Raises
    ------
    ShapeError
        If the matrix is not square.
    ValueError
        If elimination hits a zero pivot (pivoting would be required);
        use :meth:`lu` instead.
    ImportError
        If the ``_rust_core`` extension is not available (Tier-3 feature
        with no NumPy fallback; §20.1/§20.4).
    """
    a = _require_square(self, "ldu")
    rust = _core._require_rust("ldu")
    low, d, u = rust.ldu(a)
    return LDUResult(Mat(np.asarray(low)), Mat(np.diag(np.asarray(d))),
                     Mat(np.asarray(u)))


def _mat_qdr(self):
    """QDR factorization ``A = Q @ D @ R``: QR with scaling split out.

    ``Q`` is the thin QR orthogonal factor, ``D`` the diagonal of the QR
    ``R`` factor, and ``R`` the rescaled upper-triangular factor with
    unit diagonal.

    Raises
    ------
    ValueError
        If the QR ``R`` factor has a zero diagonal entry (rank
        deficiency), so the unit-diagonal rescaling does not exist.
    ImportError
        If the ``_rust_core`` extension is not available (Tier-3 feature
        with no NumPy fallback; §20.1/§20.4).
    """
    _core._require_rust("qdr")
    q, r = _mat_qr(self)
    r = np.asarray(r)
    d = np.diag(r).copy()
    if np.any(d == 0.0):
        raise ValueError(
            "QDR requires a full-rank R factor (zero diagonal entry found)"
        )
    return QDRResult(Mat(np.asarray(q)), Mat(np.diag(d)), Mat(r / d[:, None]))


def _mat_schur(self):
    """Real Schur decomposition ``A = Z @ T @ Z.T``.

    A Tier-3 feature (§20.1/§20.4) that requires the ``_rust_core``
    extension; the computation is LAPACK-delegated via SciPy pending a
    ``_rust_core`` Schur kernel (issue #84 interim).

    Raises
    ------
    ShapeError
        If the matrix is not square.
    ImportError
        If the ``_rust_core`` extension is not available (Tier-3 feature
        with no NumPy fallback), or if SciPy is not installed on the
        interim delegated path.
    """
    a = _require_square(self, "schur")
    _core._require_rust("schur")
    try:
        from scipy.linalg import schur as _scipy_schur
    except ImportError as exc:
        raise ImportError(
            "Mat.schur() currently requires SciPy (LAPACK-delegated path; "
            "_rust_core Schur kernel pending)"
        ) from exc
    t, z = _scipy_schur(a, output="real")
    return SchurResult(Mat(t), Mat(z))


def _mat_polar(self):
    """Left polar decomposition ``A = U @ P``.

    ``U`` has orthonormal columns and ``P`` is symmetric positive
    semidefinite. Derived from the thin SVD via the Rust kernel.

    Raises
    ------
    ImportError
        If the ``_rust_core`` extension is not available (Tier-3 feature
        with no NumPy fallback; §20.1/§20.4).
    """
    _core._require_rust("polar")
    u, s, vt = _mat_svd(self)
    u, s, vt = np.asarray(u), np.asarray(s).ravel(), np.asarray(vt)
    return PolarResult(Mat(u @ vt), Mat(vt.T @ np.diag(s) @ vt))


def _mat_diagonalize(self):
    """Similarity diagonalization ``A = P @ D @ P^-1``.

    Returns
    -------
    DiagonalizeResult
        ``P`` (eigenvector columns) and diagonal ``D``; nemopy types for
        a real spectrum, plain complex ndarrays otherwise (per §4).

    Raises
    ------
    ShapeError
        If the matrix is not square.
    ValueError
        If the matrix is not diagonalizable (defective eigenbasis).
    ImportError
        If the ``_rust_core`` extension is not available (Tier-3 feature
        with no NumPy fallback; §20.1/§20.4).
    """
    a = _require_square(self, "diagonalize")
    _core._require_rust("diagonalize")
    n = a.shape[0]
    result = _mat_eig(self)
    vecs = np.asarray(result.vectors)
    if np.linalg.matrix_rank(vecs) < n:
        raise ValueError("matrix is not diagonalizable")
    vals = np.asarray(result.values).ravel()
    if np.iscomplexobj(vals):
        return DiagonalizeResult(vecs, np.diag(vals))
    return DiagonalizeResult(Mat(vecs), Mat(np.diag(vals)))


def _nullspace_basis(m, tol):
    _, s, vt = np.linalg.svd(m)
    rank = int(np.sum(s > tol)) if s.size else 0
    return vt[rank:].conj().T


def _mat_jordan(self):
    """Jordan normal form ``A = P @ J @ P^-1`` (issue #84).

    Numerically delicate: eigenvalues are clustered with a tolerance and
    generalized eigenvector chains are built by the staircase algorithm.
    Intended for small, well-conditioned matrices; results degrade when
    eigenvalues are nearly defective without being exactly so.

    Returns
    -------
    JordanResult
        ``J`` (Jordan blocks) and ``P``; nemopy types for a real
        spectrum, plain complex ndarrays otherwise (per §4).

    Raises
    ------
    ShapeError
        If the matrix is not square.
    ValueError
        If a complete generalized eigenbasis cannot be assembled.
    ImportError
        If the ``_rust_core`` extension is not available (Tier-3 feature
        with no NumPy fallback; §20.1/§20.4).
    """
    a = _require_square(self, "jordan")
    _core._require_rust("jordan")
    n = a.shape[0]
    scale = max(1.0, float(np.abs(a).max()))
    tol = 1e-8 * scale
    vals = np.linalg.eigvals(a)
    clusters = []
    used = np.zeros(n, dtype=bool)
    for i in range(n):
        if used[i]:
            continue
        group = [vals[i]]
        used[i] = True
        for j in range(i + 1, n):
            if not used[j] and abs(vals[j] - vals[i]) <= 1e-6 * scale:
                used[j] = True
                group.append(vals[j])
        clusters.append((np.mean(group), len(group)))
    is_complex = any(abs(np.imag(lam)) > 1e-10 * scale for lam, _ in clusters)
    dtype = complex if is_complex else float
    eye = np.eye(n, dtype=dtype)
    jmat = np.zeros((n, n), dtype=dtype)
    pmat = np.zeros((n, n), dtype=dtype)
    col = 0
    for lam, mult in clusters:
        lam = complex(lam) if is_complex else float(np.real(lam))
        b = a.astype(dtype) - lam * eye
        null_bases = [np.zeros((n, 0), dtype=dtype)]
        bk = eye.copy()
        while null_bases[-1].shape[1] < mult and len(null_bases) <= n:
            bk = bk @ b
            null_bases.append(_nullspace_basis(bk, tol))
        grade = len(null_bases) - 1
        chains = []
        for k in range(grade, 0, -1):
            avoid_cols = [null_bases[k - 1]]
            for chain in chains:
                if len(chain) > k:
                    avoid_cols.append(chain[k - 1].reshape(-1, 1))
            avoid = np.hstack(avoid_cols)
            if avoid.shape[1]:
                ua, sa, _ = np.linalg.svd(avoid, full_matrices=False)
                basis = ua[:, : int(np.sum(sa > tol))]
            else:
                basis = np.zeros((n, 0), dtype=dtype)
            for idx in range(null_bases[k].shape[1]):
                v = null_bases[k][:, idx]
                r = v - basis @ (basis.conj().T @ v)
                if np.linalg.norm(r) > 10 * tol:
                    head = r / np.linalg.norm(r)
                    chain = [head]
                    for _ in range(k - 1):
                        chain.append(b @ chain[-1])
                    chain.reverse()
                    chains.append(chain)
                    basis = np.hstack([basis, head.reshape(-1, 1)])
        chains.sort(key=len, reverse=True)
        for chain in chains:
            m = len(chain)
            for g, vec in enumerate(chain):
                pmat[:, col + g] = vec
            jmat[col:col + m, col:col + m] = (
                lam * np.eye(m, dtype=dtype)
                + np.diag(np.ones(m - 1, dtype=dtype), 1)
            )
            col += m
    if col != n or np.linalg.matrix_rank(pmat, tol=tol) < n:
        raise ValueError(
            "Jordan form computation failed to assemble a complete "
            "generalized eigenbasis"
        )
    if is_complex:
        return JordanResult(jmat, pmat)
    return JordanResult(Mat(jmat.real), Mat(pmat.real))


Mat.ldu = _mat_ldu
Mat.qdr = _mat_qdr
Mat.schur = _mat_schur
Mat.polar = _mat_polar
Mat.diagonalize = _mat_diagonalize
Mat.jordan = _mat_jordan
