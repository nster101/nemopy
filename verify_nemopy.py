#!/usr/bin/env python3
"""
nemopy verification script.

Exercises common linear algebra operations across mathematics, physics,
and data science to confirm the library behaves correctly.

Run:  python verify_nemopy.py
"""

import numpy as np
from nemopy import _c, _m, mat, eye, as_col, as_mat, ColVec, Mat, ShapeError

PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  -- {detail}")


def close(a, b, tol=1e-10):
    return abs(a - b) < tol


def allclose(a, b, tol=1e-10):
    return np.allclose(np.asarray(a), np.asarray(b), atol=tol)


# ──────────────────────────────────────────────────────────────────────
#  SECTION 1 — PURE MATHEMATICS
# ──────────────────────────────────────────────────────────────────────
print("\n=== MATHEMATICS ===\n")

# --- Construction & types ---
u = _c[1, 2, 3]
v = _c[4, 5, 6]
check("ColVec construction via _c", isinstance(u, ColVec) and u.shape == (3, 1))

A = _m["1, 0; 0, 1; 1, 1"]  # 3 columns of length 2
check("Mat construction via _m", isinstance(A, Mat) and A.shape == (2, 3))

I3 = eye(3)
check("eye(3) is 3x3 identity", isinstance(I3, Mat) and allclose(I3, np.eye(3)))

# --- Inner product  u^T v ---
dot_val = (u.T @ v).item()
check("Inner product u^T v", close(dot_val, 1*4 + 2*5 + 3*6),
      f"expected 32, got {dot_val}")

# --- Outer product  u v^T ---
outer = u @ v.T
check("Outer product shape", isinstance(outer, Mat) and outer.shape == (3, 3))
check("Outer product values", close(outer[0, 0], 4) and close(outer[2, 2], 18))

# --- Vector norm ---
norm_u = np.sqrt((u.T @ u).item())
check("Euclidean norm ||u||", close(norm_u, np.sqrt(14)),
      f"expected {np.sqrt(14):.6f}, got {norm_u:.6f}")

# --- Orthogonality ---
e1 = _c[1, 0, 0]
e2 = _c[0, 1, 0]
check("Orthogonal basis vectors e1^T e2 = 0", close((e1.T @ e2).item(), 0))

# --- Matrix multiplication ---
B = mat([1, 4], [2, 5], [3, 6])   # 2x3
C = mat([7, 9, 11], [8, 10, 12])  # 3x2
D = B @ C                          # 2x2
check("Matrix multiply (2x3)@(3x2) shape", D.shape == (2, 2))
check("Matrix multiply values",
      close(D[0, 0], 1*7+2*9+3*11) and close(D[1, 1], 4*8+5*10+6*12))

# --- Determinant ---
M = as_mat([[3, 8], [4, 6]])
check("Determinant", close(M.det, 3*6 - 8*4), f"expected -14, got {M.det}")

# --- Matrix inverse ---
M_inv = M.inv
product = M @ M_inv
check("M @ M.inv ≈ I", allclose(product, np.eye(2)))

# --- Solving Ax = b ---
A_sys = as_mat([[2, 1], [5, 3]])
b_sys = _c[4, 7]
x = A_sys.inv @ b_sys
check("Solve 2x2 system Ax=b", allclose(x, [[5], [-6]]),
      f"expected [[5],[-6]], got {x.to_list()}")

# --- Singular matrix detection ---
S = as_mat([[1, 2], [2, 4]])
check("Singular matrix detected", S.is_singular)

# --- Shape error enforcement ---
try:
    _ = _c[1, 2] + _c[1, 2, 3]
    check("ShapeError on mismatched add", False, "no exception raised")
except ShapeError:
    check("ShapeError on mismatched add", True)

# --- Column join operator | ---
joined = _c[1, 2] | _c[3, 4] | _c[5, 6]
check("Column join |", isinstance(joined, Mat) and joined.shape == (2, 3))

# --- Transpose type rules ---
check("ColVec.T is Mat (row)", isinstance(u.T, Mat) and u.T.shape == (1, 3))
check("Mat.T shape", I3.T.shape == (3, 3) and isinstance(I3.T, Mat))


# ──────────────────────────────────────────────────────────────────────
#  SECTION 2 — PHYSICS
# ──────────────────────────────────────────────────────────────────────
print("\n=== PHYSICS ===\n")

# --- 2D rotation matrix ---
theta = np.pi / 4  # 45 degrees
R2 = as_mat([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta),  np.cos(theta)],
])
check("2D rotation matrix is orthogonal (R^T R = I)",
      allclose(R2.T @ R2, np.eye(2)))
check("2D rotation det = 1 (proper rotation)", close(R2.det, 1.0))

# Rotate the unit-x vector by 45°
ex = _c[1, 0]
rotated = R2 @ ex
check("Rotate [1,0] by 45°",
      close(rotated[0], np.cos(theta)) and close(rotated[1], np.sin(theta)))

# --- 3D rotation about z-axis ---
phi = np.pi / 6  # 30 degrees
Rz = as_mat([
    [np.cos(phi), -np.sin(phi), 0],
    [np.sin(phi),  np.cos(phi), 0],
    [0,            0,           1],
])
check("3D z-rotation det = 1", close(Rz.det, 1.0))
check("3D z-rotation is orthogonal", allclose(Rz.T @ Rz, np.eye(3)))

# --- Force decomposition / projection ---
F = _c[3, 4, 0]
n_hat = _c[1, 0, 0]  # unit normal
F_parallel_scalar = (F.T @ n_hat).item()
F_parallel = n_hat * F_parallel_scalar
F_perp = F + (n_hat * (-F_parallel_scalar))
check("Force projection: F_par + F_perp = F",
      allclose(F_parallel + F_perp, F))
check("F_parallel along n_hat", allclose(F_parallel, [[3], [0], [0]]))

# --- Moment of inertia tensor (uniform rod along z) ---
m_rod, L = 2.0, 3.0
Ixx = Iyy = (1.0/12.0) * m_rod * L**2
Izz = 0.0
I_tensor = as_mat([
    [Ixx,  0,   0],
    [0,   Iyy,  0],
    [0,    0,  Izz],
])
omega = _c[0, 0, 5]
L_angular = I_tensor @ omega
check("Angular momentum L = I·ω",
      close(L_angular[0], 0) and close(L_angular[1], 0) and close(L_angular[2], 0))

omega2 = _c[1, 0, 0]
L2 = I_tensor @ omega2
check("L for rotation about x-axis", close(L2[0], Ixx) and close(L2[1], 0))

# --- Kinetic energy  T = ½ ω^T I ω ---
KE = 0.5 * (omega2.T @ I_tensor @ omega2).item()
check("Rotational KE = ½ω^T I ω", close(KE, 0.5 * Ixx))

# --- Lorentz boost (1+1D, v/c = 0.6) ---
beta = 0.6
gamma = 1.0 / np.sqrt(1 - beta**2)
Lambda = as_mat([
    [gamma,       -gamma*beta],
    [-gamma*beta,  gamma],
])
event = _c[1, 0]  # (ct, x) = (1, 0) in rest frame
boosted = Lambda @ event
check("Lorentz boost preserves interval s²",
      close(event[0]**2 - event[1]**2,
            boosted[0]**2 - boosted[1]**2))


# ──────────────────────────────────────────────────────────────────────
#  SECTION 3 — DATA SCIENCE
# ──────────────────────────────────────────────────────────────────────
print("\n=== DATA SCIENCE ===\n")

# --- Ordinary least squares  β = (X^T X)^{-1} X^T y ---
X_raw = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_true = 2.0 * X_raw + 1.0  # y = 2x + 1 (exact)
ones_col = as_col(np.ones(5))
x_col = as_col(X_raw)
X_design = ones_col | x_col  # (5, 2) design matrix
y_vec = as_col(y_true)

beta = (X_design.T @ X_design).inv @ (X_design.T @ y_vec)
check("OLS intercept ≈ 1.0", close(beta[0], 1.0),
      f"got {beta[0]:.6f}")
check("OLS slope ≈ 2.0", close(beta[1], 2.0),
      f"got {beta[1]:.6f}")

# --- OLS with noise (verify residuals are small) ---
np.random.seed(42)
noise = as_col(np.random.randn(5) * 0.1)
y_noisy = y_vec + noise
beta_n = (X_design.T @ X_design).inv @ (X_design.T @ y_noisy)
check("Noisy OLS intercept near 1.0", abs(beta_n[0] - 1.0) < 0.5,
      f"got {beta_n[0]:.4f}")
check("Noisy OLS slope near 2.0", abs(beta_n[1] - 2.0) < 0.5,
      f"got {beta_n[1]:.4f}")

# --- Residuals ---
y_hat = X_design @ beta_n
residuals = y_noisy + (y_hat * -1)
check("Residuals are ColVec", isinstance(residuals, ColVec) and residuals.shape == (5, 1))

# --- Covariance matrix from data ---
data = mat([1, 4, 7], [2, 5, 8], [3, 6, 9])  # 3 observations, 3 features
n_obs = 3
col_means = as_col([(data[:, j].T @ as_col(np.ones(n_obs))).item() / n_obs
                     for j in range(3)])
centered = data + as_mat(-1 * np.ones((3, 1)) @ col_means.T)
cov = (centered.T @ centered) * (1.0 / (n_obs - 1))
check("Covariance matrix is 3x3 Mat", isinstance(cov, Mat) and cov.shape == (3, 3))
check("Covariance is symmetric", allclose(cov, cov.T))

# --- Projection matrix  P = X(X^T X)^{-1} X^T ---
P = X_design @ (X_design.T @ X_design).inv @ X_design.T
check("Projection matrix is idempotent (P² = P)", allclose(P @ P, P))
check("Projection matrix is symmetric", allclose(P, P.T))

# --- Whitening transform: W = Σ^{-1/2} ---
Sigma = as_mat([[4, 2], [2, 3]])
eigvals, eigvecs = np.linalg.eigh(np.asarray(Sigma))
D_inv_sqrt = as_mat(np.diag(1.0 / np.sqrt(eigvals)))
V = as_mat(eigvecs)
W = V @ D_inv_sqrt @ V.T
whitened_cov = W @ Sigma @ W.T
check("Whitening produces identity covariance",
      allclose(whitened_cov, np.eye(2), tol=1e-9))

# --- Conversion round-trips ---
w = _c[10, 20, 30]
check("to_numpy() round-trip", allclose(as_col(w.to_numpy()), w))
check("to_flat() shape", w.to_flat().shape == (3,))
check("to_list() values", w.to_list() == [10.0, 20.0, 30.0])

Q = mat([1, 3], [2, 4])  # columns [1,3] and [2,4] → rows [1,2] and [3,4]
check("Mat.to_list() gives nested rows", Q.to_list() == [[1.0, 2.0], [3.0, 4.0]])


# ──────────────────────────────────────────────────────────────────────
#  SECTION 4 — RUST CORE (two-mode load/feature smoke check)
# ──────────────────────────────────────────────────────────────────────
print("\n=== RUST CORE ===\n")

from nemopy import _core

if _core._RUST is not None:
    version = _core._RUST.rust_core_version()
    check("Rust core loaded (rust_core_version)",
          isinstance(version, str) and bool(version))
    # Tier-3 feature smoke: LDU factorization A = L @ D @ U.
    A_ldu = mat([4, 2], [2, 5])
    L, Dg, U = A_ldu.ldu()
    check("Tier-3 ldu() reconstructs A (Rust core)",
          allclose(np.asarray(L) @ np.asarray(Dg) @ np.asarray(U), A_ldu))
else:
    print("  [INFO] Rust core not built — Tier-3 features raise ImportError.")
    try:
        mat([4, 2], [2, 5]).ldu()
        check("Tier-3 ldu() raises ImportError when core absent", False,
              "no exception raised")
    except ImportError as exc:
        check("Tier-3 ldu() raises ImportError when core absent",
              "maturin develop" in str(exc))


# ──────────────────────────────────────────────────────────────────────
#  SUMMARY
# ──────────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"  Results:  {PASS} passed,  {FAIL} failed,  {PASS + FAIL} total")
print(f"{'='*50}")
if FAIL == 0:
    print("  All checks passed. nemopy is working correctly.")
else:
    print(f"  {FAIL} check(s) failed — review output above.")
print()
