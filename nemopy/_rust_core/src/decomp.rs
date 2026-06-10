//! Decomposition kernels (issues #76, #84): LU, LDU, Cholesky, thin
//! Householder QR, symmetric eigendecomposition (cyclic Jacobi) and thin
//! SVD (one-sided Jacobi). General eig/Schur/Jordan stay LAPACK-delegated
//! on the Python side pending a faer integration (issue #84 blesses a
//! fallback-primary path for the numerically delicate forms).

use ndarray::{Array1, Array2, ArrayView2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray2, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

type LuParts = (Vec<usize>, Array2<f64>, Array2<f64>);

/// LU with partial pivoting (Doolittle): A[perm[i], :] == (L @ U)[i, :].
fn lu_impl(a: ArrayView2<'_, f64>) -> LuParts {
    let n = a.nrows();
    let mut u = a.to_owned();
    let mut l = Array2::<f64>::eye(n);
    let mut perm: Vec<usize> = (0..n).collect();
    for k in 0..n {
        let mut p = k;
        let mut max = u[(k, k)].abs();
        for i in (k + 1)..n {
            if u[(i, k)].abs() > max {
                max = u[(i, k)].abs();
                p = i;
            }
        }
        if max == 0.0 {
            continue;
        }
        if p != k {
            perm.swap(p, k);
            for j in 0..n {
                u.swap((k, j), (p, j));
            }
            for j in 0..k {
                l.swap((k, j), (p, j));
            }
        }
        let piv = u[(k, k)];
        for i in (k + 1)..n {
            let f = u[(i, k)] / piv;
            l[(i, k)] = f;
            u[(i, k)] = 0.0;
            for j in (k + 1)..n {
                u[(i, j)] -= f * u[(k, j)];
            }
        }
    }
    (perm, l, u)
}

#[pyfunction]
fn lu<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
) -> PyResult<(
    Vec<usize>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
)> {
    let a = a.as_array();
    let (perm, l, u) = py.allow_threads(|| lu_impl(a));
    Ok((perm, l.to_pyarray(py), u.to_pyarray(py)))
}

/// LDU without row exchanges; errors on a (near-)zero pivot (issue #84).
#[pyfunction]
fn ldu<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
)> {
    let a = a.as_array();
    let result = py.allow_threads(|| {
        let n = a.nrows();
        let scale = a.iter().fold(0.0f64, |m, &x| m.max(x.abs())).max(1.0);
        let mut u = a.to_owned();
        let mut l = Array2::<f64>::eye(n);
        for k in 0..n {
            let piv = u[(k, k)];
            if piv.abs() < 1e-12 * scale {
                return Err(format!(
                    "LDU requires nonzero pivots without row exchanges; zero \
                     pivot at position {}. Use lu() for a pivoted factorization.",
                    k
                ));
            }
            for i in (k + 1)..n {
                let f = u[(i, k)] / piv;
                l[(i, k)] = f;
                for j in k..n {
                    u[(i, j)] -= f * u[(k, j)];
                }
            }
        }
        let d: Vec<f64> = (0..n).map(|i| u[(i, i)]).collect();
        for i in 0..n {
            let di = d[i];
            for j in 0..n {
                u[(i, j)] /= di;
            }
        }
        Ok((l, d, u))
    });
    match result {
        Ok((l, d, u)) => Ok((
            l.to_pyarray(py),
            Array1::from(d).to_pyarray(py),
            u.to_pyarray(py),
        )),
        Err(msg) => Err(PyValueError::new_err(msg)),
    }
}

#[pyfunction]
fn cholesky<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let a = a.as_array();
    let result = py.allow_threads(|| {
        let n = a.nrows();
        let mut l = Array2::<f64>::zeros((n, n));
        for j in 0..n {
            let mut s = a[(j, j)];
            for k in 0..j {
                s -= l[(j, k)] * l[(j, k)];
            }
            if s <= 0.0 {
                return Err("matrix is not positive definite".to_string());
            }
            l[(j, j)] = s.sqrt();
            for i in (j + 1)..n {
                let mut s = a[(i, j)];
                for k in 0..j {
                    s -= l[(i, k)] * l[(j, k)];
                }
                l[(i, j)] = s / l[(j, j)];
            }
        }
        Ok(l)
    });
    result
        .map(|l| l.to_pyarray(py))
        .map_err(PyValueError::new_err)
}

/// Thin Householder QR: A (m x n) -> Q (m x r), R (r x n), r = min(m, n).
fn qr_impl(a: ArrayView2<'_, f64>) -> (Array2<f64>, Array2<f64>) {
    let m = a.nrows();
    let n = a.ncols();
    let r = m.min(n);
    let mut work = a.to_owned();
    let mut vs: Vec<Array1<f64>> = Vec::with_capacity(r);
    for k in 0..r {
        let mut norm = 0.0;
        for i in k..m {
            norm += work[(i, k)] * work[(i, k)];
        }
        let norm = norm.sqrt();
        let mut v = Array1::<f64>::zeros(m);
        if norm == 0.0 {
            vs.push(v);
            continue;
        }
        let alpha = if work[(k, k)] >= 0.0 { -norm } else { norm };
        for i in k..m {
            v[i] = work[(i, k)];
        }
        v[k] -= alpha;
        let vnorm2: f64 = v.iter().map(|x| x * x).sum();
        if vnorm2 > 0.0 {
            for j in k..n {
                let mut dot = 0.0;
                for i in k..m {
                    dot += v[i] * work[(i, j)];
                }
                let f = 2.0 * dot / vnorm2;
                for i in k..m {
                    work[(i, j)] -= f * v[i];
                }
            }
        }
        vs.push(v);
    }
    let mut q = Array2::<f64>::zeros((m, r));
    for j in 0..r {
        q[(j, j)] = 1.0;
    }
    for k in (0..r).rev() {
        let v = &vs[k];
        let vnorm2: f64 = v.iter().map(|x| x * x).sum();
        if vnorm2 == 0.0 {
            continue;
        }
        for j in 0..r {
            let mut dot = 0.0;
            for i in k..m {
                dot += v[i] * q[(i, j)];
            }
            let f = 2.0 * dot / vnorm2;
            for i in k..m {
                q[(i, j)] -= f * v[i];
            }
        }
    }
    let mut rr = Array2::<f64>::zeros((r, n));
    for i in 0..r {
        for j in i..n {
            rr[(i, j)] = work[(i, j)];
        }
    }
    (q, rr)
}

#[pyfunction]
fn qr<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
    let a = a.as_array();
    let (q, r) = py.allow_threads(|| qr_impl(a));
    Ok((q.to_pyarray(py), r.to_pyarray(py)))
}

/// Symmetric eigendecomposition by cyclic Jacobi; ascending eigenvalues.
fn eigh_impl(a: ArrayView2<'_, f64>) -> (Vec<f64>, Array2<f64>) {
    let n = a.nrows();
    let mut m = a.to_owned();
    let mut v = Array2::<f64>::eye(n);
    let scale = a.iter().fold(0.0f64, |acc, &x| acc + x * x).sqrt().max(1.0);
    for _sweep in 0..100 {
        let mut off = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                off += m[(i, j)] * m[(i, j)];
            }
        }
        if off.sqrt() < 1e-14 * scale {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = m[(p, q)];
                if apq == 0.0 {
                    continue;
                }
                let theta = (m[(q, q)] - m[(p, p)]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let mkp = m[(k, p)];
                    let mkq = m[(k, q)];
                    m[(k, p)] = c * mkp - s * mkq;
                    m[(k, q)] = s * mkp + c * mkq;
                }
                for k in 0..n {
                    let mpk = m[(p, k)];
                    let mqk = m[(q, k)];
                    m[(p, k)] = c * mpk - s * mqk;
                    m[(q, k)] = s * mpk + c * mqk;
                }
                for k in 0..n {
                    let vkp = v[(k, p)];
                    let vkq = v[(k, q)];
                    v[(k, p)] = c * vkp - s * vkq;
                    v[(k, q)] = s * vkp + c * vkq;
                }
            }
        }
    }
    let w: Vec<f64> = (0..n).map(|i| m[(i, i)]).collect();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| w[i].partial_cmp(&w[j]).unwrap());
    let ws: Vec<f64> = idx.iter().map(|&i| w[i]).collect();
    let mut vs = Array2::<f64>::zeros((n, n));
    for (newj, &oldj) in idx.iter().enumerate() {
        for i in 0..n {
            vs[(i, newj)] = v[(i, oldj)];
        }
    }
    (ws, vs)
}

#[pyfunction]
fn eigh<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>)> {
    let a = a.as_array();
    let (w, v) = py.allow_threads(|| eigh_impl(a));
    Ok((Array1::from(w).to_pyarray(py), v.to_pyarray(py)))
}

type SvdParts = (Array2<f64>, Vec<f64>, Array2<f64>);

/// Thin SVD by one-sided Jacobi for m >= n; descending singular values.
fn svd_impl(a: ArrayView2<'_, f64>) -> SvdParts {
    let m = a.nrows();
    let n = a.ncols();
    let mut u = a.to_owned();
    let mut v = Array2::<f64>::eye(n);
    let eps = 1e-15;
    for _sweep in 0..60 {
        let mut converged = true;
        for p in 0..n {
            for q in (p + 1)..n {
                let mut alpha = 0.0;
                let mut beta = 0.0;
                let mut gamma = 0.0;
                for i in 0..m {
                    alpha += u[(i, p)] * u[(i, p)];
                    beta += u[(i, q)] * u[(i, q)];
                    gamma += u[(i, p)] * u[(i, q)];
                }
                if gamma == 0.0 || gamma.abs() <= eps * (alpha * beta).sqrt() {
                    continue;
                }
                converged = false;
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                for i in 0..m {
                    let up = u[(i, p)];
                    let uq = u[(i, q)];
                    u[(i, p)] = c * up - s * uq;
                    u[(i, q)] = s * up + c * uq;
                }
                for i in 0..n {
                    let vp = v[(i, p)];
                    let vq = v[(i, q)];
                    v[(i, p)] = c * vp - s * vq;
                    v[(i, q)] = s * vp + c * vq;
                }
            }
        }
        if converged {
            break;
        }
    }
    let sig: Vec<f64> = (0..n)
        .map(|j| {
            let mut s = 0.0;
            for i in 0..m {
                s += u[(i, j)] * u[(i, j)];
            }
            s.sqrt()
        })
        .collect();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| sig[j].partial_cmp(&sig[i]).unwrap());
    let mut uu = Array2::<f64>::zeros((m, n));
    let mut vt = Array2::<f64>::zeros((n, n));
    let mut ss = vec![0.0; n];
    for (newj, &oldj) in idx.iter().enumerate() {
        ss[newj] = sig[oldj];
        if sig[oldj] > 0.0 {
            for i in 0..m {
                uu[(i, newj)] = u[(i, oldj)] / sig[oldj];
            }
        }
        for i in 0..n {
            vt[(newj, i)] = v[(i, oldj)];
        }
    }
    (uu, ss, vt)
}

#[pyfunction]
fn svd<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
)> {
    let a = a.as_array();
    let (u, s, vt) = py.allow_threads(|| svd_impl(a));
    Ok((
        u.to_pyarray(py),
        Array1::from(s).to_pyarray(py),
        vt.to_pyarray(py),
    ))
}


/// Principal eigenpair of a positive matrix by power iteration with L1
/// normalization (issue #87: AHP priority extraction; shared kernel).
#[pyfunction]
fn power_iteration<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    tol: f64,
    max_iter: usize,
) -> PyResult<(f64, Bound<'py, PyArray1<f64>>)> {
    let a = a.as_array();
    let result = py.allow_threads(|| {
        let n = a.nrows();
        let mut x = vec![1.0 / n as f64; n];
        for _ in 0..max_iter {
            let mut y = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    y[i] += a[(i, j)] * x[j];
                }
            }
            let s: f64 = y.iter().sum();
            for v in y.iter_mut() {
                *v /= s;
            }
            let diff: f64 = x
                .iter()
                .zip(y.iter())
                .map(|(p, q)| (p - q).abs())
                .sum();
            x = y;
            if diff < tol {
                let mut lambda = 0.0;
                for i in 0..n {
                    for j in 0..n {
                        lambda += a[(i, j)] * x[j];
                    }
                }
                return Ok((lambda, x));
            }
        }
        Err("power iteration did not converge".to_string())
    });
    match result {
        Ok((lambda, x)) => Ok((lambda, Array1::from(x).to_pyarray(py))),
        Err(msg) => Err(PyValueError::new_err(msg)),
    }
}

/// Limit supermatrix lim W^k by repeated squaring with convergence
/// detection (issue #87 ANP).
#[pyfunction]
fn limit_supermatrix<'py>(
    py: Python<'py>,
    w: PyReadonlyArray2<'py, f64>,
    tol: f64,
    max_iter: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let w = w.as_array();
    let result = py.allow_threads(|| {
        let mut cur = w.to_owned();
        for _ in 0..max_iter {
            let next = crate::markov::matmul(&cur, &cur);
            let diff = next
                .iter()
                .zip(cur.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            cur = next;
            if diff < tol {
                return Ok(cur);
            }
        }
        Err("limit supermatrix did not converge; the supermatrix may be \
             cyclic"
            .to_string())
    });
    result
        .map(|x| x.to_pyarray(py))
        .map_err(PyValueError::new_err)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(lu, m)?)?;
    m.add_function(wrap_pyfunction!(ldu, m)?)?;
    m.add_function(wrap_pyfunction!(cholesky, m)?)?;
    m.add_function(wrap_pyfunction!(qr, m)?)?;
    m.add_function(wrap_pyfunction!(eigh, m)?)?;
    m.add_function(wrap_pyfunction!(svd, m)?)?;
    m.add_function(wrap_pyfunction!(power_iteration, m)?)?;
    m.add_function(wrap_pyfunction!(limit_supermatrix, m)?)?;
    Ok(())
}
