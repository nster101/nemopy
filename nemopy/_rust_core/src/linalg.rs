//! Elimination kernels (issue #85): REF, RREF with pivot tracking, and a
//! fused Gaussian factor+solve. Blocked row operations stay in Rust; the
//! pedagogical step-by-step mode lives in Python by design.

use ndarray::{Array1, Array2, ArrayView2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray2, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn default_tol(a: &ArrayView2<'_, f64>) -> f64 {
    let scale = a.iter().fold(0.0f64, |m, &x| m.max(x.abs())).max(1.0);
    f64::EPSILON * (a.nrows().max(a.ncols()) as f64) * scale
}

fn ref_impl(a: ArrayView2<'_, f64>, partial: bool) -> Result<Array2<f64>, String> {
    let (m, n) = (a.nrows(), a.ncols());
    let mut r = a.to_owned();
    let tol = default_tol(&a);
    let mut row = 0;
    for col in 0..n {
        if row >= m {
            break;
        }
        let mut p = row;
        if partial {
            let mut best = r[(row, col)].abs();
            for i in (row + 1)..m {
                if r[(i, col)].abs() > best {
                    best = r[(i, col)].abs();
                    p = i;
                }
            }
        }
        if r[(p, col)].abs() <= tol {
            if !partial {
                let needed = ((row + 1)..m).any(|i| r[(i, col)].abs() > tol);
                if needed {
                    return Err(format!(
                        "zero pivot in column {} requires a row exchange; \
                         use pivot=\"partial\"",
                        col
                    ));
                }
            }
            continue;
        }
        if p != row {
            for j in 0..n {
                r.swap((row, j), (p, j));
            }
        }
        let piv = r[(row, col)];
        for i in (row + 1)..m {
            let f = r[(i, col)] / piv;
            if f != 0.0 {
                for j in col..n {
                    r[(i, j)] -= f * r[(row, j)];
                }
            }
            r[(i, col)] = 0.0;
        }
        row += 1;
    }
    Ok(r)
}

#[pyfunction(name = "ref_")]
fn ref_py<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    partial: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let a = a.as_array();
    py.allow_threads(|| ref_impl(a, partial))
        .map(|r| r.to_pyarray(py))
        .map_err(PyValueError::new_err)
}

fn rref_impl(a: ArrayView2<'_, f64>) -> (Array2<f64>, Vec<usize>) {
    let (m, n) = (a.nrows(), a.ncols());
    let mut r = a.to_owned();
    let tol = default_tol(&a);
    let mut pivots = Vec::new();
    let mut row = 0;
    for col in 0..n {
        if row >= m {
            break;
        }
        let mut p = row;
        let mut best = r[(row, col)].abs();
        for i in (row + 1)..m {
            if r[(i, col)].abs() > best {
                best = r[(i, col)].abs();
                p = i;
            }
        }
        if best <= tol {
            continue;
        }
        if p != row {
            for j in 0..n {
                r.swap((row, j), (p, j));
            }
        }
        let piv = r[(row, col)];
        for j in 0..n {
            r[(row, j)] /= piv;
        }
        r[(row, col)] = 1.0;
        for i in 0..m {
            if i == row {
                continue;
            }
            let f = r[(i, col)];
            if f != 0.0 {
                for j in 0..n {
                    r[(i, j)] -= f * r[(row, j)];
                }
                r[(i, col)] = 0.0;
            }
        }
        pivots.push(col);
        row += 1;
    }
    // canonicalize negative zeros produced by eliminations
    r.mapv_inplace(|x| if x == 0.0 { 0.0 } else { x });
    (r, pivots)
}

#[pyfunction]
fn rref<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Vec<usize>)> {
    let a = a.as_array();
    let (r, pivots) = py.allow_threads(|| rref_impl(a));
    Ok((r.to_pyarray(py), pivots))
}

/// Fused Gaussian factor + solve for square systems (partial pivoting).
#[pyfunction]
fn gauss_solve<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    b: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a = a.as_array();
    let b = b.as_array();
    let result = py.allow_threads(|| {
        let n = a.nrows();
        let tol = default_tol(&a);
        let mut w = Array2::<f64>::zeros((n, n + 1));
        for i in 0..n {
            for j in 0..n {
                w[(i, j)] = a[(i, j)];
            }
            w[(i, n)] = b[(i, 0)];
        }
        for k in 0..n {
            let mut p = k;
            let mut best = w[(k, k)].abs();
            for i in (k + 1)..n {
                if w[(i, k)].abs() > best {
                    best = w[(i, k)].abs();
                    p = i;
                }
            }
            if best <= tol {
                return Err("matrix is singular to working precision".to_string());
            }
            if p != k {
                for j in 0..=n {
                    w.swap((k, j), (p, j));
                }
            }
            let piv = w[(k, k)];
            for i in (k + 1)..n {
                let f = w[(i, k)] / piv;
                if f != 0.0 {
                    for j in k..=n {
                        w[(i, j)] -= f * w[(k, j)];
                    }
                }
            }
        }
        let mut x = Array1::<f64>::zeros(n);
        for i in (0..n).rev() {
            let mut s = w[(i, n)];
            for j in (i + 1)..n {
                s -= w[(i, j)] * x[j];
            }
            x[i] = s / w[(i, i)];
        }
        Ok(x)
    });
    result
        .map(|x| x.to_pyarray(py))
        .map_err(PyValueError::new_err)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ref_py, m)?)?;
    m.add_function(wrap_pyfunction!(rref, m)?)?;
    m.add_function(wrap_pyfunction!(gauss_solve, m)?)?;
    Ok(())
}
