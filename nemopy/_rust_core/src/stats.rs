//! Column-wise statistics kernels (issues #77, #79).

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use numpy::{PyArray1, PyArray2, PyReadonlyArray2, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

/// Single-pass Welford accumulation: returns (mean, M2).
fn welford(col: ArrayView1<'_, f64>) -> (f64, f64) {
    let mut mean = 0.0;
    let mut m2 = 0.0;
    for (i, &x) in col.iter().enumerate() {
        let delta = x - mean;
        mean += delta / (i as f64 + 1.0);
        m2 += delta * (x - mean);
    }
    (mean, m2)
}

/// Fused column-wise mean and variance in one traversal (Welford).
#[pyfunction]
fn colwise_mean_var<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    ddof: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let a = a.as_array();
    let n = a.nrows() as f64;
    let (means, vars): (Vec<f64>, Vec<f64>) = py.allow_threads(|| {
        a.axis_iter(Axis(1))
            .into_par_iter()
            .map(|col| {
                let (mean, m2) = welford(col);
                (mean, m2 / (n - ddof as f64))
            })
            .unzip()
    });
    Ok((
        Array1::from(means).to_pyarray(py),
        Array1::from(vars).to_pyarray(py),
    ))
}

fn cov_impl(a: ArrayView2<'_, f64>, ddof: usize) -> Array2<f64> {
    let n = a.nrows();
    let k = a.ncols();
    let means: Vec<f64> = a
        .axis_iter(Axis(1))
        .map(|c| c.sum() / n as f64)
        .collect();
    let denom = (n - ddof) as f64;
    let pairs: Vec<(usize, usize)> = (0..k)
        .flat_map(|i| (i..k).map(move |j| (i, j)))
        .collect();
    let vals: Vec<f64> = pairs
        .par_iter()
        .map(|&(i, j)| {
            let ci = a.column(i);
            let cj = a.column(j);
            let (mi, mj) = (means[i], means[j]);
            let mut s = 0.0;
            for r in 0..n {
                s += (ci[r] - mi) * (cj[r] - mj);
            }
            s / denom
        })
        .collect();
    let mut out = Array2::<f64>::zeros((k, k));
    for (&(i, j), &v) in pairs.iter().zip(vals.iter()) {
        out[(i, j)] = v;
        out[(j, i)] = v;
    }
    out
}

/// Covariance matrix treating columns as variables (np.cov rowvar=False).
#[pyfunction]
fn cov<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    ddof: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let a = a.as_array();
    let out = py.allow_threads(|| cov_impl(a, ddof));
    Ok(out.to_pyarray(py))
}

/// Correlation matrix of columns, derived from the covariance kernel.
#[pyfunction]
fn corr<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let a = a.as_array();
    let out = py.allow_threads(|| {
        let mut c = cov_impl(a, 1);
        let k = c.nrows();
        let d: Vec<f64> = (0..k).map(|i| c[(i, i)].sqrt()).collect();
        for i in 0..k {
            for j in 0..k {
                c[(i, j)] /= d[i] * d[j];
            }
        }
        c
    });
    Ok(out.to_pyarray(py))
}

/// Fused column norm + scale: every column rescaled to unit L2 norm.
#[pyfunction]
fn colwise_normalize<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let a = a.as_array();
    let result = py.allow_threads(|| {
        let mut out = a.to_owned();
        let norms: Vec<f64> = a
            .axis_iter(Axis(1))
            .into_par_iter()
            .map(|c| c.dot(&c).sqrt())
            .collect();
        if let Some(j) = norms.iter().position(|&x| x == 0.0) {
            return Err(j);
        }
        out.axis_iter_mut(Axis(1))
            .into_par_iter()
            .zip(norms.par_iter())
            .for_each(|(mut col, &nrm)| col.mapv_inplace(|x| x / nrm));
        Ok(out)
    });
    match result {
        Ok(out) => Ok(out.to_pyarray(py)),
        Err(j) => Err(PyValueError::new_err(format!(
            "cannot normalize: column {} has zero norm",
            j
        ))),
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(colwise_mean_var, m)?)?;
    m.add_function(wrap_pyfunction!(cov, m)?)?;
    m.add_function(wrap_pyfunction!(corr, m)?)?;
    m.add_function(wrap_pyfunction!(colwise_normalize, m)?)?;
    Ok(())
}
