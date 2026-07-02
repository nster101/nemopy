//! Fused shape-guard arithmetic and batch kernels (issues #75, #80).

use numpy::{PyArray2, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
use pyo3::sync::GILOnceCell;
use pyo3::types::PyType;

static SHAPE_ERROR: GILOnceCell<Py<PyType>> = GILOnceCell::new();

/// Raise nemopy.ShapeError when registered, ValueError otherwise
/// (ShapeError subclasses ValueError, so callers see consistent types).
pub(crate) fn shape_error(py: Python<'_>, msg: String) -> PyErr {
    match SHAPE_ERROR.get(py) {
        Some(cls) => match cls.bind(py).call1((msg,)) {
            Ok(inst) => PyErr::from_value(inst),
            Err(err) => err,
        },
        None => pyo3::exceptions::PyValueError::new_err(msg),
    }
}

#[pyfunction]
fn register_shape_error(cls: Bound<'_, PyType>) {
    let py = cls.py();
    let _ = SHAPE_ERROR.set(py, cls.unbind());
}

/// Fused shape guard + elementwise subtraction (issue #75 Phase 1).
#[pyfunction]
fn fused_sub<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    b: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let a = a.as_array();
    let b = b.as_array();
    if a.shape() != b.shape() {
        return Err(shape_error(
            py,
            format!(
                "Element-wise '-' requires identical shapes, got ({}, {}) and ({}, {}). \
                 If broadcasting is intended, use np.multiply / np.add directly.",
                a.shape()[0],
                a.shape()[1],
                b.shape()[0],
                b.shape()[1]
            ),
        ));
    }
    let out = py.allow_threads(|| &a - &b);
    Ok(out.to_pyarray(py))
}

/// Fused shape guard + elementwise addition (issue #109 Phase 2).
#[pyfunction]
fn fused_add<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    b: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let a = a.as_array();
    let b = b.as_array();
    if a.shape() != b.shape() {
        return Err(shape_error(
            py,
            format!(
                "Element-wise '+' requires identical shapes, got ({}, {}) and ({}, {}). \
                 If broadcasting is intended, use np.multiply / np.add directly.",
                a.shape()[0],
                a.shape()[1],
                b.shape()[0],
                b.shape()[1]
            ),
        ));
    }
    let out = py.allow_threads(|| &a + &b);
    Ok(out.to_pyarray(py))
}

/// Fused shape guard + elementwise multiplication (issue #109 Phase 2).
#[pyfunction]
fn fused_mul<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    b: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let a = a.as_array();
    let b = b.as_array();
    if a.shape() != b.shape() {
        return Err(shape_error(
            py,
            format!(
                "Element-wise '*' requires identical shapes, got ({}, {}) and ({}, {}). \
                 If broadcasting is intended, use np.multiply / np.add directly.",
                a.shape()[0],
                a.shape()[1],
                b.shape()[0],
                b.shape()[1]
            ),
        ));
    }
    let out = py.allow_threads(|| &a * &b);
    Ok(out.to_pyarray(py))
}

/// Fused shape guard + elementwise division (issue #109 Phase 2).
#[pyfunction]
fn fused_div<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    b: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let a = a.as_array();
    let b = b.as_array();
    if a.shape() != b.shape() {
        return Err(shape_error(
            py,
            format!(
                "Element-wise '/' requires identical shapes, got ({}, {}) and ({}, {}). \
                 If broadcasting is intended, use np.multiply / np.add directly.",
                a.shape()[0],
                a.shape()[1],
                b.shape()[0],
                b.shape()[1]
            ),
        ));
    }
    let out = py.allow_threads(|| &a / &b);
    Ok(out.to_pyarray(py))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(register_shape_error, m)?)?;
    m.add_function(wrap_pyfunction!(fused_sub, m)?)?;
    m.add_function(wrap_pyfunction!(fused_add, m)?)?;
    m.add_function(wrap_pyfunction!(fused_mul, m)?)?;
    m.add_function(wrap_pyfunction!(fused_div, m)?)?;
    Ok(())
}
