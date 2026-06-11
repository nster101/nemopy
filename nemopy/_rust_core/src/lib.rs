//! nemopy `_rust_core` — Rust fast paths for nemopy (issue #75).
//!
//! The Python layer dispatches to this extension when it is importable
//! and falls back to pure NumPy for the NumPy-replacement surface per
//! DESIGN_APPENDICES.md §20.1. Module layout follows §20.2.

use pyo3::prelude::*;

mod decomp;
mod linalg;
mod markov;
mod ops;
mod optim;
mod stats;

#[pyfunction]
fn rust_core_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pymodule]
fn _rust_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rust_core_version, m)?)?;
    ops::register(m)?;
    stats::register(m)?;
    decomp::register(m)?;
    linalg::register(m)?;
    markov::register(m)?;
    optim::register(m)?;
    Ok(())
}
