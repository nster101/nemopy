//! LP/IP/MILP solver kernels (issue #91): two-phase primal simplex with
//! Bland's rule, Big-M, dual simplex, and depth-first branch-and-bound.
//! All solver logic is Rust-primary per the issue ("Python layer is
//! API/ergonomics only"); there is no Python port.

use ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub(crate) const OPTIMAL: i32 = 0;
pub(crate) const INFEASIBLE: i32 = 1;
pub(crate) const UNBOUNDED: i32 = 2;
pub(crate) const ITER_LIMIT: i32 = 3;

const TOL: f64 = 1e-9;

struct Std {
    a: Array2<f64>,
    b: Vec<f64>,
    n: usize,
    n_slack: usize,
    n_art: usize,
    basis: Vec<usize>,
    flipped: Vec<bool>,
    unit_col: Vec<usize>,
}

/// Standard form: rows = ub rows then eq rows; columns = original vars,
/// one slack per ub row, artificials for eq rows and flipped ub rows.
fn build_std(
    c_len: usize,
    a_ub: &Array2<f64>,
    b_ub: &[f64],
    a_eq: &Array2<f64>,
    b_eq: &[f64],
) -> Std {
    let m_ub = b_ub.len();
    let m_eq = b_eq.len();
    let m = m_ub + m_eq;
    let mut n_art = 0;
    for &v in b_ub {
        if v < 0.0 {
            n_art += 1;
        }
    }
    n_art += m_eq;
    let ncols = c_len + m_ub + n_art;
    let mut a = Array2::<f64>::zeros((m, ncols));
    let mut b = vec![0.0; m];
    let mut basis = vec![0usize; m];
    let mut flipped = vec![false; m];
    let mut unit_col = vec![0usize; m];
    let mut art = c_len + m_ub;
    for i in 0..m_ub {
        let flip = b_ub[i] < 0.0;
        let sgn = if flip { -1.0 } else { 1.0 };
        for j in 0..c_len {
            a[(i, j)] = sgn * a_ub[(i, j)];
        }
        a[(i, c_len + i)] = sgn;
        b[i] = sgn * b_ub[i];
        flipped[i] = flip;
        if flip {
            a[(i, art)] = 1.0;
            basis[i] = art;
            unit_col[i] = art;
            art += 1;
        } else {
            basis[i] = c_len + i;
            unit_col[i] = c_len + i;
        }
    }
    for r in 0..m_eq {
        let i = m_ub + r;
        let flip = b_eq[r] < 0.0;
        let sgn = if flip { -1.0 } else { 1.0 };
        for j in 0..c_len {
            a[(i, j)] = sgn * a_eq[(r, j)];
        }
        b[i] = sgn * b_eq[r];
        flipped[i] = flip;
        a[(i, art)] = 1.0;
        basis[i] = art;
        unit_col[i] = art;
        art += 1;
    }
    Std {
        a,
        b,
        n: c_len,
        n_slack: m_ub,
        n_art,
        basis,
        flipped,
        unit_col,
    }
}

fn pivot(a: &mut Array2<f64>, b: &mut [f64], basis: &mut [usize], r: usize, jc: usize) {
    let m = b.len();
    let ncols = a.ncols();
    let piv = a[(r, jc)];
    for j in 0..ncols {
        a[(r, j)] /= piv;
    }
    b[r] /= piv;
    for i in 0..m {
        if i != r {
            let f = a[(i, jc)];
            if f != 0.0 {
                for j in 0..ncols {
                    a[(i, j)] -= f * a[(r, j)];
                }
                b[i] -= f * b[r];
            }
        }
    }
    basis[r] = jc;
}

fn reduced_cost(a: &Array2<f64>, basis: &[usize], c: &[f64], j: usize) -> f64 {
    let mut rc = c[j];
    for (i, &bi) in basis.iter().enumerate() {
        let cb = c[bi];
        if cb != 0.0 {
            rc -= cb * a[(i, j)];
        }
    }
    rc
}

/// Primal simplex with Bland's rule on the maintained tableau.
fn iterate(
    a: &mut Array2<f64>,
    b: &mut [f64],
    basis: &mut [usize],
    c: &[f64],
    allowed: usize,
    max_iter: usize,
    iters: &mut usize,
) -> i32 {
    let m = b.len();
    loop {
        if *iters >= max_iter {
            return ITER_LIMIT;
        }
        let mut enter = None;
        for j in 0..allowed {
            if basis.contains(&j) {
                continue;
            }
            if reduced_cost(a, basis, c, j) < -TOL {
                enter = Some(j);
                break;
            }
        }
        let Some(jc) = enter else {
            return OPTIMAL;
        };
        let mut leave: Option<usize> = None;
        let mut best = f64::INFINITY;
        for i in 0..m {
            if a[(i, jc)] > TOL {
                let ratio = b[i] / a[(i, jc)];
                let replace = match leave {
                    None => true,
                    // Bland tie-break: smallest basis index on equal ratios
                    Some(l) => {
                        ratio < best - 1e-12
                            || (ratio < best + 1e-12 && basis[i] < basis[l])
                    }
                };
                if replace {
                    best = best.min(ratio);
                    leave = Some(i);
                }
            }
        }
        let Some(r) = leave else {
            return UNBOUNDED;
        };
        pivot(a, b, basis, r, jc);
        *iters += 1;
    }
}

struct LpOutcome {
    status: i32,
    x: Vec<f64>,
    obj: f64,
    dual: Vec<f64>,
    iters: usize,
    obj_range: Array2<f64>,
    rhs_range: Array2<f64>,
    reduced: Vec<f64>,
}

fn empty_outcome(status: i32, n: usize, m: usize, iters: usize) -> LpOutcome {
    LpOutcome {
        status,
        x: vec![0.0; n],
        obj: 0.0,
        dual: vec![0.0; m],
        iters,
        obj_range: Array2::zeros((n, 2)),
        rhs_range: Array2::zeros((m, 2)),
        reduced: vec![0.0; n],
    }
}

fn solve_core(
    c: &[f64],
    a_ub: &Array2<f64>,
    b_ub: &[f64],
    a_eq: &Array2<f64>,
    b_eq: &[f64],
    method: &str,
    max_iter: usize,
) -> Result<LpOutcome, String> {
    let n = c.len();
    let m = b_ub.len() + b_eq.len();
    let std_form = build_std(n, a_ub, b_ub, a_eq, b_eq);
    let orig_a = std_form.a.clone();
    let mut a = std_form.a;
    let mut b = std_form.b;
    let mut basis = std_form.basis;
    let allowed = n + std_form.n_slack;
    let total = allowed + std_form.n_art;
    let mut iters = 0usize;

    let mut c_full = vec![0.0; total];
    c_full[..n].copy_from_slice(c);

    let status = match method {
        "dual" => {
            if !b_eq.is_empty() {
                return Err(
                    "dual simplex supports inequality-only problems".into()
                );
            }
            if c.iter().any(|&v| v < 0.0) {
                return Err(
                    "dual simplex requires nonnegative objective \
                     coefficients (dual-feasible start)"
                        .into(),
                );
            }
            // restart from the slack basis on the unflipped rows
            a = Array2::zeros((m, total));
            for i in 0..m {
                for j in 0..n {
                    a[(i, j)] = a_ub[(i, j)];
                }
                a[(i, n + i)] = 1.0;
                b[i] = b_ub[i];
                basis[i] = n + i;
            }
            loop {
                if iters >= max_iter {
                    break ITER_LIMIT;
                }
                let mut r = None;
                let mut most = -TOL;
                for i in 0..m {
                    if b[i] < most {
                        most = b[i];
                        r = Some(i);
                    }
                }
                let Some(r) = r else {
                    break OPTIMAL;
                };
                let mut jc = None;
                let mut best = f64::INFINITY;
                for j in 0..allowed {
                    if a[(r, j)] < -TOL {
                        let ratio =
                            reduced_cost(&a, &basis, &c_full, j) / -a[(r, j)];
                        if ratio < best {
                            best = ratio;
                            jc = Some(j);
                        }
                    }
                }
                let Some(jc) = jc else {
                    break INFEASIBLE;
                };
                pivot(&mut a, &mut b, &mut basis, r, jc);
                iters += 1;
            }
        }
        "bigm" => {
            let scale = c.iter().fold(1.0f64, |acc, &v| acc.max(v.abs()));
            let big = 1e7 * scale;
            let mut cm = c_full.clone();
            for j in allowed..total {
                cm[j] = big;
            }
            let s = iterate(&mut a, &mut b, &mut basis, &cm, total, max_iter, &mut iters);
            if s == OPTIMAL
                && basis
                    .iter()
                    .enumerate()
                    .any(|(i, &bi)| bi >= allowed && b[i] > 1e-6)
            {
                INFEASIBLE
            } else {
                s
            }
        }
        "simplex" | "two_phase" => {
            let mut phase1 = vec![0.0; total];
            for j in allowed..total {
                phase1[j] = 1.0;
            }
            let s1 = iterate(
                &mut a, &mut b, &mut basis, &phase1, total, max_iter, &mut iters,
            );
            if s1 != OPTIMAL {
                s1
            } else {
                let p1_obj: f64 = basis
                    .iter()
                    .enumerate()
                    .filter(|(_, &bi)| bi >= allowed)
                    .map(|(i, _)| b[i])
                    .sum();
                if p1_obj > 1e-7 {
                    INFEASIBLE
                } else {
                    // Drive any artificial still basic at zero out of the
                    // basis; otherwise its (cost-0) row goes slack in
                    // phase 2 and the constraint is silently dropped.
                    for r in 0..m {
                        if basis[r] >= allowed {
                            if let Some(j) =
                                (0..allowed).find(|&j| a[(r, j)].abs() > TOL)
                            {
                                pivot(&mut a, &mut b, &mut basis, r, j);
                                iters += 1;
                            }
                        }
                    }
                    iterate(
                        &mut a, &mut b, &mut basis, &c_full, allowed, max_iter,
                        &mut iters,
                    )
                }
            }
        }
        other => return Err(format!("unknown method {:?}", other)),
    };

    if status != OPTIMAL {
        return Ok(empty_outcome(status, n, m, iters));
    }

    let mut x = vec![0.0; n];
    for (i, &bi) in basis.iter().enumerate() {
        if bi < n {
            x[bi] = b[i];
        }
    }
    let obj: f64 = c.iter().zip(x.iter()).map(|(ci, xi)| ci * xi).sum();

    // duals: solve B^T y = c_B on the original standard-form columns
    let mb = basis.len();
    let mut bt = Array2::<f64>::zeros((mb, mb));
    let mut cb = Array2::<f64>::zeros((mb, 1));
    for (i, &bi) in basis.iter().enumerate() {
        for r in 0..mb {
            bt[(i, r)] = orig_a[(r, bi)];
        }
        cb[(i, 0)] = c_full[bi];
    }
    let y = crate::markov::solve_matrix(&bt, &cb)
        .map_err(|e| format!("dual computation failed: {}", e))?;
    let mut dual = vec![0.0; m];
    for i in 0..m {
        dual[i] = if std_form.flipped[i] { -y[(i, 0)] } else { y[(i, 0)] };
    }

    // reduced costs of original variables
    let mut reduced = vec![0.0; n];
    for (j, red) in reduced.iter_mut().enumerate() {
        let mut rc = c[j];
        for i in 0..m {
            rc -= y[(i, 0)] * orig_a[(i, j)];
        }
        *red = rc;
    }

    // objective coefficient ranges
    let mut obj_range = Array2::<f64>::zeros((n, 2));
    for j in 0..n {
        if let Some(rpos) = basis.iter().position(|&bi| bi == j) {
            let mut lo = f64::NEG_INFINITY;
            let mut hi = f64::INFINITY;
            for k in 0..allowed {
                if basis.contains(&k) {
                    continue;
                }
                let alpha = a[(rpos, k)];
                if alpha.abs() <= TOL {
                    continue;
                }
                let rck = reduced_cost(&a, &basis, &c_full, k);
                let bound = rck / alpha;
                if alpha > 0.0 {
                    hi = hi.min(bound);
                } else {
                    lo = lo.max(bound);
                }
            }
            obj_range[(j, 0)] = c[j] + lo;
            obj_range[(j, 1)] = c[j] + hi;
        } else {
            obj_range[(j, 0)] = c[j] - reduced[j];
            obj_range[(j, 1)] = f64::INFINITY;
        }
    }

    // RHS ranges via B^-1 columns read off the final tableau
    let mut rhs_range = Array2::<f64>::zeros((m, 2));
    for i in 0..m {
        let col = std_form.unit_col[i];
        let mut lo = f64::NEG_INFINITY;
        let mut hi = f64::INFINITY;
        for k in 0..mb {
            let d = a[(k, col)];
            if d > TOL {
                lo = lo.max(-b[k] / d);
            } else if d < -TOL {
                hi = hi.min(-b[k] / d);
            }
        }
        let base = if std_form.flipped[i] {
            -(b_ub
                .iter()
                .chain(b_eq.iter())
                .nth(i)
                .copied()
                .unwrap_or(0.0))
        } else {
            b_ub.iter().chain(b_eq.iter()).nth(i).copied().unwrap_or(0.0)
        };
        let (mut rlo, mut rhi) = (base + lo, base + hi);
        if std_form.flipped[i] {
            let (a2, b2) = (-rhi, -rlo);
            rlo = a2;
            rhi = b2;
        }
        rhs_range[(i, 0)] = rlo;
        rhs_range[(i, 1)] = rhi;
    }

    Ok(LpOutcome {
        status,
        x,
        obj,
        dual,
        iters,
        obj_range,
        rhs_range,
        reduced,
    })
}

type LpReturn<'py> = (
    i32,
    Bound<'py, PyArray1<f64>>,
    f64,
    Bound<'py, PyArray1<f64>>,
    usize,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
);

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn lp_solve<'py>(
    py: Python<'py>,
    c: PyReadonlyArray1<'py, f64>,
    a_ub: PyReadonlyArray2<'py, f64>,
    b_ub: PyReadonlyArray1<'py, f64>,
    a_eq: PyReadonlyArray2<'py, f64>,
    b_eq: PyReadonlyArray1<'py, f64>,
    method: &str,
    max_iter: usize,
) -> PyResult<LpReturn<'py>> {
    let c = c.as_array().to_vec();
    let a_ub = a_ub.as_array().to_owned();
    let b_ub = b_ub.as_array().to_vec();
    let a_eq = a_eq.as_array().to_owned();
    let b_eq = b_eq.as_array().to_vec();
    let out = py
        .allow_threads(|| {
            solve_core(&c, &a_ub, &b_ub, &a_eq, &b_eq, method, max_iter)
        })
        .map_err(PyValueError::new_err)?;
    Ok((
        out.status,
        Array1::from(out.x).to_pyarray(py),
        out.obj,
        Array1::from(out.dual).to_pyarray(py),
        out.iters,
        out.obj_range.to_pyarray(py),
        out.rhs_range.to_pyarray(py),
        Array1::from(out.reduced).to_pyarray(py),
    ))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn milp_solve<'py>(
    py: Python<'py>,
    c: PyReadonlyArray1<'py, f64>,
    a_ub: PyReadonlyArray2<'py, f64>,
    b_ub: PyReadonlyArray1<'py, f64>,
    a_eq: PyReadonlyArray2<'py, f64>,
    b_eq: PyReadonlyArray1<'py, f64>,
    int_mask: Vec<bool>,
    gap: f64,
    max_nodes: usize,
    max_iter: usize,
) -> PyResult<(i32, Bound<'py, PyArray1<f64>>, f64, usize)> {
    let c = c.as_array().to_vec();
    let a_ub = a_ub.as_array().to_owned();
    let b_ub = b_ub.as_array().to_vec();
    let a_eq = a_eq.as_array().to_owned();
    let b_eq = b_eq.as_array().to_vec();
    let n = c.len();
    let result = py.allow_threads(|| -> Result<(i32, Vec<f64>, f64, usize), String> {
        let mut best_obj = f64::INFINITY;
        let mut best_x: Option<Vec<f64>> = None;
        let mut nodes = 0usize;
        let mut stack: Vec<(Vec<f64>, Vec<f64>)> =
            vec![(vec![0.0; n], vec![f64::INFINITY; n])];
        while let Some((lo, hi)) = stack.pop() {
            if nodes >= max_nodes {
                break;
            }
            nodes += 1;
            // assemble bound rows on top of the base constraints
            let mut extra = Vec::new();
            for j in 0..n {
                if lo[j] > 0.0 {
                    let mut row = vec![0.0; n];
                    row[j] = -1.0;
                    extra.push((row, -lo[j]));
                }
                if hi[j].is_finite() {
                    let mut row = vec![0.0; n];
                    row[j] = 1.0;
                    extra.push((row, hi[j]));
                }
            }
            let m_total = b_ub.len() + extra.len();
            let mut a2 = Array2::<f64>::zeros((m_total, n));
            let mut b2 = vec![0.0; m_total];
            for i in 0..b_ub.len() {
                for j in 0..n {
                    a2[(i, j)] = a_ub[(i, j)];
                }
                b2[i] = b_ub[i];
            }
            for (k, (row, rhs)) in extra.iter().enumerate() {
                let i = b_ub.len() + k;
                for j in 0..n {
                    a2[(i, j)] = row[j];
                }
                b2[i] = *rhs;
            }
            let out = solve_core(&c, &a2, &b2, &a_eq, &b_eq, "simplex", max_iter)?;
            match out.status {
                INFEASIBLE => continue,
                UNBOUNDED => {
                    if nodes == 1 {
                        return Ok((UNBOUNDED, vec![0.0; n], 0.0, nodes));
                    }
                    continue;
                }
                ITER_LIMIT => return Ok((ITER_LIMIT, vec![0.0; n], 0.0, nodes)),
                _ => {}
            }
            if out.obj >= best_obj - gap * best_obj.abs().max(1e-12) - 1e-9 {
                continue;
            }
            // most fractional integer variable
            let mut branch: Option<(usize, f64)> = None;
            let mut most = 1e-6;
            for j in 0..n {
                if int_mask[j] {
                    let f = out.x[j] - out.x[j].floor();
                    let dist = f.min(1.0 - f);
                    if dist > most {
                        most = dist;
                        branch = Some((j, out.x[j]));
                    }
                }
            }
            match branch {
                None => {
                    if out.obj < best_obj {
                        best_obj = out.obj;
                        best_x = Some(out.x);
                    }
                }
                Some((j, xj)) => {
                    let mut hi_child = (lo.clone(), hi.clone());
                    hi_child.1[j] = xj.floor();
                    let mut lo_child = (lo, hi);
                    lo_child.0[j] = xj.ceil();
                    stack.push(hi_child);
                    stack.push(lo_child);
                }
            }
        }
        match best_x {
            Some(x) => Ok((OPTIMAL, x, best_obj, nodes)),
            None => Ok((INFEASIBLE, vec![0.0; n], 0.0, nodes)),
        }
    });
    let (status, x, obj, nodes) = result.map_err(PyValueError::new_err)?;
    Ok((status, Array1::from(x).to_pyarray(py), obj, nodes))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(lp_solve, m)?)?;
    m.add_function(wrap_pyfunction!(milp_solve, m)?)?;
    Ok(())
}
