//! Markov chain kernels (issue #86): alias-method path simulation with
//! Rayon parallelism, hitting-time Monte Carlo, power-iteration steady
//! state, Kosaraju communication classes, BFS period, and a Padé
//! scaling-and-squaring matrix exponential for CTMCs.

use ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray2, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

struct AliasTable {
    prob: Vec<f64>,
    alias: Vec<usize>,
}

/// Walker/Vose alias table for O(1) categorical sampling.
fn build_alias(row: &[f64]) -> AliasTable {
    let n = row.len();
    let mut prob = vec![0.0; n];
    let mut alias = vec![0usize; n];
    let mut work: Vec<f64> = row.iter().map(|&p| p * n as f64).collect();
    let mut small: Vec<usize> = Vec::new();
    let mut large: Vec<usize> = Vec::new();
    for (i, &w) in work.iter().enumerate() {
        if w < 1.0 {
            small.push(i);
        } else {
            large.push(i);
        }
    }
    while !small.is_empty() && !large.is_empty() {
        let s = small.pop().unwrap();
        let l = large.pop().unwrap();
        prob[s] = work[s];
        alias[s] = l;
        work[l] -= 1.0 - work[s];
        if work[l] < 1.0 {
            small.push(l);
        } else {
            large.push(l);
        }
    }
    for l in large {
        prob[l] = 1.0;
    }
    for s in small {
        prob[s] = 1.0;
    }
    AliasTable { prob, alias }
}

fn sample(table: &AliasTable, rng: &mut SmallRng) -> usize {
    let n = table.prob.len();
    let i = rng.gen_range(0..n);
    if rng.gen::<f64>() < table.prob[i] {
        i
    } else {
        table.alias[i]
    }
}

fn path_seed(seed: u64, path: usize) -> u64 {
    seed ^ ((path as u64).wrapping_add(1).wrapping_mul(0x9E3779B97F4A7C15))
}

#[pyfunction]
fn markov_simulate<'py>(
    py: Python<'py>,
    p: PyReadonlyArray2<'py, f64>,
    start: usize,
    n_steps: usize,
    n_paths: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let p = p.as_array();
    let out = py.allow_threads(|| {
        let n = p.nrows();
        let tables: Vec<AliasTable> =
            (0..n).map(|i| build_alias(&p.row(i).to_vec())).collect();
        let cols: Vec<Vec<f64>> = (0..n_paths)
            .into_par_iter()
            .map(|path| {
                let mut rng = SmallRng::seed_from_u64(path_seed(seed, path));
                let mut state = start;
                let mut col = Vec::with_capacity(n_steps + 1);
                col.push(state as f64);
                for _ in 0..n_steps {
                    state = sample(&tables[state], &mut rng);
                    col.push(state as f64);
                }
                col
            })
            .collect();
        let mut out = Array2::<f64>::zeros((n_steps + 1, n_paths));
        for (j, col) in cols.iter().enumerate() {
            for (i, &v) in col.iter().enumerate() {
                out[(i, j)] = v;
            }
        }
        out
    });
    Ok(out.to_pyarray(py))
}

#[pyfunction]
fn markov_hitting_time<'py>(
    py: Python<'py>,
    p: PyReadonlyArray2<'py, f64>,
    start: usize,
    target: usize,
    n_sims: usize,
    seed: u64,
) -> PyResult<f64> {
    let p = p.as_array();
    let result = py.allow_threads(|| {
        let n = p.nrows();
        let tables: Vec<AliasTable> =
            (0..n).map(|i| build_alias(&p.row(i).to_vec())).collect();
        const CAP: usize = 1_000_000;
        let steps: Result<Vec<usize>, String> = (0..n_sims)
            .into_par_iter()
            .map(|sim| {
                let mut rng = SmallRng::seed_from_u64(path_seed(seed, sim));
                let mut state = start;
                let mut t = 0usize;
                while state != target {
                    state = sample(&tables[state], &mut rng);
                    t += 1;
                    if t >= CAP {
                        return Err(format!(
                            "hitting_time simulation exceeded {} steps \
                             without reaching the target",
                            CAP
                        ));
                    }
                }
                Ok(t)
            })
            .collect();
        steps.map(|s| s.iter().sum::<usize>() as f64 / s.len() as f64)
    });
    result.map_err(PyValueError::new_err)
}

/// Power iteration for the stationary distribution; None when it fails
/// to converge (e.g. periodic chains) so Python can fall back to a solve.
#[pyfunction]
fn markov_steady_state<'py>(
    py: Python<'py>,
    p: PyReadonlyArray2<'py, f64>,
    tol: f64,
    max_iter: usize,
) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
    let p = p.as_array();
    let pi = py.allow_threads(|| {
        let n = p.nrows();
        let mut x = vec![1.0 / n as f64; n];
        let mut next = vec![0.0; n];
        for _ in 0..max_iter {
            for v in next.iter_mut() {
                *v = 0.0;
            }
            for i in 0..n {
                let xi = x[i];
                if xi != 0.0 {
                    for j in 0..n {
                        next[j] += xi * p[(i, j)];
                    }
                }
            }
            let s: f64 = next.iter().sum();
            for v in next.iter_mut() {
                *v /= s;
            }
            let diff: f64 = x
                .iter()
                .zip(next.iter())
                .map(|(a, b)| (a - b).abs())
                .sum();
            std::mem::swap(&mut x, &mut next);
            if diff < tol {
                return Some(x);
            }
        }
        None
    });
    Ok(pi.map(|x| Array1::from(x).to_pyarray(py)))
}

/// Communication classes via Kosaraju SCC on the positive-probability
/// adjacency; classes sorted by smallest member.
#[pyfunction]
fn markov_sccs(p: PyReadonlyArray2<'_, f64>, tol: f64) -> Vec<Vec<usize>> {
    let p = p.as_array();
    let n = p.nrows();
    let adj = |i: usize| (0..n).filter(move |&j| p[(i, j)] > tol);
    let radj = |j: usize| (0..n).filter(move |&i| p[(i, j)] > tol);

    let mut visited = vec![false; n];
    let mut order: Vec<usize> = Vec::with_capacity(n);
    for s in 0..n {
        if visited[s] {
            continue;
        }
        let mut stack: Vec<(usize, Box<dyn Iterator<Item = usize>>)> =
            vec![(s, Box::new(adj(s)))];
        visited[s] = true;
        while let Some((node, it)) = stack.last_mut() {
            let node = *node;
            if let Some(next) = it.next() {
                if !visited[next] {
                    visited[next] = true;
                    stack.push((next, Box::new(adj(next))));
                }
            } else {
                order.push(node);
                stack.pop();
            }
        }
    }
    let mut comp = vec![usize::MAX; n];
    let mut classes: Vec<Vec<usize>> = Vec::new();
    for &s in order.iter().rev() {
        if comp[s] != usize::MAX {
            continue;
        }
        let id = classes.len();
        let mut members = vec![s];
        comp[s] = id;
        let mut stack = vec![s];
        while let Some(u) = stack.pop() {
            for v in radj(u) {
                if comp[v] == usize::MAX {
                    comp[v] = id;
                    members.push(v);
                    stack.push(v);
                }
            }
        }
        members.sort_unstable();
        classes.push(members);
    }
    classes.sort_by_key(|c| c[0]);
    classes
}

/// Period of an irreducible chain: gcd of (level[u] + 1 - level[v]) over
/// edges, with BFS levels from state 0.
#[pyfunction]
fn markov_period(p: PyReadonlyArray2<'_, f64>, tol: f64) -> PyResult<usize> {
    let p = p.as_array();
    let n = p.nrows();
    let mut level = vec![-1i64; n];
    let mut queue = std::collections::VecDeque::new();
    level[0] = 0;
    queue.push_back(0usize);
    while let Some(u) = queue.pop_front() {
        for j in 0..n {
            if p[(u, j)] > tol && level[j] < 0 {
                level[j] = level[u] + 1;
                queue.push_back(j);
            }
        }
    }
    fn gcd(a: i64, b: i64) -> i64 {
        if b == 0 {
            a.abs()
        } else {
            gcd(b, a % b)
        }
    }
    let mut g: i64 = 0;
    for u in 0..n {
        for v in 0..n {
            if p[(u, v)] > tol {
                if level[u] < 0 || level[v] < 0 {
                    return Err(PyValueError::new_err(
                        "period() requires an irreducible chain",
                    ));
                }
                g = gcd(g, level[u] + 1 - level[v]);
            }
        }
    }
    Ok(g.unsigned_abs() as usize)
}

pub(crate) fn matmul(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let n = a.nrows();
    let mut out = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for k in 0..n {
            let aik = a[(i, k)];
            if aik != 0.0 {
                for j in 0..n {
                    out[(i, j)] += aik * b[(k, j)];
                }
            }
        }
    }
    out
}

pub(crate) fn solve_matrix(a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>, String> {
    let n = a.nrows();
    let m = b.ncols();
    let mut w = Array2::<f64>::zeros((n, n + m));
    for i in 0..n {
        for j in 0..n {
            w[(i, j)] = a[(i, j)];
        }
        for j in 0..m {
            w[(i, n + j)] = b[(i, j)];
        }
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
        if best == 0.0 {
            return Err("singular system in matrix exponential".to_string());
        }
        if p != k {
            for j in 0..(n + m) {
                w.swap((k, j), (p, j));
            }
        }
        let piv = w[(k, k)];
        for i in 0..n {
            if i == k {
                continue;
            }
            let f = w[(i, k)] / piv;
            if f != 0.0 {
                for j in k..(n + m) {
                    w[(i, j)] -= f * w[(k, j)];
                }
            }
        }
    }
    let mut x = Array2::<f64>::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            x[(i, j)] = w[(i, n + j)] / w[(i, i)];
        }
    }
    Ok(x)
}

/// expm(Q * t) by [6/6] Padé with scaling and squaring.
#[pyfunction]
fn matexp<'py>(
    py: Python<'py>,
    q: PyReadonlyArray2<'py, f64>,
    t: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let q = q.as_array();
    let result = py.allow_threads(|| -> Result<Array2<f64>, String> {
        let n = q.nrows();
        let mut a = q.to_owned() * t;
        let norm = (0..n)
            .map(|i| (0..n).map(|j| a[(i, j)].abs()).sum::<f64>())
            .fold(0.0f64, f64::max);
        let s = if norm > 0.5 {
            (norm / 0.5).log2().ceil() as i32
        } else {
            0
        };
        a.mapv_inplace(|x| x / 2f64.powi(s));
        let degree = 6usize;
        let mut c = vec![1.0f64; degree + 1];
        for k in 1..=degree {
            c[k] = c[k - 1] * ((degree - k + 1) as f64)
                / ((k * (2 * degree - k + 1)) as f64);
        }
        let eye = Array2::<f64>::eye(n);
        let mut term = eye.clone();
        let mut numer = eye.clone() * c[0];
        let mut denom = eye.clone() * c[0];
        let mut sign = 1.0;
        for item in c.iter().take(degree + 1).skip(1) {
            term = matmul(&term, &a);
            sign = -sign;
            numer = numer + &term * *item;
            denom = denom + &term * (*item * sign);
        }
        let mut x = solve_matrix(&denom, &numer)?;
        for _ in 0..s {
            x = matmul(&x, &x);
        }
        Ok(x)
    });
    result
        .map(|x| x.to_pyarray(py))
        .map_err(PyValueError::new_err)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(markov_simulate, m)?)?;
    m.add_function(wrap_pyfunction!(markov_hitting_time, m)?)?;
    m.add_function(wrap_pyfunction!(markov_steady_state, m)?)?;
    m.add_function(wrap_pyfunction!(markov_sccs, m)?)?;
    m.add_function(wrap_pyfunction!(markov_period, m)?)?;
    m.add_function(wrap_pyfunction!(matexp, m)?)?;
    Ok(())
}
