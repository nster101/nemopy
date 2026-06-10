"""Markov chain constructors and Mat methods (issue #86; §20.4).

Transition matrices are plain ``Mat`` objects — no new class. Chain
methods are gated by stochasticity validation; CTMC methods validate the
rate-matrix property. Engine kernels (alias-method simulation, Kosaraju
SCC, BFS period, Padé matrix exponential) are Rust-primary in
``_rust_core.markov`` with no Python port, per the amended §20.1 — they
raise a clear error when the extension is absent. Analysis quantities
that are plain linear algebra (steady state, fundamental matrix,
absorption probabilities) evaluate through NumPy directly.
"""

import numpy as np

from nemopy import _core
from nemopy._core import ColVec, Mat, ShapeError

_TOL = 1e-9


def _is_stochastic_array(a):
    return (
        a.shape[0] == a.shape[1]
        and bool(np.all(a >= -_TOL))
        and bool(np.allclose(a.sum(axis=1), 1.0, atol=1e-8))
    )


def _check_chain(self, name):
    a = np.asarray(self)
    if a.shape[0] != a.shape[1]:
        raise ShapeError(
            f"{name}() requires a square transition matrix, got "
            f"{self.shape}."
        )
    if not _is_stochastic_array(a):
        raise ValueError(
            f"{name}() requires a row-stochastic matrix (rows sum to 1, "
            f"entries nonnegative)."
        )
    return a


def _check_rate(self, name):
    a = np.asarray(self)
    if a.shape[0] != a.shape[1]:
        raise ShapeError(
            f"{name}() requires a square rate matrix, got {self.shape}."
        )
    off = a - np.diag(np.diag(a))
    if not (
        np.allclose(a.sum(axis=1), 0.0, atol=1e-8)
        and np.all(off >= -_TOL)
        and np.all(np.diag(a) <= _TOL)
    ):
        raise ValueError(
            f"{name}() requires a CTMC rate matrix (rows sum to 0, "
            f"off-diagonal entries nonnegative)."
        )
    return a


def markov(p):
    """Construct a validated row-stochastic transition matrix.

    Parameters
    ----------
    p : array_like
        Square matrix whose rows sum to 1 with nonnegative entries.

    Returns
    -------
    Mat

    Raises
    ------
    ShapeError
        If the input is not square.
    ValueError
        If the input is not row-stochastic.
    """
    a = np.asarray(p, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ShapeError(
            f"markov() requires a square matrix, got shape {a.shape}."
        )
    if not _is_stochastic_array(a):
        raise ValueError(
            "markov() requires a row-stochastic matrix (rows sum to 1, "
            "entries nonnegative)."
        )
    return Mat(a)


def ctmc(q):
    """Construct a validated CTMC rate matrix.

    Parameters
    ----------
    q : array_like
        Square matrix with rows summing to 0, nonnegative off-diagonal
        entries and nonpositive diagonal.

    Returns
    -------
    Mat

    Raises
    ------
    ShapeError
        If the input is not square.
    ValueError
        If the input is not a valid rate matrix.
    """
    a = np.asarray(q, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ShapeError(
            f"ctmc() requires a square matrix, got shape {a.shape}."
        )
    m = Mat(a)
    _check_rate(m, "ctmc")
    return m


def _mat_is_stochastic(self):
    """Whether the matrix is row-stochastic (rows sum to 1, entries >= 0)."""
    a = np.asarray(self)
    return a.shape[0] == a.shape[1] and _is_stochastic_array(a)


def _mat_is_doubly_stochastic(self):
    """Whether both rows and columns sum to 1 with nonnegative entries."""
    a = np.asarray(self)
    return bool(
        _mat_is_stochastic(self)
        and np.allclose(a.sum(axis=0), 1.0, atol=1e-8)
    )


def _mat_steady_state(self):
    """Stationary distribution pi with ``pi.T @ P = pi.T``, as a ColVec.

    Rust power iteration when available, falling back to the direct
    linear solve of ``(P.T - I) pi = 0`` with the normalization row
    (also used when power iteration does not converge, e.g. for
    periodic chains).
    """
    a = _check_chain(self, "steady_state")
    rust = _core._RUST
    if rust is not None:
        pi = rust.markov_steady_state(a, 1e-12, 100_000)
        if pi is not None:
            return ColVec(np.asarray(pi).reshape(-1, 1))
    n = a.shape[0]
    m = a.T - np.eye(n)
    m[-1, :] = 1.0
    rhs = np.zeros(n)
    rhs[-1] = 1.0
    pi = np.linalg.solve(m, rhs)
    return ColVec(pi.reshape(-1, 1))


def _mat_n_step(self, n):
    """n-step transition matrix ``P**n``."""
    a = _check_chain(self, "n_step")
    return Mat(np.linalg.matrix_power(a, int(n)))


def _mat_absorbing_states(self):
    """Indices of absorbing states (``P[i, i] == 1``)."""
    a = _check_chain(self, "absorbing_states")
    return [int(i) for i in np.flatnonzero(np.isclose(np.diag(a), 1.0))]


def _reachability(adj):
    n = adj.shape[0]
    r = adj | np.eye(n, dtype=bool)
    k = 1
    while k < n:
        r = (r.astype(np.uint8) @ r.astype(np.uint8)) > 0
        k *= 2
    return r


def _mat_is_absorbing(self):
    """Whether absorbing states exist and are reachable from every
    transient state."""
    a = _check_chain(self, "is_absorbing")
    absorbing = _mat_absorbing_states(self)
    if not absorbing:
        return False
    reach = _reachability(np.asarray(self) > _TOL)
    transient = [i for i in range(a.shape[0]) if i not in absorbing]
    return all(any(reach[i, j] for j in absorbing) for i in transient)


def _mat_communicate(self):
    """Communication classes as a list of sets of state indices."""
    a = _check_chain(self, "communicate")
    rust = _core._require_rust("Mat.communicate()")
    return [set(c) for c in rust.markov_sccs(a, _TOL)]


def _mat_is_irreducible(self):
    """Whether all states communicate (a single communication class)."""
    return len(_mat_communicate(self)) == 1


def _mat_period(self):
    """Period of an irreducible chain (1 means aperiodic).

    Raises
    ------
    ValueError
        If the chain is not irreducible.
    """
    a = _check_chain(self, "period")
    if not _mat_is_irreducible(self):
        raise ValueError("period() requires an irreducible chain")
    rust = _core._require_rust("Mat.period()")
    return int(rust.markov_period(a, _TOL))


def _mat_is_ergodic(self):
    """Whether the chain is irreducible and aperiodic."""
    _check_chain(self, "is_ergodic")
    return _mat_is_irreducible(self) and _mat_period(self) == 1


def _transient_split(self, name):
    a = _check_chain(self, name)
    absorbing = _mat_absorbing_states(self)
    if not absorbing:
        raise ValueError(f"{name}() requires an absorbing chain")
    transient = [i for i in range(a.shape[0]) if i not in absorbing]
    q = a[np.ix_(transient, transient)]
    r = a[np.ix_(transient, absorbing)]
    return q, r, transient, absorbing


def _mat_fundamental_matrix(self):
    """Fundamental matrix ``N = (I - Q)^-1`` over the transient states
    (in their original order)."""
    q, _, transient, _ = _transient_split(self, "fundamental_matrix")
    n = np.linalg.inv(np.eye(len(transient)) - q)
    return Mat(n)


def _mat_absorption_probs(self):
    """Absorption probabilities ``B = N @ R`` (transient x absorbing)."""
    q, r, transient, _ = _transient_split(self, "absorption_probs")
    n = np.linalg.inv(np.eye(len(transient)) - q)
    return Mat(n @ r)


def _mat_expected_steps(self):
    """Expected steps to absorption from each transient state."""
    q, _, transient, _ = _transient_split(self, "expected_steps")
    n = np.linalg.inv(np.eye(len(transient)) - q)
    return ColVec((n @ np.ones(len(transient))).reshape(-1, 1))


def _mat_simulate(self, start, n_steps, n_paths=1, seed=None):
    """Simulate random walks; returns a ``(n_steps+1, n_paths)`` Mat of
    state indices with row 0 equal to ``start``.

    Parameters
    ----------
    start : int
        Initial state index.
    n_steps : int
        Number of transitions per path.
    n_paths : int, optional
        Number of independent paths (default 1).
    seed : int, optional
        RNG seed for reproducibility.

    Notes
    -----
    Rust-primary engine (amended §20.1): requires the ``_rust_core``
    extension.
    """
    a = _check_chain(self, "simulate")
    if not 0 <= int(start) < a.shape[0]:
        raise ValueError(f"start state {start} out of range")
    seed = 0 if seed is None else int(seed)
    rust = _core._require_rust("Mat.simulate()")
    walk = rust.markov_simulate(
        a, int(start), int(n_steps), int(n_paths), seed
    )
    return Mat(np.asarray(walk))


def _mat_hitting_time(self, start, target, n_sims=10000, seed=None):
    """Monte Carlo estimate of the expected first-passage time from
    ``start`` to ``target``."""
    a = _check_chain(self, "hitting_time")
    n = a.shape[0]
    if not (0 <= int(start) < n and 0 <= int(target) < n):
        raise ValueError("start/target state out of range")
    seed = 0 if seed is None else int(seed)
    rust = _core._require_rust("Mat.hitting_time()")
    return float(
        rust.markov_hitting_time(
            a, int(start), int(target), int(n_sims), seed
        )
    )


def _mat_embedded_dtmc(self):
    """Embedded jump chain of a CTMC rate matrix."""
    a = _check_rate(self, "embedded_dtmc")
    n = a.shape[0]
    p = np.zeros((n, n))
    for i in range(n):
        rate = -a[i, i]
        if rate <= _TOL:
            p[i, i] = 1.0
        else:
            p[i, :] = a[i, :] / rate
            p[i, i] = 0.0
    return Mat(p)


def _mat_transient_probs(self, t):
    """CTMC transition probabilities ``P(t) = expm(Q t)`` via Padé
    scaling-and-squaring."""
    a = _check_rate(self, "transient_probs")
    rust = _core._require_rust("Mat.transient_probs()")
    return Mat(np.asarray(rust.matexp(a, float(t))))


Mat.is_stochastic = _mat_is_stochastic
Mat.is_doubly_stochastic = _mat_is_doubly_stochastic
Mat.steady_state = _mat_steady_state
Mat.n_step = _mat_n_step
Mat.absorbing_states = _mat_absorbing_states
Mat.is_absorbing = _mat_is_absorbing
Mat.communicate = _mat_communicate
Mat.is_irreducible = _mat_is_irreducible
Mat.period = _mat_period
Mat.is_ergodic = _mat_is_ergodic
Mat.fundamental_matrix = _mat_fundamental_matrix
Mat.absorption_probs = _mat_absorption_probs
Mat.expected_steps = _mat_expected_steps
Mat.simulate = _mat_simulate
Mat.hitting_time = _mat_hitting_time
Mat.embedded_dtmc = _mat_embedded_dtmc
Mat.transient_probs = _mat_transient_probs
