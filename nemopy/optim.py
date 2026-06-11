"""Linear and integer programming (issue #91; §20.4).

Problem classes formulate ``min c.T @ x`` with ``A_ub @ x <= b_ub``,
``A_eq @ x = b_eq`` and per-variable bounds (default ``x >= 0``). All
solver engines run in ``_rust_core.optim`` (two-phase simplex with
Bland's rule, Big-M, dual simplex, branch-and-bound) — Rust-primary per
the issue, with no Python port; the Python layer standardizes bounds and
wraps results in nemopy types. ``dual`` carries shadow prices
``d obj / d b_i`` in min form. The pedagogical tableau API runs in
Python by design (like ``Mat.ref_steps``).
"""

from collections import namedtuple

import numpy as np

from nemopy import _core
from nemopy._core import ColVec, Mat, ShapeError

_STATUS = {0: "optimal", 1: "infeasible", 2: "unbounded", 3: "iteration_limit"}

SensitivityReport = namedtuple(
    "SensitivityReport", ["obj_range", "rhs_range", "reduced_cost"]
)

Step = namedtuple("Step", ["tableau", "entering", "leaving"])


class LPResult:
    """Solver result: ``x``, ``obj``, ``status``, ``dual``, ``slack``,
    ``iterations``; ``sensitivity()`` for the post-optimal report."""

    def __init__(self, status, x, obj, dual, slack, iterations, sens=None):
        self.status = status
        self.x = x
        self.obj = obj
        self.dual = dual
        self.slack = slack
        self.iterations = iterations
        self._sens = sens

    def sensitivity(self):
        """Post-optimal sensitivity report from the final basis.

        Returns
        -------
        SensitivityReport
            ``obj_range`` (list of (lo, hi) per variable), ``rhs_range``
            (list of (lo, hi) per constraint), ``reduced_cost`` (ColVec).

        Raises
        ------
        ValueError
            If the solve was not optimal or used custom bounds (the
            ranges are reported in kernel space and would not map back).
        """
        if self._sens is None:
            raise ValueError(
                "sensitivity() requires an optimal solve with default bounds"
            )
        return self._sens

    def __repr__(self):
        return (
            f"LPResult(status={self.status!r}, obj={self.obj}, "
            f"iterations={self.iterations})"
        )


def _as_col(v, name):
    if v is None:
        return np.zeros(0)
    a = np.asarray(v, dtype=float)
    if a.ndim == 2 and a.shape[1] == 1:
        return a.ravel()
    if a.ndim == 1:
        return a
    raise ShapeError(f"{name} must be a column vector, got shape {a.shape}")


def _as_mat(m, ncols, name):
    if m is None:
        return np.zeros((0, ncols))
    a = np.asarray(m, dtype=float)
    if a.ndim != 2 or a.shape[1] != ncols:
        raise ShapeError(
            f"{name} must have {ncols} columns, got shape {a.shape}"
        )
    return a


class LP:
    """Linear program ``min c.T @ x`` subject to ``A_ub @ x <= b_ub``,
    ``A_eq @ x = b_eq`` and variable bounds (default ``x >= 0``).

    Parameters
    ----------
    c : ColVec or array_like
        Objective coefficients (minimization).
    A, b : Mat, ColVec, optional
        Positional aliases for ``A_ub`` / ``b_ub`` (issue #91 sketch).
    A_ub, b_ub : Mat, ColVec, optional
        Inequality constraints ``A_ub @ x <= b_ub``.
    A_eq, b_eq : Mat, ColVec, optional
        Equality constraints.
    bounds : list of (lo, hi), optional
        Per-variable bounds; ``None`` entries mean unbounded on that
        side. Default ``(0, None)`` for every variable.
    """

    def __init__(self, c, A=None, b=None, *, A_ub=None, b_ub=None,
                 A_eq=None, b_eq=None, bounds=None):
        self.c = _as_col(c, "c")
        n = self.c.size
        if A is not None and A_ub is not None:
            raise ValueError("pass either A or A_ub, not both")
        self.A_ub = _as_mat(A_ub if A_ub is not None else A, n, "A_ub")
        self.b_ub = _as_col(b_ub if b_ub is not None else b, "b_ub")
        if self.A_ub.shape[0] != self.b_ub.size:
            raise ShapeError(
                f"A_ub has {self.A_ub.shape[0]} rows but b_ub has "
                f"{self.b_ub.size} entries"
            )
        self.A_eq = _as_mat(A_eq, n, "A_eq")
        self.b_eq = _as_col(b_eq, "b_eq")
        if self.A_eq.shape[0] != self.b_eq.size:
            raise ShapeError(
                f"A_eq has {self.A_eq.shape[0]} rows but b_eq has "
                f"{self.b_eq.size} entries"
            )
        if bounds is not None and len(bounds) != n:
            raise ShapeError(f"bounds must have {n} entries")
        self.bounds = bounds
        self.n = n

    def _default_bounds(self):
        return self.bounds is None or all(
            lo == 0 and hi is None for lo, hi in self.bounds
        )

    def _kernel_form(self):
        """Standardize bounds to kernel form (x >= 0).

        Returns (c_k, A_ub_k, b_ub_k, A_eq_k, b_eq_k, transform, offset)
        where x = transform @ x_k + offset.
        """
        if self._default_bounds():
            t = np.eye(self.n)
            return (self.c, self.A_ub, self.b_ub, self.A_eq, self.b_eq,
                    t, np.zeros(self.n))
        cols = []
        offset = np.zeros(self.n)
        upper_rows = []
        for j, (lo, hi) in enumerate(self.bounds):
            if lo is None and hi is None:
                pos = np.zeros(self.n)
                pos[j] = 1.0
                neg = np.zeros(self.n)
                neg[j] = -1.0
                cols.extend([pos, neg])
            elif lo is None:
                # only upper bound: substitute x = hi - x', x' >= 0
                col = np.zeros(self.n)
                col[j] = -1.0
                cols.append(col)
                offset[j] = hi
            else:
                col = np.zeros(self.n)
                col[j] = 1.0
                cols.append(col)
                offset[j] = lo
                if hi is not None:
                    upper_rows.append((len(cols) - 1, hi - lo))
        t = np.column_stack(cols) if cols else np.zeros((self.n, 0))
        nk = t.shape[1]
        a_ub_k = self.A_ub @ t
        b_ub_k = self.b_ub - self.A_ub @ offset
        a_eq_k = self.A_eq @ t
        b_eq_k = self.b_eq - self.A_eq @ offset
        for kcol, ub in upper_rows:
            row = np.zeros(nk)
            row[kcol] = 1.0
            a_ub_k = np.vstack([a_ub_k, row])
            b_ub_k = np.append(b_ub_k, ub)
        c_k = t.T @ self.c
        return c_k, a_ub_k, b_ub_k, a_eq_k, b_eq_k, t, offset

    def solve(self, method="simplex", max_iter=10000):
        """Solve and return an :class:`LPResult`.

        Parameters
        ----------
        method : str, optional
            ``"simplex"`` (two-phase, default), ``"two_phase"``,
            ``"bigm"``, or ``"dual"``. ``"interior"`` is part of the
            provisional issue #91 surface and not implemented yet.
        max_iter : int, optional
            Pivot limit.
        """
        if method == "interior":
            raise ValueError(
                'method "interior" is not implemented yet (issue #91 '
                "provisional surface)"
            )
        if method not in ("simplex", "two_phase", "bigm", "dual"):
            raise ValueError(f"unknown method {method!r}")
        rust = _core._require_rust("LP.solve()")
        c_k, a_ub_k, b_ub_k, a_eq_k, b_eq_k, t, offset = self._kernel_form()
        status, xk, obj, dual, iters, obj_rng, rhs_rng, red = rust.lp_solve(
            np.ascontiguousarray(c_k),
            np.ascontiguousarray(a_ub_k),
            np.ascontiguousarray(b_ub_k),
            np.ascontiguousarray(a_eq_k),
            np.ascontiguousarray(b_eq_k),
            method,
            int(max_iter),
        )
        status = _STATUS[status]
        if status != "optimal":
            return LPResult(status, None, None, None, None, int(iters))
        x = t @ np.asarray(xk) + offset
        obj = float(self.c @ x)
        m_ub = self.b_ub.size
        slack = self.b_ub - self.A_ub @ x
        dual = np.asarray(dual)[: m_ub + self.b_eq.size]
        sens = None
        if self._default_bounds():
            obj_rng = np.asarray(obj_rng)
            rhs_rng = np.asarray(rhs_rng)
            sens = SensitivityReport(
                obj_range=[tuple(r) for r in obj_rng],
                rhs_range=[tuple(r) for r in rhs_rng],
                reduced_cost=ColVec(np.asarray(red).reshape(-1, 1)),
            )
        return LPResult(
            status,
            ColVec(x.reshape(-1, 1)),
            obj,
            ColVec(dual.reshape(-1, 1)),
            ColVec(slack.reshape(-1, 1)),
            int(iters),
            sens,
        )

    def tableau(self):
        """Initial simplex tableau for the inequality system (pedagogical).

        Layout: rows are ``[A_ub | I | b]`` followed by the objective row
        ``[c | 0 | 0]``.
        """
        m = self.b_ub.size
        top = np.hstack([self.A_ub, np.eye(m), self.b_ub.reshape(-1, 1)])
        obj = np.hstack([self.c, np.zeros(m + 1)])
        return Tableau(Mat(np.vstack([top, obj])), self.n)

    def solve_steps(self):
        """Yield ``Step(tableau, entering, leaving)`` per simplex pivot.

        Pedagogical mode, Python-side by design (issue #91 tableau API);
        covers the inequality-only, nonnegative-rhs textbook case. The
        final step has ``entering=None``.
        """
        t = self.tableau()
        while True:
            entering = t._entering()
            if entering is None:
                yield Step(t.mat, None, None)
                return
            leaving = t._leaving(entering)
            if leaving is None:
                yield Step(t.mat, entering, None)
                return
            yield Step(t.mat, entering, leaving)
            t.pivot(row=leaving, col=entering)


class Tableau:
    """Simplex tableau with manual pivoting (educational/debugging)."""

    def __init__(self, mat_, n_vars):
        self.mat = mat_
        self.n_vars = n_vars

    def _entering(self):
        obj = np.asarray(self.mat)[-1, :-1]
        j = int(np.argmin(obj))
        return j if obj[j] < -1e-9 else None

    def _leaving(self, col):
        a = np.asarray(self.mat)
        body, rhs = a[:-1, col], a[:-1, -1]
        ratios = np.where(body > 1e-9, rhs / np.where(body > 1e-9, body, 1.0),
                          np.inf)
        r = int(np.argmin(ratios))
        return r if np.isfinite(ratios[r]) else None

    def pivot(self, row, col):
        """Pivot in place on (row, col)."""
        a = np.asarray(self.mat).copy()
        a[row] = a[row] / a[row, col]
        for i in range(a.shape[0]):
            if i != row and a[i, col] != 0.0:
                a[i] = a[i] - a[i, col] * a[row]
        self.mat = Mat(a)

    def is_optimal(self):
        """No improving column remains in the objective row."""
        return self._entering() is None


class MILP(LP):
    """Mixed-integer LP: ``integer_vars`` lists the integer indices."""

    def __init__(self, c, A=None, b=None, *, integer_vars, **kwargs):
        super().__init__(c, A, b, **kwargs)
        self.integer_vars = sorted(int(j) for j in integer_vars)
        if any(j < 0 or j >= self.n for j in self.integer_vars):
            raise ValueError("integer_vars indices out of range")

    def solve(self, method="branch_bound", gap=0.0, max_nodes=100000,
              max_iter=10000):
        """Solve by branch and bound (depth-first, most-fractional
        branching); ``gap`` is the relative optimality gap for pruning.
        """
        if method != "branch_bound":
            raise ValueError(
                f'method must be "branch_bound" (got {method!r}); other '
                f"issue #91 variants are not implemented yet"
            )
        if not self._default_bounds():
            raise ValueError(
                "integer programs currently require default bounds; encode "
                "bounds as constraint rows"
            )
        rust = _core._require_rust("MILP.solve()")
        mask = [j in self.integer_vars for j in range(self.n)]
        status, x, obj, nodes = rust.milp_solve(
            np.ascontiguousarray(self.c),
            np.ascontiguousarray(self.A_ub),
            np.ascontiguousarray(self.b_ub),
            np.ascontiguousarray(self.A_eq),
            np.ascontiguousarray(self.b_eq),
            mask,
            float(gap),
            int(max_nodes),
            int(max_iter),
        )
        status = _STATUS[status]
        if status != "optimal":
            return LPResult(status, None, None, None, None, int(nodes))
        x = np.asarray(x)
        slack = self.b_ub - self.A_ub @ x
        return LPResult(
            status,
            ColVec(x.reshape(-1, 1)),
            float(obj),
            None,
            ColVec(slack.reshape(-1, 1)),
            int(nodes),
        )


class IP(MILP):
    """Integer program: every variable is integer."""

    def __init__(self, c, A=None, b=None, **kwargs):
        c_arr = _as_col(c, "c")
        super().__init__(
            c, A, b, integer_vars=range(c_arr.size), **kwargs
        )


class BLP(MILP):
    """Binary LP: ``binary_vars`` are restricted to {0, 1}."""

    def __init__(self, c, A=None, b=None, *, binary_vars, **kwargs):
        super().__init__(c, A, b, integer_vars=binary_vars, **kwargs)
        # x_j <= 1 rows for the binary variables (x >= 0 is inherent)
        rows = np.zeros((len(self.integer_vars), self.n))
        for k, j in enumerate(self.integer_vars):
            rows[k, j] = 1.0
        self.A_ub = np.vstack([self.A_ub, rows])
        self.b_ub = np.append(self.b_ub, np.ones(len(self.integer_vars)))
