"""Tests for LP/IP/MILP solvers (issue #91; §20.4).

Thirteen tests are justified per CLAUDE.md §6.5: each covers one distinct
documented behaviour of the issue #91 API surface. All solver engines are
Rust-primary (the issue mandates Rust implementations; the Python layer
is API/ergonomics only), so solver tests skip when the extension is not
built and one test pins the clear-error contract.

The reference problem is Wyndor (Hillier & Lieberman) in min form:
min -3x - 5y s.t. x <= 4, 2y <= 12, 3x + 2y <= 18, x,y >= 0
with optimum x = (2, 6), obj = -36, shadow prices (0, -1.5, -1) and
slack (2, 0, 0).

## Test: test_lp_optimal
- Goal: LP.solve() finds the optimum of the reference problem with
        nemopy result types and an "optimal" status.
- Source: issue #91 core LP API (LPResult fields).
- Expected: x ColVec (2, 6); obj -36; status "optimal"; iterations > 0.

## Test: test_lp_duals_and_slack
- Goal: LPResult.dual carries shadow prices (d obj / d b_i, min form)
        and slack the constraint slacks.
- Source: issue #91 LPResult fields ("dual variables (shadow prices)").
- Expected: dual (0, -1.5, -1); slack (2, 0, 0); both ColVec.

## Test: test_lp_equality
- Goal: equality constraints (A_eq/b_eq) are honoured via the two-phase
        method.
- Source: issue #91 problem variants table.
- Expected: min x + 2y s.t. x + y = 1 gives x = (1, 0), obj = 1.

## Test: test_lp_infeasible
- Goal: infeasible problems report status "infeasible" with x None.
- Source: issue #91 LPResult.status values.
- Expected: x <= -1 with x >= 0 is infeasible.

## Test: test_lp_unbounded
- Goal: unbounded problems report status "unbounded".
- Source: issue #91 LPResult.status values.
- Expected: min -x with no constraints is unbounded.

## Test: test_lp_negative_rhs
- Goal: rows with negative right-hand sides (>= constraints in <= form)
        are standardized with artificials and solved correctly.
- Source: issue #91 two-phase method row.
- Expected: min x s.t. -x <= -3 gives x = 3.

## Test: test_lp_bounds
- Goal: variable bounds, including free variables, are transformed into
        kernel standard form and mapped back.
- Source: issue #91 problem variants (bounds parameter).
- Expected: max x with x in [0, 5] gives 5; free variable reaches its
            negative optimum.

## Test: test_lp_methods
- Goal: bigm and two_phase agree with the default simplex; dual simplex
        solves a dual-feasible problem; interior is explicitly
        unimplemented (provisional surface).
- Source: issue #91 solver methods table.
- Expected: same optima; ValueError mentioning "interior".

## Test: test_lp_sensitivity
- Goal: sensitivity() reports objective ranges, RHS ranges and reduced
        costs from the optimal basis.
- Source: issue #91 sensitivity analysis section.
- Expected: Wyndor RHS range for constraint 1 is (6, 18); ranges bracket
            the current data; reduced costs ~0 for basic variables.

## Test: test_ip_branch_and_bound
- Goal: IP.solve() finds the integer optimum by branch and bound.
- Source: issue #91 integer programming table.
- Expected: min -5x1 - 4x2 s.t. 6x1 + 4x2 <= 24, x1 + 2x2 <= 6 gives
            the integer optimum (4, 0) with obj -20 (LP relaxation is
            fractional at (3, 1.5)).

## Test: test_milp_mixed
- Goal: MILP restricts only the declared integer variables.
- Source: issue #91 problem variants (integer_vars).
- Expected: same data with only x1 integer gives (3, 1.5), obj -21.

## Test: test_blp_binary
- Goal: BLP restricts declared variables to {0, 1}.
- Source: issue #91 problem variants (binary_vars).
- Expected: max x1 + x2 s.t. x1 + x2 <= 1 gives obj -1 in min form with
            binary x.

## Test: test_tableau_api
- Goal: the pedagogical tableau API exposes the initial tableau, manual
        pivots, optimality checks, and step-by-step iteration
        (Python-side by design, like ref_steps).
- Source: issue #91 tableau API section.
- Expected: tableau() returns a Tableau whose .mat is a Mat; pivot
            mutates it; solve_steps() ends at an optimal tableau.

## Test: test_solvers_require_extension
- Goal: solver engines raise the clear build-pointer error without the
        extension (no Python port for non-NumPy functionality).
- Source: amended §20.1; issue #91 ("All solver implementations must be
          in Rust").
- Expected: RuntimeError mentioning _rust_core.
"""

import numpy as np
import pytest

from nemopy import ColVec, Mat, _c, mat
from nemopy import _core
from nemopy.optim import BLP, IP, LP, MILP

requires_rust = pytest.mark.skipif(
    _core._RUST is None, reason="_rust_core extension not built"
)


def _wyndor():
    return LP(
        c=_c[-3, -5],
        A=mat([1, 0, 3], [0, 2, 2]),
        b=_c[4, 12, 18],
    )


@requires_rust
def test_lp_optimal():
    res = _wyndor().solve()
    assert res.status == "optimal"
    assert isinstance(res.x, ColVec)
    np.testing.assert_allclose(res.x.to_flat(), [2.0, 6.0], atol=1e-9)
    assert abs(res.obj - (-36.0)) < 1e-9
    assert res.iterations > 0


@requires_rust
def test_lp_duals_and_slack():
    res = _wyndor().solve()
    assert isinstance(res.dual, ColVec) and isinstance(res.slack, ColVec)
    np.testing.assert_allclose(res.dual.to_flat(), [0.0, -1.5, -1.0], atol=1e-9)
    np.testing.assert_allclose(res.slack.to_flat(), [2.0, 0.0, 0.0], atol=1e-9)


@requires_rust
def test_lp_equality():
    p = LP(c=_c[1, 2], A_eq=mat([1], [1]), b_eq=_c[1])
    res = p.solve()
    assert res.status == "optimal"
    np.testing.assert_allclose(res.x.to_flat(), [1.0, 0.0], atol=1e-9)
    assert abs(res.obj - 1.0) < 1e-9


@requires_rust
def test_lp_infeasible():
    p = LP(c=_c[1], A=mat([1]), b=_c[-1])
    res = p.solve()
    assert res.status == "infeasible"
    assert res.x is None and res.obj is None


@requires_rust
def test_lp_unbounded():
    p = LP(c=_c[-1])
    res = p.solve()
    assert res.status == "unbounded"


@requires_rust
def test_lp_negative_rhs():
    p = LP(c=_c[1], A=mat([-1]), b=_c[-3])
    res = p.solve()
    assert res.status == "optimal"
    np.testing.assert_allclose(res.x.to_flat(), [3.0], atol=1e-9)


@requires_rust
def test_lp_bounds():
    p = LP(c=_c[-1], bounds=[(0, 5)])
    res = p.solve()
    np.testing.assert_allclose(res.x.to_flat(), [5.0], atol=1e-9)

    q = LP(c=_c[1], A=mat([-1]), b=_c[4], bounds=[(None, None)])
    res = q.solve()
    assert res.status == "optimal"
    np.testing.assert_allclose(res.x.to_flat(), [-4.0], atol=1e-9)
    assert abs(res.obj - (-4.0)) < 1e-9


@requires_rust
def test_lp_methods():
    base = _wyndor().solve()
    for method in ("bigm", "two_phase"):
        res = _wyndor().solve(method=method)
        assert res.status == "optimal"
        np.testing.assert_allclose(res.x.to_flat(), base.x.to_flat(), atol=1e-8)
    dual_ok = LP(c=_c[1, 2], A=mat([-1], [-1]), b=_c[-3])
    res = dual_ok.solve(method="dual")
    assert res.status == "optimal"
    np.testing.assert_allclose(res.x.to_flat(), [3.0, 0.0], atol=1e-9)
    with pytest.raises(ValueError, match="interior"):
        _wyndor().solve(method="interior")


@requires_rust
def test_lp_sensitivity():
    res = _wyndor().solve()
    rep = res.sensitivity()
    lo, hi = rep.rhs_range[1]
    assert abs(lo - 6.0) < 1e-8 and abs(hi - 18.0) < 1e-8
    for j, (clo, chi) in enumerate(rep.obj_range):
        assert clo - 1e-12 <= [-3.0, -5.0][j] <= chi + 1e-12
    for i, (blo, bhi) in enumerate(rep.rhs_range):
        assert blo - 1e-12 <= [4.0, 12.0, 18.0][i] <= bhi + 1e-12
    np.testing.assert_allclose(rep.reduced_cost.to_flat(), [0.0, 0.0], atol=1e-9)


@requires_rust
def test_ip_branch_and_bound():
    p = IP(c=_c[-5, -4], A=mat([6, 1], [4, 2]), b=_c[24, 6])
    relax = LP(c=_c[-5, -4], A=mat([6, 1], [4, 2]), b=_c[24, 6]).solve()
    assert np.abs(relax.x.to_flat() - np.round(relax.x.to_flat())).max() > 0.4
    res = p.solve()
    assert res.status == "optimal"
    np.testing.assert_allclose(res.x.to_flat(), [4.0, 0.0], atol=1e-7)
    assert abs(res.obj - (-20.0)) < 1e-7


@requires_rust
def test_milp_mixed():
    p = MILP(c=_c[-5, -4], A=mat([6, 1], [4, 2]), b=_c[24, 6], integer_vars=[0])
    res = p.solve()
    assert res.status == "optimal"
    np.testing.assert_allclose(res.x.to_flat(), [3.0, 1.5], atol=1e-7)
    assert abs(res.obj - (-21.0)) < 1e-7


@requires_rust
def test_blp_binary():
    p = BLP(c=_c[-1, -1], A=mat([1], [1]), b=_c[1], binary_vars=[0, 1])
    res = p.solve()
    assert res.status == "optimal"
    assert abs(res.obj - (-1.0)) < 1e-9
    x = res.x.to_flat()
    assert set(np.round(x).tolist()) <= {0.0, 1.0}


@requires_rust
def test_tableau_api():
    p = _wyndor()
    t = p.tableau()
    assert isinstance(t.mat, Mat)
    assert not t.is_optimal()
    before = np.asarray(t.mat).copy()
    t.pivot(row=1, col=1)
    assert not np.allclose(np.asarray(t.mat), before)

    steps = list(p.solve_steps())
    assert steps
    for step in steps:
        assert isinstance(step.tableau, Mat)
    assert steps[-1].entering is None  # final step is optimal


def test_solvers_require_extension(monkeypatch):
    monkeypatch.setattr(_core, "_RUST", None)
    with pytest.raises(RuntimeError, match="_rust_core"):
        _wyndor().solve()
    with pytest.raises(RuntimeError, match="_rust_core"):
        IP(c=_c[-1], A=mat([1]), b=_c[3]).solve()
