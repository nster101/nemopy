"""Tests for Markov chain support (issue #86; §20.4).

Fourteen tests are justified per CLAUDE.md §6.5: each covers one distinct
documented behaviour of the issue #86 API surface.

## Test: test_markov_constructor
- Goal: markov() builds a validated row-stochastic Mat and rejects
        non-square, negative, or non-unit-row-sum input.
- Source: issue #86 — "validates rows sum to 1".
- Expected: Mat returned; ShapeError for non-square; ValueError otherwise.

## Test: test_ctmc_constructor
- Goal: ctmc() builds a validated rate Mat (rows sum to 0, off-diagonal
        >= 0) and rejects invalid input.
- Source: issue #86 CTMC extension table.
- Expected: Mat returned; ValueError for bad rows or negative off-diag.

## Test: test_is_stochastic_flags
- Goal: is_stochastic()/is_doubly_stochastic() classify matrices.
- Source: issue #86 validation table.
- Expected: True/False per construction.

## Test: test_steady_state
- Goal: steady_state() returns the stationary distribution pi with
        pi.T @ P = pi.T, as a ColVec summing to 1.
- Source: issue #86 steady-state table.
- Expected: known 2-state chain gives pi = (5/6, 1/6).

## Test: test_n_step
- Goal: n_step(n) returns the n-step transition matrix P^n.
- Source: issue #86 ("P.n_step(n) -> P^n").
- Expected: n_step(0) = I, n_step(2) = P @ P.

## Test: test_absorbing_analysis
- Goal: absorbing_states/is_absorbing/fundamental_matrix/
        absorption_probs/expected_steps reproduce the textbook
        gambler's-ruin quantities.
- Source: issue #86 analysis table ("N = (I - Q)^-1", "B = N @ R").
- Expected: states [0, 2]; N = [[4/3]] for p=1/2 ruin? — use the
            standard 4-state ruin chain and check N, B, t values.

## Test: test_communicate_and_irreducible
- Goal: communicate() returns communication classes; is_irreducible()
        reflects single-class chains.
- Source: issue #86 analysis table.
- Expected: reducible example gives two classes; cycle chain is
            irreducible.

## Test: test_period_and_ergodic
- Goal: period() returns the chain period; is_ergodic() is True only
        for irreducible + aperiodic chains.
- Source: issue #86 analysis table ("1 = aperiodic").
- Expected: 2-cycle has period 2 (not ergodic); lazy chain period 1
            (ergodic).

## Test: test_simulate
- Goal: simulate() returns a (n_steps+1, n_paths) Mat of valid state
        indices starting at `start`, deterministic for a fixed seed
        (Rust-primary engine; skipped when the extension is absent).
- Source: issue #86 simulation table; amended §20.1.
- Expected: shape/start/state-range hold; same seed reproduces.

## Test: test_hitting_time
- Goal: hitting_time() estimates the expected first-passage time by
        Monte Carlo.
- Source: issue #86 simulation table.
- Expected: a chain that always reaches the target in one step gives
            exactly 1.0.

## Test: test_embedded_dtmc
- Goal: Q.embedded_dtmc() returns the jump chain of a CTMC.
- Source: issue #86 CTMC table.
- Expected: known 2-state rate matrix gives the known jump chain.

## Test: test_transient_probs
- Goal: Q.transient_probs(t) computes P(t) = expm(Q t).
- Source: issue #86 CTMC table.
- Expected: rows sum to 1 and values match the eigendecomposition-based
            reference.

## Test: test_validation_gating
- Goal: chain methods on a non-stochastic Mat raise ValueError (methods
        are gated by validation per the design principles).
- Source: issue #86 design principles.
- Expected: ValueError from steady_state on a non-stochastic Mat.

## Test: test_simulate_distribution
- Goal: the simulation engine samples the chain's true distribution:
        long-run state frequencies match the stationary distribution.
- Source: issue #86 Rust notes (alias-method sampling correctness).
- Expected: empirical frequency of state 1 within Monte Carlo tolerance
            of pi[1].

## Test: test_engines_require_extension
- Goal: Rust-primary engines raise a clear build-pointer error when the
        extension is absent instead of silently degrading (no Python
        port exists for non-NumPy functionality).
- Source: amended §20.1 (owner directive).
- Expected: RuntimeError mentioning _rust_core from each engine method.
"""

import numpy as np
import pytest

from nemopy import ColVec, Mat, ShapeError, markov, ctmc, mat
from nemopy import _core

requires_rust = pytest.mark.skipif(
    _core._RUST is None, reason="_rust_core extension not built"
)


def _np(x):
    return np.asarray(x)


def _two_state():
    # rows: [0.9, 0.1], [0.5, 0.5]
    return markov(mat([0.9, 0.5], [0.1, 0.5]))


def _ruin():
    # gambler's ruin on {0,1,2,3}, p=0.5; 0 and 3 absorbing
    return markov(mat(
        [1.0, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.0],
        [0.0, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.5, 1.0],
    ))


def test_markov_constructor():
    P = _two_state()
    assert isinstance(P, Mat)
    with pytest.raises(ShapeError):
        markov(np.ones((2, 3)) / 3)
    with pytest.raises(ValueError, match="(?i)stochastic"):
        markov(mat([0.5, 0.5], [0.6, 0.5]))
    with pytest.raises(ValueError, match="(?i)stochastic"):
        markov(mat([1.5, 0.0], [-0.5, 1.0]))


def test_ctmc_constructor():
    Q = ctmc(mat([-1.0, 2.0], [1.0, -2.0]))
    assert isinstance(Q, Mat)
    with pytest.raises(ValueError, match="(?i)rate"):
        ctmc(mat([-1.0, 2.0], [0.5, -2.0]))
    with pytest.raises(ValueError, match="(?i)rate"):
        ctmc(mat([1.0, -1.0], [-1.0, 1.0]))


def test_is_stochastic_flags():
    assert _two_state().is_stochastic()
    assert not mat([1, 2], [3, 4]).is_stochastic()
    doubly = markov(mat([0.5, 0.5], [0.5, 0.5]))
    assert doubly.is_doubly_stochastic()
    assert not _two_state().is_doubly_stochastic()


def test_steady_state(backend):
    pi = _two_state().steady_state()
    assert isinstance(pi, ColVec)
    np.testing.assert_allclose(pi.to_flat(), [5 / 6, 1 / 6], atol=1e-8)
    assert abs(pi.sum() - 1.0) < 1e-10


def test_n_step():
    P = _two_state()
    np.testing.assert_allclose(_np(P.n_step(0)), np.eye(2))
    np.testing.assert_allclose(_np(P.n_step(2)), _np(P) @ _np(P))
    assert isinstance(P.n_step(3), Mat)


def test_absorbing_analysis():
    P = _ruin()
    assert P.absorbing_states() == [0, 3]
    assert P.is_absorbing()
    N = P.fundamental_matrix()
    # transient states {1, 2}: N = (I - Q)^-1 with Q = [[0, .5], [.5, 0]]
    np.testing.assert_allclose(
        _np(N), [[4 / 3, 2 / 3], [2 / 3, 4 / 3]], atol=1e-12
    )
    B = P.absorption_probs()
    np.testing.assert_allclose(
        _np(B), [[2 / 3, 1 / 3], [1 / 3, 2 / 3]], atol=1e-12
    )
    t = P.expected_steps()
    assert isinstance(t, ColVec)
    np.testing.assert_allclose(t.to_flat(), [2.0, 2.0], atol=1e-12)


@requires_rust
def test_communicate_and_irreducible():
    # state 2 absorbing; {0,1} communicate
    P = markov(mat(
        [0.5, 0.5, 0.0],
        [0.4, 0.5, 0.0],
        [0.1, 0.0, 1.0],
    ))
    classes = P.communicate()
    assert {frozenset(c) for c in classes} == {frozenset({0, 1}), frozenset({2})}
    assert not P.is_irreducible()
    cycle = markov(mat([0.0, 1.0], [1.0, 0.0]))
    assert cycle.is_irreducible()


@requires_rust
def test_period_and_ergodic():
    cycle = markov(mat([0.0, 1.0], [1.0, 0.0]))
    assert cycle.period() == 2
    assert not cycle.is_ergodic()
    lazy = _two_state()
    assert lazy.period() == 1
    assert lazy.is_ergodic()


@requires_rust
def test_simulate():
    P = _two_state()
    walk = P.simulate(start=0, n_steps=50, n_paths=8, seed=123)
    assert isinstance(walk, Mat)
    assert walk.shape == (51, 8)
    w = _np(walk)
    np.testing.assert_array_equal(w[0], np.zeros(8))
    assert np.isin(w, [0.0, 1.0]).all()
    again = P.simulate(start=0, n_steps=50, n_paths=8, seed=123)
    np.testing.assert_array_equal(w, _np(again))


@requires_rust
def test_hitting_time():
    P = markov(mat([0.0, 0.0], [1.0, 1.0]))  # rows: [0,1],[0,1]
    t = P.hitting_time(start=0, target=1, n_sims=200, seed=7)
    assert t == 1.0


def test_embedded_dtmc():
    Q = ctmc(mat([-1.0, 2.0], [1.0, -2.0]))
    J = Q.embedded_dtmc()
    np.testing.assert_allclose(_np(J), [[0.0, 1.0], [1.0, 0.0]])


@requires_rust
def test_transient_probs():
    Q = ctmc(mat([-1.0, 2.0], [1.0, -2.0]))
    t = 0.7
    P_t = Q.transient_probs(t)
    assert isinstance(P_t, Mat)
    np.testing.assert_allclose(_np(P_t).sum(axis=1), [1.0, 1.0], atol=1e-10)
    q = _np(Q)
    w, v = np.linalg.eig(q)
    ref = (v @ np.diag(np.exp(w * t)) @ np.linalg.inv(v)).real
    np.testing.assert_allclose(_np(P_t), ref, atol=1e-9)


def test_validation_gating():
    A = mat([1, 2], [3, 4])
    with pytest.raises(ValueError, match="(?i)stochastic"):
        A.steady_state()


@requires_rust
def test_simulate_distribution():
    P = _two_state()
    pi = P.steady_state().to_flat()
    walk = _np(P.simulate(start=0, n_steps=4000, n_paths=4, seed=11))
    assert abs((walk == 1.0).mean() - pi[1]) < 0.03


def test_engines_require_extension(monkeypatch):
    monkeypatch.setattr(_core, "_RUST", None)
    P = _two_state()
    Q = ctmc(mat([-1.0, 2.0], [1.0, -2.0]))
    with pytest.raises(RuntimeError, match="_rust_core"):
        P.simulate(start=0, n_steps=1, seed=0)
    with pytest.raises(RuntimeError, match="_rust_core"):
        P.hitting_time(start=0, target=1, n_sims=1, seed=0)
    with pytest.raises(RuntimeError, match="_rust_core"):
        P.communicate()
    with pytest.raises(RuntimeError, match="_rust_core"):
        Q.transient_probs(0.5)
