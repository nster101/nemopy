"""Tests for the Rust core scaffolding (issue #75, Phase 1): loader contract and fused subtraction.

## Test: test_loader_contract
- Goal: Verify the loader handle nemopy._core._RUST is either None
        (extension not built) or the compiled module exposing
        rust_core_version(), giving feature code a single dispatch point.
        The bare crate source directory must never be mistaken for the
        compiled extension.
- Source: DESIGN_APPENDICES.md §20.1 (two-tier strategy); issue #75
          ("_core.py — Falls back to numpy if rust ext unavailable").
- Expected: _RUST is None, or rust_core_version() returns a non-empty str.

## Test: test_fused_sub_parity
- Goal: Verify the fused shape-guard + subtract kernel produces the same
        values and nemopy result types as the pure-NumPy operator path.
- Source: issue #75 Phase 1 (fused validation+subtraction kernel);
          DESIGN_APPENDICES.md §20.1 (results agree across paths) and
          §20.3 (type persistence — no plain ndarrays escape).
- Expected: ColVec - ColVec → ColVec, Mat - Mat → Mat; values equal to
            the NumPy result.

## Test: test_fused_sub_shape_error_message
- Goal: Verify a shape mismatch through the Rust path raises ShapeError
        with the same message contract as the Python guard.
- Source: DESIGN.md §5 shape-guarded arithmetic (message text in
          nemopy._operators._check_shapes); issue #75 Phase 1 (the kernel
          fuses the guard, so it must preserve guard semantics).
- Expected: ShapeError whose message starts with "Element-wise '-'
            requires identical shapes".

## Test: test_sub_fallback_without_extension
- Goal: Verify subtraction works with identical semantics when the
        extension is forced absent — the NumPy-replacement surface never
        gates on Rust.
- Source: DESIGN_APPENDICES.md §20.1 fallback scope (owner amendment).
- Expected: same values/types via the pure-NumPy path, and the same
            ShapeError on mismatched shapes.

## Test: test_fused_arith_parity
- Goal: Verify each forward operator (+, *, /) dispatches through its Phase 2
        Rust kernel and the Rust-path result matches the pure-NumPy fallback
        in both value and nemopy result type. A spy on the kernel proves the
        dispatch actually routes to Rust (otherwise the value parity is
        vacuous); computing the reference with the extension forced absent
        exercises the retained Tier-2 fallback.
- Source: issue #109 Phase 2 (fused_add/mul/div); DESIGN_APPENDICES.md
          §20.1 (results agree across paths; Tier-2 fallback retained),
          §20.3 (type persistence), §20.5 (Phase 2 hot-path replacement).
- Expected: kernel invoked; ColVec op ColVec → ColVec, Mat op Mat → Mat;
            Rust-path values equal the forced-fallback values.

## Test: test_fused_arith_shape_error_message
- Goal: Verify each Phase 2 kernel fuses the shape guard and raises
        ShapeError with the same per-operator message contract as fused_sub.
- Source: DESIGN.md §7 shape-guarded arithmetic (message text in
          nemopy._operators._check_shapes); issue #109 (kernels fuse the
          guard, so they must preserve guard semantics).
- Expected: ShapeError whose message starts with "Element-wise '<op>'
            requires identical shapes".
"""

import operator

import numpy as np
import pytest

from nemopy import ColVec, Mat, ShapeError, _c, mat
from nemopy import _core


requires_rust = pytest.mark.skipif(
    _core._RUST is None, reason="_rust_core extension not built"
)


def test_loader_contract():
    if _core._RUST is None:
        return
    version = _core._RUST.rust_core_version()
    assert isinstance(version, str)
    assert version


@requires_rust
def test_fused_sub_parity():
    d = _c[5, 7, 9] - _c[1, 2, 3]
    assert isinstance(d, ColVec)
    assert d.to_list() == [4.0, 5.0, 6.0]

    A = mat([5, 7], [9, 11])
    B = mat([1, 2], [3, 4])
    D = A - B
    assert isinstance(D, Mat)
    np.testing.assert_array_equal(np.asarray(D), np.asarray(A) - np.asarray(B))


@requires_rust
def test_fused_sub_shape_error_message():
    with pytest.raises(
        ShapeError, match=r"Element-wise '-' requires identical shapes"
    ):
        _c[1, 2, 3] - _c[1, 2]


def test_sub_fallback_without_extension(monkeypatch):
    monkeypatch.setattr(_core, "_RUST", None)
    d = _c[5, 7, 9] - _c[1, 2, 3]
    assert isinstance(d, ColVec)
    assert d.to_list() == [4.0, 5.0, 6.0]
    with pytest.raises(ShapeError, match=r"identical shapes"):
        mat([1, 2], [3, 4]) - mat([1, 2, 3], [4, 5, 6])


@requires_rust
@pytest.mark.parametrize(
    "op, kernel",
    [
        (operator.add, "fused_add"),
        (operator.mul, "fused_mul"),
        (operator.truediv, "fused_div"),
    ],
)
def test_fused_arith_parity(op, kernel, monkeypatch):
    col_a, col_b = _c[5, 7, 9], _c[1, 2, 3]
    mat_a, mat_b = mat([5, 7], [9, 11]), mat([1, 2], [3, 4])

    with monkeypatch.context() as m:
        m.setattr(_core, "_RUST", None)
        numpy_col, numpy_mat = op(col_a, col_b), op(mat_a, mat_b)

    invoked = {}
    original = getattr(_core._RUST, kernel)

    def spy(a, b, _orig=original):
        invoked["called"] = True
        return _orig(a, b)

    monkeypatch.setattr(_core._RUST, kernel, spy)
    rust_col, rust_mat = op(col_a, col_b), op(mat_a, mat_b)

    assert invoked.get("called")
    assert isinstance(rust_col, ColVec) and isinstance(numpy_col, ColVec)
    assert isinstance(rust_mat, Mat) and isinstance(numpy_mat, Mat)
    np.testing.assert_array_equal(np.asarray(rust_col), np.asarray(numpy_col))
    np.testing.assert_array_equal(np.asarray(rust_mat), np.asarray(numpy_mat))


@requires_rust
@pytest.mark.parametrize(
    "kernel, symbol",
    [("fused_add", "+"), ("fused_mul", "*"), ("fused_div", "/")],
)
def test_fused_arith_shape_error_message(kernel, symbol):
    a = np.asarray(_c[1, 2, 3])
    b = np.asarray(_c[1, 2])
    with pytest.raises(
        ShapeError,
        match=rf"Element-wise '\{symbol}' requires identical shapes",
    ):
        getattr(_core._RUST, kernel)(a, b)
