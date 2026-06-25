"""Build verification suite for the Rust core as a two-mode contract (issue #119).

This suite proves the Tier-3 surface end-to-end against the *state of the
build* rather than re-checking each decomposition's numerics (those live in
``test_decompositions_advanced.py`` / ``test_elimination.py``). It asserts the
two halves of the §20.1/§20.4 contract:

  * rust-PRESENT: with the compiled ``nemopy._rust_core`` extension built,
    every Tier-3 method computes without raising AND routes through the single
    ``_core._require_rust`` extension gate; the suite SKIPs cleanly when the
    extension is absent so it stays green in pure-Python CI.
  * rust-ABSENT: with ``_core._RUST`` forced to ``None`` the surfaced
    ``ImportError`` is *actionable* — it names the method and points at the
    ``maturin develop`` build step.

## Test: test_tier3_method_dispatches_when_present
- Goal: Verify that, when the extension is built, each Tier-3 method
        (ldu/qdr/schur/polar/diagonalize/jordan and
        ref/rref/gaussian_eliminate/gauss_jordan/rank/nullspace) completes
        without raising and dispatches through the `_core._require_rust`
        extension gate (proving the built core actually services the whole
        Tier-3 surface, not a silent substitute path).
- Source: issue #119 (build verification, rust-PRESENT half);
          DESIGN_APPENDICES.md §20.1/§20.4 (Rust-primary Tier-3, single
          dispatch point, no NumPy fallback).
- Expected: the call returns a non-None result and `_require_rust` is invoked
            with the method's own name; SKIPs when `_core._RUST is None`.

## Test: test_tier3_importerror_is_actionable_when_absent
- Goal: Verify that with the extension absent the ImportError surfaced from a
        Tier-3 method is actionable — it names the failing method AND gives the
        `maturin develop` build remediation. This complements the #105 tests
        (which only assert the method name appears) by pinning the build-hint
        contract; it does not modify or duplicate them.
- Source: issue #119 (rust-ABSENT half); the #105 ImportError message contract
          in `nemopy._core._require_rust`; DESIGN_APPENDICES.md §20.1/§20.4.
- Expected: ImportError whose message contains the method name and the literal
            `maturin develop` build command.
"""

import pytest

from nemopy import _c, _core, mat


requires_rust = pytest.mark.skipif(
    _core._RUST is None,
    reason="_rust_core extension not built (Tier-3 requires Rust)",
)

# The 12 Tier-3 methods named in issue #119, each with a valid receiver matrix
# and call arguments. Square, well-conditioned operands so a built extension
# computes a result rather than raising on a degenerate input.
_TIER3_CALLS = {
    "ldu": (mat([4, 2], [2, 5]), ()),
    "qdr": (mat([4, 2], [2, 5]), ()),
    "schur": (mat([4, 1, 0], [2, 3, 1], [0, 1, 2]), ()),
    "polar": (mat([3, 1, 0], [1, 2, 1]), ()),
    "diagonalize": (mat([2, 0], [1, 3]), ()),
    "jordan": (mat([2, 0], [1, 3]), ()),
    "ref": (mat([2, 1], [1, 3]), ()),
    "rref": (mat([2, 1], [1, 3]), ()),
    "rank": (mat([2, 1], [1, 3]), ()),
    "nullspace": (mat([2, 1], [1, 3]), ()),
    "gaussian_eliminate": (mat([2, 1], [1, 3]), (_c[5, 10],)),
    "gauss_jordan": (mat([2, 1], [1, 3]), (_c[5, 10],)),
}


@requires_rust
@pytest.mark.parametrize("method", sorted(_TIER3_CALLS))
def test_tier3_method_dispatches_when_present(method, monkeypatch):
    if method == "schur":
        pytest.importorskip("scipy")
    receiver, args = _TIER3_CALLS[method]

    gated = []
    original = _core._require_rust

    def spy(name, _orig=original):
        gated.append(name)
        return _orig(name)

    monkeypatch.setattr(_core, "_require_rust", spy)
    result = getattr(receiver, method)(*args)

    assert result is not None
    assert method in gated


@pytest.mark.parametrize("method", ["ldu", "ref"])
def test_tier3_importerror_is_actionable_when_absent(method, monkeypatch):
    monkeypatch.setattr(_core, "_RUST", None)
    receiver, args = _TIER3_CALLS[method]
    with pytest.raises(ImportError) as excinfo:
        getattr(receiver, method)(*args)
    message = str(excinfo.value)
    assert method in message
    assert "maturin develop" in message
