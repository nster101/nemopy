"""Shared fixtures: backend parametrization for Rust/NumPy path parity.

DESIGN_APPENDICES.md §20.1 requires that, where both a Rust fast path and
a NumPy fallback exist, both are exercised by the same test suite. Tests
that take the ``backend`` fixture run twice: once against the compiled
``_rust_core`` extension (skipped when not built) and once with the
extension forced absent so the pure-NumPy fallback executes.
"""

import pytest

from nemopy import _core


@pytest.fixture(params=["rust", "numpy"])
def backend(request, monkeypatch):
    if request.param == "rust":
        if _core._RUST is None:
            pytest.skip("_rust_core extension not built")
    else:
        monkeypatch.setattr(_core, "_RUST", None)
    return request.param
