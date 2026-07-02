# Archimedes — coordination log

<!-- Append-only; newest entries at the bottom. Only Archimedes writes to this file. -->
<!-- Entry format: ### <UTC ISO-8601 to the minute> — Archimedes — <topic> -->

### 2026-06-24T16:20Z — Archimedes — done #119 (PR #125)
Build verification suite. Branch archimedes/119-rust-build-tests; draft PR #125.
Files mine alone: new `tests/test_rust_build.py` + appended SECTION 4 to
`verify_nemopy.py`. Two-mode contract for the 12 Tier-3 methods: rust-PRESENT
(compute without raising + dispatch through the `_core._require_rust` gate,
spy-verified; skips in pure-Python CI), rust-ABSENT (actionable ImportError —
method name + `maturin develop` hint; complements #105 without touching it).
Verified both modes locally: pure-Python 2 passed/12 skipped; built the
extension (`maturin develop --release`) and exercised all 12 against the real
core (12/12, incl. schur w/ SciPy).
- Gauss (#117): compiled `.so` installs at top-level `_rust_core`, so
  `nemopy._rust_core` still resolves to the source crate dir and `_RUST` stays
  `None` after `maturin develop`. Install-layout resolution is your scope; my
  suite skips cleanly until it lands.
- Note: 17 pre-existing failures under `tests/numpy_compat/` on origin/main
  (84c15b2), unrelated to this PR (§3 report-not-fix).
Newton: gate please. #111 now ready (its preempt cleared).
