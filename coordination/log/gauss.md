# Gauss — coordination log

<!-- Append-only; newest entries at the bottom. Only Gauss writes to this file. -->
<!-- Entry format: ### <UTC ISO-8601 to the minute> — Gauss — <topic> -->

### 2026-06-24T16:05Z — Gauss — done #117 (PR #123)
Claimed + completed #117 (Rust-core reproducible build). Branch
gauss/117-rust-build, draft PR #123. New file only: scripts/build_rust.sh.
Default mode = cargo build --release + copy lib_rust_core.so →
nemopy/_rust_core.so (in-tree, resolves under editable nemopy); --wheel mode
= maturin build --release for non-editable consumers. Dropped `maturin
develop` as default — empirically gives _RUST=None under editable install
(Einstein ruled this correction authoritative on #117). Verified in a clean
uv venv on Python 3.14: assert `_RUST is not None` passes, tests/test_rust_core.py
10/10 green. No edits to root pyproject/CI/tests/docs/Cargo.toml/lib.rs;
artifacts gitignored. Newton gates. #91 (LP/IP/MILP) now unblocked — picking up next after #117 lands.

### 2026-06-25T08:23Z — Gauss — claim #129 (A[i]→column i; Mat.__iter__→columns, §6.3)
Claiming #129 (Einstein-delegated; spec gate cleared — #128 merged @0834709).
Branch gauss/129-mat-single-index-column. Files MINE: nemopy/_core.py (Mat class) +
tests/test_core.py (TestMatGetItem). Serial _core.py chain #129→#130→#131→#132 — each
blocked-until the prior MERGES, so I run them strictly in order, one Newton-gated draft
PR at a time. Disjoint from Archimedes #111 (decomp.rs/_decompositions.py).
SCOPE NOTE — maintainer-authorized scope expansion (2026-06-25, /gauss approval): on top
of the §6.3 integer intercept + Mat.__iter__, I also align Mat.__getitem__'s scalar
branch from `return float(result)` to the §6.3 code block's `return float(result) if
isinstance(result, np.generic) else result`. Why: §6.3's code block is the immutable spec
(CLAUDE.md §1) and this was a latent code/spec textual discrepancy (NOT in #128's diff).
Behaviour is unchanged on every reachable Mat path (the ndim==0 branch on a float64 Mat
always yields an np.generic scalar → float()); existing test_mat_getitem_element_returns_float
guards the live A[i,j]→float path, so no new behaviour is asserted. Declared here + on #129
+ in the PR body so Newton sees the authorization.
