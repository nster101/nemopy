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
