# Euclid — coordination log

<!-- Append-only; newest entries at the bottom. Only Euclid writes to this file. -->
<!-- Entry format: ### <UTC ISO-8601 to the minute> — Euclid — <topic> -->

### 2026-06-24T15:54Z — Euclid — claim #118 (Rust core install/usage docs, two-mode)

Claiming #118 (Einstein-delegated, ready, parallel with Gauss #117). **Files
(mine alone):** `README.md`, `docs/index.rst`. Docs-only — no code, no root
`pyproject.toml`, no CI, no `tests/`, `scripts/`, or `nemopy/_rust_core/*`.
Branch `euclid/118-rust-install-docs`. Per maintainer ruling on #118
(2026-06-24): README is canonical/full two-mode install; `docs/index.rst` gets a
condensed Installation section inserted between the intro and the `.. toctree::`
(no new `.rst`, no toctree change). Reference `scripts/build_rust.sh` (Gauss
#117) as the canonical Rust build command so this proceeds in parallel. Newton
gates the feature PR. Queued #63 (pandas compat) is ready-after this.
