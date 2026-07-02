#!/usr/bin/env bash
#
# build_rust.sh — reproducible local build/install of nemopy._rust_core (issue #117).
#
# Two modes:
#
#   1. Default (in-repo development):
#        scripts/build_rust.sh
#      Compiles the crate with cargo and copies the compiled extension into the
#      source tree at nemopy/_rust_core.so, so an editable checkout
#      (`uv pip install -e .`) can `import nemopy._rust_core` and light up the
#      Tier-3 surface (nemopy._core._RUST is not None).
#
#        cargo build --release --manifest-path nemopy/_rust_core/Cargo.toml
#        cp nemopy/_rust_core/target/release/lib_rust_core.so nemopy/_rust_core.so
#
#   2. Wheel mode (consumer / non-editable install):
#        scripts/build_rust.sh --wheel
#      Builds a distributable abi3 wheel with maturin. Install it alongside a
#      non-editable nemopy (e.g. `uv add "git+https://github.com/.../nemopy"`):
#        uv pip install <printed wheel path>
#      Both land in the same site-packages/nemopy/, so the import resolves.
#
# Why the default is the in-tree copy and NOT `maturin develop`:
#   `maturin develop` drops the extension into the active venv's site-packages,
#   but under an editable nemopy (`pip install -e .`) the `nemopy` package
#   resolves to the SOURCE TREE, whose `nemopy/_rust_core/` is an empty namespace
#   directory — so `import nemopy._rust_core` misses the site-packages copy and
#   _RUST stays None. Copying the .so into the source tree is the path that
#   actually resolves for in-repo development. (See issue #117.)
#
# The crate is built with the abi3-py39 feature, so a single build is usable on
# CPython >= 3.10 (the nemopy floor).
#
# Build artifacts (nemopy/_rust_core.so, target/) are gitignored — never commit them.

set -euo pipefail

# Resolve the repository root so the script works from any working directory.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MANIFEST="nemopy/_rust_core/Cargo.toml"

if [[ "${1:-}" == "--wheel" ]]; then
    command -v maturin >/dev/null 2>&1 || {
        echo "error: maturin not found. Install it with 'uv pip install maturin'." >&2
        exit 1
    }
    echo ">>> Building distributable wheel with maturin..."
    maturin build --release --manifest-path "${MANIFEST}"
    echo ">>> Wheel written under nemopy/_rust_core/target/wheels/."
    echo ">>> Install into a non-editable environment with: uv pip install <wheel>"
    exit 0
fi

command -v cargo >/dev/null 2>&1 || {
    echo "error: cargo not found. Install the Rust toolchain (https://rustup.rs)." >&2
    exit 1
}

echo ">>> Building nemopy._rust_core (release)..."
cargo build --release --manifest-path "${MANIFEST}"

echo ">>> Installing extension into the source tree..."
cp nemopy/_rust_core/target/release/lib_rust_core.so nemopy/_rust_core.so

echo ">>> Verifying the extension loads..."
python -c "import nemopy._core as c; assert c._RUST is not None; print('OK: _rust_core', c._RUST.rust_core_version())"

echo ">>> Done. nemopy._rust_core is built and importable."
