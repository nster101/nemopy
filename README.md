# nemopy

A column-vector-first NumPy wrapper. Vectors are `(n, 1)` by default;
matrices are constructed column-by-column; arithmetic raises a clear
`ShapeError` when shapes disagree instead of silently broadcasting.

## Install

nemopy is not on PyPI yet — install it from the git repository. It requires
Python `>=3.10` (the repo pins 3.14 via `.python-version`) and is developed
with [uv](https://docs.astral.sh/uv/). The only hard dependency is NumPy.

There are two modes. **Mode A (pure-Python)** is the default and needs no
toolchain beyond Python; **Mode B (Rust-enabled)** additionally compiles the
optional `nemopy._rust_core` extension to unlock the Tier-3 surface.

### Mode A — pure-Python

```bash
uv add "git+https://github.com/nster101/nemopy"                  # core (numpy only)
uv add "nemopy[pandas] @ git+https://github.com/nster101/nemopy" # + pandas interop
uv add "nemopy[polars] @ git+https://github.com/nster101/nemopy" # + polars interop
```

This gives you the core types (`ColVec`, `Mat`), shape-guarded arithmetic, the
Tier-2 decompositions (`svd`, `qr`, `lu`, `cholesky`, `eigh`) and the stats
helpers — everything that wraps NumPy directly. Tier-3 methods (advanced
decompositions, elimination, and future LP / network / Markov features) are
**not** available in this mode: calling one raises a clear `ImportError`
pointing at the build step, rather than silently falling back.

### Mode B — Rust-enabled

To unlock the Tier-3 surface, build the optional Rust extension after cloning
the repository. `scripts/build_rust.sh` is the canonical build command (it
wraps `maturin`):

```bash
git clone https://github.com/nster101/nemopy
cd nemopy
uv sync                             # install Python deps (incl. dev toolchain)
scripts/build_rust.sh               # compile nemopy._rust_core via maturin
```

Verify which mode is active:

```bash
python -c "import nemopy._core as c; print('rust active:', c._RUST is not None)"
```

Prints `rust active: True` once the extension is built (Mode B) and
`rust active: False` in pure-Python mode (Mode A).

For development (pytest + sphinx and the full toolchain), use the `dev` extra:

```bash
uv sync --extra dev
```

## Quick start

```python
import nemopy as nm

u = nm._c[1, 2, 3]                  # ColVec, shape (3, 1)
v = nm._c[4, 5, 6]                  # ColVec, shape (3, 1)

A = nm.mat(u, v)                    # Mat, shape (3, 2), columns are u and v
I = nm.eye(3)                       # 3x3 identity Mat

# MATLAB-style string syntax — columns separated by ';', elements by ',' or whitespace
B = nm._m["1, 2, 3; 4, 5, 6; 7, 8, 9"]   # Mat (3,3), columns [1,2,3] [4,5,6] [7,8,9]
B_rows = nm._m["1, 2, 3; 4, 5, 6; 7, 8, 9"].T   # rows [1,2,3] [4,5,6] [7,8,9]

# Inbound converters
c = nm.as_col([10, 20, 30])         # ColVec from list / Series / 1D array
M = nm.as_mat([[1, 2], [3, 4]])     # Mat from row-first nested list / DataFrame

# Outbound converters
u.to_list()                         # [1.0, 2.0, 3.0]
A.to_numpy()                        # plain ndarray, shape (3, 2)

# Matrix properties
A_sq = nm.mat(nm._c[1, 2], nm._c[3, 4])
A_sq.det                            # determinant
A_sq.inv                            # inverse Mat
A_sq.is_singular                    # bool
```

## Why column-first?

Linear algebra is column-first: a matrix-vector product `A @ x` reads
naturally when `x` is a column. nemopy enforces that convention end-to-end
so column extraction (`A[:, j]`) returns a `ColVec` you can plug straight
into `@` without reshaping.

## Documentation

The full behavioural specification lives in
[`.github/DESIGN.md`](.github/DESIGN.md) and
[`.github/DESIGN_APPENDICES.md`](.github/DESIGN_APPENDICES.md). Sphinx-built
API docs are configured in `docs/`.

## License

BSD 3-Clause. See [LICENSE](LICENSE).
