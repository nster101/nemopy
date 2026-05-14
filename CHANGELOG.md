# Changelog

All notable changes to nemopy are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] — unreleased

### Added

**Core types**
- `ColVec` — column vector type, shape `(n, 1)`, dtype `float64`, subclass of `np.ndarray`
- `Mat` — matrix type, shape `(n, k)`, dtype `float64`, subclass of `np.ndarray`
- `_VecBase` — shared non-public base class holding operator overrides and transpose properties

**Constructors**
- `_c[1, 2, 3]` — bracket-notation singleton for column vector literals
- `_m["1 2 3; 4 5 6"]` — MATLAB-style string constructor (column-first convention)
- `mat(col1, col2, ...)` — column-first matrix constructor
- `eye(n)` — `n × n` identity matrix
- `as_col(x)` — flexible inbound converter (lists, 1D arrays, scalars, pandas/polars Series)
- `as_mat(x)` — flexible inbound converter (nested lists, 2D arrays, pandas/polars DataFrames)

**Operators**
- Shape-guarded `*`, `+`, `-`, `/` — raise `ShapeError` on array shape mismatch; scalars always pass
- In-place `+=`, `-=`, `*=`, `/=` — same guards, preserve subclass label
- `|` column-join — `_c[a,b,c] | _c[d,e,f]` assembles a matrix column by column
- `ConventionWarning` on `@` when the other operand is a wide plain ndarray

**Properties on `Mat`**
- `.inv` — matrix inverse, returns `Mat`
- `.det` — determinant, returns `float`
- `.is_singular` — rank-based singularity test, returns `bool`

**Transpose and conjugate transpose** (on both `ColVec` and `Mat`)
- `.T` — transpose; return type dispatched by shape (`(n,1)` → `ColVec`, else `Mat`)
- `.H` — conjugate transpose; equals `.T` for real arrays
- `.transpose(*axes)` — method form consistent with `np.transpose`

**Outbound conversions**
- `ColVec.to_numpy()`, `.to_flat()`, `.to_list()`, `.to_series()`, `.to_polars()`
- `Mat.to_numpy()`, `.to_list()`, `.to_dataframe()`, `.to_polars()`

**Indexing**
- `ColVec[i]` → `float`; `ColVec[i:j]` / `ColVec[mask]` → `ColVec`
- `Mat[i, j]` → `float`; `Mat[:, j]` → `ColVec`; `Mat[:, j:k]` / `Mat[i, :]` → `Mat`

**Polars integration** (`nemopy.polars` submodule, optional)
- `as_col` and `as_mat` accept `polars.Series` and `polars.DataFrame`
- `ColVec.to_polars()` and `Mat.to_polars()` for outbound conversion
- `df.nemo.col(name)` and `df.nemo.mat(names)` polars DataFrame accessor
- `series.nemo.col()` polars Series accessor
- Install with: `pip install "nemopy[polars]"`

**Errors**
- `ShapeError(ValueError)` — raised on shape mismatches
- `ConventionWarning(UserWarning)` — emitted on convention-suspicious operations

**Documentation**
- Full NumPy-style docstrings on all public symbols (Examples, Parameters, Returns, Raises, See Also)
- Sphinx documentation in `docs/` with `sphinx-build -b doctest` verified examples
- Real-world examples page covering inner/outer products, OLS, column extraction, Gram–Schmidt, Polars round-trip
