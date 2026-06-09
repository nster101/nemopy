<!--
DESIGN_APPENDICES.md — companion file to DESIGN.md.

This file contains the remainder of the design specification that was split out
of DESIGN.md so that each file fits within a single LLM read pass. It contains:

  - §§11 through 16 (NumPy Interoperability, External Library Interoperability,
    Conversion Functions, Known Limitations, Summary Tables, What This Is Not)
  - Appendix A (Documentation Specification)
  - Appendix B (Implementation Checklist for Agent)

Section numbering is preserved. A reference to "Section N" refers to the same
Section N regardless of which file it resides in. This file carries the same
authority as DESIGN.md — the split is purely a file-size concern, not a change
in the status of any content.
-->

## 11. NumPy Interoperability

`ColVec` and `Mat` are `np.ndarray` subclasses and are drop-in compatible with NumPy
in all standard use cases.

### What works without any changes

```python
np.linalg.inv(A)          # matrix inverse -> plain ndarray (or use A.inv for Mat)
np.linalg.eig(A)          # eigendecomposition -> plain ndarray, plain ndarray
np.linalg.solve(A, b)     # linear solve Ax=b -> plain ndarray
np.linalg.norm(u)         # vector norm -> scalar
np.sum(u)                 # reduction -> scalar
np.exp(u)                 # ufunc, element-wise -> ColVec (via __array_wrap__)
np.zeros((n, 1))          # produces plain ndarray — wrap with ColVec() if needed
np.eye(n)                 # produces plain ndarray — use eye(n) for Mat
```

### Re-wrapping NumPy outputs

NumPy functions return plain `ndarray`. The shape is always correct; the type label is lost.
Re-wrap explicitly when the label matters:

```python
inv_A = Mat(np.linalg.inv(A))           # re-label as Mat (or just use A.inv)
x     = ColVec(np.linalg.solve(A, b))   # re-label as ColVec
```

For numerical pipelines where only the shape matters, re-wrapping is unnecessary overhead.

### Behaviour differences from plain NumPy

| Operation | NumPy default | `nemopy` behaviour |
|---|---|---|
| `u * v` mismatched shapes | Silent broadcast | `ShapeError` |
| `u + v` mismatched shapes | Silent broadcast | `ShapeError` |
| `u - v` mismatched shapes | Silent broadcast | `ShapeError` |
| `u + v` same shapes | Returns `ndarray` | Returns `ColVec` or `Mat` (type preserved) |
| `np.dot(u, v)` | Works on 1D | Works — but inputs are 2D, result is `(1,1)` |
| `np.array(u)` | Returns base `ndarray` | Loses `ColVec` label — shape `(n,1)` preserved |
| `_VecBase @ ndarray` | Silent | `ConventionWarning` if shape looks transposed |
| `np.multiply(A, B)` mismatched shapes | Silent broadcast | **Also broadcasts** — ufuncs bypass operator overrides |

### `.copy()` Behaviour

NumPy's `.copy()` on a subclass preserves the subclass in NumPy >= 1.20. Therefore:

```python
u = _c[1, 2, 3]
v = u.copy()        # v is ColVec, shape (3,1) — subclass preserved
```

This is inherited behaviour, not overridden. If NumPy changes `.copy()` semantics in a
future version, this may need revisiting.

### `np.vstack`, `np.hstack`, `np.concatenate`

These functions call `__array_wrap__` on the result (in NumPy 1.x), so the output type
is determined by shape:

```python
u = _c[1, 2, 3]
v = _c[4, 5, 6]

np.vstack([u, v])       # shape (6, 1) → ColVec (via __array_wrap__)
np.hstack([u, v])       # shape (3, 2) → Mat (via __array_wrap__)
np.concatenate([u, v])  # shape (6, 1) → ColVec (via __array_wrap__)
```

**Caveat:** This relies on `__array_wrap__` being called, which may change in NumPy 2.0+.
The implementation should verify this behaviour in its test suite (see Section 4.5 on
version targeting). If concatenation functions stop calling the persistence hook, the
output will be plain `ndarray` — shape is preserved, type label is lost.

---

## 12. External Library Interoperability

### 12.1 `scipy.linalg`

All `scipy.linalg` functions accept `ColVec` and `Mat` directly, since both are `ndarray`
subclasses. Return values are plain `ndarray`.

```python
import scipy.linalg as la

la.inv(A)           # works — returns plain ndarray
la.svd(A)           # works — returns plain ndarrays
la.solve(A, b)      # works — returns plain ndarray
la.cholesky(A)      # works — returns plain ndarray
la.expm(A)          # matrix exponential — works
```

Re-wrap outputs if `Mat`/`ColVec` labels are needed downstream.

### 12.2 `scipy.sparse`

`scipy.sparse` matrices are **not compatible**. `nemopy` wraps dense arrays only. Passing
a sparse matrix to `mat()` raises a `TypeError`. If sparse support is needed, work
directly in NumPy/SciPy and re-wrap results selectively.

### 12.3 `scipy.optimize`

Optimisation routines (e.g. `scipy.optimize.minimize`) expect 1D arrays for parameter
vectors. Pass `.flatten()` when interfacing:

```python
from scipy.optimize import minimize

def objective(x_flat):
    x = ColVec(x_flat.reshape(-1, 1))   # re-wrap inside objective
    return (x.T @ Q @ x).item()

result = minimize(objective, x0=u.flatten())
```

### 12.4 `scipy.stats`

Statistical functions generally accept 2D arrays. `ColVec` and `Mat` pass through
without issues. Note that many `scipy.stats` functions return 1D arrays — reshape
or re-wrap as needed.

### 12.5 Pandas

**DataFrame from Mat:**
```python
import pandas as pd

A = mat([1,2,3], [4,5,6], [7,8,9])
df = pd.DataFrame(A, columns=["a", "b", "c"])   # works directly
```

**Series from ColVec** — `pd.Series` expects a 1D array, flatten first:
```python
u = _c[1, 2, 3]
s = pd.Series(u.flatten(), index=["x", "y", "z"])
```

**ColVec from DataFrame column:**
```python
u = ColVec(df["a"].values.reshape(-1, 1))
```

**Mat from DataFrame:**
```python
A = Mat(df.values)   # works if df is purely numeric
```

**Indexing behaviour:** Because `ColVec` and `Mat` are `ndarray` subclasses, Pandas will
occasionally call NumPy indexing internals on them. This is safe — the subclass label may
be lost during Pandas operations, but shape is always preserved.

---

## 13. Conversion Functions

### 13.1 Rationale

`ColVec` and `Mat` are `ndarray` subclasses, so most external functions accept them
directly. However, some libraries perform exact type checks (`type(x) == np.ndarray`),
call internal constructors that strip subclass labels, or expect specific shapes (1D for
`pd.Series`, flat for `scipy.optimize`). Rather than leaving users to remember which
incantation of `.flatten()`, `.reshape()`, or `np.asarray()` applies in each case,
the library provides explicit conversion methods on `ColVec` and `Mat`, and module-level
inbound converters for creating `nemopy` types from external data.

**Design rule:** Every conversion is a one-liner under the hood. The value is not in the
implementation — it is in the named intent and the docstring that appears on hover.

### 13.2 Outbound Conversions (Methods on `ColVec` and `Mat`)

These are methods, not standalone functions, because they convert *from* a `nemopy` type.

#### On `ColVec`:

```python
# On ColVec:

def to_numpy(self):
    """Return a plain ndarray of shape (n, 1). Strips subclass label.

    Use when passing to libraries that reject ndarray subclasses.

    Returns
    -------
    np.ndarray
        Shape ``(n, 1)``, dtype ``float64``.
    """
    return np.array(self)

def to_flat(self):
    """Return a 1D ndarray of shape (n,).

    Use when interfacing with scipy.optimize, pd.Series, or any API
    that expects a 1D parameter vector.

    Returns
    -------
    np.ndarray
        Shape ``(n,)``, dtype ``float64``.
    """
    return np.asarray(self).flatten()

def to_list(self):
    """Return a plain Python list of floats.

    Returns
    -------
    list of float
        Length ``n``.
    """
    return self.flatten().tolist()

def to_series(self, index=None, name=None):
    """Return a pandas Series.

    Parameters
    ----------
    index : array-like, optional
        Index labels. Defaults to ``range(n)``.
    name : str, optional
        Series name.

    Returns
    -------
    pd.Series

    Raises
    ------
    ImportError
        If pandas is not installed.
    """
    import pandas as pd
    return pd.Series(self.flatten(), index=index, name=name)
```

#### On `Mat`:

```python
# On Mat:

def to_numpy(self):
    """Return a plain ndarray of shape (n, k). Strips subclass label.

    Returns
    -------
    np.ndarray
        Shape ``(n, k)``, dtype ``float64``.
    """
    return np.array(self)

def to_list(self):
    """Return a nested list (list of rows, each a list of floats).

    Returns
    -------
    list of list of float
        Outer list has ``n`` elements, each inner list has ``k`` elements.
    """
    return self.tolist()

def to_dataframe(self, columns=None, index=None):
    """Return a pandas DataFrame.

    Parameters
    ----------
    columns : list of str, optional
        Column labels. Defaults to integer range.
    index : array-like, optional
        Row index labels.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    ImportError
        If pandas is not installed.
    """
    import pandas as pd
    return pd.DataFrame(np.asarray(self), columns=columns, index=index)
```

**Also provided:**

```python
# On ColVec:
def to_polars(self, name=None):
    """Return a polars Series. Requires polars >= 0.19."""
    import polars as pl
    return pl.Series(name=name or "", values=self.flatten().tolist())

# On Mat:
def to_polars(self, schema=None):
    """Return a polars DataFrame. Requires polars >= 0.19."""
    import polars as pl
    arr = np.asarray(self)
    if schema is None:
        schema = [f"col_{i}" for i in range(arr.shape[1])]
    return pl.from_numpy(arr, schema=schema)
```

**Not provided:**
- `to_set()` — sets discard ordering and duplicates, which are both meaningful in vectors
  and matrices. If a user needs the unique elements, `set(u.to_list())` is explicit about
  the information loss.
- `to_ndarray()` / `to_array()` — these are aliases of `to_numpy()`. One name is enough.

### 13.3 Inbound Conversions (Module-Level Functions)

These are module-level functions that convert *to* `nemopy` types from external data.
They are more forgiving than the `ColVec()` and `Mat()` constructors: they accept
a wider range of input shapes and perform the necessary reshaping.

```python
def as_col(x):
    """Convert any array-like to a ColVec.

    Accepts 1D arrays, flat lists, pandas Series, (n,1) arrays, and
    scalar values. Performs reshaping as needed.

    Parameters
    ----------
    x : array-like
        Input data. Must be convertible to a 1D or (n,1) numeric array.

    Returns
    -------
    ColVec
        Shape ``(n, 1)``.

    Raises
    ------
    ShapeError
        If ``x`` is 2D with more than one column (ambiguous — which column?).
    TypeError
        If ``x`` cannot be converted to a numeric array.

    Examples
    --------
    >>> as_col([1, 2, 3])
    ColVec([1.0, 2.0, 3.0])

    >>> import pandas as pd
    >>> as_col(pd.Series([4, 5, 6]))
    ColVec([4.0, 5.0, 6.0])

    >>> as_col(np.array([7, 8, 9]))    # 1D ndarray — no warning, unlike mat()
    ColVec([7.0, 8.0, 9.0])

    >>> as_col(42)                      # scalar -> (1,1) ColVec
    ColVec([42.0])

    See Also
    --------
    _c : Bracket-notation constructor for column vectors from literals.
    ColVec : Direct constructor (requires shape (n,1) exactly).
    """
    # Scalar
    if isinstance(x, (int, float, complex, np.generic)):
        return ColVec(np.array([[float(x)]]))

    # Pandas Series
    try:
        import pandas as pd
        if isinstance(x, pd.Series):
            return ColVec(x.values.astype(float).reshape(-1, 1))
    except ImportError:
        pass

    arr = np.asarray(x, dtype=float)

    if arr.ndim == 0:
        return ColVec(arr.reshape(1, 1))
    if arr.ndim == 1:
        return ColVec(arr.reshape(-1, 1))
    if arr.ndim == 2 and arr.shape[1] == 1:
        return ColVec(arr)
    if arr.ndim == 2:
        raise ShapeError(
            f"as_col() received a 2D array with shape {arr.shape}. "
            f"Cannot determine which column to extract. "
            f"Pass a single column: as_col(arr[:, j])"
        )
    raise ShapeError(
        f"as_col() requires a 1D or (n,1) input, got ndim={arr.ndim}."
    )


def as_mat(x):
    """Convert any 2D array-like to a Mat.

    Accepts 2D arrays, nested lists, pandas DataFrames, and existing
    Mat instances.

    Parameters
    ----------
    x : array-like
        Input data. Must be convertible to a 2D numeric array.

    Returns
    -------
    Mat
        Shape ``(n, k)``.

    Raises
    ------
    ShapeError
        If ``x`` is not 2D after conversion.
    TypeError
        If ``x`` cannot be converted to a numeric array.

    Examples
    --------
    >>> as_mat([[1, 2], [3, 4], [5, 6]])
    Mat(3x2):
      [1, 2]
      [3, 4]
      [5, 6]

    >>> import pandas as pd
    >>> df = pd.DataFrame({"a": [1,2], "b": [3,4]})
    >>> as_mat(df)
    Mat(2x2):
      [1, 3]
      [2, 4]

    .. warning::
        ``as_mat`` on a nested list uses NumPy's row-first convention
        (each inner list is a row). This is the **opposite** of ``mat()``,
        which interprets each argument as a column. Use ``mat()`` when
        assembling from column data. Use ``as_mat()`` when converting
        existing row-first data (DataFrames, NumPy arrays, CSV data).

    See Also
    --------
    mat : Column-first matrix constructor.
    Mat : Direct constructor (requires 2D input).
    """
    # Pandas DataFrame
    try:
        import pandas as pd
        if isinstance(x, pd.DataFrame):
            return Mat(x.values.astype(float))
    except ImportError:
        pass

    arr = np.asarray(x, dtype=float)

    if arr.ndim != 2:
        raise ShapeError(
            f"as_mat() requires a 2D input, got ndim={arr.ndim} "
            f"with shape {arr.shape}."
        )
    return Mat(arr)
```

### 13.4 Conversion Summary Table

| Direction | From | To | Function / Method | Notes |
|---|---|---|---|---|
| **Outbound** | `ColVec` | `ndarray (n,1)` | `u.to_numpy()` | Strips subclass label |
| | `ColVec` | `ndarray (n,)` | `u.to_flat()` | For scipy.optimize, pd.Series input |
| | `ColVec` | `list` | `u.to_list()` | Flat list of floats |
| | `ColVec` | `pd.Series` | `u.to_series()` | Optional index and name |
| | `Mat` | `ndarray (n,k)` | `A.to_numpy()` | Strips subclass label |
| | `Mat` | nested `list` | `A.to_list()` | List of row-lists |
| | `Mat` | `pd.DataFrame` | `A.to_dataframe()` | Optional columns and index |
| **Inbound** | list, Series, 1D array, scalar | `ColVec` | `as_col(x)` | Reshapes automatically |
| | 2D array, nested list, DataFrame | `Mat` | `as_mat(x)` | Row-first (NumPy convention) |

### 13.5 `mat()` vs `as_mat()` — When to Use Which

These two functions both produce `Mat`, but they have **opposite conventions**:

| | `mat(col1, col2, ...)` | `as_mat(existing_2d)` |
|---|---|---|
| Input format | Separate column arguments | Single 2D array-like |
| Convention | **Column-first** (each arg is a column) | **Row-first** (NumPy convention) |
| Primary use | Constructing matrices from known columns | Converting external data (DataFrames, CSV, NumPy arrays) |

```python
# These produce the SAME matrix:
A = mat([1, 3], [2, 4])            # column-first: col1=[1,3], col2=[2,4]
B = as_mat([[1, 2], [3, 4]])       # row-first:    row1=[1,2], row2=[3,4]
# A == B
```

This asymmetry is intentional. `mat()` enforces the library's column-first philosophy.
`as_mat()` accepts the outside world's row-first reality without fighting it.

---

## 14. Known Limitations

### 14.1 Square array transposition is undetectable

If a square `ndarray` built row-first with `np.array([[...]])` is passed to `mat()`
or wrapped with `Mat(...)`, the transposition cannot be detected at runtime. Shape
`(n, n)` is the same whether the matrix was built row-first or column-first.
This is the one failure mode that requires authorial discipline rather than tooling.

### 14.2 Subclass label loss

NumPy functions return plain `ndarray`. Re-wrap explicitly when the label matters.
For pure numerical work, shape is sufficient and re-wrapping is unnecessary overhead.

### 14.3 `(1,1)` inner product result

`u.T @ v` returns shape `(1,1)`, not a scalar. Use `.item()` when a bare float is
needed. This is mathematically correct behaviour, not a bug.

### 14.4 Complex numbers

`_c[...]` uses `dtype=float`. For complex vectors:

```python
u = ColVec(np.array([1+2j, 3+4j]).reshape(-1, 1))   # bypass _c[]
```

The Hermitian inner product is then `u.H @ v`, not `u.T @ v`. See Section 5.7.

### 14.5 No sparse support

Dense NumPy arrays only throughout.

### 14.6 Ufuncs bypass operator overrides

Direct NumPy ufunc calls like `np.multiply(A, B)` do **not** trigger the broadcasting
guards in Section 7. Only the Python operators (`*`, `+`, `-`, `/`) are guarded.
This is an intentional escape hatch, but users should be aware that
`np.multiply(A, B)` and `A * B` may behave differently when shapes are mismatched.

### 14.7 `.inv` is a computed property

`A.inv` computes the inverse every time it is accessed. It does not cache. If the
inverse is needed multiple times, assign it: `A_inv = A.inv`.

### 14.8 Concatenation subclass persistence

`np.vstack`, `np.hstack`, and `np.concatenate` may or may not preserve the `ColVec`/`Mat`
label depending on the NumPy version (see Section 11, "Concatenation" subsection). Shape
is always preserved regardless.

---

## 15. Summary Tables

### Constructors and Converters

| Syntax | Accepts | Output type | Shape |
|---|---|---|---|
| `_c[1, 2, 3]` | Flat scalars only | `ColVec` | `(n, 1)` |
| `_c[5]` | Single scalar | `ColVec` | `(1, 1)` |
| `mat(_c[...], ...)` | `ColVec`, list, tuple, 1D ndarray | `Mat` | `(n, k)` |
| `mat([...], [...])` | Plain lists (column-first) | `Mat` | `(n, k)` |
| `mat([1,2,3])` | Single column | `Mat` | `(n, 1)` |
| `eye(n)` | Integer dimension | `Mat` | `(n, n)` |
| `ColVec(arr)` | `(n,1)` ndarray | `ColVec` | `(n, 1)` |
| `Mat(arr)` | 2D ndarray | `Mat` | `(n, k)` |
| `as_col(x)` | list, Series, 1D/2D array, scalar | `ColVec` | `(n, 1)` |
| `as_mat(x)` | 2D array, nested list, DataFrame (row-first) | `Mat` | `(n, k)` |

### Transpose and Conjugate Transpose

| Expression | Input | Output type | Output shape | Notes |
|---|---|---|---|---|
| `u.T` | `ColVec` $(n,1)$, $n>1$ | `Mat` | $(1,n)$ | Row matrix |
| `u.T` | `ColVec` $(1,1)$ | `ColVec` | $(1,1)$ | Scalar-like vector |
| `A.T` | `Mat` $(n,k)$ | `Mat` | $(k,n)$ | Standard transpose |
| `u.H` | `ColVec` $(n,1)$ | same as `.T` | same as `.T` | Conjugate transpose; equals `.T` for real |
| `A.H` | `Mat` $(n,k)$ | `Mat` | $(k,n)$ | Conjugate transpose; equals `.T` for real |

### Matrix Properties

| Expression | Requires | Returns | Raises on |
|---|---|---|---|
| `A.inv` | Square `Mat` | `Mat` $(n,n)$ | Non-square (`ShapeError`), singular (`LinAlgError`) |
| `A.det` | Square `Mat` | `float` | Non-square (`ShapeError`) |
| `A.is_singular` | Square `Mat` | `bool` | Non-square (`ShapeError`) |

### Operations

| Expression | Mathematical meaning | Output shape | Output type |
|---|---|---|---|
| `u.T @ v` | Inner product $\mathbf{u}^T\mathbf{v}$ | `(1, 1)` | `ColVec` |
| `u.H @ v` | Hermitian inner product $\mathbf{u}^H\mathbf{v}$ | `(1, 1)` | `ColVec` |
| `u @ v.T` | Outer / dyadic product $\mathbf{u}\mathbf{v}^T$ | `(n, m)` | `Mat` |
| `A @ _c[...]` | Matrix-vector product | `(n, 1)` | `ColVec` |
| `A @ B` | Matrix-matrix product | `(n, m)` | `Mat` |
| `A * B` | Hadamard product (same shape only) | `(n, k)` | `Mat` |
| `3 * u` | Scalar multiplication | `(n, 1)` | `ColVec` |
| `u + v` | Vector addition (same shape only) | `(n, 1)` | `ColVec` |
| `A.inv` | Matrix inverse $A^{-1}$ | `(n, n)` | `Mat` |
| `A.det` | Determinant $\det(A)$ | scalar | `float` |
| `np.kron(A, B)` | Kronecker product $A \otimes B$ | varies | `ndarray` (re-wrap) |
| `np.einsum(...)` | Arbitrary contraction | varies | `ndarray` (re-wrap) |

### Conversions

| From | Method / Function | To | Notes |
|---|---|---|---|
| `ColVec` | `.to_numpy()` | `ndarray (n,1)` | Strips subclass label |
| `ColVec` | `.to_flat()` | `ndarray (n,)` | For scipy.optimize, pd.Series |
| `ColVec` | `.to_list()` | `list` | Flat list of floats |
| `ColVec` | `.to_series()` | `pd.Series` | Optional index, name |
| `Mat` | `.to_numpy()` | `ndarray (n,k)` | Strips subclass label |
| `Mat` | `.to_list()` | nested `list` | List of row-lists |
| `Mat` | `.to_dataframe()` | `pd.DataFrame` | Optional columns, index |
| any array-like | `as_col(x)` | `ColVec` | Auto-reshapes |
| any 2D array-like | `as_mat(x)` | `Mat` | Row-first convention |

### Indexing

| Expression (on ColVec u) | Result type | Shape |
|---|---|---|
| `u[i]` | `float` | scalar |
| `u[i, 0]` | `float` | scalar |
| `u[i:j]` | `ColVec` | `(k, 1)` |
| `u[[i, j, k]]` | `ColVec` | `(3, 1)` |
| `u[mask]` | `ColVec` | `(k, 1)` |

| Expression (on Mat A) | Result type | Shape |
|---|---|---|
| `A[i, j]` | `float` | scalar |
| `A[:, j]` | `ColVec` | `(n, 1)` — **core contract: columns come back as ColVec** |
| `A[:, j:k]` | `Mat` | `(n, m)` — column slice |
| `A[:, [j, k]]` | `Mat` | `(n, 2)` — fancy column index |
| `A[:, mask]` | `Mat` | `(n, k)` — boolean column mask |
| `A[i, :]` | `Mat` | `(1, k)` — row as row matrix |
| `A[i:j, :]` | `Mat` | `(m, k)` — row submatrix |
| `A[i:j, p:q]` | `Mat` or `ColVec` | `(m, r)` — ColVec if single column |

### Interoperability

| Library | Compatible? | Notes |
|---|---|---|
| `numpy` ufuncs | Yes | Full compatibility, returns `ColVec`/`Mat` via `__array_wrap__` |
| `numpy.linalg` | Yes | Returns plain `ndarray` — re-wrap if needed |
| `scipy.linalg` | Yes | Returns plain `ndarray` — re-wrap if needed |
| `scipy.optimize` | Partial | Flatten `ColVec` to 1D for parameter vectors |
| `scipy.stats` | Partial | Returns 1D arrays — reshape or re-wrap as needed |
| `scipy.sparse` | No | Dense only |
| `pandas.DataFrame` | Yes | Pass `Mat` directly |
| `pandas.Series` | Partial | Flatten `ColVec` to 1D first |
| Complex dtypes | Partial | Use `ColVec(np.array(...))` directly; `.H` for conjugate transpose |

---

## 16. What This Is Not

Each exclusion traces to the three design goals (Section 1.2). If a feature does not
serve shape clarity, syntactic convenience, or notational correctness for rank-1 and
rank-2 objects, it is out of scope.

- Not a tensor algebra library — `nemopy` stops at rank 2 because the three goals do not
  extend to higher orders (Section 1.4). Use `np.einsum`, `opt_einsum`, or `tensorly`.
- Not a symbolic algebra system — use SymPy for that.
- Not a replacement for NumPy — coexistence is the goal; all NumPy functions remain
  accessible, and conversion functions (Section 13) provide clean entry/exit points.
- Not a GPU library — pure NumPy backend throughout.
- Not a sparse matrix library — dense arrays only.
- Not a type-safe system — Python has no `const`; `_c` can be shadowed, though the
  underscore prefix makes this unlikely in practice.

---

## 17. MATLAB-style String Constructor `_m`

### 17.1 Rationale

Python does not allow `[1, 2, 3; 4, 5, 6]` as a literal — semicolons are
not valid inside list brackets. Users coming from MATLAB / Octave / Julia
frequently want a one-line literal for a matrix. `_m` provides the closest
viable Python syntax: a singleton with `__getitem__` that takes a single
string and parses it.

`_m` is column-first by default to match nemopy's overall convention
(§3, §5). The `.T` property on the resulting `Mat` yields the row-first
MATLAB interpretation for users who want it.

### 17.2 Surface

```python
_m: _MatConstructor    # singleton in nemopy._constructors

# Forms that all parse identically
A = _m["1, 2, 3; 4, 5, 6; 7, 8, 9"]
B = _m["[1, 2, 3; 4, 5, 6; 7, 8, 9]"]   # MATLAB-true brackets tolerated
C = _m["1 2 3; 4 5 6; 7 8 9"]            # whitespace separators
```

All three of `A`, `B`, `C` are `Mat(3, 3)` whose **columns** are
`[1, 2, 3]`, `[4, 5, 6]`, `[7, 8, 9]`.

### 17.3 Grammar

```
expr        := optional-brackets columns optional-brackets
columns     := column ( ';' column )*
column      := element ( separator element )*
separator   := ',' | whitespace+
element     := signed numeric token convertible to float
optional-brackets := '[' ... ']'   (only at the very start and end)
```

### 17.4 Behaviour and errors

| Input                          | Result                             |
|--------------------------------|------------------------------------|
| `_m["1, 2; 3, 4"]`             | `Mat(2, 2)` cols `[1,2]`, `[3,4]`  |
| `_m["1 2; 3 4"]`               | same as above (whitespace = comma) |
| `_m["[1, 2; 3, 4]"]`           | same as above (brackets stripped)  |
| `_m["1, 2, 3"]`                | `Mat(3, 1)` (single column)        |
| `_m["1, 2; 3, 4, 5"]`          | raises `ValueError` (ragged)       |
| `_m["1, a; 3, 4"]`             | raises `TypeError` (non-numeric)   |
| `_m[""]`, `_m["[]"]`           | raises `ValueError` (empty)        |
| `_m["1, 2; ; 3, 4"]`           | raises `ValueError` (empty col)    |
| `_m[1, 2, 3]` (not a string)   | raises `TypeError`                 |

### 17.5 Row-first interpretation via `.T`

For users who want the MATLAB-true row-first reading, transpose the result:

```python
# columns [1, 2, 3] and [4, 5, 6]  — nemopy column-first default
A = _m["1, 2, 3; 4, 5, 6"]            # Mat(3, 2)

# rows [1, 2, 3] and [4, 5, 6]  — MATLAB-true row-first
A_rows = _m["1, 2, 3; 4, 5, 6"].T     # Mat(2, 3)
```

### 17.6 dtype

Always `float64`, matching `_c[]` and `mat()`. Tokens are parsed via
`float(...)`, so any literal Python `float`-convertible string is accepted
(`"1e-3"`, `"-2.5"`, `"+0"`, `"inf"`, `"nan"`).

### 17.7 `__all__` registration

`_m` is added to `__all__` in `nemopy/__init__.py` alongside `_c`.

---

## Appendix A. Documentation Specification

*This appendix specifies the documentation tooling and format. It is an implementation
deliverable, not part of the design specification. The docstring content templates below
are normative (they define the documentation contract); the Sphinx configuration is
advisory (it may be adapted to the implementer's environment).*

### A.1 Tooling: Sphinx, Not Pandoc

Pandoc is a format converter (Markdown → PDF, LaTeX → HTML, etc.). It does not generate
API documentation with cross-references, hover tooltips, or searchable indices.

The correct tool is **Sphinx** with the following extensions:

| Extension | Purpose |
|---|---|
| `sphinx.ext.autodoc` | Extracts docstrings from source code into HTML docs |
| `sphinx.ext.napoleon` | Parses NumPy-style docstrings (see Section A.3) |
| `sphinx.ext.intersphinx` | Links to NumPy/SciPy docs so that `np.ndarray` in hover resolves to NumPy's own docs |
| `sphinx.ext.viewcode` | Adds "[source]" links from docs to highlighted source |
| `sphinx_rtd_theme` | Read the Docs theme — standard, readable, supports hover cross-refs |

### A.2 Hover Behaviour

Sphinx's HTML output provides hover tooltips in two ways:

1. **Cross-references within the project:** When a docstring mentions `:class:\`ColVec\``
   or `:func:\`mat\``, Sphinx generates a hyperlink. On hover, the browser shows a tooltip
   with the target's short description. This works in the generated HTML documentation site.

2. **IDE integration:** VS Code, PyCharm, and other editors parse docstrings directly from
   source. When a user hovers over `_c[...]` or `mat(...)` in their own code, the IDE
   displays the docstring in a popup. This requires no Sphinx build — it reads the
   docstrings at edit time.

**Both channels read the same docstrings.** Writing good docstrings once serves both the
HTML documentation site and the IDE hover experience.

### A.3 Docstring Format: NumPy Style

All public symbols must use NumPy-style docstrings. Napoleon (the Sphinx extension) parses
this format into structured HTML with parameter tables, return types, and example blocks.

**Required sections per symbol type:**

| Symbol type | Required sections |
|---|---|
| Class (`ColVec`, `Mat`) | Short summary, Parameters, Attributes, Raises, Examples, See Also |
| Function (`mat`, `eye`) | Short summary, Parameters, Returns, Raises, Examples, See Also |
| Singleton (`_c`) | Short summary, Examples, Notes, Warns |
| Property (`.inv`, `.det`, `.is_singular`, `.H`) | Short summary, Returns, Raises, Examples, See Also |
| Exception (`ShapeError`) | Short summary, Notes |
| Warning (`ConventionWarning`) | Short summary, Notes |

### A.4 Sphinx Configuration Skeleton

*Implementation suggestion — adapt as needed:*

The agent should generate a `docs/` directory with the following:

```
docs/
├── conf.py
├── index.rst
├── api.rst
└── Makefile         # optional — standard Sphinx makefile
```

**`conf.py` requirements:**

```python
project = "nemopy"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
]
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}
napoleon_numpy_docstring = True
napoleon_google_docstring = False
autodoc_member_order = "bysource"
html_theme = "sphinx_rtd_theme"
```

**`api.rst`:**

```rst
API Reference
=============

Constructors
------------

.. autodata:: nemopy._c
   :no-value:

.. autofunction:: nemopy.mat

.. autofunction:: nemopy.eye

Types
-----

.. autoclass:: nemopy.ColVec
   :members:
   :inherited-members:

.. autoclass:: nemopy.Mat
   :members:
   :inherited-members:

Exceptions
----------

.. autoexception:: nemopy.ShapeError

.. autoexception:: nemopy.ConventionWarning
```

### A.5 Doctest Integration

All docstring examples must be valid doctests. The Sphinx build should run:

```bash
sphinx-build -b doctest docs/ docs/_build/doctest
```

This verifies that every example in the documentation actually produces the stated output.
The agent should ensure that `__repr__` output (Section 4.2, 4.3) exactly matches what
appears in the docstring examples.

---

## Appendix B. Implementation Checklist for Agent

*This appendix provides a sequenced implementation plan. It is guidance for the coding
agent, not part of the design specification. The verification criteria below are derived
from the behavioural contracts in the main document.*

Each step has a verification criterion.

1. **Define `ShapeError` and `ConventionWarning`**
   → Verify: `isinstance(ShapeError(), ValueError)` is `True`

2. **Define `_is_scalar()` and `_check_shapes()`**
   → Verify: `_check_shapes(np.ones((3,1)), 5.0, "*")` does not raise;
     `_check_shapes(np.ones((3,1)), np.ones((2,1)), "*")` raises `ShapeError`

3. **Define `_VecBase(np.ndarray)`** with `__array_finalize__`, `__array_wrap__`
   (or `__array_ufunc__` per Section 4.5), all operator overrides (`__mul__`,
   `__rmul__`, `__add__`, `__radd__`, `__sub__`, `__rsub__`, `__truediv__`,
   `__rtruediv__`, `__iadd__`, `__isub__`, `__imul__`, `__itruediv__`,
   `__matmul__`, `__rmatmul__`), and the `.H` property
   → Verify: Can be subclassed; operator overrides call `_check_shapes`;
     in-place operators preserve subclass label; `.H` returns conjugate transpose

4. **Define `ColVec(_VecBase)`** with `__new__`, `__repr__`, `__getitem__`,
   and outbound conversions (`to_numpy`, `to_flat`, `to_list`, `to_series`)
   → Verify: `ColVec(np.array([[1],[2],[3]]))`  has shape `(3,1)`;
     `ColVec(np.array([1,2,3]))` raises `ShapeError`

5. **Define `Mat(_VecBase)`** with `__new__`, `__repr__`, `__getitem__`,
   outbound conversions (`to_numpy`, `to_list`, `to_dataframe`),
   and properties (`.inv`, `.det`, `.is_singular`)
   → Verify: `Mat(np.eye(3))` has shape `(3,3)`;
     `Mat(np.array([1,2,3]))` raises `ShapeError`;
     `Mat(np.eye(3)).inv` returns identity;
     `Mat(np.eye(3)).det` returns `1.0`;
     `Mat(np.eye(3)).is_singular` returns `False`;
     `mat([1,2],[2,4]).is_singular` returns `True`;
     non-square `.inv` raises `ShapeError`

6. **Define `_ColConstructor` and instantiate `_c`**
   → Verify: `_c[1,2,3]` is `ColVec` with shape `(3,1)`;
     `_c[[1,2,3]]` raises `ValueError`

7. **Define `_to_colvec()`, `mat()`, and `eye()`**
   → Verify: `mat([1,2],[3,4])` has shape `(2,2)` and type `Mat`;
     `mat([1,2],[3,4,5])` raises `ValueError`;
     `eye(3)` has shape `(3,3)` and type `Mat`;
     `eye(3)` equals `Mat(np.eye(3))`

8. **Define `as_col()` and `as_mat()`** module-level inbound converters
   → Verify: `as_col([1,2,3])` is `ColVec` with shape `(3,1)`;
     `as_col(pd.Series([4,5]))` is `ColVec` with shape `(2,1)`;
     `as_mat([[1,2],[3,4]])` is `Mat` with shape `(2,2)`;
     `as_col(np.ones((3,3)))` raises `ShapeError`

9. **Write test suite** covering:
   - All constructor paths (scalars, lists, tuples, mixed, edge cases)
   - All operator overrides (shape match, shape mismatch, scalar, each operator)
   - **In-place operators:** `+=`, `-=`, `*=`, `/=` with shape guard and type preservation
   - **Comparison operators:** `==`, `<`, `>` return plain ndarray, no shape guard
   - Indexing (scalar extraction, slicing, fancy indexing, boolean masking)
   - **Column extraction specifically:** `A[:, j]` → `ColVec`; `A[:, j:k]` → `Mat`;
     `A[:, [j, k]]` → `Mat`; `A[i, :]` → `Mat` (row, not ColVec);
     verify extracted column is usable in `@` without reshape
   - **Transpose:** `u.T` type and shape for various dimensions;
     `A.T` type and shape; `.T` is a view (modification propagates)
   - **Conjugate transpose:** `u.H` equals `u.conj().T`; for real arrays, `.H == .T`
   - **Matrix properties:** `.inv` on identity, on known invertible, on singular
     (expect `LinAlgError`), on non-square (expect `ShapeError`);
     `.det` on identity (expect `1.0`), on singular (expect `0.0`);
     `.is_singular` on identity (`False`), on rank-deficient (`True`)
   - **Identity constructor:** `eye(n)` shape, type, values
   - **Tensor products:** `u @ v.T` produces `Mat`; `np.kron(A, B)` accepts `Mat`
   - **Conversion round-trips:** `as_col(u.to_list())` recovers original values;
     `as_mat(A.to_numpy())` recovers original values
   - `ConventionWarning` emission on `@` with transposed-looking plain ndarray
   - NumPy interop (`np.linalg.solve`, `np.exp`, `np.sum`)
   - **`.copy()` subclass persistence**
   - `__repr__` output format
   - Doctests pass: `python -m doctest nemopy/_core.py nemopy/_constructors.py`

10. **Write `__init__.py`** exporting `__all__`
    → Verify: `from nemopy import _c, mat, eye, as_col, as_mat, ColVec, Mat, ShapeError, ConventionWarning`

11. **Write docstrings** conforming to Appendix A templates on all public symbols
    → Verify: `python -c "from nemopy import _c; help(_c)"` shows full docstring with
    Examples, Notes, See Also

12. **Set up Sphinx documentation** per Appendix A
    → Verify: `sphinx-build -b html docs/ docs/_build/html` succeeds;
    `sphinx-build -b doctest docs/ docs/_build/doctest` passes all examples

---

## 18. Column-Join Operator `|`

### 18.1 Rationale

Python cannot parse `[a b c; d e f]` as a literal — semicolons are not valid
inside expressions. The `_m["..."]` string constructor handles literal matrices,
but requires string tokens, preventing use of runtime variables.

The `|` operator provides a Python-native column-join syntax that matches the
mathematical notation `A = [col₁ | col₂ | col₃]`:

```python
A = _c[a, b, c] | _c[d, e, f] | _c[g, h, i]   # Mat(3,3)
```

### 18.2 Semantics

```
ColVec | ColVec  →  Mat  (two-column, horizontally stacked)
ColVec | Mat     →  Mat  (column prepended on left)
Mat    | ColVec  →  Mat  (column appended on right)
Mat    | Mat     →  Mat  (horizontal stack, same row count required)
```

Shape guard: if row counts differ, `ShapeError` is raised.

### 18.3 Implementation

Defined as `__or__` and `__ror__` on `_VecBase` in `nemopy/_operators.py`.
Uses `np.hstack` internally.

```python
def __or__(self, other):
    if np.shape(self)[0] != np.shape(other)[0]:
        raise ShapeError(...)
    return Mat(np.hstack([np.asarray(self), np.asarray(other)]))
```

### 18.4 Behavioural contract

| Expression | Result type | Shape |
|---|---|---|
| `_c[1,2,3] \| _c[4,5,6]` | `Mat` | `(3, 2)` |
| `_c[1,2,3] \| _c[4,5,6] \| _c[7,8,9]` | `Mat` | `(3, 3)` |
| `A \| _c[1,2,3]` where A is `Mat(3,2)` | `Mat` | `(3, 3)` |
| `_c[1,2] \| _c[1,2,3]` | — | `ShapeError` (row mismatch) |

---

## 19. Polars Integration

### 19.1 Inbound (`as_col`, `as_mat`)

`as_col` and `as_mat` accept `polars.Series` and `polars.DataFrame` respectively,
using the same optional-import guard as the existing pandas branches:

```python
try:
    import polars as pl
    if isinstance(x, pl.Series):
        return ColVec(x.to_numpy().astype(float).reshape(-1, 1))
except ImportError:
    pass
```

### 19.2 Outbound (`.to_polars()`)

Both `ColVec` and `Mat` gain a `.to_polars()` method:

- `ColVec.to_polars(name=None)` → `polars.Series`
- `Mat.to_polars(schema=None)` → `polars.DataFrame`

These follow the same `ImportError`-on-missing-polars pattern as the pandas
methods.

### 19.3 Accessor (`nemopy.polars`)

Importing `nemopy.polars` registers two polars accessors:

- `df.nemo.col(name)` → `ColVec`
- `df.nemo.mat(names)` → `Mat`
- `series.nemo.col()` → `ColVec`

The accessor module is in `nemopy/polars.py` and raises `ImportError` at import
time if polars is not installed.

### 19.4 Optional dependency

Polars is an optional dependency declared in `pyproject.toml`:

```toml
[project.optional-dependencies]
polars = ["polars>=0.19"]
```

Install with: `pip install "nemopy[polars]"`

---

## 20. Feature Roadmap and Rust Acceleration

> **Status:** roadmap. This section records the agreed direction and the
> contract every roadmap feature must satisfy. Signatures shown in the
> referenced issues are provisional until a feature is implemented, at which
> point its final surface is promoted into the appropriate numbered section
> of this spec and becomes immutable under §1 of `CLAUDE.md`.

### 20.1 Strategy

nemopy keeps NumPy as its computational base and grows a Rust extension
layer (`_rust_core`, PyO3 + maturin) that progressively takes over the hot
paths. NumPy is the BLAS; Rust is the application layer. Features ship in
two tiers:

1. **Fast path (required):** a Rust kernel in `_rust_core`. Every roadmap
   feature below is only considered useful at real-world scale with its
   Rust implementation — this is a hard requirement, not an optimization.
2. **Fallback path (required):** a pure-Python/NumPy implementation with
   identical semantics. `nemopy` must remain pip-installable with `numpy`
   as the only hard dependency. The extension accelerates; it never gates
   functionality.

Both paths are exercised by the same test suite. A feature is not complete
until results agree across paths within documented tolerances.

### 20.2 Rust core layout

```
nemopy/
├── _rust_core/          # PyO3 extension module (maturin-managed)
│   ├── src/
│   │   ├── lib.rs       # Module entry point
│   │   ├── ops.rs       # Fused shape-guard arithmetic, batch kernels
│   │   ├── stats.rs     # Column-wise statistics, EWMA/rolling covariance
│   │   ├── decomp.rs    # Decompositions (SVD, QR, LU, LDU, Schur, Jordan,
│   │   │                #   Polar, QDR, eigen, power iteration)
│   │   ├── linalg.rs    # Gaussian elimination, REF/RREF, rank, nullspace
│   │   ├── optim.rs     # Simplex, dual simplex, interior point,
│   │   │                #   branch-and-bound (LP/IP/MILP)
│   │   ├── network.rs   # Graph algorithms (shortest path, max flow,
│   │   │                #   min cost flow, assignment, MST)
│   │   ├── markov.rs    # Markov chain analysis and simulation
│   │   ├── sim.rs       # Monte Carlo, queueing, random variates
│   │   └── lazy.rs      # Expression-graph optimizer and fused executor
│   └── Cargo.toml
└── ...
```

Key crates: `faer` (pure-Rust linear algebra, no system LAPACK), `rayon`
(data parallelism), `rand` (simulation RNG), PyO3 `numpy` (zero-copy
buffer interop).

### 20.3 Interop contract

- **Zero-copy:** `Mat`/`ColVec` buffers cross the FFI boundary as views via
  the PyO3 `numpy` crate. Kernels never copy inputs or outputs unless a
  layout change is mathematically required.
- **Type persistence:** Rust kernels return buffers that the Python layer
  wraps back into `Mat`/`ColVec` per the §4 shape rules. No plain ndarrays
  escape from nemopy APIs.
- **GIL:** kernels release the GIL for the duration of computation; Rayon
  provides intra-kernel parallelism.

### 20.4 Feature domains

| Domain | Issues | Rust modules |
|---|---|---|
| Core decompositions (SVD, QR, LU, Cholesky, eig, eigh) | #76 | `decomp.rs` |
| Advanced decompositions (LDU, QDR, Schur, Jordan, Polar, diagonalize) | #84 | `decomp.rs` |
| Gaussian elimination, Gauss-Jordan, REF/RREF, rank, nullspace | #85 | `linalg.rs` |
| Statistical methods (mean/std/var/norm/cov/corr, column-first) | #77 | `stats.rs` |
| DataFrame-as-matrix bridge (`NamedMat`) | #78 | `ops.rs` |
| Financial/risk primitives (portfolio, EWMA/rolling cov, factor models) | #79 | `stats.rs`, `sim.rs` |
| Batch operations (`BatchMat`, `BatchColVec`) | #80 | `ops.rs` |
| Lazy evaluation and expression optimization | #81 | `lazy.rs` |
| Markov chains, DTMC/CTMC, simulation | #86 | `markov.rs` |
| AHP/ANP multi-criteria decision methods | #87 | `decomp.rs` |
| Linear programming (Simplex, dual simplex, LP/IP/MILP) | #91 | `optim.rs` |
| Network optimization (shortest path, max flow, assignment) | #92 | `network.rs` |
| Stochastic OR (Monte Carlo, queueing, discrete-event simulation) | #93 | `sim.rs` |

### 20.5 Phases

1. PyO3/maturin scaffolding; one fused kernel benchmarked against the
   pure-Python path.
2. Hot-path replacement of the top profiled bottlenecks.
3. Decompositions (#76, #84) and elimination (#85).
4. LP/IP/MILP solvers (#91).
5. Network optimization (#92).
6. Stochastic simulation (#86, #93) with Rayon parallelism.
7. Financial/risk primitives (#79).
8. Optional GPU dispatch (wgpu/CUDA) — exploratory.

Ecosystem compatibility (pandas, polars, matplotlib, scikit-learn, scipy,
integration workflows) is validated continuously by the
`tests/test_compat_*.py` and `tests/test_integration_workflow.py` suites
and is a prerequisite for every phase: no Rust kernel may regress a compat
test.

### 20.6 Issue tracker as spec annex

Each issue listed in §20.4 carries the working API sketch, design
principles, and Rust kernel requirements for its domain, and is kept
current as design evolves. Issue #75 is the umbrella record for the Rust
core and its phase plan.
