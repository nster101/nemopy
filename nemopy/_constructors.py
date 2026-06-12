"""Constructors: _c singleton, _m singleton, mat(), eye(), as_col(), as_mat()."""

import warnings

import numpy as np

from nemopy._core import ColVec, ConventionWarning, Mat, ShapeError


class _ColConstructor:
    """Bracket-notation constructor for ``ColVec`` instances.

    ``_c`` is a module-level singleton of this class. Use subscript notation
    to build column vectors from scalar literals or scalar variables:

    .. code-block:: python

        _c[1, 2, 3]      →  ColVec of shape (3, 1)
        _c[x, y, z]      →  ColVec of shape (3, 1) from variables

    The leading underscore signals "do not reassign this name" — see Notes.

    Examples
    --------
    Basic construction:

    >>> _c[1, 2, 3]
    ColVec([1.0, 2.0, 3.0])

    Single element:

    >>> _c[5]
    ColVec([5.0])

    Negative values:

    >>> _c[-1, -2, -3]
    ColVec([-1.0, -2.0, -3.0])

    From scalar variables:

    >>> x, y, z = 1.0, 2.0, 3.0
    >>> _c[x, y, z]
    ColVec([1.0, 2.0, 3.0])

    Use directly in linear algebra:

    >>> u = _c[1, 2, 3]
    >>> v = _c[4, 5, 6]
    >>> (u.T @ v).item()
    32.0

    Notes
    -----
    ``_c`` is a **singleton instance**, not a class. This mirrors NumPy's
    ``np.c_`` and ``np.r_`` design. As an instance, the leading underscore
    serves as a convention signal: "do not reassign this name in your local
    scope." Writing ``_c = something_else`` in a local scope silently breaks
    all subsequent uses.

    ``_c[...]`` always produces ``dtype=float64``. For complex vectors, bypass
    ``_c`` and construct directly::

        ColVec(np.array([[1+2j], [3+4j]]))

    Raises
    ------
    ValueError
        If any element of the subscript is a ``list``, ``tuple``, or
        ``ndarray`` (use ``mat()`` for nested input).
    """

    def __getitem__(self, items):
        if not isinstance(items, tuple):
            items = (items,)
        if any(isinstance(i, (list, tuple, np.ndarray)) for i in items):
            raise ValueError(
                "_c[] takes a flat sequence of scalars. "
                "For a matrix, use mat(). "
                "Example: mat(_c[1,2,3], _c[4,5,6])"
            )
        return ColVec(np.array(items, dtype=float).reshape(-1, 1))

    def __repr__(self):
        return "_c"


_c = _ColConstructor()


class _MatConstructor:
    """Bracket-notation MATLAB-style string constructor for ``Mat`` instances.

    ``_m`` is a module-level singleton of this class. Pass a single string
    where columns are separated by ``';'`` and elements within each column
    are separated by commas or whitespace:

    .. code-block:: python

        _m["1 2 3; 4 5 6; 7 8 9"]

    produces a ``Mat(3, 3)`` whose **columns** are ``[1,2,3]``, ``[4,5,6]``,
    ``[7,8,9]``. This is **column-first**, matching nemopy's convention.
    For the row-first MATLAB reading, append ``.T``.

    Examples
    --------
    Column-first construction — three columns, three elements each:

    >>> _m["1 2 3; 4 5 6; 7 8 9"]
    Mat(3x3):
      [1, 4, 7]
      [2, 5, 8]
      [3, 6, 9]

    Commas as element separators (equivalent to whitespace):

    >>> _m["1, 2, 3; 4, 5, 6"]
    Mat(3x2):
      [1, 4]
      [2, 5]
      [3, 6]

    Optional surrounding brackets (MATLAB literal style):

    >>> _m["[1 2; 3 4]"]
    Mat(2x2):
      [1, 3]
      [2, 4]

    Row-first interpretation via ``.T`` (MATLAB convention):

    >>> _m["1 2 3; 4 5 6"].T
    Mat(2x3):
      [1, 2, 3]
      [4, 5, 6]

    Single column:

    >>> _m["1; 2; 3"]
    Mat(3x1):
      [1]
      [2]
      [3]

    Notes
    -----
    ``_m`` is a **singleton instance** for the same shadowing-safety reason
    as ``_c``. All tokens are parsed via ``float()``, so any float-convertible
    string (``"1e-3"``, ``"-2.5"``, ``"inf"``, ``"nan"``) is accepted.

    Raises
    ------
    TypeError
        If the subscript is not a string, or if a token cannot be parsed as
        a float.
    ValueError
        If the string is empty, columns have unequal lengths, or a column is
        empty (double or leading/trailing semicolon).
    """

    def __getitem__(self, item):
        if not isinstance(item, str):
            raise TypeError(
                "_m[] takes a single string. "
                'Example: _m["1, 2, 3; 4, 5, 6; 7, 8, 9"]'
            )
        text = item.strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1].strip()
        if not text:
            raise ValueError("_m[] received an empty string.")

        col_strs = [s.strip() for s in text.split(";")]
        cols = []
        for i, col_str in enumerate(col_strs):
            if not col_str:
                raise ValueError(
                    f"_m[] column {i} is empty. "
                    f"Use a single ';' between columns, no trailing or leading ';'."
                )
            tokens = [t for t in col_str.replace(",", " ").split() if t]
            try:
                values = [float(t) for t in tokens]
            except ValueError as exc:
                raise TypeError(
                    f"_m[] column {i} contains a non-numeric token: {exc}"
                ) from exc
            cols.append(values)

        lengths = {len(c) for c in cols}
        if len(lengths) > 1:
            raise ValueError(
                f"_m[] columns have unequal lengths: {[len(c) for c in cols]}. "
                f"All columns must have the same number of rows."
            )

        return Mat(np.array(cols, dtype=float).T)

    def __repr__(self):
        return "_m"


_m = _MatConstructor()


def _to_colvec(arg, index):
    """Convert a single mat() argument to ColVec.

    This is an internal helper — not exported.
    """
    if isinstance(arg, ColVec):
        return arg
    if isinstance(arg, (list, tuple)):
        arr = np.array(arg, dtype=float)
        if arr.ndim != 1:
            raise ValueError(
                f"mat() argument {index} is a nested list/tuple. "
                f"Each argument must be a flat sequence representing one column. "
                f"Got shape {arr.shape}."
            )
        return ColVec(arr.reshape(-1, 1))
    if isinstance(arg, np.ndarray):
        if arg.ndim == 1:
            warnings.warn(
                f"mat() argument {index} is a 1D ndarray of shape {arg.shape}. "
                f"Promoting to ColVec. If this came from np.array([...]), "
                f"verify it is not transposed relative to nemopy convention.",
                ConventionWarning,
                stacklevel=3,
            )
            return ColVec(arg.astype(float).reshape(-1, 1))
        if arg.ndim == 2 and arg.shape[1] == 1:
            return ColVec(arg.astype(float))
        raise TypeError(
            f"mat() argument {index} is a 2D ndarray with shape {arg.shape}. "
            f"Expected a column vector of shape (n,1). "
            f"If this is a plain NumPy matrix, it may be row-first — "
            f"check for transposition before passing to mat()."
        )
    raise TypeError(
        f"mat() argument {index} has unrecognised type {type(arg)}. "
        f"Expected _c[...], a list, a tuple, or a 1D/2D ndarray."
    )


def mat(*args):
    """Construct a ``Mat`` from column vectors.

    Each argument becomes one **column** of the resulting matrix. This is the
    column-first convention: ``mat(a, b, c)`` assembles ``A = [a | b | c]``,
    which is the mathematical convention and the inverse of NumPy's row-first
    ``np.array([[...], [...]])``.

    Parameters
    ----------
    *args : ColVec, list, tuple, or ndarray
        One argument per column. All columns must have the same number of rows.

        - ``ColVec``: used directly.
        - ``list`` or ``tuple``: must be flat (1D); nested raises ``ValueError``.
        - 1D ``numpy.ndarray``: promoted to ``ColVec`` with a ``ConventionWarning``
          (shape is ambiguous — may be transposed).
        - 2D ``numpy.ndarray`` of shape ``(n, 1)``: accepted silently.

    Returns
    -------
    Mat
        Shape ``(n, k)`` where ``n`` is the column length and ``k = len(args)``.

    Raises
    ------
    ValueError
        If called with no arguments, if columns have unequal lengths, or if
        an argument is a nested list or tuple.
    TypeError
        If an argument is an unrecognised type or a 2D ndarray that is not
        shape ``(n, 1)``.

    Warns
    -----
    ConventionWarning
        If any argument is a 1D ndarray (promoted to ``ColVec``, but may be
        transposed relative to nemopy convention).

    Examples
    --------
    Three columns as plain lists:

    >>> mat([1, 2, 3], [4, 5, 6], [7, 8, 9])
    Mat(3x3):
      [1, 4, 7]
      [2, 5, 8]
      [3, 6, 9]

    Using ``_c`` column vectors:

    >>> mat(_c[1, 2], _c[3, 4])
    Mat(2x2):
      [1, 3]
      [2, 4]

    Mixed inputs:

    >>> u = _c[1, 2, 3]
    >>> mat(u, [4, 5, 6])
    Mat(3x2):
      [1, 4]
      [2, 5]
      [3, 6]

    A single column returns ``Mat``, not ``ColVec``:

    >>> mat([1, 2, 3])
    Mat(3x1):
      [1]
      [2]
      [3]

    See Also
    --------
    _c : Construct a single column vector with bracket notation.
    _m : MATLAB-style string constructor for matrix literals.
    as_mat : Convert existing 2D data to a ``Mat`` (row-first convention).
    eye : Identity matrix constructor.
    """
    if len(args) == 0:
        raise ValueError("mat() requires at least one column argument.")

    cols = [_to_colvec(arg, i) for i, arg in enumerate(args)]
    lengths = [c.shape[0] for c in cols]

    if len(set(lengths)) > 1:
        raise ValueError(
            f"mat() columns have unequal lengths: {lengths}. "
            f"All columns must have the same number of rows."
        )

    stacked = np.hstack(cols)
    return Mat(stacked)


def eye(n, *args, **kwargs):
    """Construct an identity matrix.

    Wraps ``numpy.eye`` as a ``Mat``, accepting all of NumPy's ``eye()``
    parameters for compatibility. The output is always ``float64`` because
    ``Mat`` promotes all dtypes to ``float64``.

    Parameters
    ----------
    n : int
        Number of rows.
    *args, **kwargs
        Additional arguments forwarded to ``numpy.eye`` (e.g. ``M``,
        ``k``, ``dtype``). The ``dtype`` argument is accepted for
        compatibility but the output is always promoted to ``float64``
        by ``Mat``.

    Returns
    -------
    Mat
        Identity matrix with dtype ``float64``.

    Examples
    --------
    >>> eye(3)
    Mat(3x3):
      [1, 0, 0]
      [0, 1, 0]
      [0, 0, 1]

    The identity is its own inverse:

    >>> import numpy as np
    >>> np.allclose(eye(4).inv, eye(4))
    True

    Chain directly into expressions:

    >>> eye(2) @ _c[3, 5]
    ColVec([3.0, 5.0])

    See Also
    --------
    mat : Column-first matrix constructor.
    Mat : Direct constructor for 2D arrays.
    """
    return Mat(np.eye(int(n), *args, **kwargs))


def as_col(x):
    """Convert any array-like to a ``ColVec``.

    More permissive than the ``ColVec`` constructor: accepts 1D arrays, flat
    lists, scalars, ``(n, 1)`` 2D arrays, and ``pandas.Series``. Performs
    the necessary reshaping automatically. Use this when receiving data from
    external libraries (NumPy functions, pandas, polars) that return 1D
    arrays where a column vector is expected.

    Parameters
    ----------
    x : array-like
        Input data. Accepted forms:

        - Python scalar (``int``, ``float``): wrapped in a ``(1, 1)`` ColVec.
        - Flat ``list`` or ``tuple``: converted to ``(n, 1)``.
        - 1D ``numpy.ndarray`` of shape ``(n,)``: reshaped to ``(n, 1)``.
        - 2D ``numpy.ndarray`` of shape ``(n, 1)``: wrapped as-is.
        - ``pandas.Series``: values extracted and reshaped to ``(n, 1)``.
        - ``polars.Series``: values extracted and reshaped to ``(n, 1)``
          (when polars is installed).

    Returns
    -------
    ColVec
        Shape ``(n, 1)``, dtype ``float64``.

    Raises
    ------
    ShapeError
        If ``x`` is a 2D array with more than one column. Pass a single
        column explicitly: ``as_col(arr[:, j])``.
    TypeError
        If ``x`` cannot be converted to a numeric float array.

    Examples
    --------
    From a list:

    >>> as_col([1, 2, 3])
    ColVec([1.0, 2.0, 3.0])

    From a scalar:

    >>> as_col(42)
    ColVec([42.0])

    From a 1D NumPy array (no ``ConventionWarning``, unlike ``mat()``):

    >>> import numpy as np
    >>> as_col(np.array([7, 8, 9]))
    ColVec([7.0, 8.0, 9.0])

    From a NumPy function result:

    >>> import numpy as np
    >>> A = mat([1, 2, 3], [4, 5, 6])
    >>> as_col(np.sum(A.to_numpy(), axis=1))
    ColVec([5.0, 7.0, 9.0])

    See Also
    --------
    _c : Bracket-notation constructor for literals — shorter for inline use.
    ColVec : Direct constructor (requires shape ``(n, 1)`` exactly).
    as_mat : Analogous converter for 2D inputs.
    """
    if isinstance(x, (int, float, complex, np.generic)):
        if np.iscomplexobj(x):
            if np.imag(x) != 0:
                raise TypeError(
                    "as_col() cannot convert a complex scalar with a non-zero "
                    "imaginary part to float."
                )
            x = np.real(x)
        scalar = np.array([[x]], dtype=float)
        return ColVec(scalar)

    # polars Series
    try:
        import polars as pl

        if isinstance(x, pl.Series):
            try:
                return ColVec(x.to_numpy().astype(float).reshape(-1, 1))
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"as_col() could not convert polars Series to float."
                ) from exc
    except ImportError:
        pass

    # pandas Series
    try:
        import pandas as pd

        if isinstance(x, pd.Series):
            try:
                return ColVec(x.values.astype(float).reshape(-1, 1))
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"as_col() could not convert input of type {type(x)} to float."
                ) from exc
    except ImportError:
        pass

    try:
        arr = np.asarray(x, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"as_col() could not convert input of type {type(x)} to float.") from exc

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
    raise ShapeError(f"as_col() requires a 1D or (n,1) input, got ndim={arr.ndim}.")


def as_mat(x):
    """Convert any 2D array-like to a ``Mat``.

    Accepts 2D NumPy arrays, nested lists (row-first), ``pandas.DataFrame``,
    ``polars.DataFrame``, and existing ``Mat`` instances. Unlike ``mat()``,
    which takes separate column arguments in column-first order, ``as_mat``
    takes a single 2D object using NumPy's standard row-first layout.

    Use ``as_mat`` when converting data arriving from external sources;
    use ``mat()`` when building matrices from known column vectors.

    Parameters
    ----------
    x : array-like
        Input data. Must be convertible to a 2D numeric array. Accepted forms:

        - 2D ``numpy.ndarray``
        - Nested ``list`` or ``tuple`` of equal-length rows
        - ``pandas.DataFrame`` (numeric columns only)
        - ``polars.DataFrame`` (numeric columns only, when polars is installed)
        - Existing ``Mat`` instance

    Returns
    -------
    Mat
        Shape ``(n, k)``, dtype ``float64``.

    Raises
    ------
    ShapeError
        If ``x`` is not 2D after conversion.
    TypeError
        If ``x`` cannot be converted to a numeric float array.

    Notes
    -----
    ``as_mat`` uses **row-first** (NumPy) convention: each inner list is a row.
    This is the opposite of ``mat()``, which treats each argument as a column.
    The following two calls produce the same matrix::

        mat([1, 3], [2, 4])          # column-first: col0=[1,3], col1=[2,4]
        as_mat([[1, 2], [3, 4]])     # row-first:    row0=[1,2], row1=[3,4]

    Examples
    --------
    From nested lists (row-first):

    >>> as_mat([[1, 2], [3, 4], [5, 6]])
    Mat(3x2):
      [1, 2]
      [3, 4]
      [5, 6]

    From a 2D NumPy array:

    >>> import numpy as np
    >>> as_mat(np.eye(3))
    Mat(3x3):
      [1, 0, 0]
      [0, 1, 0]
      [0, 0, 1]

    Non-2D input raises ``ShapeError``:

    >>> as_mat([1, 2, 3])   # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    ShapeError: ...

    See Also
    --------
    mat : Column-first constructor from separate column arguments.
    Mat : Direct constructor (requires 2D input).
    as_col : Analogous converter for 1D inputs.
    """
    # polars DataFrame
    try:
        import polars as pl

        if isinstance(x, pl.DataFrame):
            try:
                return Mat(x.to_numpy().astype(float))
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "as_mat() could not convert polars DataFrame to float."
                ) from exc
    except ImportError:
        pass

    # pandas DataFrame
    try:
        import pandas as pd

        if isinstance(x, pd.DataFrame):
            try:
                return Mat(x.values.astype(float))
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"as_mat() could not convert input of type {type(x)} to float."
                ) from exc
    except ImportError:
        pass

    try:
        arr = np.asarray(x, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"as_mat() could not convert input of type {type(x)} to float.") from exc

    if arr.ndim != 2:
        raise ShapeError(
            f"as_mat() requires a 2D input, got ndim={arr.ndim} "
            f"with shape {arr.shape}."
        )
    return Mat(arr)
