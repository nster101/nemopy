"""Core types: ColVec, Mat, _VecBase, ShapeError, ConventionWarning."""

import numpy as np

_NPY_MAJOR = int(np.__version__.split(".")[0])


class ShapeError(ValueError):
    """Raised when array shapes are incompatible for the requested operation.

    A subclass of ``ValueError``, so it is caught by ``except ValueError``
    in existing code. Emitted by shape-guarded arithmetic operators,
    construction checks (``ColVec``, ``Mat``), and property guards
    (``.inv``, ``.det``, ``.is_singular``).

    Notes
    -----
    Use ``except ShapeError`` for targeted handling, or ``except ValueError``
    for broader compatibility with code that does not import nemopy.

    Examples
    --------
    >>> try:
    ...     _c[1, 2, 3] + _c[1, 2]
    ... except ShapeError as exc:
    ...     print(type(exc).__name__)
    ShapeError
    """

    pass


class ConventionWarning(UserWarning):
    """Emitted when a plain ndarray is used where a nemopy type is expected,
    indicating a possible row/column convention mismatch.

    A subclass of ``UserWarning``. Emitted in two situations:

    1. A 1D ``numpy.ndarray`` is passed to ``mat()`` — it is promoted to a
       ``ColVec``, but may be transposed relative to nemopy's column-first
       convention.
    2. A plain ndarray whose ``shape[0] < shape[1]`` (more columns than rows)
       is used as an operand of ``@`` with a ``ColVec`` or ``Mat`` — it may
       have been constructed row-first and not yet transposed.

    Notes
    -----
    To suppress intentionally in code that mixes plain arrays with nemopy types::

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConventionWarning)
            result = A @ some_numpy_array
    """

    pass


def _apply_type_rules(result):
    """Apply ColVec/Mat type persistence rules to an array result.

    Parameters
    ----------
    result : np.ndarray or scalar
        The result of a ufunc or operation.

    Returns
    -------
    ColVec, Mat, or the original result
        Type determined by output shape per §4.4.
    """
    if not isinstance(result, np.ndarray):
        return result
    if result.ndim == 2 and result.shape[1] == 1:
        return result.view(ColVec)
    if result.ndim == 2:
        return result.view(Mat)
    return np.asarray(result)


HANDLED_FUNCTIONS = {}


def implements(np_function):
    """Register an __array_function__ override."""
    def decorator(func):
        HANDLED_FUNCTIONS[np_function] = func
        return func
    return decorator


def _strip_vecbase(arg):
    if isinstance(arg, _VecBase):
        return np.asarray(arg)
    if isinstance(arg, (list, tuple)):
        return type(arg)(_strip_vecbase(a) for a in arg)
    return arg


class _VecBase(np.ndarray):
    """Non-public base class for ColVec and Mat.

    Holds shared ``__array_finalize__``, ufunc persistence, operator overrides,
    and the ``.T``, ``.H`` properties. Not exported. Not intended for direct
    instantiation.
    """

    def __array_finalize__(self, obj):
        pass

    if _NPY_MAJOR < 2:

        def __array_wrap__(self, out_arr, context=None, return_scalar=False):
            return _apply_type_rules(out_arr)

    else:

        def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
            args = [
                np.asarray(x) if isinstance(x, _VecBase) else x for x in inputs
            ]

            out = kwargs.pop("out", None)
            if out is not None:
                kwargs["out"] = tuple(
                    np.asarray(o) if isinstance(o, _VecBase) else o for o in out
                )

            results = getattr(ufunc, method)(*args, **kwargs)

            if method == "at":
                return

            if isinstance(results, tuple):
                return tuple(_apply_type_rules(r) for r in results)

            return _apply_type_rules(results)

    def __array_function__(self, func, types, args, kwargs):
        if not all(issubclass(t, _VecBase) for t in types):
            return NotImplemented
        if func in HANDLED_FUNCTIONS:
            return HANDLED_FUNCTIONS[func](*args, **kwargs)
        args_as_np = tuple(_strip_vecbase(a) for a in args)
        kwargs_as_np = {
            k: _strip_vecbase(v) for k, v in kwargs.items()
        }
        result = func(*args_as_np, **kwargs_as_np)
        if isinstance(result, np.ndarray):
            return _apply_type_rules(result)
        if isinstance(result, tuple):
            return tuple(
                _apply_type_rules(r) if isinstance(r, np.ndarray) else r
                for r in result
            )
        return result

    @property
    def T(self):
        """Transpose, with subclass label dispatched by output shape.

        Returns a **view** of the underlying data with axes reversed. The
        return type follows the shape → type rules: an output of shape
        ``(n, 1)`` becomes a ``ColVec``; any other 2D shape becomes a ``Mat``.
        NumPy's native ``.T`` is shadowed because it would preserve the source
        subclass label instead of relabelling by shape.

        Returns
        -------
        ColVec or Mat
            Type determined by output shape: ``(n, 1)`` → ``ColVec``;
            otherwise → ``Mat``.

        Notes
        -----
        ``.T`` is always a view — modifying the result modifies the original:

            >>> import numpy as np
            >>> u = _c[1, 2, 3]
            >>> row = u.T
            >>> np.shares_memory(u, row)
            True

        Examples
        --------
        Transposing a column vector yields a ``(1, n)`` row matrix:

        >>> _c[1, 2, 3].T
        Mat(1x3):
          [1, 2, 3]

        Transposing a matrix:

        >>> mat([1, 2, 3], [4, 5, 6]).T
        Mat(2x3):
          [1, 2, 3]
          [4, 5, 6]

        Inner product — the standard ``u^T v`` expression:

        >>> u = _c[1, 2, 3]
        >>> v = _c[4, 5, 6]
        >>> (u.T @ v).item()
        32.0

        Outer product:

        >>> u @ v.T
        Mat(3x3):
          [4, 5, 6]
          [8, 10, 12]
          [12, 15, 18]

        See Also
        --------
        H : Conjugate transpose (equals ``.T`` for real arrays).
        """
        return _apply_type_rules(np.asarray(self).transpose())

    def transpose(self, *axes):
        """Transpose, with subclass label dispatched by output shape.

        Semantic twin of ``.T`` covering the method spelling and, by extension,
        ``numpy.transpose(x)``. The return type follows the same shape → type
        rules as ``.T``.

        Returns
        -------
        ColVec or Mat
            Type determined by output shape.

        See Also
        --------
        T : Property spelling of the same operation.
        """
        return _apply_type_rules(np.asarray(self).transpose(*axes))

    @property
    def H(self):
        """Conjugate transpose (Hermitian adjoint).

        Returns ``self.conj().T``. For real arrays this is identical to ``.T``.
        For complex arrays it simultaneously conjugates all elements and
        transposes. Use ``.H`` wherever the mathematics requires :math:`A^H`
        — inner products in complex spaces, unitary matrices, Hermitian
        matrices.

        Returns
        -------
        ColVec or Mat
            Type determined by output shape (same rules as ``.T``).

        Notes
        -----
        For real arrays, ``.H`` and ``.T`` produce identical results because
        conjugation is the identity on the reals.

        Examples
        --------
        For a real column vector, ``.H`` equals ``.T``:

        >>> _c[1, 2, 3].H
        Mat(1x3):
          [1, 2, 3]

        For a real matrix, ``.H`` equals ``.T``:

        >>> mat([1, 2, 3], [4, 5, 6]).H
        Mat(2x3):
          [1, 2, 3]
          [4, 5, 6]

        See Also
        --------
        T : Plain transpose (no conjugation).
        """
        return self.conj().T


class ColVec(_VecBase):
    """Column vector of shape (n, 1) with dtype float64.

    The primary vector type in nemopy. All column vectors are strictly 2D
    arrays of shape ``(n, 1)``, eliminating the ambiguity between NumPy's
    ``(n,)``, ``(n, 1)``, and ``(1, n)`` representations.

    Parameters
    ----------
    input_array : array-like
        A 2D array of shape ``(n, 1)``. Any numeric dtype is accepted and
        promoted to ``float64``.

    Attributes
    ----------
    shape : tuple of int
        Always ``(n, 1)`` for some ``n >= 1``.
    dtype : numpy.dtype
        Always ``float64``.

    Raises
    ------
    ShapeError
        If ``input_array`` is not 2D with exactly one column.

    Examples
    --------
    Construct from literals using the ``_c`` bracket-notation constructor
    (the preferred form):

    >>> _c[1, 2, 3]
    ColVec([1.0, 2.0, 3.0])

    Wrap an existing ``(n, 1)`` array:

    >>> import numpy as np
    >>> ColVec(np.array([[4.0], [5.0], [6.0]]))
    ColVec([4.0, 5.0, 6.0])

    Single element:

    >>> _c[5]
    ColVec([5.0])

    Index and slice:

    >>> u = _c[10, 20, 30]
    >>> u[0]        # integer → scalar float
    10.0
    >>> u[1:]       # slice → ColVec
    ColVec([20.0, 30.0])

    Linear algebra:

    >>> u = _c[1, 2, 3]
    >>> v = _c[4, 5, 6]
    >>> (u.T @ v).item()   # inner product
    32.0
    >>> u @ v.T            # outer product → Mat
    Mat(3x3):
      [4, 5, 6]
      [8, 10, 12]
      [12, 15, 18]

    See Also
    --------
    _c : Bracket-notation constructor for literals — the usual way to build a ColVec.
    as_col : Flexible inbound converter (1D arrays, Series, scalars, lists).
    Mat : Matrix type for shape ``(n, k)`` with k ≥ 1.
    """

    def __new__(cls, input_array):
        arr = np.asarray(input_array, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 1:
            raise ShapeError(
                f"ColVec requires shape (n, 1), got {arr.shape}. "
                f"If you have a 1D array, reshape with arr.reshape(-1, 1)."
            )
        return arr.view(cls)

    def __repr__(self):
        vals = self.flatten().tolist()
        return f"ColVec({vals})"

    def __str__(self):
        return self.__repr__()

    def __getitem__(self, key):
        """Index a ColVec, enforcing column-vector semantics.

        Single-element extraction (integer key) returns a scalar float;
        structure-preserving indexing (slice, fancy index, boolean mask)
        returns a ``ColVec``.

        Parameters
        ----------
        key : int, slice, list, or ndarray
            NumPy-compatible index expression applied to the ``(n, 1)`` array.

        Returns
        -------
        float
            When ``key`` is an integer — element extracted as a Python float.
        ColVec
            When ``key`` is a slice, list, or boolean mask — result is a
            ``(k, 1)`` ColVec.

        Examples
        --------
        >>> u = _c[10, 20, 30, 40, 50]
        >>> u[0]            # integer → float
        10.0
        >>> u[1:4]          # slice → ColVec
        ColVec([20.0, 30.0, 40.0])
        >>> u[[0, 2, 4]]    # fancy index → ColVec
        ColVec([10.0, 30.0, 50.0])

        See Also
        --------
        Mat.__getitem__ : Indexing semantics for matrices.
        """
        if self.ndim != 2 or self.shape[1] != 1:
            return np.asarray(self)[key]

        if isinstance(key, (int, np.integer)):
            return float(super().__getitem__((key, 0)))

        result = super().__getitem__(key)

        if not isinstance(result, np.ndarray) or result.ndim == 0:
            return float(result)

        if result.ndim == 1:
            return ColVec(result.reshape(-1, 1))

        if result.ndim == 2 and result.shape[1] == 1:
            return result.view(ColVec)

        return np.asarray(result)

    def to_numpy(self):
        """Return a plain ndarray of shape (n, 1). Strips the ColVec subclass label.

        Use when passing to libraries that perform ``type(x) == np.ndarray``
        checks and reject ndarray subclasses.

        Returns
        -------
        numpy.ndarray
            Shape ``(n, 1)``, dtype ``float64``.

        Examples
        --------
        >>> u = _c[1, 2, 3]
        >>> import numpy as np
        >>> type(u.to_numpy()) is np.ndarray
        True
        >>> u.to_numpy().shape
        (3, 1)

        See Also
        --------
        to_flat : Returns shape ``(n,)`` (1D) instead of ``(n, 1)``.
        to_list : Returns a plain Python list.
        """
        return np.array(self)

    def to_flat(self):
        """Return a 1D ndarray of shape (n,).

        Use when interfacing with ``scipy.optimize``, ``pandas.Series``, or
        any API that expects a 1D parameter vector.

        Returns
        -------
        numpy.ndarray
            Shape ``(n,)``, dtype ``float64``.

        Examples
        --------
        >>> u = _c[1, 2, 3]
        >>> u.to_flat().shape
        (3,)

        See Also
        --------
        to_numpy : Returns shape ``(n, 1)`` with the subclass label stripped.
        to_series : Returns a ``pandas.Series``.
        """
        return np.asarray(self).flatten()

    def to_list(self):
        """Return a plain Python list of floats.

        Returns
        -------
        list of float
            Length ``n``, values as Python floats.

        Examples
        --------
        >>> _c[1, 2, 3].to_list()
        [1.0, 2.0, 3.0]

        See Also
        --------
        to_flat : Returns a 1D ndarray instead.
        to_numpy : Returns an ndarray of shape ``(n, 1)``.
        """
        return self.flatten().tolist()

    def to_series(self, index=None, name=None):
        """Return a pandas Series.

        Parameters
        ----------
        index : array-like, optional
            Index labels. Defaults to ``range(n)``.
        name : str, optional
            Name of the Series.

        Returns
        -------
        pandas.Series
            Length ``n``, dtype ``float64``.

        Raises
        ------
        ImportError
            If pandas is not installed.

        Examples
        --------
        >>> u = _c[10, 20, 30]
        >>> s = u.to_series(index=["a", "b", "c"], name="vals")
        >>> s.name
        'vals'
        >>> list(s.index)
        ['a', 'b', 'c']

        See Also
        --------
        to_flat : Extracts the underlying 1D ndarray (suitable as ``pd.Series`` input).
        to_polars : Returns a ``polars.Series`` instead.
        Mat.to_dataframe : Analogous conversion from ``Mat`` to ``DataFrame``.
        """
        import pandas as pd

        return pd.Series(self.flatten(), index=index, name=name)

    def to_polars(self, name=None):
        """Return a polars Series.

        Parameters
        ----------
        name : str, optional
            Name of the Series. Defaults to an empty string.

        Returns
        -------
        polars.Series
            Length ``n``, dtype ``Float64``.

        Raises
        ------
        ImportError
            If polars is not installed.

        Examples
        --------
        .. code-block:: python

            import polars as pl
            u = _c[1, 2, 3]
            s = u.to_polars(name="x")
            # polars.Series: name='x', values=[1.0, 2.0, 3.0]

        See Also
        --------
        to_series : Returns a ``pandas.Series`` instead.
        to_flat : Returns a plain 1D ndarray.
        """
        import polars as pl

        return pl.Series(name=name or "", values=self.flatten().tolist())


class Mat(_VecBase):
    """Matrix of shape (n, k) with dtype float64.

    The matrix type in nemopy. Always 2D, always ``float64``. Construct
    column-first with ``mat(col1, col2, ...)`` — each argument is one
    *column* of the result. Alternatively, use the MATLAB-style string
    constructor ``_m["..."]`` or the flexible inbound converter ``as_mat()``.

    Parameters
    ----------
    input_array : array-like
        A 2D array of any shape ``(n, k)``. Any numeric dtype is accepted
        and promoted to ``float64``.

    Attributes
    ----------
    shape : tuple of int
        ``(n, k)`` — rows × columns.
    dtype : numpy.dtype
        Always ``float64``.

    Raises
    ------
    ShapeError
        If ``input_array`` is not 2D.

    Examples
    --------
    Column-first construction — each argument is a column:

    >>> A = mat([1, 2, 3], [4, 5, 6])
    >>> A
    Mat(3x2):
      [1, 4]
      [2, 5]
      [3, 6]

    Column extraction returns a ``ColVec``, usable directly in ``@``:

    >>> A[:, 0]
    ColVec([1.0, 2.0, 3.0])

    Matrix multiplication:

    >>> A.T @ A
    Mat(2x2):
      [14, 32]
      [32, 77]

    Matrix inverse and determinant:

    >>> B = mat([1, 0], [0, 2])
    >>> B.inv
    Mat(2x2):
      [1, 0]
      [0, 0.5]
    >>> B.det
    2.0

    See Also
    --------
    mat : Column-first constructor (preferred over wrapping ``Mat`` directly).
    _m : MATLAB-style string constructor ``_m["1 2; 3 4"]``.
    as_mat : Flexible inbound converter for DataFrames, 2D arrays, nested lists.
    eye : Identity matrix constructor.
    """

    def __new__(cls, input_array):
        arr = np.asarray(input_array, dtype=float)
        if arr.ndim != 2:
            raise ShapeError(
                f"Mat requires a 2D array, got ndim={arr.ndim} with shape {arr.shape}."
            )
        return arr.view(cls)

    def __getitem__(self, key):
        """Index a Mat, enforcing column-extraction semantics.

        Single-column extraction (``A[:, j]``) returns a ``ColVec`` usable
        directly in ``@`` without reshape. Single-element extraction returns
        ``float``. Multi-column and row results are typed as ``Mat``.

        Parameters
        ----------
        key : int, slice, tuple, list, or ndarray
            NumPy-compatible index expression.

        Returns
        -------
        float
            When ``key`` selects a single scalar element.
        ColVec
            When ``key`` selects a single column (``A[:, j]``) or produces
            any 2D result with exactly one column.
        Mat
            When ``key`` selects a multi-column submatrix or a row.

        Examples
        --------
        >>> A = mat([1, 2, 3], [4, 5, 6], [7, 8, 9])
        >>> A[1, 2]      # scalar element
        8.0
        >>> A[:, 0]      # single column → ColVec
        ColVec([1.0, 2.0, 3.0])
        >>> A[:, 1:]     # multi-column slice → Mat
        Mat(3x2):
          [4, 7]
          [5, 8]
          [6, 9]
        >>> A[0, :]      # row → Mat
        Mat(1x3):
          [1, 4, 7]

        See Also
        --------
        ColVec.__getitem__ : Indexing semantics for column vectors.
        """
        if isinstance(key, np.ndarray) and key.dtype == np.bool_:
            return np.asarray(super().__getitem__(key))

        result = super().__getitem__(key)

        if not isinstance(result, np.ndarray) or result.ndim == 0:
            return float(result)

        if result.ndim == 1:
            if isinstance(key, tuple) and len(key) == 2:
                row_key, col_key = key
                if isinstance(col_key, (int, np.integer)):
                    return ColVec(result.reshape(-1, 1))
                if isinstance(row_key, (int, np.integer)):
                    return Mat(result.reshape(1, -1))
            return ColVec(result.reshape(-1, 1))

        if result.ndim == 2:
            if result.shape[1] == 1:
                return result.view(ColVec)
            return result.view(Mat)

        return np.asarray(result)

    def __repr__(self):
        rows = self.tolist()
        if rows and not isinstance(rows[0], list):
            rows = [rows]
        row_strs = [", ".join(f"{v:.6g}" for v in row) for row in rows]
        inner = "\n  ".join(f"[{r}]" for r in row_strs)
        nrows = len(rows)
        ncols = len(rows[0]) if rows else 0
        return f"Mat({nrows}x{ncols}):\n  {inner}"

    def __str__(self):
        return self.__repr__()

    @property
    def inv(self):
        """Matrix inverse A⁻¹.

        Returns
        -------
        Mat
            The inverse matrix, shape ``(n, n)``.

        Raises
        ------
        ShapeError
            If the matrix is not square.
        numpy.linalg.LinAlgError
            If the matrix is singular. Use ``.is_singular`` to guard before
            calling ``.inv`` if you need to handle this case.

        Examples
        --------
        The inverse of the identity is the identity:

        >>> eye(2).inv
        Mat(2x2):
          [1, 0]
          [0, 1]

        A diagonal matrix:

        >>> mat([2, 0], [0, 4]).inv
        Mat(2x2):
          [0.5, 0]
          [0, 0.25]

        Verify round-trip:

        >>> import numpy as np
        >>> A = mat([1, 2], [3, 4])
        >>> np.allclose(A @ A.inv, eye(2))
        True

        See Also
        --------
        is_singular : Check if the matrix has an inverse before calling ``.inv``.
        det : Determinant (zero iff singular).
        eye : Identity matrix.
        """
        if self.shape[0] != self.shape[1]:
            raise ShapeError(
                f"Only square matrices have inverses. "
                f"This matrix has shape {self.shape}."
            )
        return Mat(np.linalg.inv(self))

    @property
    def is_singular(self):
        """Whether the matrix is singular (non-invertible).

        Uses ``numpy.linalg.matrix_rank`` for numerical robustness. A square
        matrix is considered singular when its rank is less than its
        dimension. This is more reliable than testing ``abs(det) < tol`` for
        large matrices where the determinant can overflow or underflow.

        Returns
        -------
        bool
            ``True`` if the matrix is singular, ``False`` otherwise.

        Raises
        ------
        ShapeError
            If the matrix is not square.

        Examples
        --------
        >>> eye(3).is_singular
        False

        >>> mat([1, 2], [2, 4]).is_singular   # col1 = 2 * col0
        True

        See Also
        --------
        inv : Raises ``LinAlgError`` on singular matrices.
        det : Determinant (zero iff singular, less numerically robust).
        """
        if self.shape[0] != self.shape[1]:
            raise ShapeError(
                f"Singularity is defined only for square matrices. "
                f"This matrix has shape {self.shape}."
            )
        return int(np.linalg.matrix_rank(self)) < self.shape[0]

    @property
    def det(self):
        """Determinant of the matrix.

        Returns
        -------
        float
            The determinant as a Python ``float``.

        Raises
        ------
        ShapeError
            If the matrix is not square.

        Examples
        --------
        >>> eye(3).det
        1.0

        >>> mat([1, 2], [3, 4]).det
        -2.0

        A singular matrix has determinant zero:

        >>> mat([1, 2], [2, 4]).det
        0.0

        See Also
        --------
        is_singular : Numerically robust singularity check (preferred over ``det == 0``).
        inv : Matrix inverse (exists iff ``det != 0``).
        """
        if self.shape[0] != self.shape[1]:
            raise ShapeError(
                f"Determinant is defined only for square matrices. "
                f"This matrix has shape {self.shape}."
            )
        return float(np.linalg.det(self))

    def to_numpy(self):
        """Return a plain ndarray of shape (n, k). Strips the Mat subclass label.

        Use when passing to libraries that perform ``type(x) == np.ndarray``
        checks and reject ndarray subclasses.

        Returns
        -------
        numpy.ndarray
            Shape ``(n, k)``, dtype ``float64``.

        Examples
        --------
        >>> A = mat([1, 2], [3, 4])
        >>> import numpy as np
        >>> type(A.to_numpy()) is np.ndarray
        True

        See Also
        --------
        to_list : Returns a nested Python list of rows.
        to_dataframe : Returns a ``pandas.DataFrame``.
        """
        return np.array(self)

    def to_list(self):
        """Return a nested list (list of rows, each a list of floats).

        Returns
        -------
        list of list of float
            Outer list has ``n`` elements (rows); each inner list has ``k``
            elements (column values for that row).

        Examples
        --------
        >>> mat([1, 2], [3, 4]).to_list()
        [[1.0, 3.0], [2.0, 4.0]]

        See Also
        --------
        to_numpy : Returns a plain ndarray instead.
        to_dataframe : Returns a ``pandas.DataFrame``.
        """
        return self.tolist()

    def to_dataframe(self, columns=None, index=None):
        """Return a pandas DataFrame.

        Parameters
        ----------
        columns : list of str, optional
            Column labels. Defaults to integer range ``[0, 1, ..., k-1]``.
        index : array-like, optional
            Row index labels. Defaults to integer range ``[0, 1, ..., n-1]``.

        Returns
        -------
        pandas.DataFrame
            Shape ``(n, k)``.

        Raises
        ------
        ImportError
            If pandas is not installed.

        Examples
        --------
        >>> A = mat([1, 2, 3], [4, 5, 6])
        >>> df = A.to_dataframe(columns=["x", "y"])
        >>> list(df.columns)
        ['x', 'y']
        >>> df.shape
        (3, 2)

        See Also
        --------
        to_polars : Returns a ``polars.DataFrame`` instead.
        as_mat : Convert a DataFrame back to a ``Mat``.
        to_numpy : Returns a plain ndarray instead.
        """
        import pandas as pd

        return pd.DataFrame(np.asarray(self), columns=columns, index=index)

    def to_polars(self, schema=None):
        """Return a polars DataFrame.

        Parameters
        ----------
        schema : list of str or dict, optional
            Column names (or a polars schema). Defaults to ``["col_0", "col_1", ...]``.

        Returns
        -------
        polars.DataFrame
            Shape ``(n, k)``.

        Raises
        ------
        ImportError
            If polars is not installed.

        Examples
        --------
        .. code-block:: python

            import polars as pl
            A = mat([1, 2, 3], [4, 5, 6])
            df = A.to_polars(schema=["x", "y"])
            # polars.DataFrame with columns x, y

        See Also
        --------
        to_dataframe : Returns a ``pandas.DataFrame`` instead.
        as_mat : Convert a polars DataFrame back to a ``Mat``.
        """
        import polars as pl

        arr = np.asarray(self)
        if schema is None:
            schema = [f"col_{i}" for i in range(arr.shape[1])]
        return pl.from_numpy(arr, schema=schema)


def _load_rust_core():
    """Import the optional ``_rust_core`` extension (issue #75).

    Returns the compiled module when it is importable and structurally
    valid, else ``None``. The bare crate source directory
    ``nemopy/_rust_core/`` is importable as an empty namespace package,
    so validity is checked by attribute rather than import success alone.
    """
    import importlib

    try:
        ext = importlib.import_module("nemopy._rust_core")
    except ImportError:
        return None
    if not hasattr(ext, "rust_core_version"):
        return None
    ext.register_shape_error(ShapeError)
    return ext


_RUST = _load_rust_core()


def _require_rust(method):
    """Return the loaded ``_rust_core`` extension or raise ``ImportError``.

    Tier-3 features (DESIGN_APPENDICES.md §20.1/§20.4) have no NumPy
    equivalent, so when the compiled extension is absent there is nothing
    to fall back to. Raises a clear ``ImportError`` naming ``method`` and
    pointing at the build step instead of computing a substitute answer.
    """
    if _RUST is None:
        raise ImportError(
            f"Mat.{method}() requires the compiled nemopy._rust_core "
            f"extension, which is not available. Build it with "
            f"`maturin develop` to enable this Tier-3 feature."
        )
    return _RUST
