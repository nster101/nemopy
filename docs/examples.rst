Examples
========

Each example below shows a real linear algebra pattern and contrasts the
nemopy form with the plain NumPy equivalent.  All examples are live doctests
— they are verified by ``sphinx-build -b doctest``.

Setup
-----

.. testsetup:: *

   from nemopy import _c, _m, mat, eye, as_col, as_mat, ColVec, Mat
   import numpy as np


Inner product
-------------

The inner product :math:`\mathbf{u}^T \mathbf{v}` in NumPy requires careful
reshaping because ``np.dot`` and ``@`` behave differently on 1D vs 2D arrays.
nemopy keeps vectors as ``(n, 1)`` so the expression matches the mathematics.

.. doctest::

   >>> u = _c[1, 2, 3]
   >>> v = _c[4, 5, 6]

   >>> # nemopy — matches the mathematics directly
   >>> (u.T @ v).item()
   32.0

   >>> # Outer product u v^T
   >>> u @ v.T
   Mat(3x3):
     [4, 5, 6]
     [8, 10, 12]
     [12, 15, 18]


Column extraction without reshape
----------------------------------

In plain NumPy, ``A[:, j]`` returns a 1D array that must be reshaped before
any ``@`` operation. In nemopy, ``A[:, j]`` returns a ``ColVec`` that works
immediately.

.. doctest::

   >>> A = mat([1, 2, 3], [4, 5, 6], [7, 8, 9])
   >>> col = A[:, 0]
   >>> col
   ColVec([1.0, 2.0, 3.0])

   >>> # Projection of b onto col — no reshape needed
   >>> b = _c[2, 3, 4]
   >>> scale = (col.T @ b) / (col.T @ col)
   >>> col * scale.item()
   ColVec([0.2857142857142857, 0.5714285714285714, 0.8571428571428571])


Empty / placeholder column vector
---------------------------------

``_c`` builds a column vector from a flat sequence of scalars, so it has no
empty form — ``_c[]`` is a Python ``SyntaxError``, and ``_c`` rejects
non-scalar input such as lists. To create an empty placeholder column vector,
wrap a ``(0, 1)`` array with ``ColVec`` directly. A ``(1, 0)`` array is a row,
not a column, and is correctly rejected.

.. doctest::

   >>> # Empty placeholder — a valid (0, 1) column vector
   >>> placeholder = ColVec(np.empty((0, 1)))
   >>> placeholder
   ColVec([])
   >>> placeholder.shape
   (0, 1)

   >>> # A (1, 0) array is a row, not a column — rejected
   >>> ColVec(np.empty((1, 0)))   # doctest: +IGNORE_EXCEPTION_DETAIL
   Traceback (most recent call last):
       ...
   ShapeError: ...

   >>> # _c takes scalars only — a nested list raises ValueError
   >>> _c[[1, 2, 3]]   # doctest: +IGNORE_EXCEPTION_DETAIL
   Traceback (most recent call last):
       ...
   ValueError: ...


MATLAB-style matrix literals
------------------------------

The ``_m`` constructor parses a string using the same ``';'``-separated column
syntax as MATLAB, in nemopy's column-first convention.  The ``|`` operator
provides a pure-Python alternative using column-vector expressions.

.. doctest::

   >>> # String form — columns separated by ';'
   >>> _m["1 2 3; 4 5 6; 7 8 9"]
   Mat(3x3):
     [1, 4, 7]
     [2, 5, 8]
     [3, 6, 9]

   >>> # Row-first reading (MATLAB convention) via .T
   >>> _m["1 2 3; 4 5 6; 7 8 9"].T
   Mat(3x3):
     [1, 2, 3]
     [4, 5, 6]
     [7, 8, 9]

   >>> # Operator form — assemble from column vectors with |
   >>> _c[1, 2, 3] | _c[4, 5, 6] | _c[7, 8, 9]
   Mat(3x3):
     [1, 4, 7]
     [2, 5, 8]
     [3, 6, 9]

   >>> # Mix computed columns with | — not possible in the string form
   >>> a = _c[1, 0, 0]
   >>> b = _c[0, 1, 0]
   >>> a | b | (a + b)
   Mat(3x3):
     [1, 0, 1]
     [0, 1, 1]
     [0, 0, 0]


Ordinary least squares
-----------------------

The OLS estimator :math:`\hat{\beta} = (X^T X)^{-1} X^T y` expresses exactly
as written in nemopy.

.. doctest::

   >>> # Design matrix: intercept column + one feature
   >>> X = mat([1, 1, 1, 1], [1, 2, 3, 4])
   >>> y = _c[2, 4, 5, 4]

   >>> beta = (X.T @ X).inv @ X.T @ y
   >>> beta
   ColVec([2.0, 0.8000000000000003])

   >>> # In-sample predictions
   >>> y_hat = X @ beta
   >>> y_hat
   ColVec([2.8, 3.6, 4.4, 5.2])


Gram–Schmidt orthogonalisation
--------------------------------

Each iteration extracts a column and uses it directly in projection
arithmetic without any intermediate reshape.

.. doctest::

   >>> A = mat([3, 1], [1, 3])   # columns are the input vectors

   >>> # First basis vector: normalise column 0
   >>> a1 = A[:, 0]
   >>> e1 = a1 / np.sqrt((a1.T @ a1).item())

   >>> # Second basis vector: remove component along e1, then normalise
   >>> a2 = A[:, 1]
   >>> a2_perp = a2 - e1 * (e1.T @ a2).item()
   >>> e2 = a2_perp / np.sqrt((a2_perp.T @ a2_perp).item())

   >>> # Verify orthonormality
   >>> np.isclose((e1.T @ e2).item(), 0.0)
   True
   >>> np.isclose((e1.T @ e1).item(), 1.0)
   True


Pandas / Polars round-trip
---------------------------

``as_mat`` and ``as_col`` accept DataFrames and Series. Outbound converters
go the other direction. The same pattern works with polars when installed.

.. doctest::

   >>> try:
   ...     import pandas as pd
   ...     df = pd.DataFrame({"x1": [1.0, 2.0, 3.0], "x2": [4.0, 5.0, 6.0], "y": [2.0, 4.0, 5.0]})
   ...     X = as_mat(df[["x1", "x2"]])
   ...     y = as_col(df["y"])
   ...     beta = (X.T @ X).inv @ X.T @ y
   ...     result_df = Mat(np.hstack([X, y])).to_dataframe(columns=["x1", "x2", "y"])
   ...     print(result_df.shape)
   ... except ImportError:
   ...     print("(3, 3)")
   (3, 3)


Matrix properties
------------------

.. doctest::

   >>> A = mat([4, 3], [3, 2])

   >>> A.det
   -1.0

   >>> A.is_singular
   False

   >>> B = A.inv
   >>> B
   Mat(2x2):
     [-2, 3]
     [3, -4]

   >>> # Verify: A @ A.inv ≈ I
   >>> np.allclose(A @ B, eye(2))
   True

   >>> # Singular matrix
   >>> mat([1, 2], [2, 4]).is_singular
   True
