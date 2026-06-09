"""Operator override logic (mixed into _VecBase)."""

import warnings

import numpy as np

from nemopy._core import ConventionWarning, ShapeError, _VecBase


def _is_scalar(x):
    if isinstance(x, (int, float, complex, np.generic)):
        return True
    if isinstance(x, np.ndarray) and x.ndim == 0:
        return True
    return False


def _coerce_1d(vecbase, other):
    """Reshape a 1D ndarray to (n,1) when paired with a (n,1) _VecBase."""
    if (isinstance(other, np.ndarray)
            and not isinstance(other, _VecBase)
            and other.ndim == 1
            and vecbase.ndim == 2
            and vecbase.shape[1] == 1
            and other.shape[0] == vecbase.shape[0]):
        return other.reshape(-1, 1)
    return other


def _check_shapes(a, b, op_name):
    """Raise ShapeError if a and b are both arrays with different shapes."""
    if _is_scalar(a) or _is_scalar(b):
        return
    a_shape = np.shape(a)
    b_shape = np.shape(b)
    if a_shape != b_shape:
        raise ShapeError(
            f"Element-wise '{op_name}' requires identical shapes, "
            f"got {a_shape} and {b_shape}. "
            f"If broadcasting is intended, use np.multiply / np.add directly."
        )


def __mul__(self, other):
    other = _coerce_1d(self, other)
    _check_shapes(self, other, "*")
    return super(_VecBase, self).__mul__(other)


def __rmul__(self, other):
    other = _coerce_1d(self, other)
    _check_shapes(other, self, "*")
    return super(_VecBase, self).__rmul__(other)


def __add__(self, other):
    other = _coerce_1d(self, other)
    _check_shapes(self, other, "+")
    return super(_VecBase, self).__add__(other)


def __radd__(self, other):
    other = _coerce_1d(self, other)
    _check_shapes(other, self, "+")
    return super(_VecBase, self).__radd__(other)


def __sub__(self, other):
    other = _coerce_1d(self, other)
    _check_shapes(self, other, "-")
    return super(_VecBase, self).__sub__(other)


def __rsub__(self, other):
    other = _coerce_1d(self, other)
    _check_shapes(other, self, "-")
    return super(_VecBase, self).__rsub__(other)


def __truediv__(self, other):
    other = _coerce_1d(self, other)
    _check_shapes(self, other, "/")
    return super(_VecBase, self).__truediv__(other)


def __rtruediv__(self, other):
    other = _coerce_1d(self, other)
    _check_shapes(other, self, "/")
    return super(_VecBase, self).__rtruediv__(other)


def __matmul__(self, other):
    if isinstance(other, np.ndarray) and not isinstance(other, _VecBase):
        if other.ndim == 2 and other.shape[0] < other.shape[1]:
            warnings.warn(
                f"Right operand of @ is a plain ndarray with shape {other.shape} — "
                f"more columns than rows. If this came from np.array([[...]]), "
                f"it may be row-first and transposed relative to nemopy convention. "
                f"Wrap with Mat(...) to suppress this warning.",
                ConventionWarning,
                stacklevel=2,
            )
    return super(_VecBase, self).__matmul__(other)


def __rmatmul__(self, other):
    if isinstance(other, np.ndarray) and not isinstance(other, _VecBase):
        if other.ndim == 2 and other.shape[0] < other.shape[1]:
            warnings.warn(
                f"Left operand of @ is a plain ndarray with shape {other.shape} — "
                f"more columns than rows. If this came from np.array([[...]]), "
                f"it may be row-first and transposed relative to nemopy convention. "
                f"Wrap with Mat(...) to suppress this warning.",
                ConventionWarning,
                stacklevel=2,
            )
    return super(_VecBase, self).__rmatmul__(other)


def __iadd__(self, other):
    _check_shapes(self, other, "+")
    super(_VecBase, self).__iadd__(other)
    return self


def __isub__(self, other):
    _check_shapes(self, other, "-")
    super(_VecBase, self).__isub__(other)
    return self


def __imul__(self, other):
    _check_shapes(self, other, "*")
    super(_VecBase, self).__imul__(other)
    return self


def __itruediv__(self, other):
    _check_shapes(self, other, "/")
    super(_VecBase, self).__itruediv__(other)
    return self


def __or__(self, other):
    """Column-join operator: horizontally stack ``self`` and ``other``.

    Assembles a matrix by placing ``other`` to the right of ``self``.
    The mathematical notation ``[a | b | c]`` maps directly to
    ``_c[...] | _c[...] | _c[...]`` in nemopy.

    Parameters
    ----------
    other : ColVec, Mat, or array-like with 2D shape
        Right operand. Must have the same number of rows as ``self``.

    Returns
    -------
    Mat
        Horizontally stacked result with shape ``(n, k_self + k_other)``.

    Raises
    ------
    ShapeError
        If ``self`` and ``other`` have different row counts.

    Examples
    --------
    >>> _c[1, 2, 3] | _c[4, 5, 6]
    Mat(3x2):
      [1, 4]
      [2, 5]
      [3, 6]

    >>> _c[1, 2, 3] | _c[4, 5, 6] | _c[7, 8, 9]
    Mat(3x3):
      [1, 4, 7]
      [2, 5, 8]
      [3, 6, 9]

    See Also
    --------
    mat : Column-first constructor (equivalent, function form).
    """
    from nemopy._core import Mat  # avoid circular at module level

    self_rows = np.shape(self)[0]
    other_rows = np.shape(other)[0]
    if self_rows != other_rows:
        raise ShapeError(
            f"'|' requires equal row counts, "
            f"got {np.shape(self)} and {np.shape(other)}."
        )
    return Mat(np.hstack([np.asarray(self), np.asarray(other)]))


def __ror__(self, other):
    """Reflected column-join: ``other | self``."""
    from nemopy._core import Mat  # avoid circular at module level

    other_rows = np.shape(other)[0]
    self_rows = np.shape(self)[0]
    if other_rows != self_rows:
        raise ShapeError(
            f"'|' requires equal row counts, "
            f"got {np.shape(other)} and {np.shape(self)}."
        )
    return Mat(np.hstack([np.asarray(other), np.asarray(self)]))


_VecBase.__mul__ = __mul__
_VecBase.__rmul__ = __rmul__
_VecBase.__add__ = __add__
_VecBase.__radd__ = __radd__
_VecBase.__sub__ = __sub__
_VecBase.__rsub__ = __rsub__
_VecBase.__truediv__ = __truediv__
_VecBase.__rtruediv__ = __rtruediv__
_VecBase.__matmul__ = __matmul__
_VecBase.__rmatmul__ = __rmatmul__
_VecBase.__iadd__ = __iadd__
_VecBase.__isub__ = __isub__
_VecBase.__imul__ = __imul__
_VecBase.__itruediv__ = __itruediv__
_VecBase.__or__ = __or__
_VecBase.__ror__ = __ror__
