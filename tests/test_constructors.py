"""Tests for constructors: _c singleton, mat(), eye(), as_mat().

## Test: test_c_basic
- Goal: Verify that _c[1, 2, 3] produces a ColVec with shape (3, 1) and dtype float64.
- Source: DESIGN.md §3, §5.1 — "_c[1, 2, 3] produces a column vector of shape (3, 1)".
- Expected: result is ColVec, shape (3, 1), dtype float64, values [1.0, 2.0, 3.0].

## Test: test_c_single_element
- Goal: Verify that _c[5] produces a ColVec with shape (1, 1).
- Source: DESIGN.md §3 — "_c[5] shape (1,1) — a (1x1) column vector, not a scalar".
- Expected: result is ColVec, shape (1, 1), value 5.0.

## Test: test_c_negative_element
- Goal: Verify that _c[-1] produces a ColVec with shape (1, 1) containing -1.0,
        not an indexing operation.
- Source: DESIGN.md §3 — "_c[-1] shape (1,1) containing -1.0 — not an indexing operation".
- Expected: result is ColVec, shape (1, 1), value -1.0.

## Test: test_c_nested_raises
- Goal: Verify that _c[[1, 2, 3]] raises ValueError (nested input is forbidden).
- Source: DESIGN.md §3 — "_c[[1,2,3]] raises ValueError — use mat() instead".
- Expected: ValueError raised.

## Test: test_c_repr
- Goal: Verify that repr(_c) returns the string "_c".
- Source: DESIGN.md §3 — "def __repr__(self): return '_c'".
- Expected: repr(_c) == "_c".

## Test: test_mat_from_lists
- Goal: Verify that mat([1,2,3], [4,5,6]) produces a Mat with shape (3, 2) and
        column-first layout (first list is first column).
- Source: DESIGN.md §5.2, §5.3 — "Each argument becomes one column of the resulting matrix."
- Expected: result is Mat, shape (3, 2), column 0 is [1,2,3], column 1 is [4,5,6].

## Test: test_mat_single_column_returns_mat
- Goal: Verify that mat([1, 2, 3]) returns a Mat, NOT a ColVec, with shape (3, 1).
- Source: DESIGN.md §4.3 — "mat() with a single column returns Mat, not ColVec".
- Expected: result is Mat (not ColVec), shape (3, 1).

## Test: test_mat_unequal_lengths_raises
- Goal: Verify that mat([1, 2], [3, 4, 5]) raises ValueError for unequal column lengths.
- Source: DESIGN.md §5.3 — "mat() columns have unequal lengths: [...]. All columns must
         have the same number of rows."
- Expected: ValueError raised.

## Test: test_mat_no_args_raises
- Goal: Verify that mat() with no arguments raises ValueError.
- Source: DESIGN.md §5.3 — "mat() requires at least one column argument."
- Expected: ValueError raised.

## Test: test_eye_identity
- Goal: Verify that eye(3) returns a Mat of shape (3, 3) with identity matrix values.
- Source: DESIGN.md §5.8 — "return Mat(np.eye(int(n)))".
- Expected: result is Mat, shape (3, 3), values match np.eye(3).

## Test: test_as_col_from_list
- Goal: Verify that as_col([1, 2, 3]) returns ColVec with shape (3, 1).
- Source: DESIGN_APPENDICES.md §13.3 — as_col accepts flat lists and reshapes to (n,1).
- Expected: result is ColVec, shape (3, 1), values [1.0, 2.0, 3.0].

## Test: test_as_col_from_scalar
- Goal: Verify that as_col(7) returns ColVec with shape (1, 1).
- Source: DESIGN_APPENDICES.md §13.3 — as_col accepts scalar values and reshapes to (1,1).
- Expected: result is ColVec, shape (1, 1), value 7.0.

## Test: test_as_col_from_1d_ndarray
- Goal: Verify that as_col(np.array([1, 2, 3])) returns ColVec with shape (3, 1).
- Source: DESIGN_APPENDICES.md §13.3 — as_col accepts 1D ndarrays and reshapes to (n,1).
- Expected: result is ColVec, shape (3, 1), values [1.0, 2.0, 3.0].

## Test: test_as_col_from_2d_single_column_ndarray
- Goal: Verify that as_col(np.ones((3, 1))) returns ColVec with shape (3, 1).
- Source: DESIGN_APPENDICES.md §13.3 — as_col accepts (n,1) arrays directly.
- Expected: result is ColVec, shape (3, 1), values are all ones.

## Test: test_as_col_from_pandas_series_if_available
- Goal: Verify that as_col(pd.Series([4, 5])) returns ColVec with shape (2, 1) when pandas is installed.
- Source: DESIGN_APPENDICES.md §13.3 — as_col accepts pandas Series.
- Expected: result is ColVec, shape (2, 1), values [4.0, 5.0].

## Test: test_as_col_rejects_2d_multi_column
- Goal: Verify that as_col(np.ones((3, 3))) raises ShapeError.
- Source: DESIGN_APPENDICES.md §13.3 — 2D inputs with more than one column are rejected.
- Expected: ShapeError raised.

## Test: test_as_col_rejects_non_numeric_input
- Goal: Verify that as_col raises TypeError (not ValueError) when the input
        cannot be converted to a numeric array.
- Source: DESIGN_APPENDICES.md §13.3 — documented TypeError contract for
          non-numeric inputs.
- Expected: TypeError raised for as_col(['a']).

## Test: test_as_mat_from_nested_lists_row_first
- Goal: Verify that as_mat converts nested row-first lists into a Mat of matching shape.
- Source: DESIGN_APPENDICES.md §13.3 — "as_mat on a nested list uses NumPy's row-first convention".
- Expected: Mat shape matches input rows/cols and values preserve row order.

## Test: test_as_mat_accepts_existing_mat
- Goal: Verify that as_mat accepts an existing Mat and returns a Mat with matching shape/values.
- Source: DESIGN_APPENDICES.md §13.3 — "Accepts ... existing Mat instances."
- Expected: result is Mat with same shape and equal values.

## Test: test_as_mat_accepts_2d_ndarray
- Goal: Verify that as_mat accepts a 2D numpy.ndarray and returns matching Mat.
- Source: DESIGN_APPENDICES.md §13.3 — "Accepts 2D arrays ... and returns Mat."
- Expected: result is Mat with same shape/values as ndarray.

## Test: test_as_mat_non_2d_raises_shape_error
- Goal: Verify that non-2D input to as_mat raises ShapeError.
- Source: DESIGN_APPENDICES.md §13.3 — "ShapeError if x is not 2D after conversion."
- Expected: ShapeError raised for 1D input.

## Test: test_as_mat_accepts_pandas_dataframe_when_available
- Goal: Verify that as_mat accepts pandas.DataFrame when pandas is installed.
- Source: DESIGN_APPENDICES.md §13.3 — "Accepts ... pandas DataFrames".
- Expected: DataFrame converts to Mat with matching shape and row-first values.

## Test: test_as_mat_rejects_non_numeric_input
- Goal: Verify that as_mat raises TypeError (not ValueError) when the input
        cannot be converted to a numeric array.
- Source: DESIGN_APPENDICES.md §13.3 — documented TypeError contract for
          non-numeric inputs.
- Expected: TypeError raised for as_mat([['a', 'b'], ['c', 'd']]).
"""

import numpy as np
import pytest

from nemopy import _c, _m, as_col, as_mat, eye, mat, ColVec, Mat, ShapeError


class TestColConstructor:
    def test_c_basic(self):
        """_c[1, 2, 3] produces ColVec with shape (3, 1) and dtype float64."""
        u = _c[1, 2, 3]
        assert isinstance(u, ColVec)
        assert u.shape == (3, 1)
        assert u.dtype == np.float64
        np.testing.assert_array_equal(u.flatten(), [1.0, 2.0, 3.0])

    def test_c_single_element(self):
        """_c[5] produces ColVec with shape (1, 1)."""
        u = _c[5]
        assert isinstance(u, ColVec)
        assert u.shape == (1, 1)
        assert u.item() == 5.0

    def test_c_negative_element(self):
        """_c[-1] produces ColVec (1, 1) containing -1.0, not an indexing op."""
        u = _c[-1]
        assert isinstance(u, ColVec)
        assert u.shape == (1, 1)
        assert u.item() == -1.0

    def test_c_nested_raises(self):
        """_c[[1, 2, 3]] raises ValueError for nested input."""
        with pytest.raises(ValueError):
            _c[[1, 2, 3]]

    def test_c_repr(self):
        """repr(_c) returns '_c'."""
        assert repr(_c) == "_c"


class TestMatConstructor:
    def test_mat_from_lists(self):
        """mat([1,2,3], [4,5,6]) produces Mat (3, 2) with column-first layout."""
        A = mat([1, 2, 3], [4, 5, 6])
        assert isinstance(A, Mat)
        assert A.shape == (3, 2)
        np.testing.assert_array_equal(A[:, 0].flatten(), [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(A[:, 1].flatten(), [4.0, 5.0, 6.0])

    def test_mat_single_column_returns_mat(self):
        """mat([1, 2, 3]) returns Mat (not ColVec), shape (3, 1)."""
        A = mat([1, 2, 3])
        assert isinstance(A, Mat)
        assert not isinstance(A, ColVec)
        assert A.shape == (3, 1)

    def test_mat_unequal_lengths_raises(self):
        """mat([1, 2], [3, 4, 5]) raises ValueError for unequal lengths."""
        with pytest.raises(ValueError):
            mat([1, 2], [3, 4, 5])

    def test_mat_no_args_raises(self):
        """mat() with no arguments raises ValueError."""
        with pytest.raises(ValueError):
            mat()


class TestEyeConstructor:
    def test_eye_identity(self):
        """eye(3) returns Mat (3, 3) matching np.eye(3)."""
        I = eye(3)
        assert isinstance(I, Mat)
        assert I.shape == (3, 3)
        np.testing.assert_array_equal(np.asarray(I), np.eye(3))

    ## Test: test_eye_accepts_dtype_kwarg
    # - Goal: eye(n, dtype=...) accepts a dtype kwarg without raising
    #         TypeError, always producing float64 per §3.
    # - Source: Issue #68 — np.eye(n, dtype=...) through nemopy fails.
    # - Expected: No TypeError; result dtype is float64; shape is (n,n).
    def test_eye_accepts_dtype_kwarg(self):
        """eye(n, dtype=...) accepts dtype kwarg, result is always float64."""
        I = eye(3, dtype=np.float32)
        assert isinstance(I, Mat)
        assert I.shape == (3, 3)
        assert I.dtype == np.float64

    ## Test: test_eye_accepts_M_and_k_kwargs
    # - Goal: eye(n, M=m, k=k) accepts NumPy-compatible kwargs.
    # - Source: Issue #68 — fuller compat with np.eye() signature.
    # - Expected: Non-square identity with offset diagonal.
    def test_eye_accepts_M_and_k_kwargs(self):
        """eye(n, M=m, k=k) produces non-square identity with offset."""
        I = eye(3, M=4, k=1)
        assert isinstance(I, Mat)
        assert I.shape == (3, 4)
        expected = np.eye(3, M=4, k=1)
        assert np.array_equal(np.asarray(I), expected)


class TestAsColConstructor:
    def test_as_col_from_list(self):
        """as_col([1,2,3]) returns ColVec (3,1) with matching values."""
        u = as_col([1, 2, 3])
        assert isinstance(u, ColVec)
        assert u.shape == (3, 1)
        np.testing.assert_array_equal(np.asarray(u), np.array([[1.0], [2.0], [3.0]]))

    def test_as_col_from_scalar(self):
        """as_col(7) returns ColVec (1,1)."""
        u = as_col(7)
        assert isinstance(u, ColVec)
        assert u.shape == (1, 1)
        np.testing.assert_array_equal(np.asarray(u), np.array([[7.0]]))

    def test_as_col_from_complex_scalar(self):
        """as_col(1+0j) keeps scalar complex handling and returns ColVec (1,1)."""
        u = as_col(1 + 0j)
        assert isinstance(u, ColVec)
        assert u.shape == (1, 1)
        np.testing.assert_array_equal(np.asarray(u), np.array([[1.0]]))

    def test_as_col_rejects_non_numeric_input(self):
        """as_col(['a']) raises TypeError per the conversion contract."""
        with pytest.raises(TypeError):
            as_col(["a"])


class TestAsMatConstructor:
    def test_as_mat_from_nested_lists_row_first(self):
        """as_mat() preserves nested-list row-first layout in Mat output."""
        A = as_mat([[1, 2], [3, 4]])
        assert isinstance(A, Mat)
        assert A.shape == (2, 2)
        np.testing.assert_array_equal(np.asarray(A), np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_as_mat_accepts_existing_mat(self):
        """as_mat() accepts an existing Mat and preserves shape/values."""
        A0 = Mat(np.array([[1.0, 2.0], [3.0, 4.0]]))
        A = as_mat(A0)
        assert isinstance(A, Mat)
        assert A.shape == (2, 2)
        np.testing.assert_array_equal(np.asarray(A), np.asarray(A0))

    def test_as_mat_accepts_2d_ndarray(self):
        """as_mat() accepts a 2D ndarray and preserves shape/values."""
        arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        A = as_mat(arr)
        assert isinstance(A, Mat)
        assert A.shape == (2, 3)
        np.testing.assert_array_equal(np.asarray(A), arr)

    def test_as_mat_non_2d_raises_shape_error(self):
        """as_mat() raises ShapeError for non-2D input."""
        with pytest.raises(ShapeError):
            as_mat([1, 2, 3])

    def test_as_mat_accepts_pandas_dataframe_when_available(self):
        """as_mat() accepts DataFrame when pandas is installed."""
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        A = as_mat(df)
        assert isinstance(A, Mat)
        assert A.shape == (2, 2)
        np.testing.assert_array_equal(np.asarray(A), df.values.astype(float))

    def test_as_mat_rejects_non_numeric_input(self):
        """as_mat([['a', 'b'], ['c', 'd']]) raises TypeError per the contract."""
        with pytest.raises(TypeError):
            as_mat([["a", "b"], ["c", "d"]])


## Test: test_m_columns_default
# - Goal: Verify _m["1, 2, 3; 4, 5, 6; 7, 8, 9"] returns Mat (3,3) whose columns
#         are [1,2,3], [4,5,6], [7,8,9] per nemopy column-first convention.
# - Source: DESIGN_APPENDICES.md §17 (MATLAB-style _m string constructor).
# - Expected: shape (3, 3), column 0 == [1,2,3], column 2 == [7,8,9].
#
## Test: test_m_transpose_to_rows
# - Goal: Verify the .T modifier flips the column-first interpretation to rows,
#         giving the row-first MATLAB result.
# - Source: DESIGN_APPENDICES.md §17.
#
## Test: test_m_unequal_columns_raises
# - Goal: Verify unequal column lengths raise ValueError.
#
## Test: test_m_non_numeric_raises
# - Goal: Verify non-numeric tokens raise TypeError.
#
## Test: test_m_brackets_tolerated
# - Goal: Verify "[1, 2; 3, 4]" with surrounding [] parses identically.
#
## Test: test_m_whitespace_separator
# - Goal: Verify whitespace works as element separator (MATLAB-true).
#
## Test: test_m_empty_string_raises
# - Goal: Verify "" or "[]" raises ValueError.
#
## Test: test_m_non_string_input_raises
# - Goal: Verify non-string subscripts raise TypeError.
class TestMatlabConstructor:
    def test_m_columns_default(self):
        """_m['1, 2, 3; 4, 5, 6; 7, 8, 9'] has columns [1,2,3], [4,5,6], [7,8,9]."""
        A = _m["1, 2, 3; 4, 5, 6; 7, 8, 9"]
        assert isinstance(A, Mat)
        assert A.shape == (3, 3)
        np.testing.assert_array_equal(A[:, 0].flatten(), [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(A[:, 1].flatten(), [4.0, 5.0, 6.0])
        np.testing.assert_array_equal(A[:, 2].flatten(), [7.0, 8.0, 9.0])

    def test_m_transpose_to_rows(self):
        """_m['1, 2, 3; 4, 5, 6; 7, 8, 9'].T has rows [1,2,3], [4,5,6], [7,8,9]."""
        A = _m["1, 2, 3; 4, 5, 6; 7, 8, 9"].T
        assert A.shape == (3, 3)
        np.testing.assert_array_equal(np.asarray(A[0, :]).flatten(), [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(np.asarray(A[1, :]).flatten(), [4.0, 5.0, 6.0])
        np.testing.assert_array_equal(np.asarray(A[2, :]).flatten(), [7.0, 8.0, 9.0])

    def test_m_unequal_columns_raises(self):
        """_m['1, 2; 3, 4, 5'] raises ValueError for ragged columns."""
        with pytest.raises(ValueError):
            _m["1, 2; 3, 4, 5"]

    def test_m_non_numeric_raises(self):
        """_m['1, a; 3, 4'] raises TypeError for non-numeric tokens."""
        with pytest.raises(TypeError):
            _m["1, a; 3, 4"]

    def test_m_brackets_tolerated(self):
        """_m['[1, 2; 3, 4]'] equals _m['1, 2; 3, 4']."""
        A = _m["[1, 2; 3, 4]"]
        B = _m["1, 2; 3, 4"]
        np.testing.assert_array_equal(np.asarray(A), np.asarray(B))

    def test_m_whitespace_separator(self):
        """_m['1 2 3; 4 5 6'] parses with whitespace separators."""
        A = _m["1 2 3; 4 5 6"]
        assert A.shape == (3, 2)
        np.testing.assert_array_equal(A[:, 0].flatten(), [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(A[:, 1].flatten(), [4.0, 5.0, 6.0])

    def test_m_empty_string_raises(self):
        """_m[''] and _m['[]'] raise ValueError."""
        with pytest.raises(ValueError):
            _m[""]
        with pytest.raises(ValueError):
            _m["[]"]

    def test_m_non_string_input_raises(self):
        """_m[1, 2, 3] (non-string) raises TypeError."""
        with pytest.raises(TypeError):
            _m[1, 2, 3]
