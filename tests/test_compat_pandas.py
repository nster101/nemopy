"""Tests for pandas interoperability with ColVec and Mat.

Focuses on round-trip integrity, direct consumption via the ndarray
protocol, and edge cases (empty, single-element, type promotion, mixed
dtypes, label preservation).  Does NOT duplicate the basic to_series /
to_dataframe / as_col / as_mat tests that already live in test_core.py
and test_constructors.py.

## Test: test_colvec_roundtrip_series
- Goal: Verify ColVec -> Series -> ColVec preserves values exactly.
- Source: Issue #63 — round-trip integrity.
- Expected: as_col(colvec.to_series()) equals original ColVec element-wise.

## Test: test_mat_roundtrip_dataframe
- Goal: Verify Mat -> DataFrame -> Mat preserves values exactly.
- Source: Issue #63 — round-trip integrity.
- Expected: as_mat(mat.to_dataframe()) equals original Mat element-wise.

## Test: test_dataframe_roundtrip_mat
- Goal: Verify DataFrame -> Mat -> DataFrame preserves values exactly.
- Source: Issue #63 — round-trip integrity.
- Expected: mat.to_dataframe() values equal original DataFrame values.

## Test: test_direct_dataframe_from_mat
- Goal: Verify pd.DataFrame(Mat) works because Mat is an ndarray subclass.
- Source: DESIGN_APPENDICES.md §12.5 — "pd.DataFrame(A, ...) works directly".
- Expected: DataFrame has correct shape and values.

## Test: test_direct_series_from_colvec_flatten
- Goal: Verify pd.Series(ColVec.flatten()) works for direct consumption.
- Source: DESIGN_APPENDICES.md §12.5 — "pd.Series(u.flatten(), ...)".
- Expected: Series has correct length and values.

## Test: test_empty_colvec_to_series
- Goal: Verify an empty ColVec (0, 1) converts to an empty Series.
- Source: Issue #63 — edge case: empty ColVec.
- Expected: Series has length 0.

## Test: test_empty_mat_to_dataframe
- Goal: Verify an empty Mat (0, k) converts to a DataFrame with 0 rows.
- Source: Issue #63 — edge case: empty Mat.
- Expected: DataFrame has shape (0, k).

## Test: test_single_element_colvec_roundtrip
- Goal: Verify a (1, 1) ColVec round-trips through Series.
- Source: Issue #63 — edge case: single-element ColVec.
- Expected: Round-tripped ColVec has shape (1, 1) and same value.

## Test: test_single_column_mat_roundtrip
- Goal: Verify a (n, 1) Mat round-trips through DataFrame.
- Source: Issue #63 — edge case: single-column Mat.
- Expected: Round-tripped Mat has shape (n, 1) and same values.

## Test: test_single_row_mat_roundtrip
- Goal: Verify a (1, k) Mat round-trips through DataFrame.
- Source: Issue #63 — edge case: single-row Mat.
- Expected: Round-tripped Mat has shape (1, k) and same values.

## Test: test_integer_dataframe_promoted_to_float64
- Goal: Verify as_mat on an integer DataFrame promotes dtype to float64.
- Source: Issue #63 — non-float DataFrame; DESIGN.md §4.3 — Mat is always float64.
- Expected: Resulting Mat has dtype float64 with correct values.

## Test: test_mixed_numeric_dataframe_promoted_to_float64
- Goal: Verify as_mat on a DataFrame with mixed int and float columns
        promotes all values to float64.
- Source: Issue #63 — mixed numeric types.
- Expected: Resulting Mat has dtype float64.

## Test: test_series_index_preserved_through_to_series
- Goal: Verify that to_series(index=...) preserves named index labels,
        not just the values (complementing test_core which checks basic
        index/name; here we verify non-default index survives).
- Source: DESIGN_APPENDICES.md §13.2 — index parameter.
- Expected: Series index matches the provided labels exactly.

## Test: test_dataframe_columns_preserved_through_to_dataframe
- Goal: Verify that to_dataframe(columns=...) preserves column labels.
- Source: DESIGN_APPENDICES.md §13.2 — columns parameter.
- Expected: DataFrame columns match the provided labels exactly.

## Test: test_to_series_default_index_is_range
- Goal: Verify that to_series() without index uses integer range index.
- Source: DESIGN_APPENDICES.md §13.2 — "Defaults to range(n)".
- Expected: Series index is RangeIndex(start=0, stop=n, step=1).

## Test: test_to_dataframe_default_columns_is_range
- Goal: Verify that to_dataframe() without columns uses integer range.
- Source: DESIGN_APPENDICES.md §13.2 — "Defaults to integer range".
- Expected: DataFrame columns is RangeIndex(start=0, stop=k, step=1).

## Test: test_colvec_to_series_dtype_is_float64
- Goal: Verify that the Series produced by to_series has dtype float64.
- Source: DESIGN_APPENDICES.md §13.2 — ColVec is always float64.
- Expected: Series dtype is float64.

## Test: test_mat_to_dataframe_dtypes_are_float64
- Goal: Verify that the DataFrame produced by to_dataframe has float64 columns.
- Source: DESIGN_APPENDICES.md §13.2 — Mat is always float64.
- Expected: All DataFrame column dtypes are float64.
"""

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from nemopy import ColVec, Mat, _c, as_col, as_mat, mat


class TestRoundTrip:
    def test_colvec_roundtrip_series(self):
        """ColVec -> Series -> ColVec preserves values."""
        u = _c[1, 2, 3, 4, 5]
        s = u.to_series()
        result = as_col(s)
        assert isinstance(result, ColVec)
        assert result.shape == (5, 1)
        np.testing.assert_array_equal(np.asarray(result), np.asarray(u))

    def test_mat_roundtrip_dataframe(self):
        """Mat -> DataFrame -> Mat preserves values."""
        A = mat([1, 2, 3], [4, 5, 6])
        df = A.to_dataframe()
        result = as_mat(df)
        assert isinstance(result, Mat)
        assert result.shape == A.shape
        np.testing.assert_array_equal(np.asarray(result), np.asarray(A))

    def test_dataframe_roundtrip_mat(self):
        """DataFrame -> Mat -> DataFrame preserves values."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
        A = as_mat(df)
        df_back = A.to_dataframe()
        np.testing.assert_array_equal(df_back.values, df.values)


class TestDirectConsumption:
    def test_direct_dataframe_from_mat(self):
        """pd.DataFrame(Mat) works because Mat is an ndarray subclass."""
        A = mat([1, 2, 3], [4, 5, 6])
        df = pd.DataFrame(A, columns=["a", "b"])
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (3, 2)
        np.testing.assert_array_equal(df.values, np.asarray(A))

    def test_direct_series_from_colvec_flatten(self):
        """pd.Series(ColVec.flatten()) works for direct consumption."""
        u = _c[10, 20, 30]
        s = pd.Series(u.flatten(), index=["a", "b", "c"])
        assert isinstance(s, pd.Series)
        assert len(s) == 3
        assert s.tolist() == [10.0, 20.0, 30.0]
        assert s.index.tolist() == ["a", "b", "c"]


class TestEdgeCases:
    def test_empty_colvec_to_series(self):
        """Empty ColVec (0, 1) converts to an empty Series."""
        u = ColVec(np.empty((0, 1)))
        s = u.to_series()
        assert isinstance(s, pd.Series)
        assert len(s) == 0

    def test_empty_mat_to_dataframe(self):
        """Empty Mat (0, 3) converts to a DataFrame with 0 rows."""
        A = Mat(np.empty((0, 3)))
        df = A.to_dataframe(columns=["a", "b", "c"])
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (0, 3)
        assert list(df.columns) == ["a", "b", "c"]

    def test_single_element_colvec_roundtrip(self):
        """Single-element ColVec (1, 1) round-trips through Series."""
        u = _c[42]
        s = u.to_series()
        result = as_col(s)
        assert isinstance(result, ColVec)
        assert result.shape == (1, 1)
        assert result.item() == 42.0

    def test_single_column_mat_roundtrip(self):
        """Single-column Mat (n, 1) round-trips through DataFrame."""
        A = mat([10, 20, 30])
        df = A.to_dataframe()
        result = as_mat(df)
        assert isinstance(result, Mat)
        assert result.shape == (3, 1)
        np.testing.assert_array_equal(np.asarray(result), np.asarray(A))

    def test_single_row_mat_roundtrip(self):
        """Single-row Mat (1, k) round-trips through DataFrame."""
        A = Mat(np.array([[7.0, 8.0, 9.0]]))
        df = A.to_dataframe()
        result = as_mat(df)
        assert isinstance(result, Mat)
        assert result.shape == (1, 3)
        np.testing.assert_array_equal(np.asarray(result), np.asarray(A))


class TestTypePromotion:
    def test_integer_dataframe_promoted_to_float64(self):
        """as_mat on an integer DataFrame promotes to float64."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        assert df["a"].dtype != np.float64  # confirm input is int
        A = as_mat(df)
        assert isinstance(A, Mat)
        assert A.dtype == np.float64
        np.testing.assert_array_equal(
            np.asarray(A), np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
        )

    def test_mixed_numeric_dataframe_promoted_to_float64(self):
        """as_mat on a DataFrame with mixed int/float columns -> float64."""
        df = pd.DataFrame({"ints": [1, 2], "floats": [3.5, 4.5]})
        A = as_mat(df)
        assert isinstance(A, Mat)
        assert A.dtype == np.float64
        np.testing.assert_array_equal(
            np.asarray(A), np.array([[1.0, 3.5], [2.0, 4.5]])
        )


class TestLabelPreservation:
    def test_series_index_preserved_through_to_series(self):
        """to_series(index=...) preserves named index labels."""
        u = _c[100, 200, 300]
        idx = ["alpha", "beta", "gamma"]
        s = u.to_series(index=idx, name="measurements")
        assert s.index.tolist() == idx
        assert s.name == "measurements"

    def test_dataframe_columns_preserved_through_to_dataframe(self):
        """to_dataframe(columns=...) preserves column labels."""
        A = mat([1, 2], [3, 4], [5, 6])
        cols = ["x", "y", "z"]
        df = A.to_dataframe(columns=cols)
        assert list(df.columns) == cols

    def test_to_series_default_index_is_range(self):
        """to_series() without index uses integer range index."""
        u = _c[1, 2, 3]
        s = u.to_series()
        assert list(s.index) == [0, 1, 2]

    def test_to_dataframe_default_columns_is_range(self):
        """to_dataframe() without columns uses integer range."""
        A = mat([1, 2], [3, 4])
        df = A.to_dataframe()
        assert list(df.columns) == [0, 1]

    def test_colvec_to_series_dtype_is_float64(self):
        """Series produced by to_series has dtype float64."""
        u = _c[1, 2, 3]
        s = u.to_series()
        assert s.dtype == np.float64

    def test_mat_to_dataframe_dtypes_are_float64(self):
        """DataFrame produced by to_dataframe has float64 columns."""
        A = mat([1, 2], [3, 4])
        df = A.to_dataframe()
        for col in df.columns:
            assert df[col].dtype == np.float64
