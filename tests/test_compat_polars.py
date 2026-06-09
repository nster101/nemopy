"""Tests for Polars compatibility with ColVec and Mat.

Verifies that nemopy types round-trip through polars Series/DataFrame
correctly, that the ``nemo`` accessor works on polars objects, and that
inbound/outbound conversions preserve values and types. Per
DESIGN_APPENDICES.md §19, polars is an optional dependency.

## Test: test_df_nemo_col_returns_colvec
- Goal: df.nemo.col("name") extracts a single DataFrame column as a ColVec.
- Source: DESIGN_APPENDICES.md §19.3 — "df.nemo.col(name) → ColVec".
- Expected: result is ColVec with shape (3,1), values match the column.

## Test: test_df_nemo_mat_selected_columns
- Goal: df.nemo.mat(["a","b"]) extracts selected columns as a Mat.
- Source: DESIGN_APPENDICES.md §19.3 — "df.nemo.mat(names) → Mat".
- Expected: result is Mat with shape (3,2), values match the selected columns.

## Test: test_df_nemo_mat_all_columns
- Goal: df.nemo.mat() with no args extracts all columns as a Mat.
- Source: DESIGN_APPENDICES.md §19.3 — "df.nemo.mat(names) → Mat" (names defaults to all).
- Expected: result is Mat with shape (3,3), values match the full DataFrame.

## Test: test_series_nemo_col
- Goal: s.nemo.col() converts a polars Series to a ColVec.
- Source: DESIGN_APPENDICES.md §19.3 — "series.nemo.col() → ColVec".
- Expected: result is ColVec with shape (3,1), values match the Series.

## Test: test_as_col_from_polars_series
- Goal: as_col(pl.Series) converts a polars Series to a ColVec.
- Source: DESIGN_APPENDICES.md §19.1 — "as_col accepts polars.Series".
- Expected: result is ColVec with shape (3,1), values match.

## Test: test_as_mat_from_polars_dataframe
- Goal: as_mat(pl.DataFrame) converts a polars DataFrame to a Mat.
- Source: DESIGN_APPENDICES.md §19.1 — "as_mat accepts polars.DataFrame".
- Expected: result is Mat with shape (3,2), values match.

## Test: test_colvec_to_polars_with_name
- Goal: ColVec.to_polars(name) returns a polars Series with the given name.
- Source: DESIGN_APPENDICES.md §19.2 — "ColVec.to_polars(name=None) → polars.Series".
- Expected: result is polars.Series, name matches, values match.

## Test: test_mat_to_polars_with_schema
- Goal: Mat.to_polars(schema) returns a polars DataFrame with given column names.
- Source: DESIGN_APPENDICES.md §19.2 — "Mat.to_polars(schema=None) → polars.DataFrame".
- Expected: result is polars.DataFrame, columns match schema, values match.

## Test: test_roundtrip_colvec_through_polars
- Goal: ColVec → polars.Series → as_col → ColVec preserves values exactly.
- Source: DESIGN_APPENDICES.md §19.1, §19.2 — round-trip consistency.
- Expected: recovered ColVec equals original element-wise.

## Test: test_roundtrip_mat_through_polars
- Goal: Mat → polars.DataFrame → as_mat → Mat preserves values exactly.
- Source: DESIGN_APPENDICES.md §19.1, §19.2 — round-trip consistency.
- Expected: recovered Mat equals original element-wise.

## Test: test_integer_series_converts_to_float64
- Goal: A polars Series of integers converts to a float64 ColVec.
- Source: DESIGN_APPENDICES.md §19.1 — "as_col … astype(float)".
- Expected: result dtype is float64, values match as floats.

## Test: test_single_element_colvec_to_polars
- Goal: A single-element ColVec round-trips through to_polars correctly.
- Source: DESIGN_APPENDICES.md §19.2.
- Expected: result Series has length 1, value matches.

## Test: test_mat_to_polars_default_schema
- Goal: Mat.to_polars() with no schema uses default col_0, col_1, ... names.
- Source: DESIGN_APPENDICES.md §19.2 — schema defaults to col_0, col_1, etc.
- Expected: column names are ["col_0", "col_1"].

## Test: test_ols_workflow_from_polars
- Goal: Load data from a polars DataFrame, perform OLS via X.T @ X, predict.
- Source: DESIGN_APPENDICES.md §19.3 (accessor usage in linear algebra workflow).
- Expected: predicted values are close to true y values.
"""

import numpy as np
import pytest

pl = pytest.importorskip("polars")

from nemopy import ColVec, Mat, _c, as_col, as_mat, mat

import nemopy.polars  # noqa: F401 — registers df.nemo and s.nemo accessors


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------


class TestAccessors:

    def test_df_nemo_col_returns_colvec(self):
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
        result = df.nemo.col("x")
        assert isinstance(result, ColVec)
        assert result.shape == (3, 1)
        np.testing.assert_array_equal(result.to_flat(), [1.0, 2.0, 3.0])

    def test_df_nemo_mat_selected_columns(self):
        df = pl.DataFrame(
            {"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0], "c": [7.0, 8.0, 9.0]}
        )
        result = df.nemo.mat(["a", "b"])
        assert isinstance(result, Mat)
        assert result.shape == (3, 2)
        np.testing.assert_array_equal(
            np.asarray(result), [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
        )

    def test_df_nemo_mat_all_columns(self):
        df = pl.DataFrame(
            {"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0], "c": [7.0, 8.0, 9.0]}
        )
        result = df.nemo.mat()
        assert isinstance(result, Mat)
        assert result.shape == (3, 3)
        np.testing.assert_array_equal(
            np.asarray(result),
            [[1.0, 4.0, 7.0], [2.0, 5.0, 8.0], [3.0, 6.0, 9.0]],
        )

    def test_series_nemo_col(self):
        s = pl.Series("vals", [10.0, 20.0, 30.0])
        result = s.nemo.col()
        assert isinstance(result, ColVec)
        assert result.shape == (3, 1)
        np.testing.assert_array_equal(result.to_flat(), [10.0, 20.0, 30.0])


# ---------------------------------------------------------------------------
# Inbound conversion
# ---------------------------------------------------------------------------


class TestInboundConversion:

    def test_as_col_from_polars_series(self):
        s = pl.Series("x", [1.0, 2.0, 3.0])
        result = as_col(s)
        assert isinstance(result, ColVec)
        assert result.shape == (3, 1)
        np.testing.assert_array_equal(result.to_flat(), [1.0, 2.0, 3.0])

    def test_as_mat_from_polars_dataframe(self):
        df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        result = as_mat(df)
        assert isinstance(result, Mat)
        assert result.shape == (3, 2)
        np.testing.assert_array_equal(
            np.asarray(result), [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
        )


# ---------------------------------------------------------------------------
# Outbound conversion
# ---------------------------------------------------------------------------


class TestOutboundConversion:

    def test_colvec_to_polars_with_name(self):
        v = _c[1, 2, 3]
        s = v.to_polars(name="x")
        assert isinstance(s, pl.Series)
        assert s.name == "x"
        assert s.to_list() == [1.0, 2.0, 3.0]

    def test_mat_to_polars_with_schema(self):
        A = mat([1, 2, 3], [4, 5, 6])
        df = A.to_polars(schema=["x", "y"])
        assert isinstance(df, pl.DataFrame)
        assert df.columns == ["x", "y"]
        assert df.shape == (3, 2)
        np.testing.assert_array_equal(df["x"].to_list(), [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(df["y"].to_list(), [4.0, 5.0, 6.0])

    def test_mat_to_polars_default_schema(self):
        A = mat([1, 2], [3, 4])
        df = A.to_polars()
        assert df.columns == ["col_0", "col_1"]


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:

    def test_roundtrip_colvec_through_polars(self):
        original = _c[3.14, 2.72, 1.41]
        s = original.to_polars(name="rt")
        recovered = as_col(s)
        assert isinstance(recovered, ColVec)
        np.testing.assert_allclose(recovered.to_flat(), original.to_flat())

    def test_roundtrip_mat_through_polars(self):
        original = mat([1, 2, 3], [4, 5, 6])
        df = original.to_polars(schema=["a", "b"])
        recovered = as_mat(df)
        assert isinstance(recovered, Mat)
        np.testing.assert_allclose(np.asarray(recovered), np.asarray(original))


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:

    def test_integer_series_converts_to_float64(self):
        s = pl.Series("ints", [1, 2, 3])
        result = as_col(s)
        assert isinstance(result, ColVec)
        assert result.dtype == np.float64
        np.testing.assert_array_equal(result.to_flat(), [1.0, 2.0, 3.0])

    def test_single_element_colvec_to_polars(self):
        v = _c[42]
        s = v.to_polars(name="single")
        assert isinstance(s, pl.Series)
        assert len(s) == 1
        assert s[0] == 42.0

    def test_ols_workflow_from_polars(self):
        # y = 2*x1 + 3*x2 + 1 (with an intercept column of ones)
        df = pl.DataFrame({
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "x2": [2.0, 1.0, 3.0, 2.0, 4.0],
            "ones": [1.0, 1.0, 1.0, 1.0, 1.0],
            "y": [9.0, 8.0, 16.0, 15.0, 23.0],
        })
        X = df.nemo.mat(["x1", "x2", "ones"])
        y = df.nemo.col("y")

        # OLS: beta = (X^T X)^{-1} X^T y
        beta = (X.T @ X).inv @ X.T @ y
        assert isinstance(beta, ColVec)
        assert beta.shape == (3, 1)

        # Coefficients should be close to [2, 3, 1]
        np.testing.assert_allclose(beta.to_flat(), [2.0, 3.0, 1.0], atol=1e-10)

        # Predict and compare
        y_hat = X @ beta
        np.testing.assert_allclose(y_hat.to_flat(), y.to_flat(), atol=1e-10)
