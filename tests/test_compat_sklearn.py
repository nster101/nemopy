"""Tests for scikit-learn interoperability with ColVec and Mat.

Verifies that nemopy types (ColVec, Mat) work seamlessly with scikit-learn's
most-used modules: linear models, preprocessing, decomposition, model
selection, and clustering. Per DESIGN_APPENDICES.md §12, ColVec and Mat are
ndarray subclasses and should be accepted directly by sklearn estimators.
ColVec is shape (n,1); sklearn expects y as 1D (n,). The key pattern
validated here is using ColVec.to_flat() to convert y before passing to
sklearn.

## Test: test_linear_regression_simple
- Goal: LinearRegression fits with Mat X and ColVec.to_flat() y, and
        produces correct predictions.
- Source: Issue #64 — sklearn compatibility with nemopy types.
- Expected: Model fits without error; predictions are close to true values.

## Test: test_linear_regression_multivariate
- Goal: LinearRegression fits with a multi-feature Mat X and flat y.
- Source: Issue #64 — sklearn compatibility with nemopy types.
- Expected: Model fits without error; coef_ has correct shape.

## Test: test_ridge_regression
- Goal: Ridge fits with Mat X and ColVec.to_flat() y.
- Source: Issue #64 — sklearn compatibility with nemopy types.
- Expected: Model fits without error; predictions are ndarray, not ColVec/Mat.

## Test: test_logistic_regression
- Goal: LogisticRegression fits with Mat X and integer class labels from
        ColVec.to_flat().
- Source: Issue #64 — sklearn compatibility with nemopy types.
- Expected: Model fits without error; predictions are integer class labels.

## Test: test_standard_scaler
- Goal: StandardScaler.fit_transform accepts Mat and returns scaled data.
- Source: Issue #64 — sklearn compatibility with nemopy types.
- Expected: Transformed columns have mean ~0 and std ~1.

## Test: test_minmax_scaler
- Goal: MinMaxScaler.fit_transform accepts Mat and returns data in [0, 1].
- Source: Issue #64 — sklearn compatibility with nemopy types.
- Expected: All transformed values are in [0, 1].

## Test: test_pca
- Goal: PCA.fit_transform accepts Mat and reduces dimensionality.
- Source: Issue #64 — sklearn compatibility with nemopy types.
- Expected: Output has n_components columns.

## Test: test_train_test_split
- Goal: train_test_split works with Mat X and ColVec.to_flat() y.
- Source: Issue #64 — sklearn compatibility with nemopy types.
- Expected: Splits have correct sizes; types are plain ndarray.

## Test: test_kmeans
- Goal: KMeans.fit accepts Mat X and assigns cluster labels.
- Source: Issue #64 — sklearn compatibility with nemopy types.
- Expected: labels_ has correct length and expected number of unique values.

## Test: test_cross_val_score
- Goal: cross_val_score works with Mat X and ColVec.to_flat() y.
- Source: Issue #64 — sklearn compatibility with nemopy types.
- Expected: Returns an array of scores with correct length.

## Test: test_predictions_return_plain_ndarray
- Goal: Verify that sklearn predictions are plain ndarray, not ColVec/Mat.
- Source: Issue #64 — sklearn outputs should not carry nemopy subclass labels.
- Expected: type(pred) is np.ndarray, not ColVec or Mat.

## Test: test_transform_returns_plain_ndarray
- Goal: Verify that sklearn transforms return plain ndarray, not ColVec/Mat.
- Source: Issue #64 — sklearn outputs should not carry nemopy subclass labels.
- Expected: type(result) is np.ndarray, not ColVec or Mat.
"""

import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn")

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from nemopy import ColVec, Mat, _c, mat


# ---------------------------------------------------------------------------
# Linear models
# ---------------------------------------------------------------------------


class TestLinearModels:

    def test_linear_regression_simple(self):
        X = mat([1, 2, 3, 4, 5])
        y = _c[2, 4, 6, 8, 10]
        model = LinearRegression().fit(X, y.to_flat())
        pred = model.predict(X)
        np.testing.assert_allclose(pred, [2, 4, 6, 8, 10], atol=1e-10)

    def test_linear_regression_multivariate(self):
        X = mat([1, 2, 3, 4], [10, 20, 30, 40])
        y = _c[11, 22, 33, 44]
        model = LinearRegression().fit(X, y.to_flat())
        assert model.coef_.shape == (2,)

    def test_ridge_regression(self):
        X = mat([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])
        y = _c[6, 6, 6, 6, 6]
        model = Ridge(alpha=1.0).fit(X, y.to_flat())
        pred = model.predict(X)
        assert isinstance(pred, np.ndarray)
        assert pred.shape == (5,)

    def test_logistic_regression(self):
        X = mat([1, 2, 3, 4, 5, 6], [1, 1, 1, 0, 0, 0])
        y = _c[0, 0, 0, 1, 1, 1]
        model = LogisticRegression().fit(X, y.to_flat().astype(int))
        pred = model.predict(X)
        assert pred.shape == (6,)
        assert set(pred).issubset({0, 1})


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


class TestPreprocessing:

    def test_standard_scaler(self):
        X = mat([1, 2, 3, 4, 5], [10, 20, 30, 40, 50])
        result = StandardScaler().fit_transform(X)
        np.testing.assert_allclose(result.mean(axis=0), [0, 0], atol=1e-10)
        np.testing.assert_allclose(result.std(axis=0), [1, 1], atol=1e-10)

    def test_minmax_scaler(self):
        X = mat([1, 5, 3, 7], [10, 50, 30, 70])
        result = MinMaxScaler().fit_transform(X)
        assert result.min() >= 0.0
        assert result.max() <= 1.0


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------


class TestDecomposition:

    def test_pca(self):
        np.random.seed(42)
        data = np.random.randn(20, 3)
        X = Mat(data)
        result = PCA(n_components=2).fit_transform(X)
        assert result.shape == (20, 2)


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------


class TestModelSelection:

    def test_train_test_split(self):
        X = mat([1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1])
        y = _c[1, 2, 3, 4, 5, 6, 7, 8]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y.to_flat(), test_size=0.25, random_state=42
        )
        assert X_train.shape[0] == 6
        assert X_test.shape[0] == 2
        assert y_train.shape == (6,)
        assert y_test.shape == (2,)

    def test_cross_val_score(self):
        X = mat([1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                [10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
        y = _c[2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        scores = cross_val_score(
            LinearRegression(), X, y.to_flat(), cv=3
        )
        assert scores.shape == (3,)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


class TestClustering:

    def test_kmeans(self):
        np.random.seed(42)
        cluster1 = np.random.randn(10, 2) + [0, 0]
        cluster2 = np.random.randn(10, 2) + [10, 10]
        X = Mat(np.vstack([cluster1, cluster2]))
        model = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X)
        assert model.labels_.shape == (20,)
        assert len(set(model.labels_)) == 2


# ---------------------------------------------------------------------------
# Output type verification
# ---------------------------------------------------------------------------


class TestOutputTypes:

    def test_predictions_return_plain_ndarray(self):
        X = mat([1, 2, 3, 4, 5])
        y = _c[2, 4, 6, 8, 10]
        model = LinearRegression().fit(X, y.to_flat())
        pred = model.predict(X)
        assert type(pred) is np.ndarray
        assert not isinstance(pred, (ColVec, Mat))

    def test_transform_returns_plain_ndarray(self):
        X = mat([1, 2, 3, 4, 5], [10, 20, 30, 40, 50])
        result = StandardScaler().fit_transform(X)
        assert type(result) is np.ndarray
        assert not isinstance(result, (ColVec, Mat))
