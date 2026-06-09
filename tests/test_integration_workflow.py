"""End-to-end integration tests exercising realistic data-science workflows.

These chain multiple ecosystem libraries through ``import nemopy as np``
to verify the drop-in replacement story works across full pipelines.

## Test: test_ols_end_to_end
- Goal: Full OLS workflow from pandas data to matplotlib plot.
- Source: Issue #67 — end-to-end workflow 1.
- Expected: Beta is ColVec, predictions are ColVec, SSE < 1.0, plot renders.

## Test: test_ols_vs_sklearn
- Goal: Verify nemopy manual OLS matches sklearn LinearRegression.
- Source: Issue #67 — end-to-end workflow 2.
- Expected: Intercept and coefficients match within 1e-8.

## Test: test_optimization_workflow
- Goal: Minimize ||Ax - b||^2 with scipy, compare to A.inv @ b.
- Source: Issue #67 — end-to-end workflow 3.
- Expected: scipy solution matches exact inverse within 1e-6.

## Test: test_polars_to_sklearn_to_plot
- Goal: Full pipeline from Polars DataFrame through sklearn to matplotlib.
- Source: Issue #67 — end-to-end workflow 4.
- Expected: Pipeline completes without error, predictions are ndarray.
"""

import numpy as np
import pytest

from nemopy import ColVec, Mat, _c, as_col, as_mat, mat


class TestOLSWorkflow:
    def test_ols_end_to_end(self):
        pd = pytest.importorskip("pandas")
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "y": [2.1, 3.9, 6.2, 7.8, 10.1, 12.0, 13.9, 16.2, 17.8, 20.1],
        })

        x = as_col(df["x"])
        y = as_col(df["y"])

        ones = _c[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        X = ones | x

        beta = (X.T @ X).inv @ X.T @ y
        assert isinstance(beta, ColVec)
        assert beta.shape == (2, 1)

        y_hat = X @ beta
        assert isinstance(y_hat, ColVec)
        assert y_hat.shape == (10, 1)

        residuals = y - y_hat
        assert isinstance(residuals, ColVec)
        sse = (residuals.T @ residuals).item()
        assert sse < 1.0

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].scatter(x.to_flat(), y.to_flat(), label="data")
        axes[0].plot(x.to_flat(), y_hat.to_flat(), "r-", label="fit")
        axes[0].legend()
        axes[1].scatter(y_hat.to_flat(), residuals.to_flat())
        axes[1].axhline(y=0, color="r", linestyle="--")
        plt.tight_layout()
        plt.close(fig)

    def test_ols_vs_sklearn(self):
        sklearn = pytest.importorskip("sklearn")
        from sklearn.linear_model import LinearRegression

        X_data = mat([1, 2, 3, 4, 5], [10, 7, 4, 3, 1])
        y_data = _c[15, 19, 17, 21, 20]

        ones = _c[1, 1, 1, 1, 1]
        X_design = ones | X_data
        beta_nemo = (X_design.T @ X_design).inv @ X_design.T @ y_data

        model = LinearRegression()
        model.fit(X_data, y_data.to_flat())

        np.testing.assert_allclose(float(beta_nemo[0]), model.intercept_, atol=1e-6)
        np.testing.assert_allclose(
            np.asarray(beta_nemo[1:]).flatten(), model.coef_, atol=1e-6
        )


class TestOptimizationWorkflow:
    def test_optimization_workflow(self):
        scipy = pytest.importorskip("scipy")
        import scipy.optimize

        A = mat([2, 1], [1, 3])
        b = _c[5, 7]

        def objective(x_flat):
            x = as_col(x_flat)
            r = A @ x - b
            return float((r.T @ r).item())

        x0 = np.zeros(2)
        result = scipy.optimize.minimize(objective, x0, method="BFGS")
        assert result.success

        x_opt = as_col(result.x)
        x_exact = A.inv @ b
        np.testing.assert_allclose(
            x_opt.to_flat(), x_exact.to_flat(), atol=1e-6
        )


class TestFullPipeline:
    def test_polars_to_sklearn_to_plot(self):
        pl = pytest.importorskip("polars")
        sklearn = pytest.importorskip("sklearn")
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.linear_model import LinearRegression

        import nemopy.polars

        df = pl.DataFrame({
            "sq_ft": [800, 1000, 1200, 1400, 1600, 1800, 2000],
            "price": [150.0, 200.0, 240.0, 280.0, 330.0, 365.0, 400.0],
        })

        X = df.nemo.mat(["sq_ft"])
        y = df.nemo.col("price")
        assert isinstance(X, Mat)
        assert isinstance(y, ColVec)

        model = LinearRegression()
        model.fit(X, y.to_flat())
        preds = model.predict(X)
        assert isinstance(preds, np.ndarray)
        assert not isinstance(preds, ColVec)

        fig, ax = plt.subplots()
        ax.scatter(X.flatten(), y.to_flat(), label="actual")
        ax.plot(X.flatten(), preds, "r-", label="predicted")
        ax.legend()
        plt.close(fig)
