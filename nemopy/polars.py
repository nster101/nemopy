"""Optional Polars integration for nemopy.

Import this module to register the ``nemo`` accessor on ``polars.DataFrame``
and ``polars.Series``:

.. code-block:: python

    import polars as pl
    import nemopy.polars          # registers df.nemo and s.nemo

    df = pl.read_csv("data.csv")
    X = df.nemo.mat(["x1", "x2", "x3"])   # → Mat
    y = df.nemo.col("target")              # → ColVec

    beta = (X.T @ X).inv @ X.T @ y        # OLS in one line

Raises ``ImportError`` at import time if polars is not installed.
"""

import polars as pl

from nemopy._constructors import as_col, as_mat


@pl.api.register_dataframe_namespace("nemo")
class _NemoDataFrameNamespace:
    """The ``df.nemo`` accessor for polars DataFrames.

    Registered automatically when ``nemopy.polars`` is imported.

    Examples
    --------
    .. code-block:: python

        import polars as pl
        import nemopy.polars

        df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})

        df.nemo.col("a")          # ColVec([1.0, 2.0, 3.0])
        df.nemo.mat(["a", "b"])   # Mat(3x2)
    """

    def __init__(self, df: pl.DataFrame) -> None:
        self._df = df

    def col(self, name: str):
        """Extract one column as a ``ColVec``.

        Parameters
        ----------
        name : str
            Column name.

        Returns
        -------
        ColVec
            Shape ``(n, 1)``.
        """
        return as_col(self._df[name].to_numpy())

    def mat(self, names=None):
        """Extract one or more columns as a ``Mat``.

        Parameters
        ----------
        names : list of str, optional
            Column names to include. Defaults to all columns.

        Returns
        -------
        Mat
            Shape ``(n, k)`` where ``k = len(names)``.
        """
        sub = self._df.select(names) if names is not None else self._df
        return as_mat(sub.to_numpy())


@pl.api.register_series_namespace("nemo")
class _NemoSeriesNamespace:
    """The ``s.nemo`` accessor for polars Series.

    Registered automatically when ``nemopy.polars`` is imported.

    Examples
    --------
    .. code-block:: python

        import polars as pl
        import nemopy.polars

        s = pl.Series("x", [1.0, 2.0, 3.0])
        s.nemo.col()    # ColVec([1.0, 2.0, 3.0])
    """

    def __init__(self, s: pl.Series) -> None:
        self._s = s

    def col(self):
        """Convert this Series to a ``ColVec``.

        Returns
        -------
        ColVec
            Shape ``(n, 1)``.
        """
        return as_col(self._s.to_numpy())
