"""Tests for matplotlib interoperability with ColVec and Mat.

Verifies that nemopy types (ColVec, Mat) work seamlessly with matplotlib's
most-used plotting functions. Per DESIGN_APPENDICES.md §12, ColVec and Mat
are ndarray subclasses and should be accepted by libraries that consume
ndarrays. ColVec has shape (n, 1); some matplotlib functions accept 2D
input directly, others require 1D via .to_flat().

## Test: test_plot_xy_colvec
- Goal: plt.plot(x, y) accepts ColVec for both x and y without error.
- Source: Issue #61 — matplotlib compatibility with (n,1) ColVec.
- Expected: Plot created, returns valid Line2D list, figure/axes valid.

## Test: test_plot_single_colvec
- Goal: plt.plot(y) accepts a single ColVec arg (implicit x-axis).
- Source: Issue #61 — matplotlib compatibility with (n,1) ColVec.
- Expected: Plot created, returns valid Line2D list, figure/axes valid.

## Test: test_plot_with_np_linspace
- Goal: plt.plot works with np.linspace passed through ColVec.
- Source: Issue #61 — interop with NumPy-generated data wrapped in ColVec.
- Expected: Plot created without error, valid axes returned.

## Test: test_scatter_colvec
- Goal: plt.scatter(x, y) accepts ColVec for both arguments.
- Source: Issue #61 — matplotlib compatibility with (n,1) ColVec.
- Expected: PathCollection returned, figure/axes valid.

## Test: test_bar_colvec_with_to_flat
- Goal: plt.bar works with ColVec using .to_flat() for height (bar
        requires 1D for the height argument).
- Source: Issue #61 — bar needs 1D; DESIGN_APPENDICES.md §13.2 to_flat().
- Expected: BarContainer returned, figure/axes valid.

## Test: test_bar_colvec_both_flat
- Goal: plt.bar works with both x and height flattened via .to_flat()
        since bar requires 1D for both arguments.
- Source: Issue #61 — bar needs 1D; DESIGN_APPENDICES.md §13.2 to_flat().
- Expected: BarContainer returned, figure/axes valid.

## Test: test_imshow_mat
- Goal: plt.imshow(mat) accepts a Mat directly.
- Source: Issue #61 — matplotlib compatibility with 2D Mat.
- Expected: AxesImage returned, figure/axes valid.

## Test: test_contour_mat
- Goal: plt.contour(mat) accepts a Mat directly.
- Source: Issue #61 — matplotlib compatibility with 2D Mat.
- Expected: QuadContourSet returned, figure/axes valid.

## Test: test_pcolormesh_mat
- Goal: plt.pcolormesh(mat) accepts a Mat directly.
- Source: Issue #61 — matplotlib compatibility with 2D Mat.
- Expected: QuadMesh returned, figure/axes valid.

## Test: test_hist_colvec_flat
- Goal: plt.hist accepts ColVec flattened via .to_flat() (hist needs 1D).
- Source: Issue #61 — hist needs 1D; DESIGN_APPENDICES.md §13.2 to_flat().
- Expected: Tuple of (n, bins, patches) returned, figure/axes valid.

## Test: test_errorbar_colvec
- Goal: plt.errorbar(x, y, yerr) works with ColVec flattened via .to_flat()
        since errorbar internally broadcasts y with yerr, which triggers
        nemopy's shape guard if y is (n,1).
- Source: Issue #61 — errorbar needs 1D; DESIGN_APPENDICES.md §13.2.
- Expected: ErrorbarContainer returned, figure/axes valid.

## Test: test_plot_colvec_to_flat_explicit
- Goal: plt.plot works with ColVec converted via .to_flat() for functions
        that may reject 2D input.
- Source: Issue #61 — verifying the .to_flat() escape hatch pattern.
- Expected: Plot created, returns valid Line2D list, figure/axes valid.

## Test: test_scatter_colvec_to_flat
- Goal: plt.scatter works with ColVec flattened via .to_flat() as an
        alternative pattern.
- Source: Issue #61 — verifying .to_flat() pattern for scatter.
- Expected: PathCollection returned, figure/axes valid.
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

mpl = pytest.importorskip("matplotlib")
plt = pytest.importorskip("matplotlib.pyplot")

from nemopy import ColVec, Mat, _c, mat


# ---------------------------------------------------------------------------
# Line plots
# ---------------------------------------------------------------------------


class TestLinePlots:

    def teardown_method(self):
        plt.close("all")

    def test_plot_xy_colvec(self):
        x = _c[1, 2, 3, 4, 5]
        y = _c[2, 4, 6, 8, 10]
        fig, ax = plt.subplots()
        lines = ax.plot(x, y)
        assert len(lines) == 1
        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_plot_single_colvec(self):
        y = _c[10, 20, 30, 40]
        fig, ax = plt.subplots()
        lines = ax.plot(y)
        assert len(lines) == 1
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_plot_with_np_linspace(self):
        t = np.linspace(0, 1, 50).reshape(-1, 1)
        x = ColVec(t)
        y = ColVec(np.sin(2 * np.pi * t))
        fig, ax = plt.subplots()
        lines = ax.plot(x, y)
        assert len(lines) == 1
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_plot_colvec_to_flat_explicit(self):
        x = _c[1, 2, 3, 4]
        y = _c[1, 4, 9, 16]
        fig, ax = plt.subplots()
        lines = ax.plot(x.to_flat(), y.to_flat())
        assert len(lines) == 1
        assert isinstance(ax, matplotlib.axes.Axes)


# ---------------------------------------------------------------------------
# Scatter and bar
# ---------------------------------------------------------------------------


class TestScatterBar:

    def teardown_method(self):
        plt.close("all")

    def test_scatter_colvec(self):
        x = _c[1, 2, 3, 4, 5]
        y = _c[5, 4, 3, 2, 1]
        fig, ax = plt.subplots()
        pc = ax.scatter(x, y)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_scatter_colvec_to_flat(self):
        x = _c[1, 2, 3]
        y = _c[3, 2, 1]
        fig, ax = plt.subplots()
        pc = ax.scatter(x.to_flat(), y.to_flat())
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_bar_colvec_with_to_flat(self):
        x = _c[1, 2, 3]
        height = _c[10, 20, 30]
        fig, ax = plt.subplots()
        bars = ax.bar(x.to_flat(), height.to_flat())
        assert len(bars) == 3
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_bar_colvec_both_flat(self):
        x = _c[1, 2, 3, 4]
        height = _c[4, 3, 2, 1]
        fig, ax = plt.subplots()
        bars = ax.bar(x.to_flat(), height.to_flat())
        assert len(bars) == 4
        assert isinstance(ax, matplotlib.axes.Axes)


# ---------------------------------------------------------------------------
# Matrix plots (imshow, contour, pcolormesh)
# ---------------------------------------------------------------------------


class TestMatrixPlots:

    def teardown_method(self):
        plt.close("all")

    def test_imshow_mat(self):
        A = mat([1, 4, 7], [2, 5, 8], [3, 6, 9])
        fig, ax = plt.subplots()
        im = ax.imshow(A)
        assert isinstance(im, matplotlib.image.AxesImage)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_contour_mat(self):
        A = mat([1, 4, 7], [2, 5, 8], [3, 6, 9])
        fig, ax = plt.subplots()
        cs = ax.contour(A)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_pcolormesh_mat(self):
        A = mat([1, 4, 7], [2, 5, 8], [3, 6, 9])
        fig, ax = plt.subplots()
        pm = ax.pcolormesh(A)
        assert isinstance(ax, matplotlib.axes.Axes)


# ---------------------------------------------------------------------------
# Statistical plots (hist, errorbar)
# ---------------------------------------------------------------------------


class TestStatisticalPlots:

    def teardown_method(self):
        plt.close("all")

    def test_hist_colvec_flat(self):
        v = _c[1, 2, 2, 3, 3, 3, 4, 4, 5]
        fig, ax = plt.subplots()
        n, bins, patches = ax.hist(v.to_flat())
        assert len(patches) > 0
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_errorbar_colvec(self):
        x = _c[1, 2, 3, 4]
        y = _c[10, 20, 30, 40]
        yerr = _c[1, 2, 1, 2]
        fig, ax = plt.subplots()
        container = ax.errorbar(x.to_flat(), y.to_flat(), yerr=yerr.to_flat())
        assert isinstance(ax, matplotlib.axes.Axes)
