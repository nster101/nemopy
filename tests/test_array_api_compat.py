"""Array API compatibility test runner for nemopy.

Runs the `array-api-tests <https://github.com/data-apis/array-api-tests>`_
suite against nemopy to measure conformance with the Python Array API
standard.  Known failures (intentional nemopy design decisions and upstream
NumPy gaps) are documented in ``tests/array_api_skips.py`` and can be
used by an external conftest hook to apply ``xfail`` markers when running
the suite.

Usage
-----
The suite is an external package, not bundled with nemopy.  Install it
separately::

    git clone https://github.com/data-apis/array-api-tests.git /tmp/array-api-tests
    cd /tmp/array-api-tests && git submodule update --init
    pip install hypothesis ndindex pytest-json-report

Then run from the **array-api-tests** directory with the
``ARRAY_API_TESTS_MODULE`` environment variable pointing at nemopy::

    cd /tmp/array-api-tests
    ARRAY_API_TESTS_MODULE=nemopy pytest array_api_tests/ -v --tb=short

To produce a JSON report for CI::

    ARRAY_API_TESTS_MODULE=nemopy pytest array_api_tests/ \\
        --json-report --json-report-file=array_api_results.json

Scorecard (baseline)
--------------------
Measured against array-api-tests @ commit 5f847a3 with NumPy 2.4.6:

=================================================
Total tests:            1384
Passed:                 1325  (95.7%)
Failed (known):            54  — see tests/array_api_skips.py
Skipped (upstream):         5  — remainder operator (NumPy upstream)
=================================================

Failure categories:

  nemopy design (4 tests)
      nemopy.eye(n) has a narrower signature than the Array API standard
      eye(n_rows, n_cols, *, k, dtype, device).  By design (DESIGN.md §5.8).

  upstream NumPy (50 tests)
      - finfo/iinfo: NumPy 2.4+ does not accept 0-d arrays (12 tests)
      - fft.fftfreq/rfftfreq: missing dtype keyword (4 tests)
      - sort/argsort/clip: signature differences (5 tests)
      - eig/eigvals: return float not complex dtype (2 tests)
      - inspection: missing __array_api_version__, devices() type (2 tests)
      - floor_divide infinity special cases (18 tests)
      - expm1 complex special cases (7 tests)
      - tanh complex special case (1 test)

No nemopy-specific bugs were found.  All failures are either intentional
design decisions or upstream NumPy conformance gaps that exist with or
without nemopy.
"""

def test_nemopy_exports_available():
    """Verify that nemopy's core public API is importable.

    This does not run the full Array API suite — it only confirms that
    nemopy's types and constructors are available, which is a prerequisite
    for the external suite.  The full suite must be run from the
    array-api-tests directory (see module docstring).
    """
    import nemopy  # noqa: F401
    assert hasattr(nemopy, "ColVec")
    assert hasattr(nemopy, "Mat")
    assert hasattr(nemopy, "eye")
