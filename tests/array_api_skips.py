"""Known nemopy incompatibilities with the Array API standard.

Registry of Array API test suite failures and their root causes.  Each
entry maps a test node ID to a human-readable reason.  An external
conftest hook can consume ``ALL_KNOWN_FAILURES`` to apply ``xfail``
markers automatically when running the array-api-tests suite.

Categories
----------
nemopy_design
    Failures caused by nemopy's intentional design decisions (narrower
    ``eye()`` signature, float64-only dtype policy, broadcasting guards).
upstream_numpy
    Failures in NumPy itself (not nemopy-specific).  These fail identically
    when running the Array API test suite against plain NumPy.
"""

# ---------------------------------------------------------------------------
# nemopy design — intentional incompatibilities
# ---------------------------------------------------------------------------

NEMOPY_DESIGN = {
    # nemopy.eye(n) accepts a single int (DESIGN.md §5.8).
    # The Array API standard expects eye(n_rows, n_cols=None, *, k=0,
    # dtype=None, device=None).
    "test_creation_functions.py::test_eye": (
        "nemopy.eye(n) takes one positional argument by design (DESIGN.md "
        "§5.8); Array API expects eye(n_rows, n_cols, *, k, dtype, device)"
    ),
    "test_signatures.py::test_func_signature[eye]": (
        "nemopy.eye signature is (n,) not (n_rows, n_cols, *, k, dtype, device)"
    ),
    # Cascading from eye() signature — test helpers call eye(n, dtype=...)
    "test_linalg.py::test_cholesky": (
        "Test helper calls eye(n, dtype=...) which nemopy.eye does not accept"
    ),
    "test_linalg.py::test_matrix_power": (
        "Test helper calls eye(n, dtype=...) which nemopy.eye does not accept"
    ),
}

# ---------------------------------------------------------------------------
# Upstream NumPy — not nemopy-specific
# ---------------------------------------------------------------------------

UPSTREAM_NUMPY = {
    # np.finfo / np.iinfo do not accept 0-d array arguments in NumPy >= 2.4.
    # Confirmed: fails identically with plain NumPy (no nemopy involved).
    "test_data_type_functions.py::test_finfo[float64]": (
        "NumPy 2.4+ np.finfo(0-d array) raises ValueError (upstream bug)"
    ),
    "test_data_type_functions.py::test_finfo[complex128]": (
        "NumPy 2.4+ np.finfo(0-d array) raises ValueError (upstream bug)"
    ),
    "test_data_type_functions.py::test_finfo[float32]": (
        "NumPy np.finfo(...).eps returns np.float32, not Python float"
    ),
    "test_data_type_functions.py::test_finfo[complex64]": (
        "NumPy np.finfo(...).eps returns np.float32, not Python float"
    ),
    "test_data_type_functions.py::test_iinfo[int8]": (
        "NumPy 2.4+ np.iinfo(0-d array) raises ValueError (upstream bug)"
    ),
    "test_data_type_functions.py::test_iinfo[int16]": (
        "NumPy 2.4+ np.iinfo(0-d array) raises ValueError (upstream bug)"
    ),
    "test_data_type_functions.py::test_iinfo[int32]": (
        "NumPy 2.4+ np.iinfo(0-d array) raises ValueError (upstream bug)"
    ),
    "test_data_type_functions.py::test_iinfo[int64]": (
        "NumPy 2.4+ np.iinfo(0-d array) raises ValueError (upstream bug)"
    ),
    "test_data_type_functions.py::test_iinfo[uint8]": (
        "NumPy 2.4+ np.iinfo(0-d array) raises ValueError (upstream bug)"
    ),
    "test_data_type_functions.py::test_iinfo[uint16]": (
        "NumPy 2.4+ np.iinfo(0-d array) raises ValueError (upstream bug)"
    ),
    "test_data_type_functions.py::test_iinfo[uint32]": (
        "NumPy 2.4+ np.iinfo(0-d array) raises ValueError (upstream bug)"
    ),
    "test_data_type_functions.py::test_iinfo[uint64]": (
        "NumPy 2.4+ np.iinfo(0-d array) raises ValueError (upstream bug)"
    ),
    # numpy.fft.fftfreq / rfftfreq do not accept dtype keyword.
    # Array API 2023.12 added dtype; NumPy has not implemented it.
    "test_fft.py::test_fftfreq": (
        "NumPy fft.fftfreq does not accept dtype keyword (Array API 2023.12)"
    ),
    "test_fft.py::test_rfftfreq": (
        "NumPy fft.rfftfreq does not accept dtype keyword (Array API 2023.12)"
    ),
    "test_signatures.py::test_extension_func_signature[fft.fftfreq]": (
        "NumPy fft.fftfreq signature lacks dtype parameter"
    ),
    "test_signatures.py::test_extension_func_signature[fft.rfftfreq]": (
        "NumPy fft.rfftfreq signature lacks dtype parameter"
    ),
    # numpy.sort / numpy.argsort signatures differ from Array API standard.
    "test_signatures.py::test_func_signature[sort]": (
        "NumPy sort() signature uses 'axis' not 'descending'/'stable'"
    ),
    "test_signatures.py::test_func_signature[argsort]": (
        "NumPy argsort() signature uses 'axis' not 'descending'/'stable'"
    ),
    "test_sorting_functions.py::test_sort": (
        "NumPy sort() API differs from Array API standard"
    ),
    "test_sorting_functions.py::test_argsort": (
        "NumPy argsort() API differs from Array API standard"
    ),
    # numpy.clip signature differs from Array API standard.
    "test_signatures.py::test_func_signature[clip]": (
        "NumPy clip() signature differs from Array API standard"
    ),
    # Eigenvalue return dtypes: Array API requires complex output for float
    # input; NumPy returns float when eigenvalues are real.
    "test_linalg.py::test_eig": (
        "NumPy eig returns float eigenvalues for float input; "
        "Array API requires complex"
    ),
    "test_linalg.py::test_eigvals": (
        "NumPy eigvals returns float for float input; Array API requires complex"
    ),
    # Inspection functions: __array_api_version__ and devices() return type.
    "test_inspection_functions.py::TestInspection::test_capabilities": (
        "nemopy (via NumPy) does not expose __array_api_version__"
    ),
    "test_inspection_functions.py::TestInspection::test_devices": (
        "NumPy __array_namespace_info__.devices() returns list, not tuple"
    ),
}

# ---------------------------------------------------------------------------
# Special-case edge cases — floor_divide / expm1 / tanh with infinity & NaN
# ---------------------------------------------------------------------------

_FLOOR_DIV_SPECIAL = (
    "NumPy floor_divide infinity/NaN special cases do not match Array API spec"
)
_EXPM1_SPECIAL = (
    "NumPy complex expm1 special cases do not match Array API spec"
)
_TANH_SPECIAL = (
    "NumPy complex tanh special cases do not match Array API spec"
)

UPSTREAM_NUMPY_SPECIAL_CASES = {
    # floor_divide function-form
    "test_special_cases.py::test_binary[floor_divide(x1_i is +infinity and isfinite(x2_i) and x2_i > 0) -> +infinity]":
        _FLOOR_DIV_SPECIAL,
    "test_special_cases.py::test_binary[floor_divide(x1_i is +infinity and isfinite(x2_i) and x2_i < 0) -> -infinity]":
        _FLOOR_DIV_SPECIAL,
    "test_special_cases.py::test_binary[floor_divide(x1_i is -infinity and isfinite(x2_i) and x2_i > 0) -> -infinity]":
        _FLOOR_DIV_SPECIAL,
    "test_special_cases.py::test_binary[floor_divide(x1_i is -infinity and isfinite(x2_i) and x2_i < 0) -> +infinity]":
        _FLOOR_DIV_SPECIAL,
    "test_special_cases.py::test_binary[floor_divide(isfinite(x1_i) and x1_i > 0 and x2_i is -infinity) -> -0]":
        _FLOOR_DIV_SPECIAL,
    "test_special_cases.py::test_binary[floor_divide(isfinite(x1_i) and x1_i < 0 and x2_i is +infinity) -> -0]":
        _FLOOR_DIV_SPECIAL,
    # __floordiv__ operator-form
    "test_special_cases.py::test_binary[__floordiv__(x1_i is +infinity and isfinite(x2_i) and x2_i > 0) -> +infinity]":
        _FLOOR_DIV_SPECIAL,
    "test_special_cases.py::test_binary[__floordiv__(x1_i is +infinity and isfinite(x2_i) and x2_i < 0) -> -infinity]":
        _FLOOR_DIV_SPECIAL,
    "test_special_cases.py::test_binary[__floordiv__(x1_i is -infinity and isfinite(x2_i) and x2_i > 0) -> -infinity]":
        _FLOOR_DIV_SPECIAL,
    "test_special_cases.py::test_binary[__floordiv__(x1_i is -infinity and isfinite(x2_i) and x2_i < 0) -> +infinity]":
        _FLOOR_DIV_SPECIAL,
    "test_special_cases.py::test_binary[__floordiv__(isfinite(x1_i) and x1_i > 0 and x2_i is -infinity) -> -0]":
        _FLOOR_DIV_SPECIAL,
    "test_special_cases.py::test_binary[__floordiv__(isfinite(x1_i) and x1_i < 0 and x2_i is +infinity) -> -0]":
        _FLOOR_DIV_SPECIAL,
    # __ifloordiv__ in-place operator
    "test_special_cases.py::test_iop[__ifloordiv__(x1_i is +infinity and isfinite(x2_i) and x2_i > 0) -> +infinity]":
        _FLOOR_DIV_SPECIAL,
    "test_special_cases.py::test_iop[__ifloordiv__(x1_i is +infinity and isfinite(x2_i) and x2_i < 0) -> -infinity]":
        _FLOOR_DIV_SPECIAL,
    "test_special_cases.py::test_iop[__ifloordiv__(x1_i is -infinity and isfinite(x2_i) and x2_i > 0) -> -infinity]":
        _FLOOR_DIV_SPECIAL,
    "test_special_cases.py::test_iop[__ifloordiv__(x1_i is -infinity and isfinite(x2_i) and x2_i < 0) -> +infinity]":
        _FLOOR_DIV_SPECIAL,
    "test_special_cases.py::test_iop[__ifloordiv__(isfinite(x1_i) and x1_i > 0 and x2_i is -infinity) -> -0]":
        _FLOOR_DIV_SPECIAL,
    "test_special_cases.py::test_iop[__ifloordiv__(isfinite(x1_i) and x1_i < 0 and x2_i is +infinity) -> -0]":
        _FLOOR_DIV_SPECIAL,
    # expm1 complex special cases
    "test_special_cases.py::test_unary[expm1((real(x_i) is +0 or real(x_i) == -0) and imag(x_i) is +0) -> 0 + 0j]":
        _EXPM1_SPECIAL,
    "test_special_cases.py::test_unary[expm1(real(x_i) is +infinity and imag(x_i) is +0) -> +infinity + 0j]":
        _EXPM1_SPECIAL,
    "test_special_cases.py::test_unary[expm1(real(x_i) is -infinity and imag(x_i) is +infinity) -> -1 + 0j]":
        _EXPM1_SPECIAL,
    "test_special_cases.py::test_unary[expm1(real(x_i) is +infinity and imag(x_i) is +infinity) -> infinity + NaN j]":
        _EXPM1_SPECIAL,
    "test_special_cases.py::test_unary[expm1(real(x_i) is -infinity and imag(x_i) is NaN) -> -1 + 0j]":
        _EXPM1_SPECIAL,
    "test_special_cases.py::test_unary[expm1(real(x_i) is +infinity and imag(x_i) is NaN) -> infinity + NaN j]":
        _EXPM1_SPECIAL,
    "test_special_cases.py::test_unary[expm1(real(x_i) is NaN and imag(x_i) is +0) -> NaN + 0j]":
        _EXPM1_SPECIAL,
    # tanh complex special case
    "test_special_cases.py::test_unary[tanh(real(x_i) is +infinity and isfinite(imag(x_i)) and imag(x_i) > 0) -> 1 + 0j]":
        _TANH_SPECIAL,
}

# ---------------------------------------------------------------------------
# Combined lookup — importable by an external conftest to apply xfail markers
# ---------------------------------------------------------------------------

ALL_KNOWN_FAILURES = {}
ALL_KNOWN_FAILURES.update(NEMOPY_DESIGN)
ALL_KNOWN_FAILURES.update(UPSTREAM_NUMPY)
ALL_KNOWN_FAILURES.update(UPSTREAM_NUMPY_SPECIAL_CASES)
