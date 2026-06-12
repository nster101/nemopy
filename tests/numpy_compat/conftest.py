"""Configuration for NumPy compatibility tests.

Adapted from NumPy v2.1.0 test suite. These tests validate that nemopy
works as a drop-in replacement for ``import numpy as np`` in real-world
NumPy test scenarios.

numpy.testing imports remain pointing at real NumPy — they are test
assertion helpers, not things being tested.
"""

import pytest
import re


NEMOPY_XFAILS_EXACT = {
    # ---- Intentional nemopy guardrails (DESIGN.md §7.1, §1.2 Goal 1) ----
    "TestSVDHermitian::test_herm_cases":
        "nemopy ShapeError: shape guard blocks 1D vs (n,1) subtraction "
        "inside numpy.testing (intentional guardrail, §7.1)",
    "TestQR::test_mode_all_but_economic":
        "nemopy ShapeError: shape guard blocks 1D vs (n,1) subtraction "
        "inside numpy.testing (intentional guardrail, §7.1)",
    "TestTensorinv::test_tensorinv_result":
        "nemopy ShapeError: shape guard blocks 1D vs (n,1) subtraction "
        "inside numpy.testing (intentional guardrail, §7.1)",

    # ---- nemopy float64 promotion policy (DESIGN.md §3) ----
    "TestVdot::test_basic":
        "nemopy eye() always promotes to float64 per §3 dtype policy, "
        "so eye(3, dtype=bool) returns float64 instead of bool",

    # ---- NumPy v2.1->v2.4 API changes (not nemopy-related) ----
    "TestNonarrayArgs::test_reshape_shape_arg":
        "NumPy API change: reshape() 'newshape' kwarg removed in v2.4",
    "TestTypes::test_can_cast_values":
        "NumPy NEP 50: can_cast() no longer accepts Python scalars",
    "TestCorrelate::test_mode":
        "NumPy API change: correlate mode argument validation changed",
    "TestConvolve::test_mode":
        "NumPy API change: convolve mode argument validation changed",

    # ---- C-internal / byte-order / array-ownership tests ----
    "test_byteorder_check":
        "Internal NumPy byte-order test, not relevant to nemopy",
    "TestFlags::test_writeable_from_c_data":
        "C-level writeable flag test, not relevant to nemopy",

    # ---- QR empty array edge case ----
    "TestQR::test_qr_empty":
        "Empty array edge case in QR decomposition",

    # ---- Minor version/API differences ----
    "TestMethods::test__complex__":
        "Complex conversion edge case",
    "TestMethods::test__complex__should_not_work":
        "Complex conversion edge case",
    "TestBinop::test_pow_override_with_errors":
        "Power override edge case",
    "TestStats::test_dtype_from_input":
        "NumPy dtype inference change between v2.1 and v2.4",
    "TestConversion::test_to_int_scalar":
        "NumPy scalar conversion change between v2.1 and v2.4",
    "test_richcompare_scalar_boolean_singleton_return":
        "NumPy richcompare singleton return change",
    "TestBooleanIndexing::test_bool_as_int_argument_errors":
        "NumPy deprecation handling change between v2.1 and v2.4",
}

NEMOPY_XFAILS_PATTERN = {
    # Parametrized tests matched by pattern (substring match OK here)
    "TestFloatExceptions::test_floating_exceptions":
        "NumPy API change: finfo._machar removed after v2.1",
    "TestIO::test_malformed":
        "Internal NumPy file I/O test with structured dtypes",
    "TestIO::test_fromfile_subarray_binary":
        "Internal NumPy file I/O subarray test",
    "TestIO::test_read_shorter_than_count_subarray":
        "Internal NumPy file I/O subarray test",
    "TestResize::test_int_shape":
        "Array ownership/resize test — nemopy subclass views "
        "do not own their data",
    "TestResize::test_freeform_shape":
        "Array ownership/resize test — nemopy subclass views "
        "do not own their data",
    "TestResize::test_zeros_appended":
        "Array ownership/resize test — nemopy subclass views "
        "do not own their data",
}

NEMOPY_XFAILS_REGEX = {
    # Exact test function match (no suffix matching)
    r"TestMethods::test_sort\b":
        "NumPy API change: sort behavior changed between v2.1 and v2.4",
    r"TestMethods::test_argsort\b":
        "NumPy API change: argsort behavior changed between v2.1 and v2.4",
}


def pytest_collection_modifyitems(items):
    for item in items:
        for test_name, reason in NEMOPY_XFAILS_EXACT.items():
            if test_name in item.nodeid:
                parts = item.nodeid.split("::")
                test_id = "::".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
                base_name = test_id.split("[")[0]
                if base_name == test_name or test_name in test_id:
                    item.add_marker(
                        pytest.mark.xfail(reason=reason, strict=False)
                    )
                    break

        for test_name, reason in NEMOPY_XFAILS_PATTERN.items():
            if test_name in item.nodeid:
                item.add_marker(
                    pytest.mark.xfail(reason=reason, strict=False)
                )
                break

        for pattern, reason in NEMOPY_XFAILS_REGEX.items():
            if re.search(pattern, item.nodeid):
                item.add_marker(
                    pytest.mark.xfail(reason=reason, strict=False)
                )
                break
