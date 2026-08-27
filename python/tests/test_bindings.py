from __future__ import annotations

import numpy as np
import pytest
import pywt
import wavelets_rs


def signal(length: int, dtype: np.dtype | type) -> np.ndarray:
    index = np.arange(length, dtype=np.float64)
    values = np.sin(index * 0.013) + 0.25 * np.cos(index * 0.071)
    return values.astype(dtype)


def test_package_version_is_available() -> None:
    assert wavelets_rs.__version__ == "0.1.0a4"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_single_level_matches_pywavelets(dtype: np.dtype | type) -> None:
    values = signal(4_096, dtype)
    plan = wavelets_rs.plan_dwt(len(values), "db4", "symmetric", dtype=dtype)

    actual_approx, actual_detail = plan.forward(values)
    expected_approx, expected_detail = pywt.dwt(values, "db4", mode="symmetric")
    tolerance = 2e-5 if dtype is np.float32 else 2e-12

    np.testing.assert_allclose(
        actual_approx, expected_approx, rtol=tolerance, atol=tolerance
    )
    np.testing.assert_allclose(
        actual_detail, expected_detail, rtol=tolerance, atol=tolerance
    )
    np.testing.assert_allclose(
        plan.inverse(actual_approx, actual_detail),
        values,
        rtol=tolerance,
        atol=tolerance,
    )
    assert actual_approx.dtype == np.dtype(dtype)
    assert actual_detail.dtype == np.dtype(dtype)
    assert plan.signal_len == len(values)
    assert plan.coeff_len == len(expected_approx)
    assert plan.wavelet == "db4"
    assert plan.mode == "symmetric"


@pytest.mark.parametrize(
    "mode",
    [
        "zero",
        "constant",
        "symmetric",
        "reflect",
        "periodic",
        "smooth",
        "antisymmetric",
        "antireflect",
        "periodization",
    ],
)
def test_single_level_modes_match_pywavelets(mode: str) -> None:
    values = signal(101, np.float64)
    plan = wavelets_rs.plan_dwt(len(values), "db4", mode)
    actual = plan.forward(values)
    expected = pywt.dwt(values, "db4", mode=mode)

    for actual_band, expected_band in zip(actual, expected, strict=True):
        np.testing.assert_allclose(actual_band, expected_band, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(plan.inverse(*actual), values, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("mode", ["symmetric", "periodization"])
def test_multilevel_matches_pywavelets(dtype: np.dtype | type, mode: str) -> None:
    values = signal(4_096, dtype)
    plan = wavelets_rs.plan_wavedec(len(values), "db4", mode, dtype=dtype)
    actual = plan.forward(values)
    expected = pywt.wavedec(values, "db4", mode=mode, level=plan.levels)
    tolerance = 3e-5 if dtype is np.float32 else 3e-12

    assert len(actual) == len(expected)
    for actual_band, expected_band in zip(actual, expected, strict=True):
        np.testing.assert_allclose(
            actual_band, expected_band, rtol=tolerance, atol=tolerance
        )
    np.testing.assert_allclose(
        plan.inverse(actual), values, rtol=tolerance, atol=tolerance
    )


def test_multilevel_inverse_accepts_pywavelets_bands() -> None:
    values = signal(101, np.float64)
    plan = wavelets_rs.plan_wavedec(len(values), "bior4.4", "symmetric", 2)
    bands = pywt.wavedec(values, "bior4.4", mode="symmetric", level=2)
    np.testing.assert_allclose(plan.inverse(bands), values, rtol=3e-12, atol=3e-12)


def test_invalid_inputs_raise_python_errors() -> None:
    with pytest.raises(ValueError, match="unknown wavelet"):
        wavelets_rs.plan_dwt(16, "not-a-wavelet")
    with pytest.raises(ValueError, match="unknown boundary"):
        wavelets_rs.plan_dwt(16, mode="not-a-mode")
    with pytest.raises(TypeError, match="float32 and float64"):
        wavelets_rs.plan_dwt(16, dtype=np.int64)

    plan = wavelets_rs.plan_dwt(8)
    with pytest.raises(ValueError, match="planned length"):
        plan.forward(np.zeros(7, dtype=np.float64))
    with pytest.raises(ValueError, match="contiguous"):
        plan.forward(np.zeros(16, dtype=np.float64)[::2])
    with pytest.raises(TypeError):
        plan.forward(np.zeros(8, dtype=np.float32))
