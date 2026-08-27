from __future__ import annotations

from importlib.metadata import version
from typing import TypeAlias

import numpy as np

from ._wavelets_rs import DwtPlanF32, DwtPlanF64, WavedecPlanF32, WavedecPlanF64

DwtPlan: TypeAlias = DwtPlanF32 | DwtPlanF64
WavedecPlan: TypeAlias = WavedecPlanF32 | WavedecPlanF64
__version__ = version("wavelets-rs")


def _precision(dtype: np.dtype | type | str) -> np.dtype:
    precision = np.dtype(dtype)
    if precision not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("wavelets_rs supports only float32 and float64")
    return precision


def plan_dwt(
    length: int,
    wavelet: str = "db1",
    mode: str = "symmetric",
    *,
    dtype: np.dtype | type | str = np.float64,
) -> DwtPlan:
    """Plan a reusable single-level transform for one signal length."""
    if _precision(dtype) == np.dtype(np.float32):
        return DwtPlanF32(length, wavelet, mode)
    return DwtPlanF64(length, wavelet, mode)


def plan_wavedec(
    length: int,
    wavelet: str = "db1",
    mode: str = "symmetric",
    level: int | None = None,
    *,
    dtype: np.dtype | type | str = np.float64,
) -> WavedecPlan:
    """Plan a reusable multilevel transform for one signal length."""
    if _precision(dtype) == np.dtype(np.float32):
        return WavedecPlanF32(length, wavelet, mode, level)
    return WavedecPlanF64(length, wavelet, mode, level)


__all__ = [
    "DwtPlan",
    "DwtPlanF32",
    "DwtPlanF64",
    "WavedecPlan",
    "WavedecPlanF32",
    "WavedecPlanF64",
    "__version__",
    "plan_dwt",
    "plan_wavedec",
]
