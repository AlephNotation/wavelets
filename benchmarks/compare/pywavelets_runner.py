#!/usr/bin/env python3
"""Run PyWavelets comparison cases supplied as a JSON request on stdin."""

from __future__ import annotations

import gc
import importlib.metadata
import json
import math
import platform
import sys
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pywt

SCHEMA_VERSION = 2
MAX_BATCH_ITERATIONS = 100_000_000
BOUNDARIES = {
    "zero",
    "constant",
    "symmetric",
    "reflect",
    "periodic",
    "smooth",
    "antisymmetric",
    "antireflect",
    "periodization",
}


def main() -> None:
    request = json.load(sys.stdin)
    validate_request(request)
    config = request["config"]
    results = [run_case(case, config) for case in request["cases"]]
    json.dump(
        {
            "schema": SCHEMA_VERSION,
            "engine": {
                "name": "pywavelets",
                "version": importlib.metadata.version("PyWavelets"),
                "module_version": pywt.__version__,
                "language": "Python",
                "clock": "time.perf_counter_ns",
                "cyclic_gc": "disabled_during_timing",
                "python": platform.python_version(),
                "numpy": importlib.metadata.version("numpy"),
                "target": f"{platform.machine()}-{platform.system().lower()}",
            },
            "results": results,
        },
        sys.stdout,
        separators=(",", ":"),
    )


def validate_request(request: dict[str, Any]) -> None:
    if request.get("schema") != SCHEMA_VERSION:
        raise ValueError(f"unsupported request schema {request.get('schema')!r}")
    if set(request) != {"schema", "config", "cases"}:
        raise ValueError("request must contain exactly schema, config, and cases")

    config = request["config"]
    if set(config) != {"samples", "sample_time_ms", "warmup_batches"}:
        raise ValueError("invalid benchmark configuration fields")
    if not isinstance(config["samples"], int) or config["samples"] < 3:
        raise ValueError("at least three samples are required")
    if not math.isfinite(config["sample_time_ms"]) or config["sample_time_ms"] <= 0:
        raise ValueError("sample_time_ms must be finite and positive")
    if not isinstance(config["warmup_batches"], int) or config["warmup_batches"] < 0:
        raise ValueError("warmup_batches must be a non-negative integer")

    seen = set()
    if not request["cases"]:
        raise ValueError("at least one benchmark case is required")
    for case in request["cases"]:
        expected_fields = {
            "id",
            "scope",
            "direction",
            "dtype",
            "wavelet",
            "boundary",
            "len",
        }
        if set(case) != expected_fields:
            raise ValueError(f"invalid fields for case {case!r}")
        if case["scope"] not in {"single_level", "multilevel"}:
            raise ValueError(f"invalid scope in {case['id']}")
        if case["direction"] not in {"forward", "inverse"}:
            raise ValueError(f"invalid direction in {case['id']}")
        if case["dtype"] not in {"f32", "f64"}:
            raise ValueError(f"invalid dtype in {case['id']}")
        if case["boundary"] not in BOUNDARIES:
            raise ValueError(f"invalid boundary in {case['id']}")
        if not isinstance(case["wavelet"], str):
            raise TypeError(f"invalid wavelet name in {case['id']}")
        try:
            pywt.Wavelet(case["wavelet"])
        except ValueError as error:
            raise ValueError(f"unsupported wavelet in {case['id']}") from error
        if not isinstance(case["len"], int) or case["len"] <= 0:
            raise ValueError(f"invalid signal length in {case['id']}")
        expected_id = case_id(case)
        if case["id"] != expected_id:
            raise ValueError(f"case id {case['id']!r} does not match {expected_id!r}")
        if case["id"] in seen:
            raise ValueError(f"duplicate case id {case['id']!r}")
        seen.add(case["id"])


def run_case(case: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    dtype = np.float32 if case["dtype"] == "f32" else np.float64
    values = np.asarray(signal(case["len"]), dtype=dtype)
    wavelet = pywt.Wavelet(case["wavelet"])
    mode = case["boundary"]

    if case["scope"] == "single_level":
        coefficients = pywt.dwt(values, wavelet, mode=mode)
        if case["direction"] == "forward":
            operation = lambda: pywt.dwt(values, wavelet, mode=mode)
        else:
            operation = lambda: pywt.idwt(
                coefficients[0], coefficients[1], wavelet, mode=mode
            )[: case["len"]]
    else:
        level = pywt.dwt_max_level(case["len"], wavelet.dec_len)
        coefficients = pywt.wavedec(values, wavelet, mode=mode, level=level)
        if case["direction"] == "forward":
            operation = lambda: pywt.wavedec(values, wavelet, mode=mode, level=level)
        else:
            operation = lambda: pywt.waverec(coefficients, wavelet, mode=mode)[
                : case["len"]
            ]

    checksum = output_checksum(operation())
    batch_iterations, samples_ns = measure(operation, config)
    return {
        "case_id": case["id"],
        "api": "allocating",
        "batch_iterations": batch_iterations,
        "samples_ns": samples_ns,
        "checksum": checksum,
    }


def measure(
    operation: Callable[[], Any], config: dict[str, Any]
) -> tuple[int, list[float]]:
    target_ns = int(config["sample_time_ms"] * 1_000_000)
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        batch_iterations = calibrate(operation, target_ns)
        for _ in range(config["warmup_batches"]):
            run_batch(operation, batch_iterations)
        samples_ns = [
            run_batch(operation, batch_iterations) / batch_iterations
            for _ in range(config["samples"])
        ]
    finally:
        if was_enabled:
            gc.enable()
    return batch_iterations, samples_ns


def calibrate(operation: Callable[[], Any], target_ns: int) -> int:
    minimum_ns = target_ns // 4
    iterations = 1
    while True:
        elapsed_ns = run_batch(operation, iterations)
        if elapsed_ns >= minimum_ns or iterations >= MAX_BATCH_ITERATIONS:
            estimate = math.ceil(iterations * target_ns / max(elapsed_ns, 1))
            return min(max(estimate, 1), MAX_BATCH_ITERATIONS)
        iterations = min(iterations * 2, MAX_BATCH_ITERATIONS)


def run_batch(operation: Callable[[], Any], iterations: int) -> int:
    start = time.perf_counter_ns()
    for _ in range(iterations):
        result = operation()
        if result is None:
            raise RuntimeError("benchmark operation returned no result")
        del result
    elapsed = time.perf_counter_ns() - start
    return elapsed


def signal(length: int) -> list[float]:
    return [
        (((index * 17) % 257) - 128) / 64.0 + ((index % 11) - 5) / 16.0
        for index in range(length)
    ]


def output_checksum(output: Any) -> float:
    if isinstance(output, (tuple, list)):
        value = sum(output_checksum(part) for part in output)
    else:
        value = float(np.sum(np.abs(output), dtype=np.float64))
    if not math.isfinite(value):
        raise ValueError("benchmark output checksum is not finite")
    return value


def case_id(case: dict[str, Any]) -> str:
    return (
        f"{case['scope']}/{case['direction']}/{case['dtype']}/"
        f"{case['wavelet']}/{case['boundary']}/{case['len']}"
    )


if __name__ == "__main__":
    main()
