#!/usr/bin/env python3
"""Compare wavelets-rs and PyWavelets through the same Python interpreter."""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import importlib.metadata
import json
import math
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import compare as native_compare
import numpy as np
import pywavelets_runner
import pywt

try:
    import wavelets_rs
except ImportError as error:
    raise SystemExit(
        "wavelets_rs is not installed; run `maturin develop --release` from python/"
    ) from error

SCHEMA_VERSION = 1
HERE = Path(__file__).resolve().parent
BENCHMARKS_DIR = HERE.parent
MAX_BATCH_ITERATIONS = 100_000_000


def main() -> None:
    args = parse_args()
    cases = native_compare.canonical_cases()
    if args.case_filter:
        cases = [case for case in cases if args.case_filter in case["id"]]
    if not cases:
        raise SystemExit(f"no cases match {args.case_filter!r}")

    config = {
        "samples": args.samples,
        "sample_time_ms": args.sample_ms,
        "warmup_batches": args.warmup_batches,
    }
    results = [run_case(case, config) for case in cases]
    report = {
        "schema": SCHEMA_VERSION,
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "host": {
            **native_compare.host_metadata(),
            "numpy": importlib.metadata.version("numpy"),
            "wavelets_rs": importlib.metadata.version("wavelets-rs"),
            "pywavelets_distribution": importlib.metadata.version("PyWavelets"),
            "pywavelets_module": pywt.__version__,
        },
        "configuration": config,
        "methodology": {
            "process": "same CPython interpreter",
            "clock": "time.perf_counter_ns",
            "pywavelets_wavelet_construction": "outside timer",
            "rust_planned": "canonical wavelet construction and planning outside timer",
            "rust_cold": "canonical wavelet construction and planning inside timer",
            "input_generation": "outside timer",
            "inverse_coefficient_generation": "outside timer",
            "engine_inputs": "same NumPy arrays for each engine",
            "output_materialization": "inside timer",
            "output_destruction": "inside timed batch",
            "cyclic_gc": "disabled during timing",
            "sample_order": "rotating engines",
        },
        "cases": cases,
        "results": results,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print_results(results)
    print(f"\nRaw samples: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--sample-ms", type=float, default=10.0)
    parser.add_argument("--warmup-batches", type=int, default=3)
    parser.add_argument("--filter", dest="case_filter")
    parser.add_argument(
        "--output",
        type=Path,
        default=BENCHMARKS_DIR / "reports" / "python-api.json",
    )
    args = parser.parse_args()
    if args.samples < 3:
        parser.error("--samples must be at least 3")
    if not math.isfinite(args.sample_ms) or args.sample_ms <= 0:
        parser.error("--sample-ms must be finite and positive")
    if args.warmup_batches < 0:
        parser.error("--warmup-batches cannot be negative")
    return args


def run_case(case: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    dtype = np.float32 if case["dtype"] == "f32" else np.float64
    values = np.asarray(pywavelets_runner.signal(case["len"]), dtype=dtype)
    wavelet = pywt.Wavelet(case["wavelet"])
    mode = case["boundary"]

    if case["scope"] == "single_level":
        plan_factory = lambda: wavelets_rs.plan_dwt(
            case["len"], case["wavelet"], mode, dtype=dtype
        )
        rust_plan = plan_factory()
        pywavelets_coefficients = pywt.dwt(values, wavelet, mode=mode)
        if case["direction"] == "forward":
            rust_planned_operation = lambda: rust_plan.forward(values)
            rust_cold_operation = lambda: plan_factory().forward(values)
            pywavelets_operation = lambda: pywt.dwt(values, wavelet, mode=mode)
        else:
            rust_planned_operation = lambda: rust_plan.inverse(*pywavelets_coefficients)
            rust_cold_operation = lambda: plan_factory().inverse(
                *pywavelets_coefficients
            )
            pywavelets_operation = lambda: pywt.idwt(
                *pywavelets_coefficients, wavelet, mode=mode
            )[: case["len"]]
    else:
        plan_factory = lambda: wavelets_rs.plan_wavedec(
            case["len"], case["wavelet"], mode, dtype=dtype
        )
        rust_plan = plan_factory()
        level = rust_plan.levels
        pywavelets_coefficients = pywt.wavedec(values, wavelet, mode=mode, level=level)
        if case["direction"] == "forward":
            rust_planned_operation = lambda: rust_plan.forward(values)
            rust_cold_operation = lambda: plan_factory().forward(values)
            pywavelets_operation = lambda: pywt.wavedec(
                values, wavelet, mode=mode, level=level
            )
        else:
            rust_planned_operation = lambda: rust_plan.inverse(pywavelets_coefficients)
            rust_cold_operation = lambda: plan_factory().inverse(
                pywavelets_coefficients
            )
            pywavelets_operation = lambda: pywt.waverec(
                pywavelets_coefficients, wavelet, mode=mode
            )[: case["len"]]

    rust_output = rust_planned_operation()
    rust_cold_output = rust_cold_operation()
    pywavelets_output = pywavelets_operation()
    tolerance = 2e-5 if dtype is np.float32 else 1e-10
    assert_outputs_close(rust_output, pywavelets_output, tolerance)
    assert_outputs_close(rust_cold_output, pywavelets_output, tolerance)
    rust_checksum = output_checksum(rust_output)
    pywavelets_checksum = output_checksum(pywavelets_output)
    if not math.isclose(
        rust_checksum,
        pywavelets_checksum,
        rel_tol=tolerance,
        abs_tol=tolerance,
    ):
        raise ValueError(
            f"checksum mismatch for {case['id']}: "
            f"wavelets-rs={rust_checksum}, PyWavelets={pywavelets_checksum}"
        )

    measurements = measure_engines(
        [
            ("wavelets_rs_planned", rust_planned_operation),
            ("wavelets_rs_cold", rust_cold_operation),
            ("pywavelets", pywavelets_operation),
        ],
        config,
    )
    return {
        "case_id": case["id"],
        "checksum": rust_checksum,
        **measurements,
    }


def measure_engines(
    engines: list[tuple[str, Callable[[], Any]]],
    config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    target_ns = int(config["sample_time_ms"] * 1_000_000)
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        iterations = {
            name: calibrate(operation, target_ns) for name, operation in engines
        }
        for _ in range(config["warmup_batches"]):
            for name, operation in engines:
                run_batch(operation, iterations[name])

        samples: dict[str, list[float]] = {name: [] for name, _ in engines}
        for sample in range(config["samples"]):
            offset = sample % len(engines)
            for name, operation in engines[offset:] + engines[:offset]:
                elapsed = run_batch(operation, iterations[name])
                samples[name].append(elapsed / iterations[name])
    finally:
        if was_enabled:
            gc.enable()
    return {
        name: {
            "batch_iterations": iterations[name],
            "samples_ns": samples[name],
            "summary": summarize(samples[name]),
        }
        for name, _ in engines
    }


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
    return time.perf_counter_ns() - start


def assert_outputs_close(actual: Any, expected: Any, tolerance: float) -> None:
    if isinstance(expected, (tuple, list)):
        if not isinstance(actual, (tuple, list)) or len(actual) != len(expected):
            raise ValueError("coefficient structures differ")
        for actual_band, expected_band in zip(actual, expected, strict=True):
            np.testing.assert_allclose(
                actual_band, expected_band, rtol=tolerance, atol=tolerance
            )
    else:
        np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)


def output_checksum(output: Any) -> float:
    if isinstance(output, (tuple, list)):
        return sum(output_checksum(part) for part in output)
    value = float(np.sum(np.abs(output), dtype=np.float64))
    if not math.isfinite(value):
        raise ValueError("benchmark output checksum is not finite")
    return value


def summarize(samples: list[float]) -> dict[str, float]:
    return {
        "median_ns": statistics.median(samples),
        "p95_ns": native_compare.percentile(samples, 0.95),
        "min_ns": min(samples),
        "max_ns": max(samples),
    }


def print_results(results: list[dict[str, Any]]) -> None:
    print(
        f"{'case':<67} {'planned':>12} {'cold':>12} {'PyWavelets':>12} "
        f"{'planned x':>10} {'cold x':>8}"
    )
    for result in results:
        planned = result["wavelets_rs_planned"]["summary"]["median_ns"]
        cold = result["wavelets_rs_cold"]["summary"]["median_ns"]
        pywavelets = result["pywavelets"]["summary"]["median_ns"]
        print(
            f"{result['case_id']:<67} "
            f"{native_compare.format_ns(planned):>12} "
            f"{native_compare.format_ns(cold):>12} "
            f"{native_compare.format_ns(pywavelets):>12} "
            f"{pywavelets / planned:>9.2f}x "
            f"{pywavelets / cold:>7.2f}x"
        )


if __name__ == "__main__":
    main()
