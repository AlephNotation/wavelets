#!/usr/bin/env python3
"""Run and combine cross-library wavelet benchmark engines."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import platform
import shlex
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2
HERE = Path(__file__).resolve().parent
BENCHMARKS_DIR = HERE.parent
REPOSITORY_ROOT = BENCHMARKS_DIR.parent
RUST_MANIFEST = BENCHMARKS_DIR / "Cargo.toml"
RUST_RUNNER = (
    BENCHMARKS_DIR
    / "target"
    / "release"
    / ("wavelets-compare-runner.exe" if os.name == "nt" else "wavelets-compare-runner")
)
PYWAVELETS_RUNNER = HERE / "pywavelets_runner.py"
GSL_SOURCE = HERE / "gsl_runner.c"
GSL_RUNNER = (
    BENCHMARKS_DIR
    / "target"
    / "cross"
    / ("gsl_runner.exe" if os.name == "nt" else "gsl_runner")
)

BOUNDARIES = [
    "zero",
    "constant",
    "symmetric",
    "reflect",
    "periodic",
    "smooth",
    "antisymmetric",
    "antireflect",
    "periodization",
]


def main() -> None:
    args = parse_args()
    cases = canonical_cases()
    if args.case_filter:
        cases = [case for case in cases if args.case_filter in case["id"]]
    if not cases:
        raise SystemExit(f"no cases match {args.case_filter!r}")

    request = {
        "schema": SCHEMA_VERSION,
        "config": {
            "samples": args.samples,
            "sample_time_ms": args.sample_ms,
            "warmup_batches": args.warmup_batches,
        },
        "cases": cases,
    }

    if not args.no_build:
        run_checked(
            [
                "cargo",
                "build",
                "--release",
                "--manifest-path",
                str(RUST_MANIFEST),
                "--bin",
                "wavelets-compare-runner",
            ]
        )
    if not RUST_RUNNER.is_file():
        raise SystemExit(f"Rust runner does not exist: {RUST_RUNNER}")

    responses = [
        run_engine([str(RUST_RUNNER)], request),
        run_engine([sys.executable, str(PYWAVELETS_RUNNER)], request),
    ]
    if args.gsl:
        responses.append(run_gsl(cases, request["config"], args.no_build))
    validate_responses(responses, cases)
    results = combine_results(responses)
    validate_checksums(results, cases)

    report = {
        "schema": SCHEMA_VERSION,
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "host": host_metadata(),
        "configuration": request["config"],
        "cases": cases,
        "engines": [response["engine"] for response in responses],
        "results": results,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print_comparison(results, cases)
    print(f"\nRaw samples: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--sample-ms", type=float, default=10.0)
    parser.add_argument("--warmup-batches", type=int, default=3)
    parser.add_argument("--filter", dest="case_filter")
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument(
        "--gsl",
        action="store_true",
        help="include the compatible full-depth Haar subset using system GSL",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BENCHMARKS_DIR / "reports" / "comparison.json",
    )
    args = parser.parse_args()
    if args.samples < 3:
        parser.error("--samples must be at least 3")
    if not math.isfinite(args.sample_ms) or args.sample_ms <= 0:
        parser.error("--sample-ms must be finite and positive")
    if args.warmup_batches < 0:
        parser.error("--warmup-batches cannot be negative")
    return args


def canonical_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(
        scope: str,
        dtype: str,
        wavelet: str,
        boundary: str,
        length: int,
    ) -> None:
        for direction in ("forward", "inverse"):
            case = {
                "scope": scope,
                "direction": direction,
                "dtype": dtype,
                "wavelet": wavelet,
                "boundary": boundary,
                "len": length,
            }
            case["id"] = case_id(case)
            if case["id"] not in seen:
                seen.add(case["id"])
                cases.append(case)

    for wavelet in ("db1", "db2", "db4"):
        for length in (16, 64, 256, 1_024, 4_096, 16_384):
            add("single_level", "f64", wavelet, "symmetric", length)
        for length in (64, 256, 4_096):
            add("single_level", "f32", wavelet, "symmetric", length)
    for wavelet in (
        "db20",
        "db38",
        "sym4",
        "coif3",
        "bior4.4",
        "rbio4.4",
    ):
        add("single_level", "f64", wavelet, "symmetric", 4_096)
    for boundary in BOUNDARIES:
        add("single_level", "f64", "db4", boundary, 4_096)
    for boundary in ("symmetric", "periodization"):
        add("single_level", "f64", "db4", boundary, 101)
    for wavelet in ("db38", "coif17"):
        for boundary in ("symmetric", "antireflect"):
            for length in (16, 4_096):
                add("single_level", "f64", wavelet, boundary, length)
            add("multilevel", "f64", wavelet, boundary, 4_096)
    for dtype in ("f32", "f64"):
        for boundary in ("symmetric", "periodization"):
            add("multilevel", dtype, "db4", boundary, 4_096)
    add("multilevel", "f64", "sym4", "symmetric", 4_096)
    add("multilevel", "f64", "coif3", "symmetric", 4_096)
    add("multilevel", "f64", "bior4.4", "symmetric", 4_096)
    add("multilevel", "f64", "rbio4.4", "symmetric", 4_096)
    add("multilevel", "f64", "db4", "symmetric", 16_384)
    for length in (1_024, 4_096, 16_384):
        add("multilevel", "f64", "db1", "periodization", length)
    return cases


def run_engine(command: list[str], request: dict[str, Any]) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        input=json.dumps(request),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(
            f"benchmark engine failed ({' '.join(command)}):\n{completed.stderr}"
        )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise SystemExit(
            f"benchmark engine emitted invalid JSON ({' '.join(command)}): {error}\n"
            f"stdout: {completed.stdout}\nstderr: {completed.stderr}"
        ) from error


def run_gsl(
    cases: list[dict[str, Any]], config: dict[str, Any], no_build: bool
) -> dict[str, Any]:
    compatible = [case for case in cases if is_gsl_compatible(case)]
    if not compatible:
        raise SystemExit(
            "--gsl was requested, but no selected cases are full-depth f64 Haar "
            "transforms with periodization and power-of-two lengths"
        )

    cflags = shlex.split(command_output(["pkg-config", "--cflags", "--libs", "gsl"]))
    compiler = os.environ.get("CC", "cc")
    compiler_flags = [
        "-O3",
        "-DNDEBUG",
        "-std=c11",
        "-Wall",
        "-Wextra",
        "-Wpedantic",
        "-Werror",
    ]
    if not no_build:
        GSL_RUNNER.parent.mkdir(parents=True, exist_ok=True)
        run_checked(
            [
                compiler,
                *compiler_flags,
                str(GSL_SOURCE),
                "-o",
                str(GSL_RUNNER),
                *cflags,
            ]
        )
    if not GSL_RUNNER.is_file():
        raise SystemExit(f"GSL runner does not exist: {GSL_RUNNER}")

    responses = []
    for case in compatible:
        responses.append(
            run_engine(
                [
                    str(GSL_RUNNER),
                    case["id"],
                    case["direction"],
                    str(case["len"]),
                    str(config["samples"]),
                    str(config["sample_time_ms"]),
                    str(config["warmup_batches"]),
                ],
                {},
            )
        )
    engine = responses[0]["engine"]
    engine["compiler"] = command_output([compiler, "--version"]).splitlines()[0]
    engine["compiler_flags"] = [*compiler_flags, *cflags]
    return {
        "schema": SCHEMA_VERSION,
        "engine": engine,
        "results": [result for response in responses for result in response["results"]],
    }


def is_gsl_compatible(case: dict[str, Any]) -> bool:
    length = case["len"]
    return (
        case["scope"] == "multilevel"
        and case["dtype"] == "f64"
        and case["wavelet"] == "db1"
        and case["boundary"] == "periodization"
        and length > 0
        and length & (length - 1) == 0
    )


def run_checked(command: list[str]) -> None:
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)


def validate_responses(
    responses: list[dict[str, Any]], cases: list[dict[str, Any]]
) -> None:
    expected_ids = {case["id"] for case in cases}
    expected_apis = {
        "wavelets": {"into", "allocating"},
        "pywavelets": {"allocating"},
        "gsl": {"into"},
    }
    for response in responses:
        if response.get("schema") != SCHEMA_VERSION:
            raise ValueError(f"invalid response schema: {response!r}")
        engine = response.get("engine", {}).get("name")
        if engine not in expected_apis:
            raise ValueError(f"unexpected engine {engine!r}")
        engine_case_ids = (
            {case["id"] for case in cases if is_gsl_compatible(case)}
            if engine == "gsl"
            else expected_ids
        )
        observed: dict[str, set[str]] = {case_id: set() for case_id in engine_case_ids}
        for result in response.get("results", []):
            case_id_value = result.get("case_id")
            if case_id_value not in engine_case_ids:
                raise ValueError(f"{engine} returned unknown case {case_id_value!r}")
            api = result.get("api")
            if api in observed[case_id_value]:
                raise ValueError(f"{engine} returned duplicate {case_id_value}/{api}")
            observed[case_id_value].add(api)
            samples = result.get("samples_ns")
            if not samples or any(
                not math.isfinite(value) or value <= 0 for value in samples
            ):
                raise ValueError(
                    f"{engine} returned invalid samples for {case_id_value}"
                )
        for case_id_value, apis in observed.items():
            if apis != expected_apis[engine]:
                raise ValueError(
                    f"{engine} returned APIs {sorted(apis)} for {case_id_value}; "
                    f"expected {sorted(expected_apis[engine])}"
                )


def combine_results(responses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    combined = []
    for response in responses:
        engine = response["engine"]["name"]
        for raw in response["results"]:
            samples = raw["samples_ns"]
            combined.append(
                {
                    "engine": engine,
                    **raw,
                    "summary": {
                        "median_ns": statistics.median(samples),
                        "p95_ns": percentile(samples, 0.95),
                        "min_ns": min(samples),
                        "max_ns": max(samples),
                    },
                }
            )
    return combined


def validate_checksums(
    results: list[dict[str, Any]], cases: list[dict[str, Any]]
) -> None:
    indexed = {
        (result["case_id"], result["engine"], result["api"]): result
        for result in results
    }
    for case in cases:
        case_id_value = case["id"]
        into = indexed[(case_id_value, "wavelets", "into")]["checksum"]
        allocating = indexed[(case_id_value, "wavelets", "allocating")]["checksum"]
        pywavelets = indexed[(case_id_value, "pywavelets", "allocating")]["checksum"]
        relative_tolerance = 2e-5 if case["dtype"] == "f32" else 1e-10
        for label, value in (
            ("wavelets allocating", allocating),
            ("PyWavelets", pywavelets),
        ):
            if not math.isclose(
                into,
                value,
                rel_tol=relative_tolerance,
                abs_tol=relative_tolerance,
            ):
                raise ValueError(
                    f"checksum mismatch for {case_id_value}: wavelets into={into}, "
                    f"{label}={value}"
                )
        gsl = indexed.get((case_id_value, "gsl", "into"))
        if gsl is not None and not math.isclose(
            into, gsl["checksum"], rel_tol=1e-10, abs_tol=1e-10
        ):
            raise ValueError(
                f"checksum mismatch for {case_id_value}: wavelets into={into}, "
                f"GSL={gsl['checksum']}"
            )


def print_comparison(
    results: list[dict[str, Any]], cases: list[dict[str, Any]]
) -> None:
    indexed = {
        (result["case_id"], result["engine"], result["api"]): result
        for result in results
    }
    has_gsl = any(result["engine"] == "gsl" for result in results)
    header = (
        f"{'case':<67} {'Rust alloc':>12} {'Rust into':>12} "
        f"{'PyWavelets':>12} {'alloc x':>8} {'into x':>8}"
    )
    if has_gsl:
        header += f" {'GSL':>12} {'GSL/Rust':>9}"
    print(header)
    for case in cases:
        case_id_value = case["id"]
        rust_alloc = indexed[(case_id_value, "wavelets", "allocating")]["summary"][
            "median_ns"
        ]
        rust_into = indexed[(case_id_value, "wavelets", "into")]["summary"]["median_ns"]
        pywavelets = indexed[(case_id_value, "pywavelets", "allocating")]["summary"][
            "median_ns"
        ]
        row = (
            f"{case_id_value:<67} {format_ns(rust_alloc):>12} "
            f"{format_ns(rust_into):>12} {format_ns(pywavelets):>12} "
            f"{pywavelets / rust_alloc:>7.2f}x {pywavelets / rust_into:>7.2f}x"
        )
        if has_gsl:
            gsl = indexed.get((case_id_value, "gsl", "into"))
            if gsl is None:
                row += f" {'-':>12} {'-':>9}"
            else:
                gsl_ns = gsl["summary"]["median_ns"]
                row += f" {format_ns(gsl_ns):>12} {gsl_ns / rust_into:>8.2f}x"
        print(row)


def host_metadata() -> dict[str, Any]:
    cargo_metadata = json.loads(
        command_output(
            [
                "cargo",
                "metadata",
                "--no-deps",
                "--format-version",
                "1",
                "--manifest-path",
                str(REPOSITORY_ROOT / "Cargo.toml"),
            ]
        )
    )
    wavelets_package = next(
        package
        for package in cargo_metadata["packages"]
        if package["name"] == "wavelets"
    )
    return {
        "operating_system": platform.platform(),
        "architecture": platform.machine(),
        "cpu": cpu_model(),
        "python": platform.python_version(),
        "rustc": command_output(["rustc", "--version"]),
        "cargo": command_output(["cargo", "--version"]),
        "wavelets": wavelets_package["version"],
        "rustflags": os.environ.get("RUSTFLAGS", ""),
        "rust_profile": "release",
        **source_metadata(),
    }


def source_metadata() -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if revision.returncode != 0 or status.returncode != 0:
        return {"git_revision": None, "git_dirty": None}
    return {
        "git_revision": revision.stdout.strip(),
        "git_dirty": bool(status.stdout.strip()),
    }


def cpu_model() -> str:
    if sys.platform == "darwin":
        for key in ("machdep.cpu.brand_string", "hw.model"):
            completed = subprocess.run(
                ["sysctl", "-n", key], text=True, capture_output=True, check=False
            )
            if completed.returncode == 0 and completed.stdout.strip():
                return completed.stdout.strip()
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith(("model name", "hardware")):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or "unknown"


def command_output(command: list[str]) -> str:
    return subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def format_ns(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f} ms"
    if value >= 1_000:
        return f"{value / 1_000:.2f} us"
    return f"{value:.2f} ns"


def case_id(case: dict[str, Any]) -> str:
    return (
        f"{case['scope']}/{case['direction']}/{case['dtype']}/"
        f"{case['wavelet']}/{case['boundary']}/{case['len']}"
    )


if __name__ == "__main__":
    main()
