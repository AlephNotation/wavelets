#!/usr/bin/env python3
"""Generate DWT references without copying any wavelet coefficient table."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pywt


LENGTHS = [*range(1, 17), 17, 31, 100, 101]
WAVELETS = ["haar", *(f"db{order}" for order in range(1, 39))]


def signal(length: int) -> list[float]:
    return [math.sin(index * 0.37) + (index % 7) - 3.0 for index in range(length)]


def generate() -> dict[str, object]:
    signals = [{"len": length, "values": signal(length)} for length in LENGTHS]
    cases = []
    for wavelet_name in WAVELETS:
        for mode in pywt.Modes.modes:
            for length in LENGTHS:
                values = signal(length)
                try:
                    approx, detail = pywt.dwt(values, wavelet_name, mode)
                except ValueError:
                    # PyWavelets rejects length-one [anti]reflect transforms.
                    continue
                cases.append(
                    {
                        "wavelet": wavelet_name,
                        "mode": mode,
                        "len": length,
                        "approx": approx.tolist(),
                        "detail": detail.tolist(),
                    }
                )
    return {
        "generator": f"PyWavelets {pywt.__version__}",
        "signals": signals,
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=Path("tests/fixtures/pywavelets-1.8.0.json"),
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(generate(), separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
