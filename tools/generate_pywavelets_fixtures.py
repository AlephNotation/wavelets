#!/usr/bin/env python3
"""Generate DWT references without copying any wavelet coefficient table."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mpmath as mp
import pywt
from generate_builtin_coefficients import (
    BIORTHOGONAL_ORDERS,
    biorthogonal_low_passes,
    coiflet,
    daubechies,
    symlet,
)

SINGLE_LEVEL_LENGTHS = [*range(1, 17), 17, 31, 100, 101]
MULTILEVEL_LENGTHS = [31, 100, 101, 1000]
WAVELET_NAMES = [
    "haar",
    *(f"db{order}" for order in range(1, 39)),
    *(f"sym{order}" for order in range(2, 21)),
    *(f"coif{order}" for order in range(1, 18)),
    *(
        f"bior{reconstruction}.{decomposition}"
        for reconstruction, decomposition in BIORTHOGONAL_ORDERS
    ),
    *(
        f"rbio{reconstruction}.{decomposition}"
        for reconstruction, decomposition in BIORTHOGONAL_ORDERS
    ),
]


def signal(length: int) -> list[float]:
    return [math.sin(index * 0.37) + (index % 7) - 3.0 for index in range(length)]


def orthogonal_filter_bank(dec_lo: list[float]) -> list[list[float]]:
    dec_hi = [
        -coefficient if index % 2 == 0 else coefficient
        for index, coefficient in enumerate(reversed(dec_lo))
    ]
    rec_lo = list(reversed(dec_lo))
    rec_hi = list(reversed(dec_hi))
    return [dec_lo, dec_hi, rec_lo, rec_hi]


def biorthogonal_filter_bank(
    reconstruction: int, decomposition: int
) -> list[list[float]]:
    authored_dec_lo, authored_rec_lo = biorthogonal_low_passes(
        reconstruction, decomposition
    )
    dec_lo = [float(value) for value in authored_dec_lo]
    rec_lo = [float(value) for value in authored_rec_lo]
    dec_hi = [
        -coefficient if index % 2 == 0 else coefficient
        for index, coefficient in enumerate(rec_lo)
    ]
    rec_hi = [
        coefficient if index % 2 == 0 else -coefficient
        for index, coefficient in enumerate(dec_lo)
    ]
    return [dec_lo, dec_hi, rec_lo, rec_hi]


def authored_filter_bank(name: str) -> list[list[float]]:
    if name.startswith(("bior", "rbio")):
        prefix = "rbio" if name.startswith("rbio") else "bior"
        reconstruction, decomposition = map(int, name.removeprefix(prefix).split("."))
        filters = biorthogonal_filter_bank(reconstruction, decomposition)
        if prefix == "rbio":
            dec_lo, dec_hi, rec_lo, rec_hi = filters
            return [
                list(reversed(rec_lo)),
                list(reversed(rec_hi)),
                list(reversed(dec_lo)),
                list(reversed(dec_hi)),
            ]
        return filters

    if name == "haar":
        coefficients = daubechies(1)
    elif name.startswith("db"):
        coefficients = daubechies(int(name.removeprefix("db")))
    elif name.startswith("sym"):
        coefficients = symlet(int(name.removeprefix("sym")))
    elif name.startswith("coif"):
        coefficients = coiflet(int(name.removeprefix("coif")))
    else:
        raise ValueError(f"unsupported fixture wavelet {name}")
    return orthogonal_filter_bank([float(value) for value in coefficients])


def authored_wavelet(name: str) -> pywt.Wavelet:
    filter_bank = authored_filter_bank(name)

    # PyWavelets' catalog establishes the canonical family and orientation,
    # while the transforms below use our independently authored binary64
    # coefficients. Its older Symlet and CDF tables agree to at least ten
    # decimals.
    canonical = pywt.Wavelet(name).filter_bank
    if any(
        len(authored) != len(reference)
        for authored, reference in zip(filter_bank, canonical, strict=True)
    ):
        raise ArithmeticError(
            f"{name} filter lengths do not match the canonical PyWavelets bank"
        )
    maximum_error = max(
        abs(authored - reference)
        for authored_filter, reference_filter in zip(
            filter_bank, canonical, strict=True
        )
        for authored, reference in zip(authored_filter, reference_filter, strict=True)
    )
    if maximum_error > 2.0e-11:
        raise ArithmeticError(
            f"{name} does not match the canonical PyWavelets filter: "
            f"maximum coefficient error {maximum_error:.3e}"
        )

    return pywt.Wavelet(f"wavelets-{name}", filter_bank=filter_bank)


def generate() -> dict[str, object]:
    mp.mp.dps = 100
    wavelets = {name: authored_wavelet(name) for name in WAVELET_NAMES}
    signals = [
        {"len": length, "values": signal(length)}
        for length in sorted(set(SINGLE_LEVEL_LENGTHS + MULTILEVEL_LENGTHS))
    ]
    cases = []
    for wavelet_name, wavelet in wavelets.items():
        for mode in pywt.Modes.modes:
            for length in SINGLE_LEVEL_LENGTHS:
                values = signal(length)
                try:
                    approx, detail = pywt.dwt(values, wavelet, mode)
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

    multilevel_cases = []
    for wavelet_name, wavelet in wavelets.items():
        for mode in pywt.Modes.modes:
            for length in MULTILEVEL_LENGTHS:
                values = signal(length)
                level = pywt.dwt_max_level(length, wavelet.dec_len)
                bands = pywt.wavedec(values, wavelet, mode=mode, level=level)
                multilevel_cases.append(
                    {
                        "wavelet": wavelet_name,
                        "mode": mode,
                        "len": length,
                        "bands": [band.tolist() for band in bands],
                    }
                )
    return {
        "generator": f"PyWavelets {pywt.__version__}",
        "coefficient_source": "wavelets high-precision constructions",
        "signals": signals,
        "cases": cases,
        "multilevel_cases": multilevel_cases,
    }


def assert_equivalent(checked: object, generated: object, path: str = "$") -> None:
    """Compare fixtures with a normwise cross-platform floating-point bound."""
    if isinstance(checked, float) and isinstance(generated, float):
        if not math.isclose(checked, generated, rel_tol=1e-13, abs_tol=1e-13):
            raise ValueError(f"{path}: {checked!r} != {generated!r}")
        return
    if isinstance(checked, list) and isinstance(generated, list):
        if len(checked) != len(generated):
            raise ValueError(f"{path}: list lengths differ")
        if checked and all(isinstance(value, float) for value in checked + generated):
            scale = max(1.0, *(abs(value) for value in checked + generated))
            tolerance = 1e-13 * scale
            for index, (checked_item, generated_item) in enumerate(
                zip(checked, generated, strict=True)
            ):
                if abs(checked_item - generated_item) > tolerance:
                    raise ValueError(
                        f"{path}[{index}]: {checked_item!r} != {generated_item!r} "
                        f"(normwise tolerance {tolerance:.3e})"
                    )
            return
        for index, (checked_item, generated_item) in enumerate(
            zip(checked, generated, strict=True)
        ):
            assert_equivalent(checked_item, generated_item, f"{path}[{index}]")
        return
    if isinstance(checked, dict) and isinstance(generated, dict):
        if checked.keys() != generated.keys():
            raise ValueError(f"{path}: object keys differ")
        for key in checked:
            assert_equivalent(checked[key], generated[key], f"{path}.{key}")
        return
    if checked != generated:
        raise ValueError(f"{path}: {checked!r} != {generated!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=Path("tests/fixtures/pywavelets-1.8.0.json"),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail when the checked-in fixture is not reproducible",
    )
    args = parser.parse_args()
    if pywt.__version__ != "1.8.0":
        raise SystemExit(
            f"fixture authoring requires PyWavelets 1.8.0, got {pywt.__version__}"
        )
    generated = generate()
    if args.check:
        try:
            assert_equivalent(json.loads(args.output.read_text()), generated)
        except ValueError as error:
            raise SystemExit(
                f"{args.output} is stale; regenerate it with {__file__}: {error}"
            ) from error
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(generated, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
