#!/usr/bin/env python3
"""Author exact binary64 built-in coefficients from high-precision constructions."""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path

import mpmath as mp

OUTPUT = Path("src/coefficients.rs")

# Canonical least-asymmetric spectral factors. Each bit selects one root from a
# reciprocal pair of the Daubechies half-band polynomial: 0 selects the root
# inside the unit circle and 1 its reciprocal. Real auxiliary roots come first
# in ascending order, followed by positive-imaginary roots ordered by
# (real, imaginary) part. This compact, exact phase definition is the source of
# truth; the floating-point filter tables below are derived from it.
SYMLET_ROOT_SELECTIONS = {
    2: "0",
    3: "0",
    4: "01",
    5: "10",
    6: "101",
    7: "100",
    8: "0101",
    9: "0110",
    10: "10101",
    11: "01100",
    12: "101010",
    13: "001110",
    14: "0011010",
    15: "0011100",
    16: "10011010",
    17: "01110001",
    18: "101100101",
    19: "001011100",
    20: "1010011010",
}

# Canonical Cohen-Daubechies-Feauveau pairs. The tuple values are the exact
# zero multiplicities at z = -1 for the reconstruction and decomposition
# low-pass filters, followed by a mask assigning auxiliary-root groups to the
# reconstruction filter. A zero assigns the group to the decomposition filter.
#
# The spline pairs therefore use all-zero masks. The 4.4, 5.5, and 6.8 masks
# define the standard compact symmetric factorizations. The nominal 5.5 pair
# uses multiplicities 6 and 4: odd-length symmetric filters necessarily have
# even multiplicity at z = -1.
BIORTHOGONAL_SPECS = {
    (1, 1): (1, 1, ""),
    (1, 3): (1, 3, "0"),
    (1, 5): (1, 5, "0"),
    (2, 2): (2, 2, "0"),
    (2, 4): (2, 4, "0"),
    (2, 6): (2, 6, "00"),
    (2, 8): (2, 8, "00"),
    (3, 1): (3, 1, "0"),
    (3, 3): (3, 3, "0"),
    (3, 5): (3, 5, "00"),
    (3, 7): (3, 7, "00"),
    (3, 9): (3, 9, "000"),
    (4, 4): (4, 4, "10"),
    (5, 5): (6, 4, "10"),
    (6, 8): (6, 8, "010"),
}
BIORTHOGONAL_ORDERS = tuple(BIORTHOGONAL_SPECS)


def rust_bits(value: mp.mpf) -> str:
    bits = struct.unpack("!Q", struct.pack("!d", float(value)))[0]
    groups = f"{bits:016x}"
    return "0x" + "_".join(groups[index : index + 4] for index in range(0, 16, 4))


def render_array(name: str, values: list[mp.mpf]) -> list[str]:
    lines = [f"pub(crate) const {name}: [f64; {len(values)}] = ["]
    lines.extend(f"    f64::from_bits({rust_bits(value)})," for value in values)
    lines.append("];")
    return lines


def convolve(left: list[mp.mpc], right: list[mp.mpc]) -> list[mp.mpc]:
    result = [mp.mpc(0)] * (len(left) + len(right) - 1)
    for left_index, left_value in enumerate(left):
        for right_index, right_value in enumerate(right):
            result[left_index + right_index] += left_value * right_value
    return result


def auxiliary_roots(order: int) -> list[mp.mpc]:
    """Return the roots of Daubechies' auxiliary polynomial."""
    if order == 1:
        return []

    # P_N(y) = sum(k=0..N-1) binomial(N-1+k, k) y^k.
    auxiliary_ascending = [
        mp.mpf(math.comb(order - 1 + index, index)) for index in range(order)
    ]
    return mp.polyroots(
        list(reversed(auxiliary_ascending)),
        maxsteps=1000,
        error=False,
        extraprec=100,
    )


def reciprocal_root(root_y: mp.mpc, outside: bool) -> mp.mpc:
    """Map an auxiliary root to its inside or outside unit-circle root."""
    linear = 4 * root_y - 2
    discriminant = mp.sqrt(linear * linear - 4)
    candidates = [(-linear + discriminant) / 2, (-linear - discriminant) / 2]
    select = max if outside else min
    return select(candidates, key=abs)


def root_groups(roots_y: list[mp.mpc]) -> list[tuple[mp.mpc, bool]]:
    """Group real roots and one representative from each conjugate pair."""
    tolerance = mp.mpf("1e-70")
    real_roots = sorted(
        mp.re(root) for root in roots_y if abs(mp.im(root)) <= tolerance
    )
    positive_roots = sorted(
        (root for root in roots_y if mp.im(root) > tolerance),
        key=lambda root: (mp.re(root), mp.im(root)),
    )
    negative_roots = [root for root in roots_y if mp.im(root) < -tolerance]
    if len(positive_roots) != len(negative_roots):
        raise ArithmeticError("auxiliary roots did not form conjugate pairs")
    for root in positive_roots:
        if (
            min(abs(mp.conj(root) - candidate) for candidate in negative_roots)
            > tolerance
        ):
            raise ArithmeticError("auxiliary roots did not form conjugate pairs")
    return [(mp.mpc(root), False) for root in real_roots] + [
        (root, True) for root in positive_roots
    ]


def spectral_factor(order: int, selections: str, label: str) -> list[mp.mpf]:
    """Return one normalized real spectral factor in PyWavelets order."""
    if order == 1:
        if selections:
            raise ValueError("db1 has no reciprocal roots to select")
        return [1 / mp.sqrt(2)] * 2

    groups = root_groups(auxiliary_roots(order))
    if len(selections) != len(groups) or set(selections) - {"0", "1"}:
        raise ValueError(
            f"{label} needs {len(groups)} binary root selections, got {selections!r}"
        )

    # y = -(1-z)^2/(4z) gives reciprocal z-root pairs. Selecting the root
    # on either side of the unit circle controls phase without changing the
    # wavelet's magnitude response.
    roots_z: list[mp.mpc] = []
    for selection, (root_y, conjugate_pair) in zip(selections, groups, strict=True):
        root_z = reciprocal_root(root_y, outside=selection == "1")
        roots_z.append(root_z)
        if conjugate_pair:
            roots_z.append(mp.conj(root_z))

    coefficients = [mp.mpc(1)]
    for _ in range(order):
        coefficients = convolve(coefficients, [mp.mpc(1), mp.mpc(1)])
    for root_z in roots_z:
        coefficients = convolve(coefficients, [-root_z, mp.mpc(1)])

    maximum_imaginary = max(abs(mp.im(value)) for value in coefficients)
    if maximum_imaginary > mp.mpf("1e-70"):
        raise ArithmeticError(
            f"{label} spectral factor retained imaginary residue {maximum_imaginary}"
        )
    real_coefficients = [mp.re(value) for value in coefficients]
    scale = mp.sqrt(2) / mp.fsum(real_coefficients)
    return [value * scale for value in real_coefficients]


def daubechies(order: int) -> list[mp.mpf]:
    """Return the extremal-phase low-pass filter in PyWavelets order."""
    group_count = len(root_groups(auxiliary_roots(order)))
    return spectral_factor(order, "0" * group_count, f"db{order}")


def symlet(order: int) -> list[mp.mpf]:
    """Return the canonical least-asymmetric low-pass filter."""
    try:
        selections = SYMLET_ROOT_SELECTIONS[order]
    except KeyError as error:
        raise ValueError(f"unsupported Symlet order {order}") from error
    return spectral_factor(order, selections, f"sym{order}")


def add_polynomials(
    left: dict[int, mp.mpf], right: dict[int, mp.mpf]
) -> dict[int, mp.mpf]:
    result = left.copy()
    for exponent, coefficient in right.items():
        result[exponent] = result.get(exponent, mp.mpf(0)) + coefficient
    return result


def multiply_polynomials(
    left: dict[int, mp.mpf], right: dict[int, mp.mpf]
) -> dict[int, mp.mpf]:
    result: dict[int, mp.mpf] = {}
    for left_exponent, left_coefficient in left.items():
        for right_exponent, right_coefficient in right.items():
            exponent = left_exponent + right_exponent
            result[exponent] = (
                result.get(exponent, mp.mpf(0)) + left_coefficient * right_coefficient
            )
    return result


def coiflet_raw_coefficients(order: int, unknowns: tuple[mp.mpf, ...]) -> list[mp.mpf]:
    """Construct the normalized Laurent coefficients for one Newton iterate."""
    if len(unknowns) != order:
        raise ValueError(f"coif{order} needs {order} Newton unknowns")

    # Equations (4.7) and (4.12)-(4.14) of Daubechies, "Orthonormal
    # Bases of Compactly Supported Wavelets II" (1993), DOI
    # 10.1137/0524031. The first half of g is exact; the second half is the
    # square quadratic system's unknown vector.
    g = [
        2 * mp.mpf(math.comb(2 * order - 1 + index, order + index))
        for index in range(order)
    ] + list(unknowns)
    f = []
    for index in range(2 * order):
        if index == 0:
            value = (
                mp.fsum(
                    mp.mpf(math.comb(2 * degree, degree))
                    * mp.power(4, -degree)
                    * g[degree]
                    for degree in range(2 * order)
                )
                / 2
            )
        else:
            value = (-1) ** index * mp.fsum(
                mp.mpf(math.comb(2 * degree, degree - index))
                * mp.power(4, -degree)
                * g[degree]
                for degree in range(index, 2 * order)
            )
        f.append(value)

    quarter = mp.mpf(1) / 4
    cosine_squared = {-1: quarter, 0: 2 * quarter, 1: quarter}
    sine_squared = {-1: -quarter, 0: 2 * quarter, 1: -quarter}
    cosine_power = {0: mp.mpf(1)}
    sine_power = {0: mp.mpf(1)}
    bracket: dict[int, mp.mpf] = {}
    for degree in range(order):
        multiplier = mp.mpf(math.comb(order - 1 + degree, degree))
        bracket = add_polynomials(
            bracket,
            {
                exponent: multiplier * coefficient
                for exponent, coefficient in sine_power.items()
            },
        )
        cosine_power = multiply_polynomials(cosine_power, cosine_squared)
        sine_power = multiply_polynomials(sine_power, sine_squared)

    bracket = add_polynomials(
        bracket,
        multiply_polynomials(
            sine_power,
            {exponent: coefficient for exponent, coefficient in enumerate(f)},
        ),
    )
    polynomial = multiply_polynomials(cosine_power, bracket)
    return [
        polynomial.get(exponent, mp.mpf(0)) for exponent in range(-2 * order, 4 * order)
    ]


def coiflet_affine_filter(
    order: int,
) -> tuple[list[mp.mpf], list[list[mp.mpf]]]:
    zero = tuple(mp.mpf(0) for _ in range(order))
    base = coiflet_raw_coefficients(order, zero)
    basis = []
    for index in range(order):
        unit = list(zero)
        unit[index] = mp.mpf(1)
        basis.append(
            [
                coefficient - base_coefficient
                for coefficient, base_coefficient in zip(
                    coiflet_raw_coefficients(order, tuple(unit)), base, strict=True
                )
            ]
        )
    return base, basis


def solve_coiflet_stage(
    order: int, start: tuple[mp.mpf, ...], tolerance: mp.mpf
) -> tuple[mp.mpf, ...]:
    """Apply Newton's method to the independent high-lag QMF equations."""
    base, basis = coiflet_affine_filter(order)
    length = len(base)
    unknowns = mp.matrix(start)

    for _ in range(64):
        coefficients = [
            base[index]
            + mp.fsum(
                unknowns[column] * basis[column][index] for column in range(order)
            )
            for index in range(length)
        ]
        residuals = []
        jacobian = []
        for shift in range(2 * order, 3 * order):
            lag = 2 * shift
            residuals.append(
                2
                * mp.fsum(
                    coefficients[index] * coefficients[index + lag]
                    for index in range(length - lag)
                )
            )
            jacobian.append(
                [
                    2
                    * mp.fsum(
                        basis[column][index] * coefficients[index + lag]
                        + coefficients[index] * basis[column][index + lag]
                        for index in range(length - lag)
                    )
                    for column in range(order)
                ]
            )

        if max(abs(residual) for residual in residuals) <= tolerance:
            return tuple(unknowns)
        unknowns += mp.lu_solve(mp.matrix(jacobian), -mp.matrix(residuals))

    raise ArithmeticError(f"coif{order} Newton iteration did not converge")


def coiflet(order: int) -> list[mp.mpf]:
    """Return the canonical Coiflet low-pass filter in PyWavelets order."""
    if not 1 <= order <= 17:
        raise ValueError(f"unsupported Coiflet order {order}")

    # Daubechies specifies Newton iteration from the exact zero vector. Staged
    # precision makes that deterministic path well-conditioned through coif17.
    unknowns = tuple(mp.mpf(0) for _ in range(order))
    for precision, tolerance in ((60, "1e-35"), (110, "1e-75"), (180, "1e-130")):
        with mp.workdps(precision):
            unknowns = solve_coiflet_stage(order, unknowns, mp.mpf(tolerance))

    with mp.workdps(180):
        coefficients = [
            mp.sqrt(2) * coefficient
            for coefficient in reversed(coiflet_raw_coefficients(order, unknowns))
        ]
        qmf_error = max(
            abs(
                mp.fsum(
                    coefficients[index] * coefficients[index + 2 * shift]
                    for index in range(len(coefficients) - 2 * shift)
                )
                - (1 if shift == 0 else 0)
            )
            for shift in range(3 * order)
        )
        alternating_moment_error = max(
            abs(
                mp.fsum(
                    (-1) ** index * mp.power(index, degree) * coefficient
                    for index, coefficient in enumerate(coefficients)
                )
            )
            for degree in range(2 * order)
        )
        shift = 4 * order - 1
        scaling_moment_error = max(
            abs(
                mp.fsum(
                    mp.power(index - shift, degree) * coefficient
                    for index, coefficient in enumerate(coefficients)
                )
            )
            for degree in range(1, 2 * order)
        )
        maximum_error = max(qmf_error, alternating_moment_error, scaling_moment_error)
        if maximum_error > mp.mpf("1e-100"):
            raise ArithmeticError(
                f"coif{order} construction residual {maximum_error} is too large"
            )
        return coefficients


def auxiliary_factor(
    order: int, selections: str, selected: bool, label: str
) -> list[mp.mpf]:
    """Factor the Daubechies auxiliary polynomial by root-group selection."""
    groups = root_groups(auxiliary_roots(order))
    if len(selections) != len(groups) or set(selections) - {"0", "1"}:
        raise ValueError(
            f"{label} needs {len(groups)} binary root selections, got {selections!r}"
        )

    coefficients = [mp.mpc(1)]
    for selection, (root, conjugate_pair) in zip(selections, groups, strict=True):
        if (selection == "1") != selected:
            continue
        coefficients = convolve(coefficients, [-root, mp.mpc(1)])
        if conjugate_pair:
            coefficients = convolve(coefficients, [-mp.conj(root), mp.mpc(1)])

    maximum_imaginary = max(abs(mp.im(value)) for value in coefficients)
    if maximum_imaginary > mp.mpf("1e-70"):
        raise ArithmeticError(
            f"{label} auxiliary factor retained imaginary residue {maximum_imaginary}"
        )
    real_coefficients = [mp.re(value) for value in coefficients]
    scale = 1 / real_coefficients[0]
    return [value * scale for value in real_coefficients]


def cdf_low_pass(zero_order: int, factor: list[mp.mpf]) -> list[mp.mpf]:
    """Expand cos(x/2)^N * factor(sin(x/2)^2) as FIR coefficients."""
    half = mp.mpf(1) / 2
    quarter = mp.mpf(1) / 4
    cosine = {0: half, 1: half}
    sine_squared = {-1: -quarter, 0: 2 * quarter, 1: -quarter}

    cosine_power = {0: mp.mpf(1)}
    for _ in range(zero_order):
        cosine_power = multiply_polynomials(cosine_power, cosine)

    factor_response: dict[int, mp.mpf] = {}
    sine_power = {0: mp.mpf(1)}
    for coefficient in factor:
        factor_response = add_polynomials(
            factor_response,
            {exponent: coefficient * value for exponent, value in sine_power.items()},
        )
        sine_power = multiply_polynomials(sine_power, sine_squared)

    response = multiply_polynomials(cosine_power, factor_response)
    return [
        mp.sqrt(2) * response[exponent]
        for exponent in range(min(response), max(response) + 1)
    ]


def pad_filter(values: list[mp.mpf], length: int, left: int) -> list[mp.mpf]:
    right = length - left - len(values)
    if left < 0 or right < 0:
        raise ValueError("filter padding cannot be negative")
    return [mp.mpf(0)] * left + values + [mp.mpf(0)] * right


def biorthogonal_low_passes(
    reconstruction_order: int, decomposition_order: int
) -> tuple[list[mp.mpf], list[mp.mpf]]:
    """Return canonical (decomposition, reconstruction) CDF low-pass filters."""
    try:
        reconstruction_zeros, decomposition_zeros, selections = BIORTHOGONAL_SPECS[
            (reconstruction_order, decomposition_order)
        ]
    except KeyError as error:
        raise ValueError(
            "unsupported biorthogonal pair "
            f"{reconstruction_order}.{decomposition_order}"
        ) from error

    # Equations (6.10) and (6.13) of Cohen, Daubechies, and Feauveau,
    # "Biorthogonal Bases of Compactly Supported Wavelets" (1992), DOI
    # 10.1002/cpa.3160450502. Factoring the unique lowest-degree half-band
    # polynomial distributes its roots between the two symmetric low passes.
    half_order = (reconstruction_zeros + decomposition_zeros) // 2
    with mp.workdps(180):
        reconstruction_factor = auxiliary_factor(
            half_order,
            selections,
            True,
            f"bior{reconstruction_order}.{decomposition_order}",
        )
        decomposition_factor = auxiliary_factor(
            half_order,
            selections,
            False,
            f"bior{reconstruction_order}.{decomposition_order}",
        )
        expected_product = [
            mp.mpf(math.comb(half_order - 1 + index, index))
            for index in range(half_order)
        ]
        actual_product = [
            mp.re(value)
            for value in convolve(
                [mp.mpc(value) for value in reconstruction_factor],
                [mp.mpc(value) for value in decomposition_factor],
            )
        ]
        factor_error = max(
            abs(actual - expected)
            for actual, expected in zip(actual_product, expected_product, strict=True)
        )

        reconstruction = cdf_low_pass(reconstruction_zeros, reconstruction_factor)
        decomposition = cdf_low_pass(decomposition_zeros, decomposition_factor)
        common_length = max(len(decomposition), len(reconstruction))
        common_length += common_length % 2
        decomposition = pad_filter(
            decomposition,
            common_length,
            (common_length - len(decomposition) + 1) // 2,
        )
        reconstruction = pad_filter(
            reconstruction,
            common_length,
            (common_length - len(reconstruction)) // 2,
        )

        normalization_error = max(
            abs(mp.fsum(values) - mp.sqrt(2))
            for values in (decomposition, reconstruction)
        )
        moment_error = max(
            max(
                abs(
                    mp.fsum(
                        (-1) ** index * mp.power(index, degree) * coefficient
                        for index, coefficient in enumerate(values)
                    )
                )
                for degree in range(zero_order)
            )
            for values, zero_order in (
                (decomposition, decomposition_zeros),
                (reconstruction, reconstruction_zeros),
            )
        )
        reversed_reconstruction = list(reversed(reconstruction))
        biorthogonality_error = max(
            abs(
                mp.fsum(
                    decomposition[index] * reversed_reconstruction[index - lag]
                    for index in range(common_length)
                    if 0 <= index - lag < common_length
                )
                - (1 if lag == 0 else 0)
            )
            for lag in range(-2 * common_length, 2 * common_length + 1, 2)
        )
        maximum_error = max(
            factor_error,
            normalization_error,
            moment_error,
            biorthogonality_error,
        )
        if maximum_error > mp.mpf("1e-100"):
            raise ArithmeticError(
                f"bior{reconstruction_order}.{decomposition_order} construction "
                f"residual {maximum_error} is too large"
            )
        return decomposition, reconstruction


def render() -> str:
    mp.mp.dps = 100
    daubechies_filters = {order: daubechies(order) for order in range(1, 39)}
    symlet_filters = {order: symlet(order) for order in range(2, 21)}
    coiflet_filters = {order: coiflet(order) for order in range(1, 18)}
    biorthogonal_filters = {
        orders: biorthogonal_low_passes(*orders) for orders in BIORTHOGONAL_ORDERS
    }

    lines = [
        "// @generated by tools/generate_builtin_coefficients.py; do not edit by hand.",
        "",
    ]
    for order, coefficients in daubechies_filters.items():
        lines.extend(render_array(f"DB{order}_DEC_LO", coefficients))
        lines.append("")
    for order, coefficients in symlet_filters.items():
        lines.extend(render_array(f"SYM{order}_DEC_LO", coefficients))
        lines.append("")
    for order, coefficients in coiflet_filters.items():
        lines.extend(render_array(f"COIF{order}_DEC_LO", coefficients))
        lines.append("")
    for (reconstruction, decomposition), (
        dec_lo,
        rec_lo,
    ) in biorthogonal_filters.items():
        prefix = f"BIOR{reconstruction}_{decomposition}"
        lines.extend(render_array(f"{prefix}_DEC_LO", dec_lo))
        lines.append("")
        lines.extend(render_array(f"{prefix}_REC_LO", rec_lo))
        lines.append("")
    lines.extend(
        [
            "pub(crate) fn daubechies(order: usize) -> Option<&'static [f64]> {",
            "    match order {",
            *(
                f"        {order} => Some(&DB{order}_DEC_LO),"
                for order in daubechies_filters
            ),
            "        _ => None,",
            "    }",
            "}",
            "",
            "pub(crate) fn symlet(order: usize) -> Option<&'static [f64]> {",
            "    match order {",
            *(
                f"        {order} => Some(&SYM{order}_DEC_LO),"
                for order in symlet_filters
            ),
            "        _ => None,",
            "    }",
            "}",
            "",
            "pub(crate) fn coiflet(order: usize) -> Option<&'static [f64]> {",
            "    match order {",
            *(
                f"        {order} => Some(&COIF{order}_DEC_LO),"
                for order in coiflet_filters
            ),
            "        _ => None,",
            "    }",
            "}",
            "",
            "pub(crate) fn biorthogonal(",
            "    reconstruction: usize,",
            "    decomposition: usize,",
            ") -> Option<(&'static [f64], &'static [f64])> {",
            "    match (reconstruction, decomposition) {",
            *(
                f"        ({reconstruction}, {decomposition}) => "
                f"Some((&BIOR{reconstruction}_{decomposition}_DEC_LO, "
                f"&BIOR{reconstruction}_{decomposition}_REC_LO)),"
                for reconstruction, decomposition in biorthogonal_filters
            ),
            "        _ => None,",
            "    }",
            "}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail when the checked-in Rust source is not reproducible",
    )
    args = parser.parse_args()
    generated = render()
    if args.check:
        if OUTPUT.read_text() != generated:
            raise SystemExit(f"{OUTPUT} is stale; regenerate it with {__file__}")
    else:
        OUTPUT.write_text(generated)


if __name__ == "__main__":
    main()
