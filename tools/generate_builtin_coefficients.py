#!/usr/bin/env python3
"""Author exact binary64 built-in coefficients by spectral factorization."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import struct

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
        (mp.re(root) for root in roots_y if abs(mp.im(root)) <= tolerance)
    )
    positive_roots = sorted(
        (root for root in roots_y if mp.im(root) > tolerance),
        key=lambda root: (mp.re(root), mp.im(root)),
    )
    negative_roots = [root for root in roots_y if mp.im(root) < -tolerance]
    if len(positive_roots) != len(negative_roots):
        raise ArithmeticError("auxiliary roots did not form conjugate pairs")
    for root in positive_roots:
        if min(abs(mp.conj(root) - candidate) for candidate in negative_roots) > tolerance:
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
            f"{label} spectral factor retained imaginary residue "
            f"{maximum_imaginary}"
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


def render() -> str:
    mp.mp.dps = 100
    daubechies_filters = {order: daubechies(order) for order in range(1, 39)}
    symlet_filters = {order: symlet(order) for order in range(2, 21)}

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
            *(f"        {order} => Some(&SYM{order}_DEC_LO)," for order in symlet_filters),
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
