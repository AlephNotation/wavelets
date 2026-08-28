#!/usr/bin/env python3
"""Author conditioned paraunitary lattice factors for selected built-ins."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import mpmath as mp

import generate_builtin_coefficients as filters

OUTPUT = Path("src/lattice_coefficients.rs")


@dataclass(frozen=True)
class Section:
    q: mp.mpf
    chart: int
    determinant: int


def transpose(matrix: list[list[mp.mpf]]) -> list[list[mp.mpf]]:
    return [[matrix[column][row] for column in range(2)] for row in range(2)]


def multiply(
    left: list[list[mp.mpf]], right: list[list[mp.mpf]]
) -> list[list[mp.mpf]]:
    return [
        [
            mp.fsum(left[row][inner] * right[inner][column] for inner in range(2))
            for column in range(2)
        ]
        for row in range(2)
    ]


def add(
    left: list[list[mp.mpf]], right: list[list[mp.mpf]]
) -> list[list[mp.mpf]]:
    return [
        [left[row][column] + right[row][column] for column in range(2)]
        for row in range(2)
    ]


def outer(vector: list[mp.mpf]) -> list[list[mp.mpf]]:
    return [
        [vector[row] * vector[column] for column in range(2)] for row in range(2)
    ]


def subtract(
    left: list[list[mp.mpf]], right: list[list[mp.mpf]]
) -> list[list[mp.mpf]]:
    return [
        [left[row][column] - right[row][column] for column in range(2)]
        for row in range(2)
    ]


def polyphase(low_pass: list[mp.mpf]) -> list[list[list[mp.mpf]]]:
    """Return the crate's odd-newest analysis polyphase matrix."""
    high_pass = [
        -coefficient if index % 2 == 0 else coefficient
        for index, coefficient in enumerate(reversed(low_pass))
    ]
    return [
        [
            [low_pass[2 * index + 1], low_pass[2 * index]],
            [high_pass[2 * index + 1], high_pass[2 * index]],
        ]
        for index in range(len(low_pass) // 2)
    ]


def endpoint_direction(matrix: list[list[mp.mpf]]) -> list[mp.mpf]:
    row = max(matrix, key=lambda candidate: mp.fsum(value * value for value in candidate))
    norm = mp.sqrt(mp.fsum(value * value for value in row))
    vector = [value / norm for value in row]
    if vector[max(range(2), key=lambda index: abs(vector[index]))] < 0:
        vector = [-value for value in vector]
    return vector


def factor(polyphase_matrix: list[list[list[mp.mpf]]]) -> tuple[list[Section], mp.mpf]:
    """Factor P(z) into conditioned constant sections separated by delays."""
    identity = [[mp.mpf(1), mp.mpf(0)], [mp.mpf(0), mp.mpf(1)]]
    # Endpoint elimination peels right polynomial factors. Apply it to P^T,
    # then transpose every resulting constant section so execution remains the
    # conventional column-vector cascade A_K D ... D A_0.
    current = [transpose(matrix) for matrix in polyphase_matrix]
    directions: list[list[mp.mpf]] = []
    while len(current) > 1:
        direction = endpoint_direction(current[-1])
        projector = outer(direction)
        complement = subtract(identity, projector)
        current = [
            add(
                multiply(current[index], complement),
                multiply(current[index + 1], projector),
            )
            for index in range(len(current) - 1)
        ]
        directions.append(direction)

    rotations = [
        [[direction[1], direction[0]], [-direction[0], direction[1]]]
        for direction in reversed(directions)
    ]
    constants = [multiply(current[0], rotations[0])]
    constants.extend(
        multiply(transpose(left), right)
        for left, right in zip(rotations, rotations[1:])
    )
    constants.append(transpose(rotations[-1]))
    constants = [transpose(matrix) for matrix in constants]

    sections = []
    scale = mp.mpf(1)
    for matrix in constants:
        a = matrix[0][0]
        c = matrix[1][0]
        determinant = 1 if mp.det(mp.matrix(matrix)) > 0 else -1
        if abs(a) >= abs(c):
            chart = 0
            section_scale = a
            q = c / a
        else:
            chart = 1
            section_scale = c
            q = a / c
        if abs(q) > 1 + mp.mpf("1e-120"):
            raise ArithmeticError(f"unconditioned lattice section q={q}")
        scale *= section_scale
        sections.append(Section(q, chart, determinant))

    verify(polyphase_matrix, sections, scale)
    return sections, scale


def apply(section: Section, first: mp.mpf, second: mp.mpf) -> tuple[mp.mpf, mp.mpf]:
    q = section.q
    determinant = section.determinant
    if section.chart == 0:
        return first - determinant * q * second, q * first + determinant * second
    return q * first - determinant * second, first + determinant * q * second


def verify(
    expected: list[list[list[mp.mpf]]], sections: list[Section], scale: mp.mpf
) -> None:
    maximum_error = mp.mpf(0)
    for input_channel in range(2):
        state = [mp.mpf(0)] * (len(sections) - 1)
        for index in range(len(sections)):
            first = mp.mpf(input_channel == 0 and index == 0)
            second = mp.mpf(input_channel == 1 and index == 0)
            for section_index, section in enumerate(sections):
                if section_index:
                    previous = second
                    second = state[section_index - 1]
                    state[section_index - 1] = previous
                first, second = apply(section, first, second)
            maximum_error = max(
                maximum_error,
                abs(scale * first - expected[index][0][input_channel]),
                abs(scale * second - expected[index][1][input_channel]),
            )
    if maximum_error > mp.mpf("1e-100"):
        raise ArithmeticError(f"lattice reconstruction residual {maximum_error}")


def rust_bits(value: mp.mpf) -> str:
    return filters.rust_bits(value)


def render_sections(name: str, sections: list[Section]) -> list[str]:
    lines = [f"const {name}_SECTIONS: [LatticeSection; {len(sections)}] = ["]
    for section in sections:
        lines.extend(
            [
                "    LatticeSection {",
                f"        q: f64::from_bits({rust_bits(section.q)}),",
                f"        chart: {section.chart},",
                f"        determinant: {section.determinant},",
                "    },",
            ]
        )
    lines.append("];")
    return lines


def render() -> str:
    mp.mp.dps = 180
    selected = {
        "DB20": filters.daubechies(20),
        "SYM20": filters.symlet(20),
        "DB38": filters.daubechies(38),
        "COIF17": filters.coiflet(17),
    }
    factored = {
        name: factor(polyphase(low_pass)) for name, low_pass in selected.items()
    }

    lines = [
        "// @generated by tools/generate_lattice_coefficients.py; do not edit by hand.",
        "",
        "use crate::coefficients;",
        "",
        "#[derive(Clone, Copy, Debug)]",
        "pub(crate) struct LatticeSection {",
        "    pub(crate) q: f64,",
        "    pub(crate) chart: u8,",
        "    pub(crate) determinant: i8,",
        "}",
        "",
        "#[derive(Debug)]",
        "pub(crate) struct LatticeFactors {",
        "    pub(crate) sections: &'static [LatticeSection],",
        "    pub(crate) scale: f64,",
        "}",
        "",
    ]
    for name, (sections, scale) in factored.items():
        lines.extend(render_sections(name, sections))
        lines.extend(
            [
                f"const {name}: LatticeFactors = LatticeFactors {{",
                f"    sections: &{name}_SECTIONS,",
                f"    scale: f64::from_bits({rust_bits(scale)}),",
                "};",
                "",
            ]
        )
    lines.extend(
        [
            "pub(crate) fn analysis(dec_lo: &[f64]) -> Option<&'static LatticeFactors> {",
            "    match dec_lo.len() {",
            "        40 if dec_lo == coefficients::DB20_DEC_LO => Some(&DB20),",
            "        40 if dec_lo == coefficients::SYM20_DEC_LO => Some(&SYM20),",
            "        76 if dec_lo == coefficients::DB38_DEC_LO => Some(&DB38),",
            "        102 if dec_lo == coefficients::COIF17_DEC_LO => Some(&COIF17),",
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
