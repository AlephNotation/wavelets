use fearless_simd::{Simd, prelude::*};

use super::SimdSample;
use crate::lattice::LatticeSection;

pub struct LatticeAnalysis<'a, T> {
    pub(crate) signal: &'a [T],
    pub(crate) first_pair: usize,
    pub(crate) sections: &'a [LatticeSection<T>],
    pub(crate) scale: T,
}

pub(crate) const MIN_LATTICE_OUTPUTS: usize = 512;
const LATTICE_TILE: usize = 8;
const MAX_LATTICE_SECTIONS: usize = 51;
const MAX_LATTICE_LANES: usize = 8;

#[inline(always)]
pub(crate) fn forward_lattice<S: Simd, T: SimdSample<S>>(
    simd: S,
    analysis: LatticeAnalysis<'_, T>,
    approx: &mut [T],
    detail: &mut [T],
) -> usize {
    if approx.len() < MIN_LATTICE_OUTPUTS {
        return 0;
    }
    match T::Vector::N {
        2 => forward_lattice_width_2(simd, analysis, approx, detail),
        4 | 8 => forward_lattice_wide(simd, analysis, approx, detail),
        _ => 0,
    }
}

#[inline(always)]
fn apply_lattice_section<S: Simd, T: SimdSample<S>>(
    simd: S,
    section: LatticeSection<T>,
    first: T::Vector,
    second: T::Vector,
) -> (T::Vector, T::Vector) {
    let q = T::Vector::splat(simd, section.q);
    match (section.chart, section.determinant) {
        (0, 1) => ((-second).mul_add(q, first), first.mul_add(q, second)),
        (0, -1) => (second.mul_add(q, first), first.mul_add(q, -second)),
        (1, 1) => (first.mul_add(q, -second), second.mul_add(q, first)),
        (1, -1) => (first.mul_add(q, second), (-second).mul_add(q, first)),
        _ => unreachable!("generated lattice sections use two charts and unit determinants"),
    }
}

#[inline(always)]
fn apply_lattice_0_positive<S: Simd, T: SimdSample<S>>(
    _simd: S,
    q: T::Vector,
    first: T::Vector,
    second: T::Vector,
) -> (T::Vector, T::Vector) {
    ((-second).mul_add(q, first), first.mul_add(q, second))
}

#[inline(always)]
fn apply_lattice_0_negative<S: Simd, T: SimdSample<S>>(
    _simd: S,
    q: T::Vector,
    first: T::Vector,
    second: T::Vector,
) -> (T::Vector, T::Vector) {
    (second.mul_add(q, first), first.mul_add(q, -second))
}

#[inline(always)]
fn apply_lattice_1_positive<S: Simd, T: SimdSample<S>>(
    _simd: S,
    q: T::Vector,
    first: T::Vector,
    second: T::Vector,
) -> (T::Vector, T::Vector) {
    (first.mul_add(q, -second), second.mul_add(q, first))
}

#[inline(always)]
fn apply_lattice_1_negative<S: Simd, T: SimdSample<S>>(
    _simd: S,
    q: T::Vector,
    first: T::Vector,
    second: T::Vector,
) -> (T::Vector, T::Vector) {
    (first.mul_add(q, second), (-second).mul_add(q, first))
}

#[inline(always)]
fn load_lattice_pair_width_2<S: Simd, T: SimdSample<S>>(
    simd: S,
    signal: &[T],
    first_pair: usize,
    second_pair: usize,
) -> (T::Vector, T::Vector) {
    let first_offset = 2 * first_pair;
    let second_offset = 2 * second_pair;
    let first = T::Vector::from_slice(simd, &signal[first_offset..first_offset + 2]);
    let second = T::Vector::from_slice(simd, &signal[second_offset..second_offset + 2]);
    first.deinterleave(second)
}

#[inline(always)]
fn forward_lattice_width_2<S: Simd, T: SimdSample<S>>(
    simd: S,
    analysis: LatticeAnalysis<'_, T>,
    approx: &mut [T],
    detail: &mut [T],
) -> usize {
    debug_assert_eq!(T::Vector::N, 2);
    debug_assert_eq!(approx.len(), detail.len());
    debug_assert!(analysis.sections.len() <= MAX_LATTICE_SECTIONS);

    let processed = approx.len() - approx.len() % (2 * LATTICE_TILE);
    let segment_len = processed / 2;
    let delay_count = analysis.sections.len() - 1;
    debug_assert!(analysis.first_pair >= delay_count);

    let zero = T::Vector::splat(simd, T::default());
    let mut state = [zero; MAX_LATTICE_SECTIONS];
    for predecessor in analysis.first_pair - delay_count..analysis.first_pair {
        let (mut first, mut second) = load_lattice_pair_width_2(
            simd,
            analysis.signal,
            predecessor,
            predecessor + segment_len,
        );
        (first, second) = apply_lattice_section(simd, analysis.sections[0], first, second);
        for (stage, &section) in analysis.sections[1..].iter().enumerate() {
            std::mem::swap(&mut second, &mut state[stage]);
            (first, second) = apply_lattice_section(simd, section, first, second);
        }
    }

    for offset in (0..segment_len).step_by(LATTICE_TILE) {
        let mut first = [zero; LATTICE_TILE];
        let mut second = [zero; LATTICE_TILE];
        for time in 0..LATTICE_TILE {
            (first[time], second[time]) = load_lattice_pair_width_2(
                simd,
                analysis.signal,
                analysis.first_pair + offset + time,
                analysis.first_pair + segment_len + offset + time,
            );
        }

        let initial = analysis.sections[0];
        let initial_q = T::Vector::splat(simd, initial.q);
        macro_rules! apply_initial {
            ($apply:ident) => {
                for time in 0..LATTICE_TILE {
                    (first[time], second[time]) =
                        $apply::<S, T>(simd, initial_q, first[time], second[time]);
                }
            };
        }
        match (initial.chart, initial.determinant) {
            (0, 1) => apply_initial!(apply_lattice_0_positive),
            (0, -1) => apply_initial!(apply_lattice_0_negative),
            (1, 1) => apply_initial!(apply_lattice_1_positive),
            (1, -1) => apply_initial!(apply_lattice_1_negative),
            _ => unreachable!("generated lattice section kind"),
        }

        for (stage, &section) in analysis.sections[1..].iter().enumerate() {
            let final_previous = second[LATTICE_TILE - 1];
            let q = T::Vector::splat(simd, section.q);
            macro_rules! apply_stage {
                ($apply:ident) => {
                    for time in (0..LATTICE_TILE).rev() {
                        let delayed = if time == 0 {
                            state[stage]
                        } else {
                            second[time - 1]
                        };
                        (first[time], second[time]) = $apply::<S, T>(simd, q, first[time], delayed);
                    }
                };
            }
            match (section.chart, section.determinant) {
                (0, 1) => apply_stage!(apply_lattice_0_positive),
                (0, -1) => apply_stage!(apply_lattice_0_negative),
                (1, 1) => apply_stage!(apply_lattice_1_positive),
                (1, -1) => apply_stage!(apply_lattice_1_negative),
                _ => unreachable!("generated lattice section kind"),
            }
            state[stage] = final_previous;
        }

        for time in (0..LATTICE_TILE).step_by(2) {
            let (first_segment, second_segment) =
                (first[time] * analysis.scale).interleave(first[time + 1] * analysis.scale);
            first_segment.store_slice(&mut approx[offset + time..offset + time + 2]);
            second_segment.store_slice(
                &mut approx[segment_len + offset + time..segment_len + offset + time + 2],
            );

            let (first_segment, second_segment) =
                (second[time] * analysis.scale).interleave(second[time + 1] * analysis.scale);
            first_segment.store_slice(&mut detail[offset + time..offset + time + 2]);
            second_segment.store_slice(
                &mut detail[segment_len + offset + time..segment_len + offset + time + 2],
            );
        }
    }

    processed
}

#[inline(always)]
fn transpose_lattice_vectors<S: Simd, T: SimdSample<S>>(
    vectors: &mut [T::Vector; MAX_LATTICE_LANES],
    width: usize,
) {
    debug_assert!(matches!(width, 4 | 8));
    let mut first = [vectors[0]; MAX_LATTICE_LANES];
    for pair in (0..width).step_by(2) {
        (first[pair], first[pair + 1]) = vectors[pair].deinterleave(vectors[pair + 1]);
    }

    let mut second = first;
    for group in (0..width).step_by(4) {
        (second[group], second[group + 2]) = first[group].deinterleave(first[group + 2]);
        (second[group + 1], second[group + 3]) = first[group + 1].deinterleave(first[group + 3]);
    }

    if width == 4 {
        vectors[..width].copy_from_slice(&second[..width]);
        return;
    }

    (vectors[0], vectors[4]) = second[0].deinterleave(second[4]);
    (vectors[1], vectors[5]) = second[1].deinterleave(second[5]);
    (vectors[2], vectors[6]) = second[2].deinterleave(second[6]);
    (vectors[3], vectors[7]) = second[3].deinterleave(second[7]);
}

#[inline(always)]
fn load_lattice_predecessor<S: Simd, T: SimdSample<S>>(
    simd: S,
    signal: &[T],
    pair: usize,
    segment_len: usize,
) -> (T::Vector, T::Vector) {
    let width = T::Vector::N;
    let mut first = [T::default(); MAX_LATTICE_LANES];
    let mut second = first;
    for segment in 0..width {
        let offset = 2 * (pair + segment * segment_len);
        first[segment] = signal[offset];
        second[segment] = signal[offset + 1];
    }
    (
        T::Vector::from_slice(simd, &first[..width]),
        T::Vector::from_slice(simd, &second[..width]),
    )
}

#[inline(always)]
fn load_lattice_tile<S: Simd, T: SimdSample<S>>(
    simd: S,
    signal: &[T],
    first_pair: usize,
    segment_len: usize,
    first: &mut [T::Vector; MAX_LATTICE_LANES],
    second: &mut [T::Vector; MAX_LATTICE_LANES],
) {
    let width = T::Vector::N;
    for segment in 0..width {
        let offset = 2 * (first_pair + segment * segment_len);
        let first_half = T::Vector::from_slice(simd, &signal[offset..offset + width]);
        let second_half = T::Vector::from_slice(simd, &signal[offset + width..offset + 2 * width]);
        (first[segment], second[segment]) = first_half.deinterleave(second_half);
    }
    transpose_lattice_vectors::<S, T>(first, width);
    transpose_lattice_vectors::<S, T>(second, width);
}

#[inline(always)]
fn forward_lattice_wide<S: Simd, T: SimdSample<S>>(
    simd: S,
    analysis: LatticeAnalysis<'_, T>,
    approx: &mut [T],
    detail: &mut [T],
) -> usize {
    let width = T::Vector::N;
    debug_assert!(matches!(width, 4 | 8));
    debug_assert_eq!(approx.len(), detail.len());
    debug_assert!(analysis.sections.len() <= MAX_LATTICE_SECTIONS);

    let output_block = width * width;
    let processed = approx.len() - approx.len() % output_block;
    let segment_len = processed / width;
    let delay_count = analysis.sections.len() - 1;
    debug_assert!(analysis.first_pair >= delay_count);

    let zero = T::Vector::splat(simd, T::default());
    let mut state = [zero; MAX_LATTICE_SECTIONS];
    for predecessor in analysis.first_pair - delay_count..analysis.first_pair {
        let (mut first, mut second) =
            load_lattice_predecessor(simd, analysis.signal, predecessor, segment_len);
        (first, second) = apply_lattice_section(simd, analysis.sections[0], first, second);
        for (stage, &section) in analysis.sections[1..].iter().enumerate() {
            std::mem::swap(&mut second, &mut state[stage]);
            (first, second) = apply_lattice_section(simd, section, first, second);
        }
    }

    for offset in (0..segment_len).step_by(width) {
        let mut first = [zero; MAX_LATTICE_LANES];
        let mut second = first;
        load_lattice_tile(
            simd,
            analysis.signal,
            analysis.first_pair + offset,
            segment_len,
            &mut first,
            &mut second,
        );

        let initial = analysis.sections[0];
        let initial_q = T::Vector::splat(simd, initial.q);
        macro_rules! apply_initial {
            ($apply:ident) => {
                for time in 0..width {
                    (first[time], second[time]) =
                        $apply::<S, T>(simd, initial_q, first[time], second[time]);
                }
            };
        }
        match (initial.chart, initial.determinant) {
            (0, 1) => apply_initial!(apply_lattice_0_positive),
            (0, -1) => apply_initial!(apply_lattice_0_negative),
            (1, 1) => apply_initial!(apply_lattice_1_positive),
            (1, -1) => apply_initial!(apply_lattice_1_negative),
            _ => unreachable!("generated lattice section kind"),
        }

        for (stage, &section) in analysis.sections[1..].iter().enumerate() {
            let final_previous = second[width - 1];
            let q = T::Vector::splat(simd, section.q);
            macro_rules! apply_stage {
                ($apply:ident) => {
                    for time in (0..width).rev() {
                        let delayed = if time == 0 {
                            state[stage]
                        } else {
                            second[time - 1]
                        };
                        (first[time], second[time]) = $apply::<S, T>(simd, q, first[time], delayed);
                    }
                };
            }
            match (section.chart, section.determinant) {
                (0, 1) => apply_stage!(apply_lattice_0_positive),
                (0, -1) => apply_stage!(apply_lattice_0_negative),
                (1, 1) => apply_stage!(apply_lattice_1_positive),
                (1, -1) => apply_stage!(apply_lattice_1_negative),
                _ => unreachable!("generated lattice section kind"),
            }
            state[stage] = final_previous;
        }

        transpose_lattice_vectors::<S, T>(&mut first, width);
        transpose_lattice_vectors::<S, T>(&mut second, width);
        for segment in 0..width {
            let output = segment * segment_len + offset;
            (first[segment] * analysis.scale).store_slice(&mut approx[output..output + width]);
            (second[segment] * analysis.scale).store_slice(&mut detail[output..output + width]);
        }
    }

    processed
}
