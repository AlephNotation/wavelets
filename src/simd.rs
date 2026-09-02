use std::mem::size_of;

use fearless_simd::{Simd, SimdFloatElement, prelude::*};

#[cfg(feature = "experimental-kernels")]
use crate::lattice::LatticeSection;
use crate::plan::EdgeTerm;

#[cfg(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64"))]
pub(crate) mod axis_fusion;

pub(crate) trait SimdSample<S: Simd>: SimdFloatElement {
    type Vector: SimdFloat<S, Element = Self>;
}

impl<S: Simd> SimdSample<S> for f32 {
    type Vector = S::f32s;
}

impl<S: Simd> SimdSample<S> for f64 {
    type Vector = S::f64s;
}

pub struct AnalysisInterior<'a, T> {
    pub(crate) dec_lo: &'a [T],
    pub(crate) dec_hi: &'a [T],
    pub(crate) signal: &'a [T],
    pub(crate) first_newest: usize,
}

pub struct AxisAnalysis<'a, T> {
    pub(crate) signal: &'a [T],
    pub(crate) dec_lo: &'a [T],
    pub(crate) dec_hi: &'a [T],
    pub(crate) edge_row_offsets: &'a [usize],
    pub(crate) edge_terms: &'a [EdgeTerm<T>],
    pub(crate) signal_len: usize,
    pub(crate) coeff_len: usize,
    pub(crate) outer: usize,
    pub(crate) inner: usize,
    pub(crate) prefix_len: usize,
    pub(crate) interior_first_newest: usize,
    pub(crate) interior_len: usize,
}

pub struct AxisSynthesis<'a, T> {
    pub(crate) approx: &'a [T],
    pub(crate) detail: &'a [T],
    pub(crate) rec_lo: &'a [T],
    pub(crate) rec_hi: &'a [T],
    pub(crate) signal_len: usize,
    pub(crate) coeff_len: usize,
    pub(crate) outer: usize,
    pub(crate) inner: usize,
    pub(crate) periodized_initial: Option<usize>,
    pub(crate) periodized_phases_are_swapped: bool,
}

pub struct ButterflyAnalysis<'a, T> {
    pub(crate) signal: &'a [T],
    pub(crate) first_newest: usize,
    pub(crate) low_scale: T,
    pub(crate) high_scale: T,
}

pub struct ButterflyPairAnalysis<'a, T> {
    pub(crate) signal: &'a [T],
    pub(crate) first_low_scale: T,
    pub(crate) first_high_scale: T,
    pub(crate) second_low_scale: T,
    pub(crate) second_high_scale: T,
}

#[cfg(feature = "experimental-kernels")]
pub struct LatticeAnalysis<'a, T> {
    pub(crate) signal: &'a [T],
    pub(crate) first_pair: usize,
    pub(crate) sections: &'a [LatticeSection<T>],
    pub(crate) scale: T,
}

#[derive(Clone, Copy)]
struct AnalysisAccumulators<V> {
    low_earlier: V,
    low_later: V,
    high_earlier: V,
    high_later: V,
}

pub struct LinearSynthesis<'a, T> {
    pub(crate) rec_lo: &'a [T],
    pub(crate) rec_hi: &'a [T],
    pub(crate) approx: &'a [T],
    pub(crate) detail: &'a [T],
}

pub struct ButterflySynthesis<'a, T> {
    pub(crate) approx: &'a [T],
    pub(crate) detail: &'a [T],
    pub(crate) low_scale: T,
    pub(crate) high_scale: T,
}

pub struct ButterflyPairSynthesis<'a, T> {
    pub(crate) approx: &'a [T],
    pub(crate) first_detail: &'a [T],
    pub(crate) second_detail: &'a [T],
    pub(crate) first_low_scale: T,
    pub(crate) first_high_scale: T,
    pub(crate) second_low_scale: T,
    pub(crate) second_high_scale: T,
}

pub struct PeriodizedInterior<'a, T> {
    pub(crate) first_lo: &'a [T],
    pub(crate) first_hi: &'a [T],
    pub(crate) second_lo: &'a [T],
    pub(crate) second_hi: &'a [T],
    pub(crate) approx: &'a [T],
    pub(crate) detail: &'a [T],
    pub(crate) first_coefficient: usize,
    pub(crate) second_offset: usize,
}

#[inline(always)]
fn analyze_axis_edge_row<S: Simd, T: SimdSample<S>>(
    simd: S,
    signal: &[T],
    inner: usize,
    vectorized: usize,
    terms: &[EdgeTerm<T>],
    approx: &mut [T],
    detail: &mut [T],
) {
    let lanes = T::Vector::N;
    for lane in (0..vectorized).step_by(lanes) {
        let mut low = T::Vector::splat(simd, T::default());
        let mut high = low;
        for term in terms {
            let offset = term.input * inner + lane;
            let sample = T::Vector::from_slice(simd, &signal[offset..offset + lanes]);
            low = sample.mul_add(term.low, low);
            high = sample.mul_add(term.high, high);
        }
        low.store_slice(&mut approx[lane..lane + lanes]);
        high.store_slice(&mut detail[lane..lane + lanes]);
    }
}

#[inline(always)]
fn analyze_axis_interior_row<S: Simd, T: SimdSample<S>>(
    simd: S,
    analysis: &AxisAnalysis<'_, T>,
    signal: &[T],
    vectorized: usize,
    newest: usize,
    approx: &mut [T],
    detail: &mut [T],
) {
    let lanes = T::Vector::N;
    for lane in (0..vectorized).step_by(lanes) {
        let mut low = T::Vector::splat(simd, T::default());
        let mut high = low;
        for tap in 0..analysis.dec_lo.len() {
            let offset = (newest - tap) * analysis.inner + lane;
            let sample = T::Vector::from_slice(simd, &signal[offset..offset + lanes]);
            low = sample.mul_add(analysis.dec_lo[tap], low);
            high = sample.mul_add(analysis.dec_hi[tap], high);
        }
        low.store_slice(&mut approx[lane..lane + lanes]);
        high.store_slice(&mut detail[lane..lane + lanes]);
    }
}

#[inline(always)]
fn analyze_axis_interior_rows<S: Simd, T: SimdSample<S>>(
    simd: S,
    analysis: &AxisAnalysis<'_, T>,
    signal: &[T],
    vectorized: usize,
    outputs: std::ops::Range<usize>,
    approx: &mut [T],
    detail: &mut [T],
) {
    for interior_output in outputs {
        let output = analysis.prefix_len + interior_output;
        let output_start = output * analysis.inner;
        analyze_axis_interior_row(
            simd,
            analysis,
            signal,
            vectorized,
            analysis.interior_first_newest + 2 * interior_output,
            &mut approx[output_start..output_start + analysis.inner],
            &mut detail[output_start..output_start + analysis.inner],
        );
    }
}

#[inline(always)]
fn analyze_axis_interior_batches<S: Simd, T: SimdSample<S>, const OUTPUTS_PER_BATCH: usize>(
    simd: S,
    analysis: &AxisAnalysis<'_, T>,
    signal: &[T],
    vectorized: usize,
    approx: &mut [T],
    detail: &mut [T],
) {
    let batched_outputs = analysis.interior_len - analysis.interior_len % OUTPUTS_PER_BATCH;
    for first_output in (0..batched_outputs).step_by(OUTPUTS_PER_BATCH) {
        analyze_axis_interior_batch::<S, T, OUTPUTS_PER_BATCH>(
            simd,
            analysis,
            signal,
            vectorized,
            first_output,
            approx,
            detail,
        );
    }
    analyze_axis_interior_rows(
        simd,
        analysis,
        signal,
        vectorized,
        batched_outputs..analysis.interior_len,
        approx,
        detail,
    );
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn analyze_axis_interior_batch<S: Simd, T: SimdSample<S>, const OUTPUTS_PER_BATCH: usize>(
    simd: S,
    analysis: &AxisAnalysis<'_, T>,
    signal: &[T],
    vectorized: usize,
    first_output: usize,
    approx: &mut [T],
    detail: &mut [T],
) {
    let lanes = T::Vector::N;
    let first_newest = analysis.interior_first_newest + 2 * first_output;
    let last_newest = first_newest + 2 * (OUTPUTS_PER_BATCH - 1);
    let first_input = first_newest + 1 - analysis.dec_lo.len();

    for lane in (0..vectorized).step_by(lanes) {
        let zero = T::Vector::splat(simd, T::default());
        let mut low = [zero; OUTPUTS_PER_BATCH];
        let mut high = low;

        for input in (first_input..=last_newest).rev() {
            let offset = input * analysis.inner + lane;
            let sample = T::Vector::from_slice(simd, &signal[offset..offset + lanes]);

            for output in 0..OUTPUTS_PER_BATCH {
                let newest = first_newest + 2 * output;
                if input <= newest {
                    let tap = newest - input;
                    if tap < analysis.dec_lo.len() {
                        low[output] = sample.mul_add(analysis.dec_lo[tap], low[output]);
                        high[output] = sample.mul_add(analysis.dec_hi[tap], high[output]);
                    }
                }
            }
        }

        for output in 0..OUTPUTS_PER_BATCH {
            let output_start =
                (analysis.prefix_len + first_output + output) * analysis.inner + lane;
            low[output].store_slice(&mut approx[output_start..output_start + lanes]);
            high[output].store_slice(&mut detail[output_start..output_start + lanes]);
        }
    }
}

#[inline(always)]
pub(crate) fn forward_axis<S: Simd, T: SimdSample<S>>(
    simd: S,
    analysis: AxisAnalysis<'_, T>,
    approx: &mut [T],
    detail: &mut [T],
) -> usize {
    forward_axis_with_batch::<S, T, 1>(simd, analysis, approx, detail)
}

#[inline(always)]
#[cfg(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64"))]
pub(crate) fn forward_axis_fused<S: Simd, T: SimdSample<S>, const OUTPUTS_PER_BATCH: usize>(
    simd: S,
    analysis: AxisAnalysis<'_, T>,
    approx: &mut [T],
    detail: &mut [T],
) -> usize {
    forward_axis_with_batch::<S, T, OUTPUTS_PER_BATCH>(simd, analysis, approx, detail)
}

#[inline(always)]
fn forward_axis_with_batch<S: Simd, T: SimdSample<S>, const OUTPUTS_PER_BATCH: usize>(
    simd: S,
    analysis: AxisAnalysis<'_, T>,
    approx: &mut [T],
    detail: &mut [T],
) -> usize {
    let lanes = T::Vector::N;
    let vectorized = analysis.inner - analysis.inner % lanes;
    let interior_end = analysis.prefix_len + analysis.interior_len;

    for outer in 0..analysis.outer {
        let signal_start = outer * analysis.signal_len * analysis.inner;
        let signal =
            &analysis.signal[signal_start..signal_start + analysis.signal_len * analysis.inner];
        let output_start = outer * analysis.coeff_len * analysis.inner;
        let approx = &mut approx[output_start..output_start + analysis.coeff_len * analysis.inner];
        let detail = &mut detail[output_start..output_start + analysis.coeff_len * analysis.inner];

        for output in 0..analysis.prefix_len {
            let term_start = analysis.edge_row_offsets[output];
            let term_end = analysis.edge_row_offsets[output + 1];
            let output_start = output * analysis.inner;
            analyze_axis_edge_row(
                simd,
                signal,
                analysis.inner,
                vectorized,
                &analysis.edge_terms[term_start..term_end],
                &mut approx[output_start..output_start + analysis.inner],
                &mut detail[output_start..output_start + analysis.inner],
            );
        }

        if OUTPUTS_PER_BATCH == 1 {
            analyze_axis_interior_rows(
                simd,
                &analysis,
                signal,
                vectorized,
                0..analysis.interior_len,
                approx,
                detail,
            );
        } else {
            analyze_axis_interior_batches::<S, T, OUTPUTS_PER_BATCH>(
                simd, &analysis, signal, vectorized, approx, detail,
            );
        }

        for output in interior_end..analysis.coeff_len {
            let edge_row = analysis.prefix_len + output - interior_end;
            let term_start = analysis.edge_row_offsets[edge_row];
            let term_end = analysis.edge_row_offsets[edge_row + 1];
            let output_start = output * analysis.inner;
            analyze_axis_edge_row(
                simd,
                signal,
                analysis.inner,
                vectorized,
                &analysis.edge_terms[term_start..term_end],
                &mut approx[output_start..output_start + analysis.inner],
                &mut detail[output_start..output_start + analysis.inner],
            );
        }
    }

    vectorized
}

#[inline(always)]
pub(crate) fn inverse_axis_batched<S: Simd, T: SimdSample<S>>(
    simd: S,
    synthesis: AxisSynthesis<'_, T>,
    out: &mut [T],
) -> usize {
    // AVX-512 and AArch64 both have enough vector registers for sixteen output
    // accumulators. Narrower x86 backends use eight to avoid register spills.
    if cfg!(target_arch = "aarch64") || size_of::<T::Vector>() >= 64 {
        inverse_axis_linear_batched::<S, T, 8>(simd, synthesis, out)
    } else {
        inverse_axis_linear_batched::<S, T, 4>(simd, synthesis, out)
    }
}

#[inline(always)]
fn inverse_axis_linear_batched<S: Simd, T: SimdSample<S>, const PAIRS_PER_BATCH: usize>(
    simd: S,
    synthesis: AxisSynthesis<'_, T>,
    out: &mut [T],
) -> usize {
    let lanes = T::Vector::N;
    let vectorized = synthesis.inner - synthesis.inner % lanes;
    let half_filter_len = synthesis.rec_lo.len() / 2;
    let (first_lo, second_lo) = synthesis.rec_lo.split_at(half_filter_len);
    let (first_hi, second_hi) = synthesis.rec_hi.split_at(half_filter_len);
    let output_pairs = synthesis.signal_len.div_ceil(2);

    for outer in 0..synthesis.outer {
        let coeff_start = outer * synthesis.coeff_len * synthesis.inner;
        let coeff_end = coeff_start + synthesis.coeff_len * synthesis.inner;
        let approx = &synthesis.approx[coeff_start..coeff_end];
        let detail = &synthesis.detail[coeff_start..coeff_end];
        let output_start = outer * synthesis.signal_len * synthesis.inner;
        let out = &mut out[output_start..output_start + synthesis.signal_len * synthesis.inner];

        for batch_start in (0..output_pairs).step_by(PAIRS_PER_BATCH) {
            let pairs = (output_pairs - batch_start).min(PAIRS_PER_BATCH);
            let first_coefficient = batch_start + half_filter_len - 1;
            let last_coefficient = first_coefficient + pairs - 1;

            for lane in (0..vectorized).step_by(lanes) {
                let zero = T::Vector::splat(simd, T::default());
                let mut first = [zero; PAIRS_PER_BATCH];
                let mut second = first;

                for coefficient in (batch_start..=last_coefficient).rev() {
                    let offset = coefficient * synthesis.inner + lane;
                    let approx = T::Vector::from_slice(simd, &approx[offset..offset + lanes]);
                    let detail = T::Vector::from_slice(simd, &detail[offset..offset + lanes]);

                    for batch_pair in 0..pairs {
                        let pair_coefficient = first_coefficient + batch_pair;
                        if coefficient <= pair_coefficient {
                            let tap = pair_coefficient - coefficient;
                            if tap < half_filter_len {
                                first[batch_pair] =
                                    approx.mul_add(first_lo[tap], first[batch_pair]);
                                first[batch_pair] =
                                    detail.mul_add(first_hi[tap], first[batch_pair]);
                                second[batch_pair] =
                                    approx.mul_add(second_lo[tap], second[batch_pair]);
                                second[batch_pair] =
                                    detail.mul_add(second_hi[tap], second[batch_pair]);
                            }
                        }
                    }
                }

                for batch_pair in 0..pairs {
                    let pair = batch_start + batch_pair;
                    let first_output = 2 * pair * synthesis.inner + lane;
                    first[batch_pair].store_slice(&mut out[first_output..first_output + lanes]);
                    if 2 * pair + 1 < synthesis.signal_len {
                        let second_output = first_output + synthesis.inner;
                        second[batch_pair]
                            .store_slice(&mut out[second_output..second_output + lanes]);
                    }
                }
            }
        }
    }

    vectorized
}

#[inline(always)]
pub(crate) fn inverse_axis<S: Simd, T: SimdSample<S>>(
    simd: S,
    synthesis: AxisSynthesis<'_, T>,
    out: &mut [T],
) -> usize {
    let lanes = T::Vector::N;
    let vectorized = synthesis.inner - synthesis.inner % lanes;
    let half_filter_len = synthesis.rec_lo.len() / 2;
    let (first_lo, second_lo) = synthesis.rec_lo.split_at(half_filter_len);
    let (first_hi, second_hi) = synthesis.rec_hi.split_at(half_filter_len);
    let output_pairs = synthesis.signal_len.div_ceil(2);
    for outer in 0..synthesis.outer {
        let coeff_start = outer * synthesis.coeff_len * synthesis.inner;
        let coeff_end = coeff_start + synthesis.coeff_len * synthesis.inner;
        let approx = &synthesis.approx[coeff_start..coeff_end];
        let detail = &synthesis.detail[coeff_start..coeff_end];
        let output_start = outer * synthesis.signal_len * synthesis.inner;
        let out = &mut out[output_start..output_start + synthesis.signal_len * synthesis.inner];

        for pair in 0..output_pairs {
            let first_coefficient = synthesis
                .periodized_initial
                .map_or(pair + half_filter_len - 1, |initial| {
                    (initial + pair) % synthesis.coeff_len
                });
            let second_coefficient = if synthesis.periodized_initial.is_some()
                && synthesis.periodized_phases_are_swapped
            {
                (first_coefficient + 1) % synthesis.coeff_len
            } else {
                first_coefficient
            };

            for lane in (0..vectorized).step_by(lanes) {
                let mut first = T::Vector::splat(simd, T::default());
                let mut second = first;
                for tap in 0..half_filter_len {
                    let first_index = if synthesis.periodized_initial.is_some() {
                        (first_coefficient + synthesis.coeff_len - tap % synthesis.coeff_len)
                            % synthesis.coeff_len
                    } else {
                        first_coefficient - tap
                    };
                    let second_index = if synthesis.periodized_initial.is_some() {
                        (second_coefficient + synthesis.coeff_len - tap % synthesis.coeff_len)
                            % synthesis.coeff_len
                    } else {
                        second_coefficient - tap
                    };
                    let first_offset = first_index * synthesis.inner + lane;
                    let second_offset = second_index * synthesis.inner + lane;
                    let first_approx =
                        T::Vector::from_slice(simd, &approx[first_offset..first_offset + lanes]);
                    let first_detail =
                        T::Vector::from_slice(simd, &detail[first_offset..first_offset + lanes]);
                    let second_approx =
                        T::Vector::from_slice(simd, &approx[second_offset..second_offset + lanes]);
                    let second_detail =
                        T::Vector::from_slice(simd, &detail[second_offset..second_offset + lanes]);
                    first = first_approx.mul_add(first_lo[tap], first);
                    first = first_detail.mul_add(first_hi[tap], first);
                    second = second_approx.mul_add(second_lo[tap], second);
                    second = second_detail.mul_add(second_hi[tap], second);
                }

                let first_output = 2 * pair * synthesis.inner + lane;
                first.store_slice(&mut out[first_output..first_output + lanes]);
                if 2 * pair + 1 < synthesis.signal_len {
                    let second_output = first_output + synthesis.inner;
                    second.store_slice(&mut out[second_output..second_output + lanes]);
                }
            }
        }
    }

    vectorized
}

#[inline(always)]
pub(crate) fn forward_interior<S: Simd, T: SimdSample<S>>(
    simd: S,
    interior: AnalysisInterior<'_, T>,
    approx: &mut [T],
    detail: &mut [T],
) -> usize {
    // Peeling consumes the entire two-tap filter, leaving no FMA dependency
    // chain for a wider batch to hide.
    if interior.dec_lo.len() == 2 {
        forward_interior_batches::<_, _, 1>(simd, &interior, approx, detail)
    } else {
        forward_interior_batches::<_, _, 2>(simd, &interior, approx, detail)
    }
}

#[inline(always)]
pub(crate) fn forward_butterfly<S: Simd, T: SimdSample<S>>(
    simd: S,
    analysis: ButterflyAnalysis<'_, T>,
    approx: &mut [T],
    detail: &mut [T],
) -> usize {
    let lanes = T::Vector::N;
    let vectorized_outputs = approx.len() - approx.len() % lanes;

    for output in (0..vectorized_outputs).step_by(lanes) {
        let input = analysis.first_newest - 1 + 2 * output;
        let first = T::Vector::from_slice(simd, &analysis.signal[input..input + lanes]);
        let second =
            T::Vector::from_slice(simd, &analysis.signal[input + lanes..input + 2 * lanes]);
        let (earlier, later) = first.deinterleave(second);
        // Keep the stored FIR evaluation order. Collapsing these into
        // `(earlier +/- later) * scale` is algebraically equivalent, but can
        // differ by one ulp from applying tap 0 to `later` and tap 1 to
        // `earlier`, as the generic convolution does.
        (later * analysis.low_scale + earlier * analysis.low_scale)
            .store_slice(&mut approx[output..output + lanes]);
        (later * (T::default() - analysis.high_scale) + earlier * analysis.high_scale)
            .store_slice(&mut detail[output..output + lanes]);
    }

    vectorized_outputs
}

#[inline(always)]
pub(crate) fn forward_butterfly_pair<S: Simd, T: SimdSample<S>>(
    simd: S,
    analysis: ButterflyPairAnalysis<'_, T>,
    approx: &mut [T],
    first_detail: &mut [T],
    second_detail: &mut [T],
) -> usize {
    let lanes = T::Vector::N;
    let vectorized_outputs = approx.len() - approx.len() % lanes;

    for output in (0..vectorized_outputs).step_by(lanes) {
        let input = 4 * output;
        let first = T::Vector::from_slice(simd, &analysis.signal[input..input + lanes]);
        let second =
            T::Vector::from_slice(simd, &analysis.signal[input + lanes..input + 2 * lanes]);
        let third =
            T::Vector::from_slice(simd, &analysis.signal[input + 2 * lanes..input + 3 * lanes]);
        let fourth =
            T::Vector::from_slice(simd, &analysis.signal[input + 3 * lanes..input + 4 * lanes]);

        let (even_first, odd_first) = first.deinterleave(second);
        let (even_second, odd_second) = third.deinterleave(fourth);
        let (first_sample, third_sample) = even_first.deinterleave(even_second);
        let (second_sample, fourth_sample) = odd_first.deinterleave(odd_second);

        let first_low = (first_sample + second_sample) * analysis.first_low_scale;
        let second_low = (third_sample + fourth_sample) * analysis.first_low_scale;
        let first_high = (first_sample - second_sample) * analysis.first_high_scale;
        let second_high = (third_sample - fourth_sample) * analysis.first_high_scale;
        let (first_high, second_high) = first_high.interleave(second_high);
        let detail_output = 2 * output;
        first_high.store_slice(&mut first_detail[detail_output..detail_output + lanes]);
        second_high
            .store_slice(&mut first_detail[detail_output + lanes..detail_output + 2 * lanes]);

        ((first_low + second_low) * analysis.second_low_scale)
            .store_slice(&mut approx[output..output + lanes]);
        ((first_low - second_low) * analysis.second_high_scale)
            .store_slice(&mut second_detail[output..output + lanes]);
    }

    vectorized_outputs
}

#[cfg(feature = "experimental-kernels")]
pub(crate) const MIN_LATTICE_OUTPUTS: usize = 512;
#[cfg(feature = "experimental-kernels")]
const LATTICE_TILE: usize = 8;
#[cfg(feature = "experimental-kernels")]
const MAX_LATTICE_SECTIONS: usize = 51;
#[cfg(feature = "experimental-kernels")]
const MAX_LATTICE_LANES: usize = 8;

#[cfg(feature = "experimental-kernels")]
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

#[cfg(feature = "experimental-kernels")]
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

#[cfg(feature = "experimental-kernels")]
#[inline(always)]
fn apply_lattice_0_positive<S: Simd, T: SimdSample<S>>(
    _simd: S,
    q: T::Vector,
    first: T::Vector,
    second: T::Vector,
) -> (T::Vector, T::Vector) {
    ((-second).mul_add(q, first), first.mul_add(q, second))
}

#[cfg(feature = "experimental-kernels")]
#[inline(always)]
fn apply_lattice_0_negative<S: Simd, T: SimdSample<S>>(
    _simd: S,
    q: T::Vector,
    first: T::Vector,
    second: T::Vector,
) -> (T::Vector, T::Vector) {
    (second.mul_add(q, first), first.mul_add(q, -second))
}

#[cfg(feature = "experimental-kernels")]
#[inline(always)]
fn apply_lattice_1_positive<S: Simd, T: SimdSample<S>>(
    _simd: S,
    q: T::Vector,
    first: T::Vector,
    second: T::Vector,
) -> (T::Vector, T::Vector) {
    (first.mul_add(q, -second), second.mul_add(q, first))
}

#[cfg(feature = "experimental-kernels")]
#[inline(always)]
fn apply_lattice_1_negative<S: Simd, T: SimdSample<S>>(
    _simd: S,
    q: T::Vector,
    first: T::Vector,
    second: T::Vector,
) -> (T::Vector, T::Vector) {
    (first.mul_add(q, second), (-second).mul_add(q, first))
}

#[cfg(feature = "experimental-kernels")]
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

#[cfg(feature = "experimental-kernels")]
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

#[cfg(feature = "experimental-kernels")]
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

#[cfg(feature = "experimental-kernels")]
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

#[cfg(feature = "experimental-kernels")]
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

#[cfg(feature = "experimental-kernels")]
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

#[inline(always)]
fn forward_interior_batches<S: Simd, T: SimdSample<S>, const BATCHES: usize>(
    simd: S,
    interior: &AnalysisInterior<'_, T>,
    approx: &mut [T],
    detail: &mut [T],
) -> usize {
    let lanes = T::Vector::N;
    let vectorized_outputs = approx.len() - approx.len() % lanes;
    let batch_width = BATCHES * lanes;
    let batched_outputs = vectorized_outputs - vectorized_outputs % batch_width;

    for output in (0..batched_outputs).step_by(batch_width) {
        forward_interior_batch::<_, _, BATCHES>(simd, interior, approx, detail, output);
    }
    if batched_outputs < vectorized_outputs {
        forward_interior_batch::<_, _, 1>(simd, interior, approx, detail, batched_outputs);
    }

    vectorized_outputs
}

#[inline(always)]
fn forward_interior_batch<S: Simd, T: SimdSample<S>, const BATCHES: usize>(
    simd: S,
    interior: &AnalysisInterior<'_, T>,
    approx: &mut [T],
    detail: &mut [T],
    first_output: usize,
) {
    let dec_lo = interior.dec_lo;
    let dec_hi = interior.dec_hi;
    let signal = interior.signal;
    let lanes = T::Vector::N;
    let (low_pairs, low_remainder) = dec_lo.as_chunks::<2>();
    let (high_pairs, high_remainder) = dec_hi.as_chunks::<2>();
    debug_assert!(low_remainder.is_empty());
    debug_assert!(high_remainder.is_empty());
    let (first_low, low_pairs) = low_pairs
        .split_first()
        .expect("wavelet filters contain at least one tap pair");
    let (first_high, high_pairs) = high_pairs
        .split_first()
        .expect("wavelet filters contain at least one tap pair");
    let mut input_windows: [_; BATCHES] = std::array::from_fn(|batch| {
        let newest = interior.first_newest + 2 * (first_output + batch * lanes);
        let batch_start = newest + 1 - dec_lo.len();
        let batch_end = newest + 2 * lanes - 1;
        signal[batch_start..batch_end]
            .windows(2 * lanes)
            .rev()
            .step_by(2)
    });
    let mut accumulators: [AnalysisAccumulators<T::Vector>; BATCHES] =
        std::array::from_fn(|batch| {
            let input = input_windows[batch]
                .next()
                .expect("filter and input windows have equal lengths");
            let (first, second) = input.split_at(lanes);
            let first = T::Vector::from_slice(simd, first);
            let second = T::Vector::from_slice(simd, second);
            let (earlier, later) = first.deinterleave(second);
            AnalysisAccumulators {
                low_earlier: earlier * first_low[1],
                low_later: later * first_low[0],
                high_earlier: earlier * first_high[1],
                high_later: later * first_high[0],
            }
        });

    for (low, high) in low_pairs.iter().zip(high_pairs) {
        for (input_windows, accumulators) in input_windows.iter_mut().zip(&mut accumulators) {
            let input = input_windows
                .next()
                .expect("filter and input windows have equal lengths");
            let (first, second) = input.split_at(lanes);
            let first = T::Vector::from_slice(simd, first);
            let second = T::Vector::from_slice(simd, second);
            let (earlier, later) = first.deinterleave(second);

            accumulators.low_earlier = earlier.mul_add(low[1], accumulators.low_earlier);
            accumulators.low_later = later.mul_add(low[0], accumulators.low_later);
            accumulators.high_earlier = earlier.mul_add(high[1], accumulators.high_earlier);
            accumulators.high_later = later.mul_add(high[0], accumulators.high_later);
        }
    }

    for (batch, accumulators) in accumulators.into_iter().enumerate() {
        let output = first_output + batch * lanes;
        (accumulators.low_earlier + accumulators.low_later)
            .store_slice(&mut approx[output..output + lanes]);
        (accumulators.high_earlier + accumulators.high_later)
            .store_slice(&mut detail[output..output + lanes]);
    }
}

#[inline(always)]
pub(crate) fn inverse_periodized<S: Simd, T: SimdSample<S>>(
    simd: S,
    interior: PeriodizedInterior<'_, T>,
    out: &mut [T],
) -> usize {
    match interior.second_offset {
        0 => inverse_periodized_offset::<_, _, 0>(simd, interior, out),
        1 => inverse_periodized_offset::<_, _, 1>(simd, interior, out),
        _ => unreachable!("periodized synthesis phases are at most one coefficient apart"),
    }
}

#[inline(always)]
pub(crate) fn inverse_butterfly<S: Simd, T: SimdSample<S>>(
    simd: S,
    synthesis: ButterflySynthesis<'_, T>,
    out: &mut [T],
) -> usize {
    let lanes = T::Vector::N;
    let pair_count = out.len() / 2;
    let vectorized_pairs = pair_count - pair_count % lanes;

    for pair in (0..vectorized_pairs).step_by(lanes) {
        let low = T::Vector::from_slice(simd, &synthesis.approx[pair..pair + lanes])
            * synthesis.low_scale;
        let high = T::Vector::from_slice(simd, &synthesis.detail[pair..pair + lanes])
            * synthesis.high_scale;
        let (first, second) = (low + high).interleave(low - high);
        let output = 2 * pair;
        first.store_slice(&mut out[output..output + lanes]);
        second.store_slice(&mut out[output + lanes..output + 2 * lanes]);
    }

    vectorized_pairs
}

#[inline(always)]
pub(crate) fn inverse_butterfly_pair<S: Simd, T: SimdSample<S>>(
    simd: S,
    synthesis: ButterflyPairSynthesis<'_, T>,
    out: &mut [T],
) -> usize {
    let lanes = T::Vector::N;
    let vectorized_inputs = synthesis.approx.len() - synthesis.approx.len() % lanes;

    for input in (0..vectorized_inputs).step_by(lanes) {
        let second_low = T::Vector::from_slice(simd, &synthesis.approx[input..input + lanes])
            * synthesis.second_low_scale;
        let second_high =
            T::Vector::from_slice(simd, &synthesis.second_detail[input..input + lanes])
                * synthesis.second_high_scale;
        let first_approx = second_low + second_high;
        let second_approx = second_low - second_high;

        let detail_input = 2 * input;
        let first_detail = T::Vector::from_slice(
            simd,
            &synthesis.first_detail[detail_input..detail_input + lanes],
        );
        let second_detail = T::Vector::from_slice(
            simd,
            &synthesis.first_detail[detail_input + lanes..detail_input + 2 * lanes],
        );
        let (first_detail, second_detail) = first_detail.deinterleave(second_detail);

        let first_low = first_approx * synthesis.first_low_scale;
        let first_high = first_detail * synthesis.first_high_scale;
        let second_low = second_approx * synthesis.first_low_scale;
        let second_high = second_detail * synthesis.first_high_scale;
        let first_sample = first_low + first_high;
        let second_sample = first_low - first_high;
        let third_sample = second_low + second_high;
        let fourth_sample = second_low - second_high;

        let (even_first, even_second) = first_sample.interleave(third_sample);
        let (odd_first, odd_second) = second_sample.interleave(fourth_sample);
        let (first, second) = even_first.interleave(odd_first);
        let (third, fourth) = even_second.interleave(odd_second);
        let output = 4 * input;
        first.store_slice(&mut out[output..output + lanes]);
        second.store_slice(&mut out[output + lanes..output + 2 * lanes]);
        third.store_slice(&mut out[output + 2 * lanes..output + 3 * lanes]);
        fourth.store_slice(&mut out[output + 3 * lanes..output + 4 * lanes]);
    }

    vectorized_inputs
}

#[inline(always)]
fn inverse_periodized_offset<S: Simd, T: SimdSample<S>, const OFFSET: usize>(
    simd: S,
    interior: PeriodizedInterior<'_, T>,
    out: &mut [T],
) -> usize {
    let lanes = T::Vector::N;
    let pair_count = out.len() / 2;
    let vectorized_pairs = pair_count - pair_count % lanes;
    let paired_batch = 2 * lanes;
    let paired_pairs = pair_count - pair_count % paired_batch;

    for pair in (0..paired_pairs).step_by(paired_batch) {
        inverse_periodized_batch::<_, _, 2, OFFSET>(
            simd,
            &interior,
            out,
            interior.first_coefficient + pair,
        );
    }
    if paired_pairs < vectorized_pairs {
        inverse_periodized_batch::<_, _, 1, OFFSET>(
            simd,
            &interior,
            out,
            interior.first_coefficient + paired_pairs,
        );
    }

    vectorized_pairs
}

#[inline(always)]
fn inverse_periodized_batch<
    S: Simd,
    T: SimdSample<S>,
    const BATCHES: usize,
    const OFFSET: usize,
>(
    simd: S,
    interior: &PeriodizedInterior<'_, T>,
    out: &mut [T],
    first_coefficient: usize,
) {
    let first_lo = interior.first_lo;
    let first_hi = interior.first_hi;
    let second_lo = interior.second_lo;
    let second_hi = interior.second_hi;
    let approx = interior.approx;
    let detail = interior.detail;
    let lanes = T::Vector::N;
    let filter_len = first_lo.len();
    let first_start = first_coefficient + 1 - filter_len;
    let input_len = filter_len + BATCHES * lanes - 1 + OFFSET;
    let approx = &approx[first_start..first_start + input_len];
    let detail = &detail[first_start..first_start + input_len];
    let (first_first_lo, first_lo) = first_lo
        .split_first()
        .expect("wavelet filters contain at least one polyphase tap");
    let (first_first_hi, first_hi) = first_hi
        .split_first()
        .expect("wavelet filters contain at least one polyphase tap");
    let (first_second_lo, second_lo) = second_lo
        .split_first()
        .expect("wavelet filters contain at least one polyphase tap");
    let (first_second_hi, second_hi) = second_hi
        .split_first()
        .expect("wavelet filters contain at least one polyphase tap");
    let mut first_low: [T::Vector; BATCHES] =
        std::array::from_fn(|_| T::Vector::splat(simd, T::default()));
    let mut first_high: [T::Vector; BATCHES] =
        std::array::from_fn(|_| T::Vector::splat(simd, T::default()));
    let mut second_low: [T::Vector; BATCHES] =
        std::array::from_fn(|_| T::Vector::splat(simd, T::default()));
    let mut second_high: [T::Vector; BATCHES] =
        std::array::from_fn(|_| T::Vector::splat(simd, T::default()));

    let first_start = filter_len - 1;
    let first_approximations: [T::Vector; BATCHES] = std::array::from_fn(|batch| {
        let start = first_start + batch * lanes;
        T::Vector::from_slice(simd, &approx[start..start + lanes])
    });
    let first_details: [T::Vector; BATCHES] = std::array::from_fn(|batch| {
        let start = first_start + batch * lanes;
        T::Vector::from_slice(simd, &detail[start..start + lanes])
    });
    for batch in 0..BATCHES {
        let first_approximation = first_approximations[batch];
        let first_detail = first_details[batch];
        let second_approximation = if OFFSET == 0 {
            first_approximation
        } else if batch + 1 < BATCHES {
            first_approximation.slide::<1>(first_approximations[batch + 1])
        } else {
            let start = first_start + batch * lanes;
            T::Vector::from_slice(simd, &approx[start + OFFSET..start + OFFSET + lanes])
        };
        let second_detail = if OFFSET == 0 {
            first_detail
        } else if batch + 1 < BATCHES {
            first_detail.slide::<1>(first_details[batch + 1])
        } else {
            let start = first_start + batch * lanes;
            T::Vector::from_slice(simd, &detail[start + OFFSET..start + OFFSET + lanes])
        };
        first_low[batch] = first_approximation * *first_first_lo;
        first_high[batch] = first_detail * *first_first_hi;
        second_low[batch] = second_approximation * *first_second_lo;
        second_high[batch] = second_detail * *first_second_hi;
    }

    for (tap, (((first_lo, first_hi), second_lo), second_hi)) in first_lo
        .iter()
        .zip(first_hi)
        .zip(second_lo)
        .zip(second_hi)
        .enumerate()
    {
        let window_start = filter_len - 2 - tap;
        let first_approximations: [T::Vector; BATCHES] = std::array::from_fn(|batch| {
            let start = window_start + batch * lanes;
            T::Vector::from_slice(simd, &approx[start..start + lanes])
        });
        let first_details: [T::Vector; BATCHES] = std::array::from_fn(|batch| {
            let start = window_start + batch * lanes;
            T::Vector::from_slice(simd, &detail[start..start + lanes])
        });
        for batch in 0..BATCHES {
            let first_approximation = first_approximations[batch];
            let first_detail = first_details[batch];
            let second_approximation = if OFFSET == 0 {
                first_approximation
            } else if batch + 1 < BATCHES {
                first_approximation.slide::<1>(first_approximations[batch + 1])
            } else {
                let start = window_start + batch * lanes;
                T::Vector::from_slice(simd, &approx[start + OFFSET..start + OFFSET + lanes])
            };
            let second_detail = if OFFSET == 0 {
                first_detail
            } else if batch + 1 < BATCHES {
                first_detail.slide::<1>(first_details[batch + 1])
            } else {
                let start = window_start + batch * lanes;
                T::Vector::from_slice(simd, &detail[start + OFFSET..start + OFFSET + lanes])
            };
            first_low[batch] = first_approximation.mul_add(*first_lo, first_low[batch]);
            first_high[batch] = first_detail.mul_add(*first_hi, first_high[batch]);
            second_low[batch] = second_approximation.mul_add(*second_lo, second_low[batch]);
            second_high[batch] = second_detail.mul_add(*second_hi, second_high[batch]);
        }
    }

    let first_pair = first_coefficient - (filter_len - 1);
    for batch in 0..BATCHES {
        let first = first_low[batch] + first_high[batch];
        let second = second_low[batch] + second_high[batch];
        let (first, second) = first.interleave(second);
        let output = 2 * (first_pair + batch * lanes);
        first.store_slice(&mut out[output..output + lanes]);
        second.store_slice(&mut out[output + lanes..output + 2 * lanes]);
    }
}

#[inline(always)]
pub(crate) fn inverse_linear<S: Simd, T: SimdSample<S>>(
    simd: S,
    synthesis: LinearSynthesis<'_, T>,
    out: &mut [T],
) -> usize {
    let LinearSynthesis {
        rec_lo,
        rec_hi,
        approx,
        detail,
    } = synthesis;
    let half_filter_len = rec_lo.len() / 2;
    let (even_lo, odd_lo) = rec_lo.split_at(half_filter_len);
    let (even_hi, odd_hi) = rec_hi.split_at(half_filter_len);
    let lanes = T::Vector::N;
    let pair_count = out.len() / 2;
    let vectorized_pairs = pair_count - pair_count % lanes;
    let paired_batch = 2 * lanes;
    let paired_pairs = pair_count - pair_count % paired_batch;

    for pair in (0..paired_pairs).step_by(paired_batch) {
        inverse_linear_batch::<_, _, 2>(
            simd,
            (even_lo, even_hi),
            (odd_lo, odd_hi),
            approx,
            detail,
            out,
            pair,
        );
    }
    if paired_pairs < vectorized_pairs {
        inverse_linear_batch::<_, _, 1>(
            simd,
            (even_lo, even_hi),
            (odd_lo, odd_hi),
            approx,
            detail,
            out,
            paired_pairs,
        );
    }

    vectorized_pairs
}

#[inline(always)]
fn inverse_linear_batch<S: Simd, T: SimdSample<S>, const BATCHES: usize>(
    simd: S,
    (even_lo, even_hi): (&[T], &[T]),
    (odd_lo, odd_hi): (&[T], &[T]),
    approx: &[T],
    detail: &[T],
    out: &mut [T],
    first_pair: usize,
) {
    let lanes = T::Vector::N;
    let half_filter_len = even_lo.len();
    let input_len = half_filter_len + BATCHES * lanes - 1;
    let approx = &approx[first_pair..first_pair + input_len];
    let detail = &detail[first_pair..first_pair + input_len];
    let (first_even_lo, even_lo) = even_lo
        .split_first()
        .expect("wavelet filters contain at least one polyphase tap");
    let (first_even_hi, even_hi) = even_hi
        .split_first()
        .expect("wavelet filters contain at least one polyphase tap");
    let (first_odd_lo, odd_lo) = odd_lo
        .split_first()
        .expect("wavelet filters contain at least one polyphase tap");
    let (first_odd_hi, odd_hi) = odd_hi
        .split_first()
        .expect("wavelet filters contain at least one polyphase tap");
    let mut even_low: [T::Vector; BATCHES] =
        std::array::from_fn(|_| T::Vector::splat(simd, T::default()));
    let mut even_high: [T::Vector; BATCHES] =
        std::array::from_fn(|_| T::Vector::splat(simd, T::default()));
    let mut odd_low: [T::Vector; BATCHES] =
        std::array::from_fn(|_| T::Vector::splat(simd, T::default()));
    let mut odd_high: [T::Vector; BATCHES] =
        std::array::from_fn(|_| T::Vector::splat(simd, T::default()));

    for batch in 0..BATCHES {
        let start = half_filter_len - 1 + batch * lanes;
        let approximation = T::Vector::from_slice(simd, &approx[start..start + lanes]);
        let detail = T::Vector::from_slice(simd, &detail[start..start + lanes]);
        even_low[batch] = approximation * *first_even_lo;
        even_high[batch] = detail * *first_even_hi;
        odd_low[batch] = approximation * *first_odd_lo;
        odd_high[batch] = detail * *first_odd_hi;
    }

    for (tap, (((even_lo, even_hi), odd_lo), odd_hi)) in even_lo
        .iter()
        .zip(even_hi)
        .zip(odd_lo)
        .zip(odd_hi)
        .enumerate()
    {
        let window_start = half_filter_len - 2 - tap;
        for batch in 0..BATCHES {
            let start = window_start + batch * lanes;
            let approximation = T::Vector::from_slice(simd, &approx[start..start + lanes]);
            let detail = T::Vector::from_slice(simd, &detail[start..start + lanes]);
            even_low[batch] = approximation.mul_add(*even_lo, even_low[batch]);
            even_high[batch] = detail.mul_add(*even_hi, even_high[batch]);
            odd_low[batch] = approximation.mul_add(*odd_lo, odd_low[batch]);
            odd_high[batch] = detail.mul_add(*odd_hi, odd_high[batch]);
        }
    }

    for batch in 0..BATCHES {
        let even = even_low[batch] + even_high[batch];
        let odd = odd_low[batch] + odd_high[batch];
        let (first, second) = even.interleave(odd);
        let output = 2 * (first_pair + batch * lanes);
        first.store_slice(&mut out[output..output + lanes]);
        second.store_slice(&mut out[output + lanes..output + 2 * lanes]);
    }
}

#[cfg(test)]
mod tests {
    use fearless_simd::{Level, dispatch};

    use super::{
        AnalysisInterior, ButterflyAnalysis, ButterflyPairAnalysis, ButterflyPairSynthesis,
        ButterflySynthesis, LinearSynthesis, PeriodizedInterior, forward_butterfly,
        forward_butterfly_pair, forward_interior, inverse_butterfly, inverse_butterfly_pair,
        inverse_linear, inverse_periodized,
    };

    macro_rules! kernel_test {
        ($name:ident, $sample:ty, $tolerance:expr) => {
            #[test]
            fn $name() {
                let dec_lo: [$sample; 8] =
                    [0.17, -0.31, 0.53, 0.79, -0.11, 0.23, 0.41, -0.67];
                let dec_hi: [$sample; 8] =
                    [-0.37, 0.19, 0.73, -0.29, 0.61, -0.43, 0.13, 0.47];
                let signal: Vec<$sample> = (0..128)
                    .map(|index| index as $sample * 0.13 - 1.7)
                    .collect();
                let mut approx = vec![-12_345.0; 41];
                let mut detail = vec![-12_345.0; 41];

                let forward_outputs = dispatch!(Level::new(), simd => forward_interior(
                    simd,
                    AnalysisInterior {
                        dec_lo: &dec_lo,
                        dec_hi: &dec_hi,
                        signal: &signal,
                        first_newest: 7,
                    },
                    &mut approx,
                    &mut detail
                ));

                assert!(forward_outputs > 0);
                assert!(forward_outputs < approx.len());
                for output in 0..forward_outputs {
                    let newest = 7 + 2 * output;
                    let mut low_earlier: $sample = 0.0;
                    let mut low_later: $sample = 0.0;
                    let mut high_earlier: $sample = 0.0;
                    let mut high_later: $sample = 0.0;
                    for tap in (0..dec_lo.len()).step_by(2) {
                        low_earlier =
                            signal[newest - tap - 1].mul_add(dec_lo[tap + 1], low_earlier);
                        low_later = signal[newest - tap].mul_add(dec_lo[tap], low_later);
                        high_earlier =
                            signal[newest - tap - 1].mul_add(dec_hi[tap + 1], high_earlier);
                        high_later = signal[newest - tap].mul_add(dec_hi[tap], high_later);
                    }
                    assert!((approx[output] - (low_earlier + low_later)).abs() <= $tolerance);
                    assert!((detail[output] - (high_earlier + high_later)).abs() <= $tolerance);
                }
                assert!(approx[forward_outputs..]
                    .iter()
                    .all(|&sample| sample == -12_345.0));
                assert!(detail[forward_outputs..]
                    .iter()
                    .all(|&sample| sample == -12_345.0));

                let mut butterfly_approx = vec![-12_345.0; 41];
                let mut butterfly_detail = vec![-12_345.0; 41];
                let butterfly_outputs = dispatch!(Level::new(), simd => forward_butterfly(
                    simd,
                    ButterflyAnalysis {
                        signal: &signal,
                        first_newest: 1,
                        low_scale: 0.5,
                        high_scale: 0.25,
                    },
                    &mut butterfly_approx,
                    &mut butterfly_detail,
                ));
                assert!(butterfly_outputs > 0);
                assert!(butterfly_outputs < butterfly_approx.len());
                for output in 0..butterfly_outputs {
                    let earlier = signal[2 * output];
                    let later = signal[2 * output + 1];
                    assert_eq!(butterfly_approx[output], later * 0.5 + earlier * 0.5);
                    assert_eq!(butterfly_detail[output], later * -0.25 + earlier * 0.25);
                }
                assert!(butterfly_approx[butterfly_outputs..]
                    .iter()
                    .all(|&sample| sample == -12_345.0));
                assert!(butterfly_detail[butterfly_outputs..]
                    .iter()
                    .all(|&sample| sample == -12_345.0));

                let mut pair_approx = vec![-12_345.0; 23];
                let mut first_pair_detail = vec![-12_345.0; 46];
                let mut second_pair_detail = vec![-12_345.0; 23];
                let pair_outputs = dispatch!(Level::new(), simd => forward_butterfly_pair(
                    simd,
                    ButterflyPairAnalysis {
                        signal: &signal,
                        first_low_scale: 0.5,
                        first_high_scale: 0.25,
                        second_low_scale: 0.75,
                        second_high_scale: 0.125,
                    },
                    &mut pair_approx,
                    &mut first_pair_detail,
                    &mut second_pair_detail,
                ));
                assert!(pair_outputs > 0);
                assert!(pair_outputs < pair_approx.len());
                for output in 0..pair_outputs {
                    let input = 4 * output;
                    let first_low = (signal[input] + signal[input + 1]) * 0.5;
                    let second_low = (signal[input + 2] + signal[input + 3]) * 0.5;
                    assert_eq!(pair_approx[output], (first_low + second_low) * 0.75);
                    assert_eq!(
                        first_pair_detail[2 * output],
                        (signal[input] - signal[input + 1]) * 0.25
                    );
                    assert_eq!(
                        first_pair_detail[2 * output + 1],
                        (signal[input + 2] - signal[input + 3]) * 0.25
                    );
                    assert_eq!(
                        second_pair_detail[output],
                        (first_low - second_low) * 0.125
                    );
                }
                assert!(pair_approx[pair_outputs..]
                    .iter()
                    .all(|&sample| sample == -12_345.0));
                assert!(first_pair_detail[2 * pair_outputs..]
                    .iter()
                    .all(|&sample| sample == -12_345.0));
                assert!(second_pair_detail[pair_outputs..]
                    .iter()
                    .all(|&sample| sample == -12_345.0));

                let rec_lo = dec_lo;
                let rec_hi = dec_hi;
                let coefficients: Vec<$sample> = (0..44)
                    .map(|index| index as $sample * -0.07 + 0.9)
                    .collect();
                let mut out = vec![-12_345.0; 83];
                let inverse_pairs = dispatch!(Level::new(), simd => inverse_linear(
                    simd,
                    LinearSynthesis {
                        rec_lo: &rec_lo,
                        rec_hi: &rec_hi,
                        approx: &coefficients,
                        detail: &signal[..44],
                    },
                    &mut out
                ));

                assert!(inverse_pairs > 0);
                assert!(inverse_pairs < out.len() / 2);
                let half = rec_lo.len() / 2;
                let (even_lo, odd_lo) = rec_lo.split_at(half);
                let (even_hi, odd_hi) = rec_hi.split_at(half);
                for pair in 0..inverse_pairs {
                    let mut even_low: $sample = 0.0;
                    let mut even_high: $sample = 0.0;
                    let mut odd_low: $sample = 0.0;
                    let mut odd_high: $sample = 0.0;
                    for tap in 0..half {
                        let coefficient = pair + half - 1 - tap;
                        even_low = coefficients[coefficient].mul_add(even_lo[tap], even_low);
                        even_high = signal[coefficient].mul_add(even_hi[tap], even_high);
                        odd_low = coefficients[coefficient].mul_add(odd_lo[tap], odd_low);
                        odd_high = signal[coefficient].mul_add(odd_hi[tap], odd_high);
                    }
                    assert!((out[2 * pair] - (even_low + even_high)).abs() <= $tolerance);
                    assert!((out[2 * pair + 1] - (odd_low + odd_high)).abs() <= $tolerance);
                }
                assert!(out[2 * inverse_pairs..]
                    .iter()
                    .all(|&sample| sample == -12_345.0));

                let mut butterfly_out = vec![-12_345.0; 83];
                let butterfly_pairs = dispatch!(Level::new(), simd => inverse_butterfly(
                    simd,
                    ButterflySynthesis {
                        approx: &coefficients,
                        detail: &signal[..44],
                        low_scale: 0.5,
                        high_scale: 0.25,
                    },
                    &mut butterfly_out,
                ));
                assert!(butterfly_pairs > 0);
                assert!(butterfly_pairs < butterfly_out.len() / 2);
                for pair in 0..butterfly_pairs {
                    let low = coefficients[pair] * 0.5;
                    let high = signal[pair] * 0.25;
                    assert_eq!(butterfly_out[2 * pair], low + high);
                    assert_eq!(butterfly_out[2 * pair + 1], low - high);
                }
                assert!(butterfly_out[2 * butterfly_pairs..]
                    .iter()
                    .all(|&sample| sample == -12_345.0));

                let mut pair_out = vec![-12_345.0; 4 * pair_approx.len()];
                let inverse_pair_outputs = dispatch!(Level::new(), simd => inverse_butterfly_pair(
                    simd,
                    ButterflyPairSynthesis {
                        approx: &pair_approx,
                        first_detail: &first_pair_detail,
                        second_detail: &second_pair_detail,
                        first_low_scale: 0.75,
                        first_high_scale: 0.125,
                        second_low_scale: 0.5,
                        second_high_scale: 0.25,
                    },
                    &mut pair_out,
                ));
                assert_eq!(inverse_pair_outputs, pair_outputs);
                for input in 0..inverse_pair_outputs {
                    let second_low = pair_approx[input] * 0.5;
                    let second_high = second_pair_detail[input] * 0.25;
                    let first_approx = second_low + second_high;
                    let second_approx = second_low - second_high;
                    let first_low = first_approx * 0.75;
                    let first_high = first_pair_detail[2 * input] * 0.125;
                    let second_low = second_approx * 0.75;
                    let second_high = first_pair_detail[2 * input + 1] * 0.125;
                    let output = 4 * input;
                    assert_eq!(pair_out[output], first_low + first_high);
                    assert_eq!(pair_out[output + 1], first_low - first_high);
                    assert_eq!(pair_out[output + 2], second_low + second_high);
                    assert_eq!(pair_out[output + 3], second_low - second_high);
                }
                assert!(pair_out[4 * inverse_pair_outputs..]
                    .iter()
                    .all(|&sample| sample == -12_345.0));

                let first_lo: [$sample; 4] = [0.13, -0.29, 0.47, 0.71];
                let first_hi: [$sample; 4] = [-0.61, 0.43, 0.17, -0.31];
                let second_lo: [$sample; 4] = [0.23, 0.67, -0.37, 0.11];
                let second_hi: [$sample; 4] = [0.53, -0.19, 0.41, -0.73];
                let approx: Vec<$sample> = (0..64)
                    .map(|index| index as $sample * 0.09 - 0.8)
                    .collect();
                let detail: Vec<$sample> = (0..64)
                    .map(|index| index as $sample * -0.04 + 1.3)
                    .collect();

                for second_offset in 0..=1 {
                    // Exercises at least one vector on AVX-512 f32 (the widest
                    // supported backend) while retaining an untouched tail.
                    let mut out = vec![-12_345.0; 83];
                    let inverse_pairs = dispatch!(Level::new(), simd => inverse_periodized(
                        simd,
                        PeriodizedInterior {
                            first_lo: &first_lo,
                            first_hi: &first_hi,
                            second_lo: &second_lo,
                            second_hi: &second_hi,
                            approx: &approx,
                            detail: &detail,
                            first_coefficient: first_lo.len() - 1,
                            second_offset,
                        },
                        &mut out
                    ));

                    assert!(inverse_pairs > 0);
                    assert!(inverse_pairs <= out.len() / 2);
                    for pair in 0..inverse_pairs {
                        let newest = first_lo.len() - 1 + pair;
                        let mut first_low: $sample = 0.0;
                        let mut first_high: $sample = 0.0;
                        let mut second_low: $sample = 0.0;
                        let mut second_high: $sample = 0.0;
                        for tap in 0..first_lo.len() {
                            let first_coefficient = newest - tap;
                            let second_coefficient = first_coefficient + second_offset;
                            first_low = approx[first_coefficient]
                                .mul_add(first_lo[tap], first_low);
                            first_high = detail[first_coefficient]
                                .mul_add(first_hi[tap], first_high);
                            second_low = approx[second_coefficient]
                                .mul_add(second_lo[tap], second_low);
                            second_high = detail[second_coefficient]
                                .mul_add(second_hi[tap], second_high);
                        }
                        assert!((out[2 * pair] - (first_low + first_high)).abs() <= $tolerance);
                        assert!(
                            (out[2 * pair + 1] - (second_low + second_high)).abs() <= $tolerance
                        );
                    }
                    assert!(out[2 * inverse_pairs..]
                        .iter()
                        .all(|&sample| sample == -12_345.0));
                }
            }
        };
    }

    kernel_test!(f32_kernels_match_scalar_and_leave_tails, f32, 8.0e-6);
    kernel_test!(f64_kernels_match_scalar_and_leave_tails, f64, 2.0e-14);
}
