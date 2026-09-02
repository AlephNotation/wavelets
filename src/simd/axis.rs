use std::mem::size_of;

use fearless_simd::{Simd, prelude::*};

use super::SimdSample;
use crate::plan::EdgeTerm;

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
