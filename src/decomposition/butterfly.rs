use std::ops::Range;

use fearless_simd::Level as SimdLevel;

use super::layout::Decomposition;
use super::plan::WavedecPlan;
use crate::num::{forward_butterfly_pair_simd, inverse_butterfly_pair_simd};
use crate::plan::PlannedDwt;
use crate::simd::{ButterflyPairAnalysis, ButterflyPairSynthesis};
use crate::{Dwt, WaveletNum};

#[derive(Clone, Copy)]
pub(super) struct ButterflyAnalysisCascade<T> {
    simd_level: SimdLevel,
    low_scale: T,
    high_scale: T,
}

#[derive(Clone, Copy)]
pub(super) struct ButterflySynthesisCascade<T> {
    simd_level: SimdLevel,
    low_scale: T,
    high_scale: T,
}

// Four-way lane rearrangement has a fixed cost. Sixty-four fused outputs are
// enough to amortize it on NEON; this remains conservative for wider vectors.
const MIN_FUSED_ANALYSIS_OUTPUTS: usize = 64;

pub(super) fn select_analysis<T: WaveletNum>(
    level_plans: &[PlannedDwt<T>],
    simd_level: SimdLevel,
) -> Option<ButterflyAnalysisCascade<T>> {
    if level_plans.len() < 2
        || !level_plans.len().is_multiple_of(2)
        || level_plans[1].coeff_len() < MIN_FUSED_ANALYSIS_OUTPUTS
    {
        return None;
    }
    let (low_scale, high_scale) = level_plans.first()?.full_butterfly_analysis()?;
    level_plans
        .iter()
        .all(|plan| plan.full_butterfly_analysis().is_some())
        .then_some(ButterflyAnalysisCascade {
            simd_level,
            low_scale,
            high_scale,
        })
}

pub(super) fn select_synthesis<T: WaveletNum>(
    level_plans: &[PlannedDwt<T>],
    simd_level: SimdLevel,
) -> Option<ButterflySynthesisCascade<T>> {
    if level_plans.len() < 2 || !level_plans.len().is_multiple_of(2) {
        return None;
    }
    let (low_scale, high_scale) = level_plans.first()?.full_butterfly_synthesis()?;
    level_plans
        .iter()
        .all(|plan| plan.full_butterfly_synthesis().is_some())
        .then_some(ButterflySynthesisCascade {
            simd_level,
            low_scale,
            high_scale,
        })
}

pub(super) fn forward<T: WaveletNum>(
    plan: &WavedecPlan<T>,
    cascade: ButterflyAnalysisCascade<T>,
    signal: &[T],
    decomposition: &mut Decomposition<T>,
    scratch: &mut [T],
) {
    let scratch = &mut scratch[..plan.scratch_len()];
    let (temp_a, scratch) = scratch.split_at_mut(plan.temp_a_len);
    let (temp_b, _) = scratch.split_at_mut(plan.temp_b_len);
    let pair_count = plan.levels() / 2;

    for pair in 0..pair_count {
        let first_level = 2 * pair;
        let first_detail_range = plan.layout.details[first_level].clone();
        let second_detail_range = plan.layout.details[first_level + 1].clone();
        let final_pair = pair + 1 == pair_count;

        if final_pair {
            let approx_end = plan.layout.approx.end;
            let (approx, details) = decomposition.buffer.split_at_mut(approx_end);
            let (first_detail, second_detail) =
                detail_pair_mut(details, approx_end, first_detail_range, second_detail_range);
            match pair {
                0 => forward_pair(cascade, signal, approx, first_detail, second_detail),
                pair if pair % 2 == 1 => forward_pair(
                    cascade,
                    &temp_a[..4 * approx.len()],
                    approx,
                    first_detail,
                    second_detail,
                ),
                _ => forward_pair(
                    cascade,
                    &temp_b[..4 * approx.len()],
                    approx,
                    first_detail,
                    second_detail,
                ),
            }
        } else {
            let (first_detail, second_detail) = detail_pair_mut(
                &mut decomposition.buffer,
                0,
                first_detail_range,
                second_detail_range,
            );
            let output_len = second_detail.len();
            if pair == 0 {
                forward_pair(
                    cascade,
                    signal,
                    &mut temp_a[..output_len],
                    first_detail,
                    second_detail,
                );
            } else if pair % 2 == 1 {
                forward_pair(
                    cascade,
                    &temp_a[..4 * output_len],
                    &mut temp_b[..output_len],
                    first_detail,
                    second_detail,
                );
            } else {
                forward_pair(
                    cascade,
                    &temp_b[..4 * output_len],
                    &mut temp_a[..output_len],
                    first_detail,
                    second_detail,
                );
            }
        }
    }
}

pub(super) fn inverse<T: WaveletNum>(
    plan: &WavedecPlan<T>,
    cascade: ButterflySynthesisCascade<T>,
    decomposition: &Decomposition<T>,
    output: &mut [T],
    scratch: &mut [T],
) {
    let scratch = &mut scratch[..plan.scratch_len()];
    let (temp_a, scratch) = scratch.split_at_mut(plan.temp_a_len);
    let (temp_b, _) = scratch.split_at_mut(plan.temp_b_len);
    let pair_count = plan.levels() / 2;

    for pair in (0..pair_count).rev() {
        let first_level = 2 * pair;
        let first_detail = decomposition.detail(first_level + 1);
        let second_detail = decomposition.detail(first_level + 2);
        let output_len = 4 * second_detail.len();

        if pair == 0 {
            let approx = if pair + 1 == pair_count {
                decomposition.approx()
            } else if pair % 2 == 0 {
                &temp_a[..second_detail.len()]
            } else {
                &temp_b[..second_detail.len()]
            };
            inverse_pair(cascade, approx, first_detail, second_detail, output);
        } else {
            let coarsest_pair = pair + 1 == pair_count;
            if pair % 2 == 1 {
                let approx = if coarsest_pair {
                    decomposition.approx()
                } else {
                    &temp_b[..second_detail.len()]
                };
                inverse_pair(
                    cascade,
                    approx,
                    first_detail,
                    second_detail,
                    &mut temp_a[..output_len],
                );
            } else {
                let approx = if coarsest_pair {
                    decomposition.approx()
                } else {
                    &temp_a[..second_detail.len()]
                };
                inverse_pair(
                    cascade,
                    approx,
                    first_detail,
                    second_detail,
                    &mut temp_b[..output_len],
                );
            }
        }
    }
}

fn detail_pair_mut<T>(
    buffer: &mut [T],
    buffer_offset: usize,
    first_range: Range<usize>,
    second_range: Range<usize>,
) -> (&mut [T], &mut [T]) {
    debug_assert!(second_range.end <= first_range.start);
    let second_start = second_range.start - buffer_offset;
    let first_start = first_range.start - buffer_offset;
    let (_, from_second) = buffer.split_at_mut(second_start);
    let (second_detail, after_second) = from_second.split_at_mut(second_range.len());
    let first_start = first_start - second_start - second_range.len();
    let first_detail = &mut after_second[first_start..first_start + first_range.len()];
    (first_detail, second_detail)
}

fn forward_pair<T: WaveletNum>(
    cascade: ButterflyAnalysisCascade<T>,
    signal: &[T],
    approx: &mut [T],
    first_detail: &mut [T],
    second_detail: &mut [T],
) {
    debug_assert_eq!(signal.len(), 4 * approx.len());
    debug_assert_eq!(first_detail.len(), 2 * approx.len());
    debug_assert_eq!(second_detail.len(), approx.len());
    let vectorized = forward_butterfly_pair_simd(
        cascade.simd_level,
        ButterflyPairAnalysis {
            signal,
            first_low_scale: cascade.low_scale,
            first_high_scale: cascade.high_scale,
            second_low_scale: cascade.low_scale,
            second_high_scale: cascade.high_scale,
        },
        approx,
        first_detail,
        second_detail,
    );

    for output in vectorized..approx.len() {
        let input = 4 * output;
        let first_low = (signal[input] + signal[input + 1]) * cascade.low_scale;
        let second_low = (signal[input + 2] + signal[input + 3]) * cascade.low_scale;
        first_detail[2 * output] = (signal[input] - signal[input + 1]) * cascade.high_scale;
        first_detail[2 * output + 1] = (signal[input + 2] - signal[input + 3]) * cascade.high_scale;
        approx[output] = (first_low + second_low) * cascade.low_scale;
        second_detail[output] = (first_low - second_low) * cascade.high_scale;
    }
}

fn inverse_pair<T: WaveletNum>(
    cascade: ButterflySynthesisCascade<T>,
    approx: &[T],
    first_detail: &[T],
    second_detail: &[T],
    out: &mut [T],
) {
    debug_assert_eq!(out.len(), 4 * approx.len());
    debug_assert_eq!(first_detail.len(), 2 * approx.len());
    debug_assert_eq!(second_detail.len(), approx.len());
    let vectorized = inverse_butterfly_pair_simd(
        cascade.simd_level,
        ButterflyPairSynthesis {
            approx,
            first_detail,
            second_detail,
            first_low_scale: cascade.low_scale,
            first_high_scale: cascade.high_scale,
            second_low_scale: cascade.low_scale,
            second_high_scale: cascade.high_scale,
        },
        out,
    );

    for input in vectorized..approx.len() {
        let second_low = approx[input] * cascade.low_scale;
        let second_high = second_detail[input] * cascade.high_scale;
        let first_approx = second_low + second_high;
        let second_approx = second_low - second_high;
        let first_low = first_approx * cascade.low_scale;
        let first_high = first_detail[2 * input] * cascade.high_scale;
        let second_low = second_approx * cascade.low_scale;
        let second_high = first_detail[2 * input + 1] * cascade.high_scale;
        let output = 4 * input;
        out[output] = first_low + first_high;
        out[output + 1] = first_low - first_high;
        out[output + 2] = second_low + second_high;
        out[output + 3] = second_low - second_high;
    }
}
