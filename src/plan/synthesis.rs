use fearless_simd::Level as SimdLevel;

use crate::num::{
    checked_from_f64, inverse_butterfly_simd, inverse_linear_simd, inverse_periodized_simd,
};
use crate::simd::{ButterflySynthesis, LinearSynthesis, PeriodizedInterior};
use crate::{Wavelet, WaveletError, WaveletNum};

use super::{Butterfly, F64Butterfly, PlannedDwt};

#[derive(Debug)]
pub(super) struct PeriodizedSynthesis<T> {
    pub(super) initial_coefficient: usize,
    simd_start: usize,
    simd_available: usize,
    pub(super) phases_are_swapped: bool,
    pub(super) folded_filters: Option<Box<[T]>>,
}

impl<T: WaveletNum> PeriodizedSynthesis<T> {
    pub(super) fn new(signal_len: usize, coeff_len: usize, rec_lo: &[T], rec_hi: &[T]) -> Self {
        let original_half_filter_len = rec_lo.len() / 2;
        let shift = original_half_filter_len - 1;
        let phases_are_swapped = periodized_phases_are_swapped(rec_lo.len());
        let initial_coefficient = (shift / 2) % coeff_len;
        // A cyclic convolution whose filter phase is longer than the band
        // revisits the same coefficient several times. Compose those repeated
        // taps while planning: this is the same real linear operator with at
        // most one multiply per input coefficient, and avoids the rounding
        // growth of repeatedly traversing a very short band.
        let folded_filters = (coeff_len < original_half_filter_len).then(|| {
            let (first_lo, second_lo) = rec_lo.split_at(original_half_filter_len);
            let (first_hi, second_hi) = rec_hi.split_at(original_half_filter_len);
            let mut filters = vec![T::zero(); 4 * coeff_len];
            for (source, folded) in [first_lo, second_lo, first_hi, second_hi]
                .into_iter()
                .zip(filters.chunks_mut(coeff_len))
            {
                for (tap, &value) in source.iter().enumerate() {
                    folded[tap % coeff_len] += value;
                }
            }
            filters.into_boxed_slice()
        });
        let half_filter_len = folded_filters
            .as_ref()
            .map_or(original_half_filter_len, |_| coeff_len);
        let complete_pairs = signal_len / 2;
        let (simd_start, simd_available) = if coeff_len >= half_filter_len {
            let start = half_filter_len - 1 - initial_coefficient;
            let final_coefficient = coeff_len - usize::from(phases_are_swapped);
            let available = final_coefficient
                .saturating_sub(half_filter_len - 1)
                .min(complete_pairs.saturating_sub(start));
            (start, available)
        } else {
            (0, 0)
        };
        Self {
            initial_coefficient,
            simd_start,
            simd_available,
            phases_are_swapped,
            folded_filters,
        }
    }

    pub(super) fn filters<'a>(&'a self, rec_lo: &'a [T], rec_hi: &'a [T]) -> (&'a [T], &'a [T]) {
        let Some(filters) = &self.folded_filters else {
            return (rec_lo, rec_hi);
        };
        let filter_len = filters.len() / 2;
        filters.split_at(filter_len)
    }
}

pub(super) fn periodized_phases_are_swapped(filter_len: usize) -> bool {
    !(filter_len / 2 - 1).is_multiple_of(2)
}

pub(super) fn synthesis_butterfly(wavelet: &Wavelet) -> Option<F64Butterfly> {
    // Synthesis may qualify independently from analysis for a custom bank.
    let [low_first, low_second] = wavelet.rec_lo() else {
        return None;
    };
    let [high_first, high_second] = wavelet.rec_hi() else {
        return None;
    };
    (*low_first == *low_second && *high_first == -*high_second).then_some(F64Butterfly {
        low_scale: *low_first,
        high_scale: *high_first,
    })
}

pub(super) fn inverse_linear<T: WaveletNum>(
    plan: &PlannedDwt<T>,
    approx: &[T],
    detail: &[T],
    out: &mut [T],
) {
    let (rec_lo, rec_hi) = plan.filters.synthesis();
    let half_filter_len = rec_lo.len() / 2;
    let (even_lo, odd_lo) = rec_lo.split_at(half_filter_len);
    let (even_hi, odd_hi) = rec_hi.split_at(half_filter_len);

    let vectorized_pairs = inverse_linear_simd(
        plan.simd_level,
        LinearSynthesis {
            rec_lo,
            rec_hi,
            approx,
            detail,
        },
        out,
    );

    // Cropping the full convolution by `filter_len - 2` makes each output
    // pair consume the same reversed coefficient window. Fusing both
    // polyphase dots keeps that window hot and loads it only once.
    for (tail_coefficient, samples) in out[2 * vectorized_pairs..].chunks_mut(2).enumerate() {
        let coefficient = vectorized_pairs + tail_coefficient;
        let coefficient_end = coefficient + half_filter_len;
        let approx = &approx[coefficient..coefficient_end];
        let detail = &detail[coefficient..coefficient_end];
        let (first, second) = synthesis_pair(even_lo, even_hi, odd_lo, odd_hi, approx, detail);
        samples[0] = first;
        if samples.len() == 2 {
            samples[1] = second;
        }
    }
}

pub(super) fn inverse_periodized<T: WaveletNum>(
    plan: &PlannedDwt<T>,
    layout: &PeriodizedSynthesis<T>,
    approx: &[T],
    detail: &[T],
    out: &mut [T],
) {
    let (rec_lo, rec_hi) = plan.synthesis_filters();
    let half_filter_len = rec_lo.len() / 2;
    let (first_lo, second_lo) = rec_lo.split_at(half_filter_len);
    let (first_hi, second_hi) = rec_hi.split_at(half_filter_len);

    let (scalar_prefix, remainder) = out[..2 * layout.simd_start].as_chunks_mut::<2>();
    debug_assert!(remainder.is_empty());
    for (pair, samples) in scalar_prefix.iter_mut().enumerate() {
        let first_coefficient = (layout.initial_coefficient + pair) % plan.coeff_len;
        let (first, second) = synthesis_periodized_pair(
            (first_lo, first_hi),
            (second_lo, second_hi),
            approx,
            detail,
            first_coefficient,
            layout.phases_are_swapped,
        );
        samples[0] = first;
        samples[1] = second;
    }

    let vectorized = inverse_periodized_simd(
        plan.simd_level,
        PeriodizedInterior {
            first_lo,
            first_hi,
            second_lo,
            second_hi,
            approx,
            detail,
            first_coefficient: half_filter_len - 1,
            second_offset: usize::from(layout.phases_are_swapped),
        },
        &mut out[2 * layout.simd_start..2 * (layout.simd_start + layout.simd_available)],
    );

    let scalar_start = layout.simd_start + vectorized;
    for (tail_pair, samples) in out[2 * scalar_start..].chunks_mut(2).enumerate() {
        let pair = scalar_start + tail_pair;
        let first_coefficient = (layout.initial_coefficient + pair) % plan.coeff_len;
        let (first, second) = synthesis_periodized_pair(
            (first_lo, first_hi),
            (second_lo, second_hi),
            approx,
            detail,
            first_coefficient,
            layout.phases_are_swapped,
        );
        samples[0] = first;
        if samples.len() == 2 {
            samples[1] = second;
        }
    }
}

pub(super) fn inverse_butterfly<T: WaveletNum>(
    simd_level: SimdLevel,
    butterfly: Butterfly<T>,
    approx: &[T],
    detail: &[T],
    out: &mut [T],
) {
    let vectorized_pairs = inverse_butterfly_simd(
        simd_level,
        ButterflySynthesis {
            approx,
            detail,
            low_scale: butterfly.low_scale,
            high_scale: butterfly.high_scale,
        },
        out,
    );

    for (pair, samples) in out[2 * vectorized_pairs..].chunks_mut(2).enumerate() {
        let coefficient = vectorized_pairs + pair;
        let low = approx[coefficient] * butterfly.low_scale;
        let high = detail[coefficient] * butterfly.high_scale;
        samples[0] = low + high;
        if samples.len() == 2 {
            samples[1] = low - high;
        }
    }
}

pub(super) fn extend_polyphase<T: WaveletNum>(
    out: &mut Vec<T>,
    filter: &[f64],
) -> Result<(), WaveletError> {
    debug_assert!(filter.len().is_multiple_of(2));
    for &tap in filter
        .iter()
        .step_by(2)
        .chain(filter.iter().skip(1).step_by(2))
    {
        out.push(checked_from_f64(tap)?);
    }
    Ok(())
}

#[inline(always)]
fn synthesis_pair<T: WaveletNum>(
    even_lo: &[T],
    even_hi: &[T],
    odd_lo: &[T],
    odd_hi: &[T],
    approx: &[T],
    detail: &[T],
) -> (T, T) {
    let mut even = T::zero();
    let mut odd = T::zero();
    for ((((even_low, even_high), (odd_low, odd_high)), approximation), detail) in even_lo
        .iter()
        .zip(even_hi)
        .zip(odd_lo.iter().zip(odd_hi))
        .zip(approx.iter().rev())
        .zip(detail.iter().rev())
    {
        even += *even_low * *approximation;
        even += *even_high * *detail;
        odd += *odd_low * *approximation;
        odd += *odd_high * *detail;
    }
    (even, odd)
}

#[inline(always)]
fn synthesis_pair_windows<T: WaveletNum>(
    (first_lo, first_hi): (&[T], &[T]),
    (second_lo, second_hi): (&[T], &[T]),
    (first_approx, first_detail): (&[T], &[T]),
    (second_approx, second_detail): (&[T], &[T]),
) -> (T, T) {
    let mut first = T::zero();
    let mut second = T::zero();
    for (
        ((first_low, first_high), (second_low, second_high)),
        ((first_approx, first_detail), (second_approx, second_detail)),
    ) in first_lo
        .iter()
        .zip(first_hi)
        .zip(second_lo.iter().zip(second_hi))
        .zip(
            first_approx
                .iter()
                .rev()
                .zip(first_detail.iter().rev())
                .zip(second_approx.iter().rev().zip(second_detail.iter().rev())),
        )
    {
        first += *first_low * *first_approx;
        first += *first_high * *first_detail;
        second += *second_low * *second_approx;
        second += *second_high * *second_detail;
    }
    (first, second)
}

#[inline(always)]
fn synthesis_pair_cyclic<T: WaveletNum>(
    (first_lo, first_hi): (&[T], &[T]),
    (second_lo, second_hi): (&[T], &[T]),
    approx: &[T],
    detail: &[T],
    first_coefficient: usize,
    second_coefficient: usize,
) -> (T, T) {
    let mut first = T::zero();
    let mut second = T::zero();
    let mut first_coefficient = first_coefficient;
    let mut second_coefficient = second_coefficient;
    for (((first_low, first_high), second_low), second_high) in
        first_lo.iter().zip(first_hi).zip(second_lo).zip(second_hi)
    {
        first += *first_low * approx[first_coefficient];
        first += *first_high * detail[first_coefficient];
        second += *second_low * approx[second_coefficient];
        second += *second_high * detail[second_coefficient];
        first_coefficient = decrement_wrapping(first_coefficient, approx.len());
        second_coefficient = decrement_wrapping(second_coefficient, approx.len());
    }
    (first, second)
}

#[inline(always)]
fn synthesis_periodized_pair<T: WaveletNum>(
    first_filters: (&[T], &[T]),
    second_filters: (&[T], &[T]),
    approx: &[T],
    detail: &[T],
    first_coefficient: usize,
    phases_are_swapped: bool,
) -> (T, T) {
    let second_coefficient = if phases_are_swapped {
        increment_wrapping(first_coefficient, approx.len())
    } else {
        first_coefficient
    };
    let filter_len = first_filters.0.len();

    if first_coefficient + 1 >= filter_len && second_coefficient + 1 >= filter_len {
        let first_start = first_coefficient + 1 - filter_len;
        let second_start = second_coefficient + 1 - filter_len;
        synthesis_pair_windows(
            first_filters,
            second_filters,
            (
                &approx[first_start..=first_coefficient],
                &detail[first_start..=first_coefficient],
            ),
            (
                &approx[second_start..=second_coefficient],
                &detail[second_start..=second_coefficient],
            ),
        )
    } else {
        synthesis_pair_cyclic(
            first_filters,
            second_filters,
            approx,
            detail,
            first_coefficient,
            second_coefficient,
        )
    }
}

#[inline]
pub(super) fn increment_wrapping(value: usize, len: usize) -> usize {
    if value + 1 == len { 0 } else { value + 1 }
}

#[inline]
fn decrement_wrapping(value: usize, len: usize) -> usize {
    if value == 0 { len - 1 } else { value - 1 }
}
