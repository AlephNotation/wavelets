use fearless_simd::{Simd, prelude::*};

use super::SimdSample;

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
