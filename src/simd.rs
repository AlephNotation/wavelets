use fearless_simd::{Simd, SimdFloatElement, prelude::*};

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

pub struct LinearSynthesis<'a, T> {
    pub(crate) rec_lo: &'a [T],
    pub(crate) rec_hi: &'a [T],
    pub(crate) approx: &'a [T],
    pub(crate) detail: &'a [T],
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
pub(crate) fn forward_interior<S: Simd, T: SimdSample<S>>(
    simd: S,
    interior: AnalysisInterior<'_, T>,
    approx: &mut [T],
    detail: &mut [T],
) -> usize {
    let AnalysisInterior {
        dec_lo,
        dec_hi,
        signal,
        first_newest,
    } = interior;
    let lanes = T::Vector::N;
    let vectorized_outputs = approx.len() - approx.len() % lanes;

    for output in (0..vectorized_outputs).step_by(lanes) {
        let newest = first_newest + 2 * output;
        let batch_start = newest + 1 - dec_lo.len();
        let batch_end = newest + 2 * lanes - 1;
        let input = &signal[batch_start..batch_end];
        let mut low_earlier = T::Vector::splat(simd, T::default());
        let mut low_later = T::Vector::splat(simd, T::default());
        let mut high_earlier = T::Vector::splat(simd, T::default());
        let mut high_later = T::Vector::splat(simd, T::default());
        let (low_pairs, low_remainder) = dec_lo.as_chunks::<2>();
        let (high_pairs, high_remainder) = dec_hi.as_chunks::<2>();
        debug_assert!(low_remainder.is_empty());
        debug_assert!(high_remainder.is_empty());

        for ((low, high), input) in low_pairs
            .iter()
            .zip(high_pairs)
            .zip(input.windows(2 * lanes).rev().step_by(2))
        {
            let (first, second) = input.split_at(lanes);
            let first = T::Vector::from_slice(simd, first);
            let second = T::Vector::from_slice(simd, second);
            let (earlier, later) = first.deinterleave(second);

            low_earlier = earlier.mul_add(low[1], low_earlier);
            low_later = later.mul_add(low[0], low_later);
            high_earlier = earlier.mul_add(high[1], high_earlier);
            high_later = later.mul_add(high[0], high_later);
        }

        (low_earlier + low_later).store_slice(&mut approx[output..output + lanes]);
        (high_earlier + high_later).store_slice(&mut detail[output..output + lanes]);
    }

    vectorized_outputs
}

#[inline(always)]
pub(crate) fn inverse_periodized<S: Simd, T: SimdSample<S>>(
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
        inverse_periodized_batch::<_, _, 2>(
            simd,
            &interior,
            out,
            interior.first_coefficient + pair,
        );
    }
    if paired_pairs < vectorized_pairs {
        inverse_periodized_batch::<_, _, 1>(
            simd,
            &interior,
            out,
            interior.first_coefficient + paired_pairs,
        );
    }

    vectorized_pairs
}

#[inline(always)]
fn inverse_periodized_batch<S: Simd, T: SimdSample<S>, const BATCHES: usize>(
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
    let second_offset = interior.second_offset;
    let lanes = T::Vector::N;
    let filter_len = first_lo.len();
    let first_start = first_coefficient + 1 - filter_len;
    let second_start = first_start + second_offset;
    let input_len = filter_len + BATCHES * lanes - 1;
    let first_approx = &approx[first_start..first_start + input_len];
    let first_detail = &detail[first_start..first_start + input_len];
    let second_approx = &approx[second_start..second_start + input_len];
    let second_detail = &detail[second_start..second_start + input_len];
    let mut first_low: [T::Vector; BATCHES] =
        std::array::from_fn(|_| T::Vector::splat(simd, T::default()));
    let mut first_high: [T::Vector; BATCHES] =
        std::array::from_fn(|_| T::Vector::splat(simd, T::default()));
    let mut second_low: [T::Vector; BATCHES] =
        std::array::from_fn(|_| T::Vector::splat(simd, T::default()));
    let mut second_high: [T::Vector; BATCHES] =
        std::array::from_fn(|_| T::Vector::splat(simd, T::default()));

    for (tap, (((first_lo, first_hi), second_lo), second_hi)) in first_lo
        .iter()
        .zip(first_hi)
        .zip(second_lo)
        .zip(second_hi)
        .enumerate()
    {
        let window_start = filter_len - 1 - tap;
        for batch in 0..BATCHES {
            let start = window_start + batch * lanes;
            let first_approximation =
                T::Vector::from_slice(simd, &first_approx[start..start + lanes]);
            let first_detail = T::Vector::from_slice(simd, &first_detail[start..start + lanes]);
            let second_approximation =
                T::Vector::from_slice(simd, &second_approx[start..start + lanes]);
            let second_detail = T::Vector::from_slice(simd, &second_detail[start..start + lanes]);
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
    let mut even_low: [T::Vector; BATCHES] =
        std::array::from_fn(|_| T::Vector::splat(simd, T::default()));
    let mut even_high: [T::Vector; BATCHES] =
        std::array::from_fn(|_| T::Vector::splat(simd, T::default()));
    let mut odd_low: [T::Vector; BATCHES] =
        std::array::from_fn(|_| T::Vector::splat(simd, T::default()));
    let mut odd_high: [T::Vector; BATCHES] =
        std::array::from_fn(|_| T::Vector::splat(simd, T::default()));

    for (tap, (((even_lo, even_hi), odd_lo), odd_hi)) in even_lo
        .iter()
        .zip(even_hi)
        .zip(odd_lo)
        .zip(odd_hi)
        .enumerate()
    {
        let window_start = half_filter_len - 1 - tap;
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

    use super::{AnalysisInterior, LinearSynthesis, forward_interior, inverse_linear};

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
            }
        };
    }

    kernel_test!(f32_kernels_match_scalar_and_leave_tails, f32, 8.0e-6);
    kernel_test!(f64_kernels_match_scalar_and_leave_tails, f64, 2.0e-14);
}
