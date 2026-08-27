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
    // Peeling consumes the entire two-tap filter, leaving no FMA dependency
    // chain for a wider batch to hide.
    if interior.dec_lo.len() == 2 {
        forward_interior_batches::<_, _, 1>(simd, &interior, approx, detail)
    } else {
        forward_interior_batches::<_, _, 2>(simd, &interior, approx, detail)
    }
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
        AnalysisInterior, LinearSynthesis, PeriodizedInterior, forward_interior, inverse_linear,
        inverse_periodized,
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
                    let mut out = vec![-12_345.0; 27];
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
