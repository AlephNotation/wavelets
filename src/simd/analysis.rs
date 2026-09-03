use fearless_simd::{Simd, prelude::*};

use super::SimdSample;

pub struct AnalysisInterior<'a, T> {
    pub(crate) dec_lo: &'a [T],
    pub(crate) dec_hi: &'a [T],
    pub(crate) signal: &'a [T],
    pub(crate) first_newest: usize,
}

pub struct PlanarAnalysis<'a, T> {
    pub(crate) dec_lo: &'a [T],
    pub(crate) dec_hi: &'a [T],
    pub(crate) even: &'a [T],
    pub(crate) odd: &'a [T],
    pub(crate) first_newest: usize,
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

#[derive(Clone, Copy)]
struct AnalysisAccumulators<V> {
    low_earlier: V,
    low_later: V,
    high_earlier: V,
    high_later: V,
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
pub(crate) fn forward_planar<S: Simd, T: SimdSample<S>>(
    simd: S,
    analysis: PlanarAnalysis<'_, T>,
    approx: &mut [T],
    detail: &mut [T],
) -> usize {
    let lanes = T::Vector::N;
    let vectorized_outputs = approx.len() - approx.len() % lanes;
    let batch_width = 4 * lanes;
    let batched_outputs = vectorized_outputs - vectorized_outputs % batch_width;

    for output in (0..batched_outputs).step_by(batch_width) {
        forward_planar_batch::<_, _, 4>(simd, &analysis, approx, detail, output);
    }
    for output in (batched_outputs..vectorized_outputs).step_by(lanes) {
        forward_planar_batch::<_, _, 1>(simd, &analysis, approx, detail, output);
    }

    vectorized_outputs
}

#[inline(always)]
fn forward_planar_batch<S: Simd, T: SimdSample<S>, const BATCHES: usize>(
    simd: S,
    analysis: &PlanarAnalysis<'_, T>,
    approx: &mut [T],
    detail: &mut [T],
    first_output: usize,
) {
    let lanes = T::Vector::N;
    let zero = T::Vector::splat(simd, T::default());
    let mut low = [zero; BATCHES];
    let mut high = low;

    for tap in 0..analysis.dec_lo.len() {
        let position = analysis.first_newest - tap;
        let plane = if position.is_multiple_of(2) {
            analysis.even
        } else {
            analysis.odd
        };
        let first_sample = position / 2 + first_output;
        for batch in 0..BATCHES {
            let sample = first_sample + batch * lanes;
            let input = T::Vector::from_slice(simd, &plane[sample..sample + lanes]);
            low[batch] = input.mul_add(analysis.dec_lo[tap], low[batch]);
            high[batch] = input.mul_add(analysis.dec_hi[tap], high[batch]);
        }
    }

    for batch in 0..BATCHES {
        let output = first_output + batch * lanes;
        low[batch].store_slice(&mut approx[output..output + lanes]);
        high[batch].store_slice(&mut detail[output..output + lanes]);
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
