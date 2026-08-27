use fearless_simd::{Simd, prelude::*};

#[inline(always)]
pub(crate) fn inverse_linear_f32<S: Simd>(
    simd: S,
    rec_lo: &[f32],
    rec_hi: &[f32],
    approx: &[f32],
    detail: &[f32],
    out: &mut [f32],
) -> usize {
    let half_filter_len = rec_lo.len() / 2;
    let (even_lo, odd_lo) = rec_lo.split_at(half_filter_len);
    let (even_hi, odd_hi) = rec_hi.split_at(half_filter_len);
    let lanes = S::f32s::N;
    let pair_count = out.len() / 2;
    let vectorized_pairs = pair_count - pair_count % lanes;

    for pair in (0..vectorized_pairs).step_by(lanes) {
        // Four independent accumulators expose enough instruction-level
        // parallelism for FMA without creating one long dependency chain.
        let mut even_low = S::f32s::splat(simd, 0.0);
        let mut even_high = S::f32s::splat(simd, 0.0);
        let mut odd_low = S::f32s::splat(simd, 0.0);
        let mut odd_high = S::f32s::splat(simd, 0.0);
        for tap in 0..half_filter_len {
            let coefficient = pair + half_filter_len - 1 - tap;
            let coefficient_end = coefficient + lanes;
            let approximation = S::f32s::from_slice(simd, &approx[coefficient..coefficient_end]);
            let detail = S::f32s::from_slice(simd, &detail[coefficient..coefficient_end]);
            even_low = approximation.mul_add(even_lo[tap], even_low);
            even_high = detail.mul_add(even_hi[tap], even_high);
            odd_low = approximation.mul_add(odd_lo[tap], odd_low);
            odd_high = detail.mul_add(odd_hi[tap], odd_high);
        }

        let even = even_low + even_high;
        let odd = odd_low + odd_high;
        let (first, second) = even.interleave(odd);
        let output = 2 * pair;
        first.store_slice(&mut out[output..output + lanes]);
        second.store_slice(&mut out[output + lanes..output + 2 * lanes]);
    }

    vectorized_pairs
}

#[inline(always)]
pub(crate) fn inverse_linear_f64<S: Simd>(
    simd: S,
    rec_lo: &[f64],
    rec_hi: &[f64],
    approx: &[f64],
    detail: &[f64],
    out: &mut [f64],
) -> usize {
    let half_filter_len = rec_lo.len() / 2;
    let (even_lo, odd_lo) = rec_lo.split_at(half_filter_len);
    let (even_hi, odd_hi) = rec_hi.split_at(half_filter_len);
    let lanes = S::f64s::N;
    let pair_count = out.len() / 2;
    let vectorized_pairs = pair_count - pair_count % lanes;

    for pair in (0..vectorized_pairs).step_by(lanes) {
        // Four independent accumulators expose enough instruction-level
        // parallelism for FMA without creating one long dependency chain.
        let mut even_low = S::f64s::splat(simd, 0.0);
        let mut even_high = S::f64s::splat(simd, 0.0);
        let mut odd_low = S::f64s::splat(simd, 0.0);
        let mut odd_high = S::f64s::splat(simd, 0.0);
        for tap in 0..half_filter_len {
            let coefficient = pair + half_filter_len - 1 - tap;
            let coefficient_end = coefficient + lanes;
            let approximation = S::f64s::from_slice(simd, &approx[coefficient..coefficient_end]);
            let detail = S::f64s::from_slice(simd, &detail[coefficient..coefficient_end]);
            even_low = approximation.mul_add(even_lo[tap], even_low);
            even_high = detail.mul_add(even_hi[tap], even_high);
            odd_low = approximation.mul_add(odd_lo[tap], odd_low);
            odd_high = detail.mul_add(odd_hi[tap], odd_high);
        }

        let even = even_low + even_high;
        let odd = odd_low + odd_high;
        let (first, second) = even.interleave(odd);
        let output = 2 * pair;
        first.store_slice(&mut out[output..output + lanes]);
        second.store_slice(&mut out[output + lanes..output + 2 * lanes]);
    }

    vectorized_pairs
}

#[cfg(test)]
mod tests {
    use fearless_simd::{Level, dispatch};

    use super::{inverse_linear_f32, inverse_linear_f64};

    #[test]
    fn f32_kernel_matches_scalar_reference_and_leaves_tail() {
        let rec_lo = [0.17_f32, -0.31, 0.53, 0.79, -0.11, 0.23, 0.41, -0.67];
        let rec_hi = [-0.37_f32, 0.19, 0.73, -0.29, 0.61, -0.43, 0.13, 0.47];
        let approx: Vec<_> = (0..44).map(|index| index as f32 * 0.13 - 1.7).collect();
        let detail: Vec<_> = (0..44).map(|index| index as f32 * -0.07 + 0.9).collect();
        let mut out = vec![-12_345.0; 83];

        let processed = dispatch!(Level::new(), simd => inverse_linear_f32(
            simd, &rec_lo, &rec_hi, &approx, &detail, &mut out
        ));

        assert!(processed > 0);
        assert!(processed < out.len() / 2);
        for pair in 0..processed {
            let (even, odd) = scalar_pair_f32(pair, &rec_lo, &rec_hi, &approx, &detail);
            assert!((out[2 * pair] - even).abs() <= 8.0e-6);
            assert!((out[2 * pair + 1] - odd).abs() <= 8.0e-6);
        }
        assert!(
            out[2 * processed..]
                .iter()
                .all(|&sample| sample == -12_345.0)
        );
    }

    #[test]
    fn f64_kernel_matches_scalar_reference_and_leaves_tail() {
        let rec_lo = [0.17_f64, -0.31, 0.53, 0.79, -0.11, 0.23, 0.41, -0.67];
        let rec_hi = [-0.37_f64, 0.19, 0.73, -0.29, 0.61, -0.43, 0.13, 0.47];
        let approx: Vec<_> = (0..44).map(|index| index as f64 * 0.13 - 1.7).collect();
        let detail: Vec<_> = (0..44).map(|index| index as f64 * -0.07 + 0.9).collect();
        let mut out = vec![-12_345.0; 83];

        let processed = dispatch!(Level::new(), simd => inverse_linear_f64(
            simd, &rec_lo, &rec_hi, &approx, &detail, &mut out
        ));

        assert!(processed > 0);
        assert!(processed < out.len() / 2);
        for pair in 0..processed {
            let (even, odd) = scalar_pair_f64(pair, &rec_lo, &rec_hi, &approx, &detail);
            assert!((out[2 * pair] - even).abs() <= 2.0e-14);
            assert!((out[2 * pair + 1] - odd).abs() <= 2.0e-14);
        }
        assert!(
            out[2 * processed..]
                .iter()
                .all(|&sample| sample == -12_345.0)
        );
    }

    fn scalar_pair_f32(
        pair: usize,
        rec_lo: &[f32],
        rec_hi: &[f32],
        approx: &[f32],
        detail: &[f32],
    ) -> (f32, f32) {
        let half = rec_lo.len() / 2;
        let (even_lo, odd_lo) = rec_lo.split_at(half);
        let (even_hi, odd_hi) = rec_hi.split_at(half);
        let mut even_low = 0.0_f32;
        let mut even_high = 0.0_f32;
        let mut odd_low = 0.0_f32;
        let mut odd_high = 0.0_f32;
        for tap in 0..half {
            let coefficient = pair + half - 1 - tap;
            even_low = approx[coefficient].mul_add(even_lo[tap], even_low);
            even_high = detail[coefficient].mul_add(even_hi[tap], even_high);
            odd_low = approx[coefficient].mul_add(odd_lo[tap], odd_low);
            odd_high = detail[coefficient].mul_add(odd_hi[tap], odd_high);
        }
        (even_low + even_high, odd_low + odd_high)
    }

    fn scalar_pair_f64(
        pair: usize,
        rec_lo: &[f64],
        rec_hi: &[f64],
        approx: &[f64],
        detail: &[f64],
    ) -> (f64, f64) {
        let half = rec_lo.len() / 2;
        let (even_lo, odd_lo) = rec_lo.split_at(half);
        let (even_hi, odd_hi) = rec_hi.split_at(half);
        let mut even_low = 0.0_f64;
        let mut even_high = 0.0_f64;
        let mut odd_low = 0.0_f64;
        let mut odd_high = 0.0_f64;
        for tap in 0..half {
            let coefficient = pair + half - 1 - tap;
            even_low = approx[coefficient].mul_add(even_lo[tap], even_low);
            even_high = detail[coefficient].mul_add(even_hi[tap], even_high);
            odd_low = approx[coefficient].mul_add(odd_lo[tap], odd_low);
            odd_high = detail[coefficient].mul_add(odd_hi[tap], odd_high);
        }
        (even_low + even_high, odd_low + odd_high)
    }
}
