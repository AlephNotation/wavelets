use fearless_simd::{Level, dispatch};

use super::{
    AnalysisInterior, ButterflyAnalysis, ButterflyPairAnalysis, ButterflyPairSynthesis,
    ButterflySynthesis, LinearSynthesis, PeriodizedInterior, PlanarAnalysis, forward_butterfly,
    forward_butterfly_pair, forward_interior, forward_planar, inverse_butterfly,
    inverse_butterfly_pair, inverse_linear, inverse_periodized,
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

            let even: Vec<_> = signal.iter().step_by(2).copied().collect();
            let odd: Vec<_> = signal.iter().skip(1).step_by(2).copied().collect();
            let mut planar_approx = vec![-12_345.0; 47];
            let mut planar_detail = vec![-12_345.0; 47];
            let planar_outputs = dispatch!(Level::new(), simd => forward_planar(
                simd,
                PlanarAnalysis {
                    dec_lo: &dec_lo,
                    dec_hi: &dec_hi,
                    even: &even,
                    odd: &odd,
                    first_newest: 7,
                },
                &mut planar_approx,
                &mut planar_detail,
            ));
            assert!(planar_outputs > 0);
            assert!(planar_outputs < planar_approx.len());
            for output in 0..planar_outputs {
                let newest = 7 + 2 * output;
                let mut low: $sample = 0.0;
                let mut high: $sample = 0.0;
                for tap in 0..dec_lo.len() {
                    low = signal[newest - tap].mul_add(dec_lo[tap], low);
                    high = signal[newest - tap].mul_add(dec_hi[tap], high);
                }
                assert!((planar_approx[output] - low).abs() <= $tolerance);
                assert!((planar_detail[output] - high).abs() <= $tolerance);
            }
            assert!(planar_approx[planar_outputs..]
                .iter()
                .all(|&sample| sample == -12_345.0));
            assert!(planar_detail[planar_outputs..]
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
