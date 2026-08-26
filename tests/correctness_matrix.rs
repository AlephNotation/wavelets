use wavelets::{
    Boundary, DwtPlanner, Level, Wavelet, WaveletError, dwt_max_level, wavedec, waverec,
};

const BOUNDARIES: [Boundary; 9] = [
    Boundary::Zero,
    Boundary::Constant,
    Boundary::Symmetric,
    Boundary::Reflect,
    Boundary::Periodic,
    Boundary::Smooth,
    Boundary::Antisymmetric,
    Boundary::Antireflect,
    Boundary::Periodization,
];

// Includes all short edge cases from the design, even and odd production
// sizes, and small/large primes on both sides of the production sizes.
const LENGTHS: [usize; 25] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 31, 97, 100, 101, 997, 1000, 4093,
    4096,
];

fn signal_f64(len: usize) -> Vec<f64> {
    (0..len)
        .map(|index| (index as f64 * 0.37).sin() + (index % 7) as f64 - 3.0)
        .collect()
}

fn assert_reconstruction_f64(actual: &[f64], expected: &[f64], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}");
    let scale = expected.iter().copied().map(f64::abs).fold(1.0, f64::max);
    let tolerance = 1.0e-12 * scale;
    let (index, error) = actual
        .iter()
        .zip(expected)
        .enumerate()
        .map(|(index, (&actual, &expected))| (index, (actual - expected).abs()))
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .unwrap_or((0, 0.0));
    assert!(
        error <= tolerance,
        "{context}: reconstruction[{index}] error {error:.3e} exceeds {tolerance:.3e}"
    );
}

fn assert_reconstruction_f32(actual: &[f32], expected: &[f32], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}");
    let scale = expected.iter().copied().map(f32::abs).fold(1.0, f32::max);
    let tolerance = 5.0e-5 * scale;
    let (index, error) = actual
        .iter()
        .zip(expected)
        .enumerate()
        .map(|(index, (&actual, &expected))| (index, (actual - expected).abs()))
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .unwrap_or((0, 0.0));
    assert!(
        error <= tolerance,
        "{context}: reconstruction[{index}] error {error:.3e} exceeds {tolerance:.3e}"
    );
}

#[test]
fn daubechies_single_level_round_trip_matrix() {
    let mut planner_f64 = DwtPlanner::<f64>::new();
    let mut planner_f32 = DwtPlanner::<f32>::new();

    for order in 1..=38 {
        let wavelet = Wavelet::daubechies(order).unwrap();
        for boundary in BOUNDARIES {
            for len in LENGTHS {
                let context = format!("db{order} {boundary:?} len={len}");
                let signal_f64 = signal_f64(len);
                let plan_f64 = planner_f64.plan_dwt(len, &wavelet, boundary);
                let plan_f32 = planner_f32.plan_dwt(len, &wavelet, boundary);

                if len == 1 && matches!(boundary, Boundary::Reflect | Boundary::Antireflect) {
                    assert!(matches!(
                        plan_f64,
                        Err(WaveletError::BoundaryRequiresLongerSignal { .. })
                    ));
                    assert!(matches!(
                        plan_f32,
                        Err(WaveletError::BoundaryRequiresLongerSignal { .. })
                    ));
                    continue;
                }

                let plan_f64 = plan_f64.unwrap();
                let (approx_f64, detail_f64) = plan_f64.forward(&signal_f64);
                assert_eq!(approx_f64.len(), plan_f64.coeff_len(), "{context}");
                assert_eq!(detail_f64.len(), plan_f64.coeff_len(), "{context}");
                assert_reconstruction_f64(
                    &plan_f64.inverse(&approx_f64, &detail_f64),
                    &signal_f64,
                    &context,
                );

                let signal_f32: Vec<_> = signal_f64.iter().map(|&value| value as f32).collect();
                let plan_f32 = plan_f32.unwrap();
                let (approx_f32, detail_f32) = plan_f32.forward(&signal_f32);
                assert_eq!(approx_f32.len(), plan_f32.coeff_len(), "{context}");
                assert_eq!(detail_f32.len(), plan_f32.coeff_len(), "{context}");
                assert_reconstruction_f32(
                    &plan_f32.inverse(&approx_f32, &detail_f32),
                    &signal_f32,
                    &context,
                );
            }
        }
    }
}

#[test]
fn daubechies_multilevel_round_trip_matrix() {
    for order in 1..=38 {
        let wavelet = Wavelet::daubechies(order).unwrap();
        for boundary in BOUNDARIES {
            for len in LENGTHS {
                let context = format!("db{order} {boundary:?} len={len}");
                let signal_f64 = signal_f64(len);
                let decomposition_f64 =
                    wavedec(&signal_f64, &wavelet, boundary, Level::Max).unwrap();
                assert_eq!(
                    decomposition_f64.levels(),
                    dwt_max_level(len, wavelet.filter_len()),
                    "{context}"
                );
                assert_reconstruction_f64(
                    &waverec(&decomposition_f64).unwrap(),
                    &signal_f64,
                    &context,
                );

                let signal_f32: Vec<_> = signal_f64.iter().map(|&value| value as f32).collect();
                let decomposition_f32 =
                    wavedec(&signal_f32, &wavelet, boundary, Level::Max).unwrap();
                assert_reconstruction_f32(
                    &waverec(&decomposition_f32).unwrap(),
                    &signal_f32,
                    &context,
                );
            }
        }
    }
}

#[test]
fn every_daubechies_wavelet_preserves_periodized_energy() {
    const EVEN_LENGTHS: [usize; 6] = [2, 16, 100, 1000, 4094, 4096];
    let mut planner = DwtPlanner::<f64>::new();

    for order in 1..=38 {
        let wavelet = Wavelet::daubechies(order).unwrap();
        for len in EVEN_LENGTHS {
            let signal = signal_f64(len);
            let plan = planner
                .plan_dwt(len, &wavelet, Boundary::Periodization)
                .unwrap();
            let (approx, detail) = plan.forward(&signal);
            let signal_energy: f64 = signal.iter().map(|value| value * value).sum();
            let coefficient_energy: f64 = approx
                .iter()
                .chain(&detail)
                .map(|value| value * value)
                .sum();
            let tolerance = 1.0e-12 * signal_energy.max(1.0);
            assert!(
                (coefficient_energy - signal_energy).abs() <= tolerance,
                "db{order} len={len}: periodized energy {coefficient_energy:.17e} != {signal_energy:.17e}"
            );
        }
    }
}

#[test]
fn daubechies_details_annihilate_polynomials_away_from_boundaries() {
    let mut planner = DwtPlanner::<f64>::new();

    for order in 1..=38 {
        let wavelet = Wavelet::daubechies(order).unwrap();
        let filter_len = wavelet.filter_len();
        let signal_len = 4 * filter_len + 1;
        let plan = planner
            .plan_dwt(signal_len, &wavelet, Boundary::Symmetric)
            .unwrap();
        let interior_coefficient = filter_len;

        for degree in 0..order {
            let center = (signal_len - 1) as f64 / 2.0;
            let signal: Vec<_> = (0..signal_len)
                .map(|index| ((index as f64 - center) / center).powi(degree as i32))
                .collect();
            let (_, detail) = plan.forward(&signal);
            let residual = detail[interior_coefficient].abs();
            assert!(
                residual <= 5.0e-12,
                "db{order} degree={degree}: interior detail residual {residual:.3e}"
            );
        }
    }
}
