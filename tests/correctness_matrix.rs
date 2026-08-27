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

const BIORTHOGONAL_ORDERS: [(usize, usize); 15] = [
    (1, 1),
    (1, 3),
    (1, 5),
    (2, 2),
    (2, 4),
    (2, 6),
    (2, 8),
    (3, 1),
    (3, 3),
    (3, 5),
    (3, 7),
    (3, 9),
    (4, 4),
    (5, 5),
    (6, 8),
];

fn signal_f64(len: usize) -> Vec<f64> {
    (0..len)
        .map(|index| (index as f64 * 0.37).sin() + (index % 7) as f64 - 3.0)
        .collect()
}

fn orthogonal_wavelets() -> Vec<Wavelet> {
    (1..=38)
        .map(|order| Wavelet::daubechies(order).unwrap())
        .chain((2..=20).map(|order| Wavelet::symlet(order).unwrap()))
        .chain((1..=17).map(|order| Wavelet::coiflet(order).unwrap()))
        .collect()
}

fn built_in_wavelets() -> Vec<Wavelet> {
    orthogonal_wavelets()
        .into_iter()
        .chain(BIORTHOGONAL_ORDERS.map(|(nr, nd)| Wavelet::biorthogonal(nr, nd).unwrap()))
        .chain(BIORTHOGONAL_ORDERS.map(|(nr, nd)| Wavelet::reverse_biorthogonal(nr, nd).unwrap()))
        .collect()
}

fn assert_reconstruction_f64(actual: &[f64], expected: &[f64], stages: usize, context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}");
    let scale = expected.iter().copied().map(f64::abs).fold(1.0, f64::max);
    // Deep reverse-biorthogonal reconstruction with derivative-extrapolated
    // boundaries has the same conditioning limit in PyWavelets. Scale the
    // uniform contract with depth instead of adding family-specific cases.
    let tolerance = 1.0e-12 * stages.max(1) as f64 * scale;
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

fn assert_reconstruction_f32(actual: &[f32], expected: &[f32], stages: usize, context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}");
    let scale = expected.iter().copied().map(f32::abs).fold(1.0, f32::max);
    // Symmetric biorthogonal banks can amplify binary32 rounding during deep
    // reconstruction, especially with derivative-extrapolated boundaries.
    // PyWavelets reaches the same order of error for the identical cases.
    let tolerance = 2.0e-4 * stages.max(1) as f32 * scale;
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
fn every_built_in_single_level_round_trip_matrix() {
    let mut planner_f64 = DwtPlanner::<f64>::new();
    let mut planner_f32 = DwtPlanner::<f32>::new();

    for wavelet in built_in_wavelets() {
        for boundary in BOUNDARIES {
            for len in LENGTHS {
                let context = format!("{} {boundary:?} len={len}", wavelet.name());
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
                    1,
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
                    1,
                    &context,
                );
            }
        }
    }
}

#[test]
fn every_built_in_multilevel_round_trip_matrix() {
    for wavelet in built_in_wavelets() {
        for boundary in BOUNDARIES {
            for len in LENGTHS {
                let context = format!("{} {boundary:?} len={len}", wavelet.name());
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
                    decomposition_f64.levels(),
                    &context,
                );

                let signal_f32: Vec<_> = signal_f64.iter().map(|&value| value as f32).collect();
                let decomposition_f32 =
                    wavedec(&signal_f32, &wavelet, boundary, Level::Max).unwrap();
                assert_reconstruction_f32(
                    &waverec(&decomposition_f32).unwrap(),
                    &signal_f32,
                    decomposition_f32.levels(),
                    &context,
                );
            }
        }
    }
}

#[test]
fn every_orthogonal_wavelet_preserves_periodized_energy() {
    const EVEN_LENGTHS: [usize; 6] = [2, 16, 100, 1000, 4094, 4096];
    let mut planner = DwtPlanner::<f64>::new();

    for wavelet in orthogonal_wavelets() {
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
                "{} len={len}: periodized energy {coefficient_energy:.17e} != {signal_energy:.17e}",
                wavelet.name()
            );
        }
    }
}

#[test]
fn orthogonal_details_annihilate_polynomials_away_from_boundaries() {
    let mut planner = DwtPlanner::<f64>::new();

    for wavelet in orthogonal_wavelets() {
        let moments = wavelet
            .vanishing_moments()
            .expect("built-in orthogonal wavelets declare their moments");
        let filter_len = wavelet.filter_len();
        let signal_len = 4 * filter_len + 1;
        let plan = planner
            .plan_dwt(signal_len, &wavelet, Boundary::Symmetric)
            .unwrap();
        let interior_coefficient = filter_len;

        for degree in 0..moments {
            let center = (signal_len - 1) as f64 / 2.0;
            let signal: Vec<_> = (0..signal_len)
                .map(|index| ((index as f64 - center) / center).powi(degree as i32))
                .collect();
            let (_, detail) = plan.forward(&signal);
            let residual = detail[interior_coefficient].abs();
            assert!(
                residual <= 5.0e-12,
                "{} degree={degree}: interior detail residual {residual:.3e}",
                wavelet.name()
            );
        }
    }
}
