use proptest::prelude::*;
use wavelets::{Boundary, DwtPlanner, Level, Wavelet, wavedec, waverec};

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

proptest! {
    #[test]
    fn db2_single_level_round_trip(
        signal in prop::collection::vec(-1.0e6_f64..1.0e6, 2..128),
        boundary_index in 0_usize..BOUNDARIES.len(),
    ) {
        let boundary = BOUNDARIES[boundary_index];
        let wavelet = Wavelet::daubechies(2).unwrap();
        let mut planner = DwtPlanner::<f64>::new();
        let plan = planner.plan_dwt(signal.len(), &wavelet, boundary).unwrap();
        let (approx, detail) = plan.forward(&signal);
        let reconstructed = plan.inverse(&approx, &detail);
        let scale = signal.iter().copied().map(f64::abs).fold(1.0, f64::max);
        let tolerance = 1.0e-12 * scale;
        for (actual, expected) in reconstructed.into_iter().zip(signal) {
            prop_assert!((actual - expected).abs() <= tolerance);
        }
    }

    #[test]
    fn haar_multilevel_round_trip(
        signal in prop::collection::vec(-1.0e6_f64..1.0e6, 2..128),
        boundary_index in 0_usize..BOUNDARIES.len(),
    ) {
        let boundary = BOUNDARIES[boundary_index];
        let wavelet = Wavelet::haar();
        let dec = wavedec(&signal, &wavelet, boundary, Level::Max).unwrap();
        let reconstructed = waverec(&dec).unwrap();
        let scale = signal.iter().copied().map(f64::abs).fold(1.0, f64::max);
        let tolerance = 1.0e-12 * scale;
        for (actual, expected) in reconstructed.into_iter().zip(signal) {
            prop_assert!((actual - expected).abs() <= tolerance);
        }
    }

    #[test]
    fn db2_multilevel_round_trip(
        signal in prop::collection::vec(-1.0e6_f64..1.0e6, 2..128),
        boundary_index in 0_usize..BOUNDARIES.len(),
    ) {
        let boundary = BOUNDARIES[boundary_index];
        let wavelet = Wavelet::daubechies(2).unwrap();
        let dec = wavedec(&signal, &wavelet, boundary, Level::Max).unwrap();
        let reconstructed = waverec(&dec).unwrap();
        let scale = signal.iter().copied().map(f64::abs).fold(1.0, f64::max);
        let tolerance = 1.0e-12 * scale;
        for (actual, expected) in reconstructed.into_iter().zip(signal) {
            prop_assert!((actual - expected).abs() <= tolerance);
        }
    }

    #[test]
    fn db2_f32_single_level_round_trip(
        signal in prop::collection::vec(-1.0e3_f32..1.0e3, 2..128),
        boundary_index in 0_usize..BOUNDARIES.len(),
    ) {
        let boundary = BOUNDARIES[boundary_index];
        let wavelet = Wavelet::daubechies(2).unwrap();
        let mut planner = DwtPlanner::<f32>::new();
        let plan = planner.plan_dwt(signal.len(), &wavelet, boundary).unwrap();
        let (approx, detail) = plan.forward(&signal);
        let reconstructed = plan.inverse(&approx, &detail);
        let scale = signal.iter().copied().map(f32::abs).fold(1.0, f32::max);
        let tolerance = 1.0e-5 * scale;
        for (actual, expected) in reconstructed.into_iter().zip(signal) {
            prop_assert!((actual - expected).abs() <= tolerance);
        }
    }

    #[test]
    fn db2_periodization_preserves_energy(
        signal in prop::collection::vec(-1.0e3_f64..1.0e3, 1..64)
            .prop_filter("periodization Parseval test uses even lengths", |values| values.len() % 2 == 0),
    ) {
        let wavelet = Wavelet::daubechies(2).unwrap();
        let mut planner = DwtPlanner::<f64>::new();
        let plan = planner
            .plan_dwt(signal.len(), &wavelet, Boundary::Periodization)
            .unwrap();
        let (approx, detail) = plan.forward(&signal);
        let signal_energy: f64 = signal.iter().map(|value| value * value).sum();
        let coefficient_energy: f64 = approx
            .iter()
            .chain(&detail)
            .map(|value| value * value)
            .sum();
        prop_assert!(
            (coefficient_energy - signal_energy).abs()
                <= 1.0e-9_f64.max(signal_energy * 1.0e-12)
        );
    }
}
