use super::*;
use crate::{Boundary, DwtPlanner, Wavelet, wavedec};

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

#[test]
fn max_level_matches_known_values() {
    assert_eq!(dwt_max_level(1, 2), 0);
    assert_eq!(dwt_max_level(2, 2), 1);
    assert_eq!(dwt_max_level(4, 4), 0);
    assert_eq!(dwt_max_level(6, 4), 1);
    assert_eq!(dwt_max_level(12, 4), 2);
    assert_eq!(dwt_max_level(1000, 8), 7);
}

#[test]
fn butterfly_cascade_selection_depends_on_algebra_and_geometry() {
    let wavelet =
        Wavelet::from_filters(&[0.5, 0.5], &[-0.25, 0.25], &[0.75, 0.75], &[0.125, -0.125])
            .unwrap();
    let mut planner = DwtPlanner::<f64>::new();

    for boundary in BOUNDARIES {
        let plan = planner
            .plan_wavedec(256, &wavelet, boundary, Level::Exact(8))
            .unwrap();
        assert!(plan.butterfly_analysis_cascade.is_some(), "{boundary:?}");
        assert!(plan.butterfly_synthesis_cascade.is_some(), "{boundary:?}");
        assert_eq!(plan.scratch_len(), 80, "{boundary:?}");
    }

    let short = planner
        .plan_wavedec(64, &wavelet, Boundary::Symmetric, Level::Exact(6))
        .unwrap();
    assert!(short.butterfly_analysis_cascade.is_none());
    assert!(short.butterfly_synthesis_cascade.is_some());
    assert_eq!(short.scratch_len(), 48);

    let odd_level_count = planner
        .plan_wavedec(64, &wavelet, Boundary::Symmetric, Level::Exact(5))
        .unwrap();
    assert!(odd_level_count.butterfly_analysis_cascade.is_none());
    assert!(odd_level_count.butterfly_synthesis_cascade.is_none());
    let edge_bearing_level = planner
        .plan_wavedec(20, &wavelet, Boundary::Symmetric, Level::Exact(4))
        .unwrap();
    assert!(edge_bearing_level.butterfly_analysis_cascade.is_none());
    assert!(edge_bearing_level.butterfly_synthesis_cascade.is_none());
}

#[test]
fn custom_butterfly_cascade_matches_two_single_level_plans() {
    let wavelet =
        Wavelet::from_filters(&[0.5, 0.5], &[-0.25, 0.25], &[0.75, 0.75], &[0.125, -0.125])
            .unwrap();
    let signal: Vec<_> = (0..256)
        .map(|index| (index as f64 * 0.19).sin() + index as f64 * 0.03)
        .collect();
    let mut planner = DwtPlanner::<f64>::new();
    let first = planner
        .plan_dwt(signal.len(), &wavelet, Boundary::Symmetric)
        .unwrap();
    let second = planner
        .plan_dwt(first.coeff_len(), &wavelet, Boundary::Symmetric)
        .unwrap();
    let cascade = planner
        .plan_wavedec(signal.len(), &wavelet, Boundary::Symmetric, Level::Exact(2))
        .unwrap();
    assert_eq!(cascade.scratch_len(), 0);

    let (first_approx, first_detail) = first.forward(&signal);
    let (expected_approx, expected_second_detail) = second.forward(&first_approx);
    let actual = cascade.forward(&signal);
    assert_eq!(actual.approx(), expected_approx);
    assert_eq!(actual.detail(1), first_detail);
    assert_eq!(actual.detail(2), expected_second_detail);

    let expected_first_approx = second.inverse(actual.approx(), actual.detail(2));
    let expected_signal = first.inverse(&expected_first_approx, actual.detail(1));
    assert_eq!(cascade.inverse(&actual), expected_signal);
}

#[test]
fn decomposition_uses_natural_detail_level_numbers() {
    let signal: Vec<_> = (0..16).map(f64::from).collect();
    let wavelet = Wavelet::haar();
    let dec = wavedec(&signal, &wavelet, Boundary::Symmetric, Level::Exact(3)).unwrap();
    assert_eq!(dec.levels(), 3);
    assert_eq!(dec.detail(1).len(), 8);
    assert_eq!(dec.detail(2).len(), 4);
    assert_eq!(dec.detail(3).len(), 2);
    assert_eq!(dec.as_slice().len(), 16);

    let bands: Vec<_> = dec.bands().collect();
    assert_eq!(bands.len(), 4);
    assert_eq!(bands[0], dec.approx());
    assert_eq!(bands[1], dec.detail(3));
    assert_eq!(bands[2], dec.detail(2));
    assert_eq!(bands[3], dec.detail(1));

    let mut mutable = dec.clone();
    mutable.as_mut_slice()[0] = 42.0;
    assert_eq!(mutable.approx()[0], 42.0);
}
