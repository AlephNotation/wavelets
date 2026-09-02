use super::super::analysis::coefficient_len;
use super::super::annihilator::AnnihilatorAnalysis;
use super::super::*;

#[test]
fn annihilator_selection_depends_on_filter_support() {
    let db20 = equivalent_custom_wavelet(&Wavelet::daubechies(20).unwrap());
    let db38 = equivalent_custom_wavelet(&Wavelet::daubechies(38).unwrap());
    let coif17 = equivalent_custom_wavelet(&Wavelet::coiflet(17).unwrap());
    let short = PreparedFilterBank::<f64>::new(&db20, false);
    let long = PreparedFilterBank::<f64>::new(&db38, false);

    assert!(short.analysis_annihilator.is_none());
    assert!(long.analysis_annihilator.is_some());

    let f32_db38 = PreparedFilterBank::<f32>::new(&db38, false);
    let f32_coif17 = PreparedFilterBank::<f32>::new(&coif17, false);
    assert!(f32_db38.analysis_annihilator.is_none());
    assert!(f32_coif17.analysis_annihilator.is_some());
}

#[test]
fn dense_signal_rejects_annihilator_execution() {
    let wavelet = equivalent_custom_wavelet(&Wavelet::daubechies(38).unwrap());
    let annihilator = annihilator_analysis(4_096, &wavelet, Boundary::Symmetric);
    let signal: Vec<_> = (0..4_096)
        .map(|index| (index as f64 * 0.173).sin())
        .collect();

    assert!(!annihilator.should_execute(&signal));
}

#[test]
fn non_finite_or_overflowing_differences_use_direct_execution() {
    let wavelet = equivalent_custom_wavelet(&Wavelet::daubechies(38).unwrap());
    let annihilator = annihilator_analysis(4_096, &wavelet, Boundary::Symmetric);

    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut signal = vec![1.0; 4_096];
        signal[128] = value;
        assert!(!annihilator.should_execute(&signal));
    }
    let mut overflowing = vec![f64::MAX; 4_096];
    overflowing[128..].fill(-f64::MAX);
    assert!(!annihilator.should_execute(&overflowing));
}

#[test]
fn annihilator_matches_direct_kernel_for_every_boundary() {
    let boundaries = [
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
    for wavelet in [
        Wavelet::daubechies(38).unwrap(),
        Wavelet::coiflet(17).unwrap(),
    ] {
        for boundary in boundaries {
            let f64_signal: Vec<_> = (0..4_096)
                .map(|index| {
                    let run = index / 64;
                    1.0 + (run as f64 * 0.17).sin() + 0.1 * run as f64
                })
                .collect();
            assert_annihilator_matches_direct(
                &wavelet,
                boundary,
                f64_signal,
                |actual: f64, expected: f64| {
                    (actual - expected).abs() <= 2e-12_f64.max(2e-15 * expected.abs())
                },
            );

            if wavelet.filter_len() >= super::super::annihilator::MIN_ANNIHILATOR_FILTER_LEN_F32 {
                let f32_signal: Vec<_> = (0..4_096)
                    .map(|index| {
                        let run = index / 64;
                        (1.0 + (run as f64 * 0.17).sin() + 0.1 * run as f64) as f32
                    })
                    .collect();
                assert_annihilator_matches_direct(
                    &wavelet,
                    boundary,
                    f32_signal,
                    |actual: f32, expected: f32| {
                        (actual - expected).abs() <= 2e-4_f32.max(2e-6 * expected.abs())
                    },
                );
            }
        }
    }
}

#[test]
fn lattice_matches_direct_kernel_for_every_supported_boundary() {
    let boundaries = [
        Boundary::Zero,
        Boundary::Constant,
        Boundary::Symmetric,
        Boundary::Reflect,
        Boundary::Periodic,
        Boundary::Smooth,
        Boundary::Antisymmetric,
        Boundary::Antireflect,
    ];
    for wavelet in [
        Wavelet::daubechies(20).unwrap(),
        Wavelet::symlet(20).unwrap(),
        Wavelet::daubechies(38).unwrap(),
        Wavelet::coiflet(17).unwrap(),
    ] {
        for boundary in boundaries {
            let signal: Vec<_> = (0..4_096)
                .map(|index| {
                    let index = index as f64;
                    (index * 0.173).sin() + 0.25 * (index * 0.037).cos()
                })
                .collect();
            assert_lattice_matches_direct(&wavelet, boundary, signal, 4.0e-13);
        }
    }
}

#[test]
fn lattice_remains_finite_over_wide_dynamic_range() {
    for wavelet in [
        Wavelet::daubechies(38).unwrap(),
        Wavelet::coiflet(17).unwrap(),
    ] {
        let signal: Vec<_> = (0_usize..4_096)
            .map(|index| {
                let exponent = ((index * 811) % 1_801) as i32 - 900;
                let mantissa = 1.0 + ((index * 37) % 997) as f64 / 997.0;
                let sign = if index.is_multiple_of(2) { 1.0 } else { -1.0 };
                sign * mantissa * 2.0_f64.powi(exponent)
            })
            .collect();
        assert_lattice_matches_direct(&wavelet, Boundary::Symmetric, signal, 2.0e-12);
    }
}

#[test]
fn avx512_lattice_preempts_the_dominated_structure_scan() {
    let level = SimdLevel::new();
    if !lattice_preempts_annihilator(level) {
        return;
    }
    let plan = create_dwt_plan::<f64>(
        4_096,
        &Wavelet::daubechies(38).unwrap(),
        Boundary::Symmetric,
        level,
    )
    .unwrap();
    assert!(plan.analysis.annihilator.is_none());
    assert!(matches!(
        plan.analysis
            .interior
            .as_ref()
            .map(|interior| &interior.kernel),
        Some(AnalysisKernel::Lattice(_))
    ));
}

fn assert_lattice_matches_direct(
    wavelet: &Wavelet,
    boundary: Boundary,
    signal: Vec<f64>,
    relative_tolerance: f64,
) {
    let mut accelerated =
        create_dwt_plan::<f64>(signal.len(), wavelet, boundary, SimdLevel::new()).unwrap();
    accelerated.analysis.annihilator = None;
    if lattice_simd_supported(SimdLevel::new()) {
        assert!(matches!(
            accelerated
                .analysis
                .interior
                .as_ref()
                .map(|interior| &interior.kernel),
            Some(AnalysisKernel::Lattice(_))
        ));
    }

    let mut direct =
        create_dwt_plan::<f64>(signal.len(), wavelet, boundary, SimdLevel::new()).unwrap();
    direct.analysis.annihilator = None;
    direct.analysis.interior.as_mut().unwrap().kernel = AnalysisKernel::Direct;

    let mut actual_approx = vec![0.0; accelerated.coeff_len];
    let mut actual_detail = vec![0.0; accelerated.coeff_len];
    accelerated.forward_into(&signal, &mut actual_approx, &mut actual_detail, &mut []);
    let mut expected_approx = vec![0.0; direct.coeff_len];
    let mut expected_detail = vec![0.0; direct.coeff_len];
    direct.forward_into(&signal, &mut expected_approx, &mut expected_detail, &mut []);

    let scale = expected_approx
        .iter()
        .chain(&expected_detail)
        .copied()
        .map(f64::abs)
        .fold(1.0, f64::max);
    let mut maximum_error = 0.0_f64;
    for (&actual, &expected) in actual_approx
        .iter()
        .chain(&actual_detail)
        .zip(expected_approx.iter().chain(&expected_detail))
    {
        assert!(actual.is_finite());
        maximum_error = maximum_error.max((actual - expected).abs());
    }
    assert!(
        maximum_error <= relative_tolerance * scale,
        "{} {boundary:?} maximum relative error {:.3e} exceeds {relative_tolerance:.3e}",
        wavelet.name(),
        maximum_error / scale,
    );
}

fn assert_annihilator_matches_direct<T: WaveletNum>(
    wavelet: &Wavelet,
    boundary: Boundary,
    signal: Vec<T>,
    close: impl Fn(T, T) -> bool,
) {
    let wavelet = equivalent_custom_wavelet(wavelet);
    let annihilator = annihilator_analysis(signal.len(), &wavelet, boundary);
    let mut direct =
        create_dwt_plan::<T>(signal.len(), &wavelet, boundary, SimdLevel::new()).unwrap();
    direct.analysis.annihilator = None;
    direct.analysis.interior.as_mut().unwrap().kernel = AnalysisKernel::Direct;

    let mut actual_approx = vec![T::zero(); direct.coeff_len];
    let mut actual_detail = vec![T::zero(); direct.coeff_len];
    annihilator.forward_into(&signal, &mut actual_approx, &mut actual_detail);
    let mut expected_approx = vec![T::zero(); direct.coeff_len];
    let mut expected_detail = vec![T::zero(); direct.coeff_len];
    direct.forward_into(&signal, &mut expected_approx, &mut expected_detail, &mut []);

    for (coefficient, (&actual, &expected)) in actual_approx
        .iter()
        .chain(&actual_detail)
        .zip(expected_approx.iter().chain(&expected_detail))
        .enumerate()
    {
        assert!(
            close(actual, expected),
            "{boundary:?} coefficient {coefficient}: {actual:?} != {expected:?}"
        );
    }
}

fn annihilator_analysis<T: WaveletNum>(
    signal_len: usize,
    wavelet: &Wavelet,
    boundary: Boundary,
) -> AnnihilatorAnalysis<T> {
    let filters = PreparedFilterBank::<T>::new(wavelet, boundary == Boundary::Periodization);
    let filter = filters
        .analysis_annihilator
        .expect("test wavelet must support annihilator analysis");
    AnnihilatorAnalysis::new(
        signal_len,
        coefficient_len(signal_len, wavelet.filter_len(), boundary),
        boundary,
        filter,
    )
}

fn equivalent_custom_wavelet(wavelet: &Wavelet) -> Wavelet {
    Wavelet::from_filters(
        wavelet.dec_lo(),
        wavelet.dec_hi(),
        wavelet.rec_lo(),
        wavelet.rec_hi(),
    )
    .unwrap()
}
