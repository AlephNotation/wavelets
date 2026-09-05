use super::super::*;
use crate::DwtPlanner;

#[test]
fn butterfly_scales_are_checked_in_the_target_precision() {
    for value in [1e40, -1e40, f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
        for (low_scale, high_scale) in [(value, 0.5), (0.5, value)] {
            assert!(matches!(
                F64Butterfly {
                    low_scale,
                    high_scale
                }
                .prepare::<f32>(),
                Err(WaveletError::InvalidFilterBank(_))
            ));
        }
    }
    let scales = F64Butterfly {
        low_scale: 1e40,
        high_scale: -1e40,
    }
    .prepare::<f64>()
    .unwrap();
    assert_eq!(scales.low_scale, 1e40);
    assert_eq!(scales.high_scale, -1e40);
}

#[test]
fn butterfly_preserves_two_tap_fir_evaluation_order() {
    let signal = [0.0_f64, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let wavelet = Wavelet::haar();
    let mut planner = DwtPlanner::<f64>::new();
    let plan = planner
        .plan_dwt(signal.len(), &wavelet, Boundary::Symmetric)
        .unwrap();

    let (approx, detail) = plan.forward(&signal);
    let (sample_pairs, remainder) = signal.as_chunks::<2>();
    assert!(remainder.is_empty());
    for (output, samples) in sample_pairs.iter().enumerate() {
        let low = wavelet.dec_lo()[0] * samples[1] + wavelet.dec_lo()[1] * samples[0];
        let high = wavelet.dec_hi()[0] * samples[1] + wavelet.dec_hi()[1] * samples[0];
        assert_eq!(approx[output].to_bits(), low.to_bits());
        assert_eq!(detail[output].to_bits(), high.to_bits());
    }
}

#[test]
fn short_periodized_synthesis_folds_cyclic_filter_taps() {
    let wavelet = Wavelet::coiflet(5).unwrap();
    let plan =
        create_dwt_plan::<f32>(8, &wavelet, Boundary::Periodization, SimdLevel::new()).unwrap();
    let layout = plan.periodized_synthesis.as_ref().unwrap();
    let folded = layout.folded_filters.as_ref().unwrap();

    assert_eq!(plan.coeff_len, 4);
    assert_eq!(folded.len(), 4 * plan.coeff_len);
    let (rec_lo, rec_hi) = plan.synthesis_filters();
    assert_eq!(rec_lo.len(), 2 * plan.coeff_len);
    assert_eq!(rec_hi.len(), 2 * plan.coeff_len);

    let long_plan =
        create_dwt_plan::<f32>(64, &wavelet, Boundary::Periodization, SimdLevel::new()).unwrap();
    assert!(
        long_plan
            .periodized_synthesis
            .as_ref()
            .unwrap()
            .folded_filters
            .is_none()
    );
}

#[test]
fn butterfly_selection_depends_on_filter_algebra() {
    let wavelet =
        Wavelet::from_filters(&[0.5, 0.5], &[-0.25, 0.25], &[0.75, 0.75], &[0.125, -0.125])
            .unwrap();
    let plan =
        create_dwt_plan::<f64>(128, &wavelet, Boundary::Symmetric, SimdLevel::new()).unwrap();

    assert!(matches!(
        plan.analysis
            .interior
            .as_ref()
            .map(|interior| &interior.kernel),
        Some(AnalysisKernel::Butterfly {
            low_scale: 0.5,
            high_scale: 0.25,
        })
    ));
    assert!(matches!(
        plan.filters.synthesis_butterfly,
        Some(Butterfly {
            low_scale: 0.75,
            high_scale: 0.125,
        })
    ));
}

#[test]
fn f32_long_filter_smooth_round_trip_remains_stable() {
    let wavelet = Wavelet::daubechies(28).unwrap();
    let plan = create_dwt_plan::<f32>(4, &wavelet, Boundary::Smooth, SimdLevel::new()).unwrap();
    let signal = [0.595_544_7, 0.964_514_5, 0.653_177_1, 0.748_906_6];
    let (approx, detail) = plan.forward(&signal);
    let reconstructed = plan.inverse(&approx, &detail);
    let mean_squared_error = signal
        .iter()
        .zip(&reconstructed)
        .map(|(expected, actual)| (expected - actual).powi(2))
        .sum::<f32>()
        / signal.len() as f32;
    let rms = mean_squared_error.sqrt();

    assert!(rms < 6.0e-7, "f32 round-trip RMS was {rms:e}");
}
