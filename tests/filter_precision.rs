use wavelets::{Boundary, DwtPlanner, Level, Wavelet, WaveletError, dwt, idwt, wavedec};

fn assert_invalid_filter<T>(result: Result<T, WaveletError>) {
    assert!(matches!(result, Err(WaveletError::InvalidFilterBank(_))));
}

#[test]
fn f32_planning_rejects_overflow_in_every_filter_and_tap() {
    let mut planner = DwtPlanner::<f32>::new();
    for length in [2, 4] {
        for bank in 0..4 {
            for tap in 0..length {
                for value in [1e40, -1e40, f64::MAX, -f64::MAX] {
                    let mut filters = vec![vec![0.25; length]; 4];
                    filters[bank][tap] = value;
                    let wavelet =
                        Wavelet::from_filters(&filters[0], &filters[1], &filters[2], &filters[3])
                            .unwrap();
                    for mode in [Boundary::Zero, Boundary::Symmetric, Boundary::Periodization] {
                        assert_invalid_filter(planner.plan_dwt(32, &wavelet, mode));
                        assert_invalid_filter(planner.plan_wavedec(32, &wavelet, mode, Level::Max));
                    }
                }
            }
        }
    }
    // Rejected plans do not prevent reuse of this planner for valid filters.
    assert!(
        planner
            .plan_dwt(32, &Wavelet::haar(), Boundary::Zero)
            .is_ok()
    );
}

#[test]
fn allocating_forward_inverse_and_multilevel_report_conversion_errors() {
    let wavelet =
        Wavelet::from_filters(&[1e40, 1e40], &[-1e40, 1e40], &[1e40, 1e40], &[1e40, -1e40])
            .unwrap();
    assert_invalid_filter(dwt(&[1.0_f32; 8], &wavelet, Boundary::Symmetric));
    assert_invalid_filter(idwt(
        &[1.0_f32; 4],
        &[1.0_f32; 4],
        &wavelet,
        Boundary::Symmetric,
    ));
    assert_invalid_filter(wavedec(
        &[1.0_f32; 8],
        &wavelet,
        Boundary::Symmetric,
        Level::Max,
    ));
}

#[test]
fn f32_accepts_finite_rounded_taps_including_limits_and_underflow() {
    let max = f64::from(f32::MAX);
    for value in [max, -max, max.next_up(), -max.next_up(), f64::from_bits(1)] {
        assert!((value as f32).is_finite());
        let wavelet =
            Wavelet::from_filters(&[value, 0.0], &[0.0, value], &[value, 0.0], &[0.0, value])
                .unwrap();
        let plan = DwtPlanner::<f32>::new()
            .plan_dwt(8, &wavelet, Boundary::Symmetric)
            .unwrap();
        let (approx, detail) = plan.forward(&[1e-40; 8]);
        assert!(approx.iter().chain(&detail).all(|value| value.is_finite()));
        assert!(
            plan.inverse(&[1e-40; 4], &[1e-40; 4])
                .iter()
                .all(|value| value.is_finite())
        );
    }
}

#[test]
fn f64_accepts_filters_outside_the_f32_range() {
    let wavelet =
        Wavelet::from_filters(&[1e40, 0.0], &[0.0, -1e40], &[1e40, 0.0], &[0.0, -1e40]).unwrap();
    let plan = DwtPlanner::<f64>::new()
        .plan_dwt(8, &wavelet, Boundary::Symmetric)
        .unwrap();
    let (approx, detail) = plan.forward(&[1.0; 8]);
    assert!(approx.iter().chain(&detail).all(|value| value.is_finite()));
    assert!(
        plan.inverse(&[1.0; 4], &[1.0; 4])
            .iter()
            .all(|value| value.is_finite())
    );
}
