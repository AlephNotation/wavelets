#![no_main]

use libfuzzer_sys::fuzz_target;
use wavelets::{Boundary, DwtPlanner, WaveletError};
use wavelets_fuzz::{
    TransformCase, assert_reconstruction_f32, assert_reconstruction_f64, assert_same_f32,
    assert_same_f64,
};

fuzz_target!(|data: &[u8]| {
    let case = TransformCase::decode(data, 2);
    if case.use_f32 {
        run_f32(case);
    } else {
        run_f64(case);
    }
});

fn run_f64(case: TransformCase) {
    let mut planner = DwtPlanner::<f64>::new();
    let plan = planner.plan_dwt(case.samples.len(), &case.wavelet, case.boundary);
    if case.samples.is_empty() {
        assert!(matches!(plan, Err(WaveletError::EmptySignal)));
        return;
    }
    if case.samples.len() == 1 && matches!(case.boundary, Boundary::Reflect | Boundary::Antireflect)
    {
        assert!(matches!(
            plan,
            Err(WaveletError::BoundaryRequiresLongerSignal { .. })
        ));
        return;
    }

    let plan = plan.expect("normalized transform case is plannable");
    let (approx, detail) = plan.forward(&case.samples);
    let mut approx_into = vec![0.0; plan.coeff_len()];
    let mut detail_into = vec![0.0; plan.coeff_len()];
    let mut scratch = vec![0.0; plan.scratch_len()];
    plan.forward_into(
        &case.samples,
        &mut approx_into,
        &mut detail_into,
        &mut scratch,
    );
    assert_same_f64(&approx_into, &approx);
    assert_same_f64(&detail_into, &detail);

    let reconstructed = plan.inverse(&approx, &detail);
    let mut reconstructed_into = vec![0.0; plan.signal_len()];
    plan.inverse_into(&approx, &detail, &mut reconstructed_into, &mut scratch);
    assert_same_f64(&reconstructed_into, &reconstructed);
    assert_reconstruction_f64(&reconstructed, &case.samples);
}

fn run_f32(case: TransformCase) {
    let samples: Vec<_> = case.samples.iter().map(|&value| value as f32).collect();
    let mut planner = DwtPlanner::<f32>::new();
    let plan = planner.plan_dwt(samples.len(), &case.wavelet, case.boundary);
    if samples.is_empty() {
        assert!(matches!(plan, Err(WaveletError::EmptySignal)));
        return;
    }
    if samples.len() == 1 && matches!(case.boundary, Boundary::Reflect | Boundary::Antireflect) {
        assert!(matches!(
            plan,
            Err(WaveletError::BoundaryRequiresLongerSignal { .. })
        ));
        return;
    }

    let plan = plan.expect("normalized transform case is plannable");
    let (approx, detail) = plan.forward(&samples);
    let mut approx_into = vec![0.0; plan.coeff_len()];
    let mut detail_into = vec![0.0; plan.coeff_len()];
    let mut scratch = vec![0.0; plan.scratch_len()];
    plan.forward_into(&samples, &mut approx_into, &mut detail_into, &mut scratch);
    assert_same_f32(&approx_into, &approx);
    assert_same_f32(&detail_into, &detail);

    let reconstructed = plan.inverse(&approx, &detail);
    let mut reconstructed_into = vec![0.0; plan.signal_len()];
    plan.inverse_into(&approx, &detail, &mut reconstructed_into, &mut scratch);
    assert_same_f32(&reconstructed_into, &reconstructed);
    assert_reconstruction_f32(&reconstructed, &samples);
}
