use wavelets::{Boundary, DwtPlanner, Level, Wavelet, wavedec, waverec};

fn assert_same_f64(actual: &[f64], expected: &[f64], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(actual.to_bits(), expected.to_bits(), "{context}[{index}]");
    }
}

fn assert_same_f32(actual: &[f32], expected: &[f32], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(actual.to_bits(), expected.to_bits(), "{context}[{index}]");
    }
}

fn run_f64(wavelet: &Wavelet, len: usize, boundary: Boundary, level: Level) {
    let context = format!("{} {boundary:?} len={len} {level:?}", wavelet.name());
    let signal: Vec<_> = (0..len)
        .map(|index| (index as f64 * 0.19).sin() + (index % 5) as f64)
        .collect();
    let mut planner = DwtPlanner::<f64>::new();
    let plan = planner.plan_wavedec(len, wavelet, boundary, level).unwrap();

    let expected = wavedec(&signal, wavelet, boundary, level).unwrap();
    let allocating = plan.forward(&signal);
    let mut reused = plan.allocate_decomposition();
    let mut scratch = vec![0.0; plan.scratch_len()];
    plan.forward_into(&signal, &mut reused, &mut scratch);
    assert_eq!(reused.as_slice().len(), plan.coeff_len(), "{context}");
    assert_same_f64(allocating.as_slice(), expected.as_slice(), &context);
    assert_same_f64(reused.as_slice(), expected.as_slice(), &context);

    let expected_signal = waverec(&expected).unwrap();
    assert_same_f64(&plan.inverse(&reused), &expected_signal, &context);
    let mut output = vec![0.0; len];
    plan.inverse_into(&reused, &mut output, &mut scratch);
    assert_same_f64(&output, &expected_signal, &context);

    let second_signal: Vec<_> = signal.iter().rev().copied().collect();
    let second_expected = plan.forward(&second_signal);
    plan.forward_into(&second_signal, &mut reused, &mut scratch);
    assert_same_f64(reused.as_slice(), second_expected.as_slice(), &context);
}

fn run_f32(wavelet: &Wavelet, len: usize, boundary: Boundary, level: Level) {
    let context = format!("{} {boundary:?} len={len} {level:?}", wavelet.name());
    let signal: Vec<_> = (0..len)
        .map(|index| (index as f32 * 0.19).sin() + (index % 5) as f32)
        .collect();
    let mut planner = DwtPlanner::<f32>::new();
    let plan = planner.plan_wavedec(len, wavelet, boundary, level).unwrap();

    let expected = wavedec(&signal, wavelet, boundary, level).unwrap();
    let allocating = plan.forward(&signal);
    let mut reused = plan.allocate_decomposition();
    let mut scratch = vec![0.0; plan.scratch_len()];
    plan.forward_into(&signal, &mut reused, &mut scratch);
    assert_eq!(reused.as_slice().len(), plan.coeff_len(), "{context}");
    assert_same_f32(allocating.as_slice(), expected.as_slice(), &context);
    assert_same_f32(reused.as_slice(), expected.as_slice(), &context);

    let expected_signal = waverec(&expected).unwrap();
    assert_same_f32(&plan.inverse(&reused), &expected_signal, &context);
    let mut output = vec![0.0; len];
    plan.inverse_into(&reused, &mut output, &mut scratch);
    assert_same_f32(&output, &expected_signal, &context);
}

#[test]
fn reusable_plan_matches_allocating_wrappers_at_every_scratch_depth() {
    let cases = [
        (Wavelet::haar(), 1, Boundary::Reflect, Level::Max),
        (Wavelet::haar(), 2, Boundary::Symmetric, Level::Max),
        (
            Wavelet::daubechies(2).unwrap(),
            12,
            Boundary::Symmetric,
            Level::Max,
        ),
        (
            Wavelet::daubechies(2).unwrap(),
            32,
            Boundary::Periodization,
            Level::Max,
        ),
        (
            Wavelet::daubechies(4).unwrap(),
            101,
            Boundary::Antireflect,
            Level::Max,
        ),
        (
            Wavelet::daubechies(38).unwrap(),
            4096,
            Boundary::Smooth,
            Level::Max,
        ),
    ];

    for (wavelet, len, boundary, level) in cases {
        run_f64(&wavelet, len, boundary, level);
        run_f32(&wavelet, len, boundary, level);
    }
}

#[test]
fn equivalent_filter_bank_is_compatible_with_plan() {
    let plan_wavelet = Wavelet::daubechies(4).unwrap();
    let decomposition_wavelet = Wavelet::daubechies(4).unwrap();
    let signal: Vec<_> = (0..128).map(f64::from).collect();
    let mut planner = DwtPlanner::<f64>::new();
    let plan = planner
        .plan_wavedec(128, &plan_wavelet, Boundary::Periodic, Level::Max)
        .unwrap();
    let decomposition = wavedec(
        &signal,
        &decomposition_wavelet,
        Boundary::Periodic,
        Level::Max,
    )
    .unwrap();
    let expected = waverec(&decomposition).unwrap();
    let reconstructed = plan.inverse(&decomposition);
    assert_same_f64(&reconstructed, &expected, "equivalent db4 filter banks");
}
