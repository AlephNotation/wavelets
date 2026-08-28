use std::hint::black_box;
use std::time::{Duration, Instant};

use wavelets::{Boundary, DwtPlanner, Wavelet};

const SAMPLES: usize = 9;
const CALIBRATION_TARGET: Duration = Duration::from_millis(3);

struct DegreeZeroPlan {
    signal_len: usize,
    coeff_len: usize,
    filter_len: usize,
    boundary: Boundary,
    phase: isize,
    first_extended_index: isize,
    low_base: f64,
    high_base: f64,
    low_correction: Box<[f64]>,
    high_correction: Box<[f64]>,
    factor_error: f64,
}

impl DegreeZeroPlan {
    fn new(signal_len: usize, coeff_len: usize, wavelet: &Wavelet, boundary: Boundary) -> Self {
        let low = wavelet.dec_lo();
        let high = wavelet.dec_hi();
        assert_eq!(low.len(), high.len());
        let filter_len = low.len();
        let phase = if boundary == Boundary::Periodization {
            (filter_len / 2) as isize
        } else {
            1
        };
        let first_extended_index = phase - (filter_len - 1) as isize;
        let (low_base, low_correction, low_error) = factor_degree_zero(low);
        let (high_base, high_correction, high_error) = factor_degree_zero(high);

        Self {
            signal_len,
            coeff_len,
            filter_len,
            boundary,
            phase,
            first_extended_index,
            low_base,
            high_base,
            low_correction: low_correction.into_boxed_slice(),
            high_correction: high_correction.into_boxed_slice(),
            factor_error: low_error.max(high_error),
        }
    }

    fn scratch_len(&self) -> usize {
        2 * self.coeff_len - 2 + self.filter_len
    }

    fn forward_into(
        &self,
        signal: &[f64],
        approx: &mut [f64],
        detail: &mut [f64],
        extended: &mut [f64],
    ) -> usize {
        assert_eq!(signal.len(), self.signal_len);
        assert_eq!(approx.len(), self.coeff_len);
        assert_eq!(detail.len(), self.coeff_len);
        assert_eq!(extended.len(), self.scratch_len());

        for (offset, sample) in extended.iter_mut().enumerate() {
            let index = self.first_extended_index + offset as isize;
            *sample = extended_sample(signal, index, self.boundary);
        }

        for coefficient in 0..self.coeff_len {
            let base_sample = extended[2 * coefficient];
            approx[coefficient] = self.low_base * base_sample;
            detail[coefficient] = self.high_base * base_sample;
        }

        let mut events = 0;
        for event_offset in 1..extended.len() {
            let amplitude = extended[event_offset] - extended[event_offset - 1];
            if amplitude == 0.0 {
                continue;
            }
            events += 1;
            let event = self.first_extended_index + event_offset as isize;
            self.scatter_event(event, amplitude, approx, detail);
        }
        events
    }

    fn forward_streaming_into(
        &self,
        signal: &[f64],
        approx: &mut [f64],
        detail: &mut [f64],
    ) -> usize {
        assert_eq!(signal.len(), self.signal_len);
        assert_eq!(approx.len(), self.coeff_len);
        assert_eq!(detail.len(), self.coeff_len);

        for coefficient in 0..self.coeff_len {
            let base_index = self.first_extended_index + 2 * coefficient as isize;
            let base_sample = extended_sample(signal, base_index, self.boundary);
            approx[coefficient] = self.low_base * base_sample;
            detail[coefficient] = self.high_base * base_sample;
        }

        let mut events = 0;
        let mut previous = extended_sample(signal, self.first_extended_index, self.boundary);
        for offset in 1..self.scratch_len() {
            let event = self.first_extended_index + offset as isize;
            let current = extended_sample(signal, event, self.boundary);
            let amplitude = current - previous;
            previous = current;
            if amplitude == 0.0 {
                continue;
            }
            events += 1;
            self.scatter_event(event, amplitude, approx, detail);
        }
        events
    }

    #[inline]
    fn scatter_event(&self, event: isize, amplitude: f64, approx: &mut [f64], detail: &mut [f64]) {
        let first_tap = (self.phase - event).rem_euclid(2) as usize;
        for tap in (first_tap..self.low_correction.len()).step_by(2) {
            let newest = event + tap as isize;
            let output_offset = newest - self.phase;
            if output_offset < 0 {
                continue;
            }
            let coefficient = output_offset as usize / 2;
            if coefficient >= self.coeff_len {
                break;
            }
            approx[coefficient] = amplitude.mul_add(self.low_correction[tap], approx[coefficient]);
            detail[coefficient] = amplitude.mul_add(self.high_correction[tap], detail[coefficient]);
        }
    }
}

fn factor_degree_zero(filter: &[f64]) -> (f64, Vec<f64>, f64) {
    assert!(filter.len() >= 2);
    let base = filter.iter().copied().sum::<f64>();
    let mut running = 0.0;
    let correction: Vec<_> = filter[..filter.len() - 1]
        .iter()
        .map(|&tap| {
            running += tap;
            running
        })
        .collect();

    let mut max_error = 0.0_f64;
    for (tap, &expected) in filter.iter().enumerate() {
        let previous = if tap == 0 { 0.0 } else { correction[tap - 1] };
        let current = correction.get(tap).copied().unwrap_or(0.0);
        let reconstructed = current - previous + if tap + 1 == filter.len() { base } else { 0.0 };
        max_error = max_error.max((reconstructed - expected).abs());
    }
    (base, correction, max_error)
}

fn extended_sample(signal: &[f64], index: isize, boundary: Boundary) -> f64 {
    if (0..signal.len() as isize).contains(&index) {
        return signal[index as usize];
    }

    match boundary {
        Boundary::Symmetric => {
            let period = 2 * signal.len();
            let wrapped = index.rem_euclid(period as isize) as usize;
            let reflected = if wrapped < signal.len() {
                wrapped
            } else {
                period - 1 - wrapped
            };
            signal[reflected]
        }
        Boundary::Periodization => {
            let periodic_len = signal.len() + signal.len() % 2;
            let wrapped = index.rem_euclid(periodic_len as isize) as usize;
            signal[wrapped.min(signal.len() - 1)]
        }
        _ => panic!("the experiment only implements symmetric and periodization boundaries"),
    }
}

enum InputKind {
    Runs(usize),
    Dense,
}

impl InputKind {
    fn name(&self) -> String {
        match self {
            Self::Runs(len) => format!("runs-{len}"),
            Self::Dense => "dense".to_owned(),
        }
    }
}

fn make_signal(len: usize, kind: &InputKind) -> Vec<f64> {
    match kind {
        InputKind::Runs(run_len) => {
            let mut signal = vec![0.0; len];
            for (run, samples) in signal.chunks_mut(*run_len).enumerate() {
                let value = 1.0 + (run as f64 * 0.17).sin() + 0.1 * run as f64;
                samples.fill(value);
            }
            signal
        }
        InputKind::Dense => {
            let mut state = 0x9e37_79b9_7f4a_7c15_u64;
            (0..len)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    (state as i64) as f64 / i64::MAX as f64
                })
                .collect()
        }
    }
}

fn prefer_annihilator(signal: &[f64]) -> bool {
    let maximum_events = signal.len() / 8;
    let mut events = 0;
    for samples in signal.windows(2) {
        events += usize::from(samples[0] != samples[1]);
        if events > maximum_events {
            return false;
        }
    }
    true
}

fn calibrate(mut execute: impl FnMut()) -> usize {
    let mut iterations = 1;
    loop {
        let start = Instant::now();
        for _ in 0..iterations {
            execute();
        }
        if start.elapsed() >= CALIBRATION_TARGET || iterations >= 1 << 24 {
            return iterations;
        }
        iterations *= 2;
    }
}

fn measure(iterations: usize, mut execute: impl FnMut()) -> f64 {
    let start = Instant::now();
    for _ in 0..iterations {
        execute();
    }
    start.elapsed().as_nanos() as f64 / iterations as f64
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn errors(
    expected_approx: &[f64],
    expected_detail: &[f64],
    actual_approx: &[f64],
    actual_detail: &[f64],
) -> (f64, f64) {
    let mut max_absolute = 0.0_f64;
    let mut squared_error = 0.0;
    let mut squared_reference = 0.0;
    for (&expected, &actual) in expected_approx
        .iter()
        .chain(expected_detail)
        .zip(actual_approx.iter().chain(actual_detail))
    {
        let error = actual - expected;
        max_absolute = max_absolute.max(error.abs());
        squared_error += error * error;
        squared_reference += expected * expected;
    }
    let relative_l2 = if squared_reference == 0.0 {
        if squared_error == 0.0 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        (squared_error / squared_reference).sqrt()
    };
    (max_absolute, relative_l2)
}

fn main() {
    let arguments: Vec<_> = std::env::args().collect();
    let full = arguments.iter().any(|argument| argument == "--full");
    let small = arguments.iter().any(|argument| argument == "--small");
    let lengths: &[usize] = if small {
        &[64, 256, 1_024, 4_096]
    } else if full {
        &[4_096, 16_384, 262_144, 1_048_576]
    } else {
        &[4_096, 262_144]
    };
    let wavelet_names: &[&str] = if full {
        &["db20", "db38", "coif17"]
    } else {
        &["db38", "coif17"]
    };
    let input_kinds = [
        InputKind::Runs(16),
        InputKind::Runs(64),
        InputKind::Runs(256),
        InputKind::Runs(1_024),
        InputKind::Runs(4_096),
        InputKind::Dense,
    ];
    let boundary = Boundary::Symmetric;

    println!(
        "wavelet,n,input,events,low_base,high_base,factor_error,max_abs_error,relative_l2_error,direct_ns,materialized_ns,streaming_ns,adaptive_ns,materialized_speedup,streaming_speedup,adaptive_speedup"
    );

    for &wavelet_name in wavelet_names {
        let wavelet = Wavelet::from_name(wavelet_name).expect("wavelet exists");
        for &len in lengths {
            let mut planner = DwtPlanner::<f64>::new();
            let direct = planner
                .plan_dwt(len, &wavelet, boundary)
                .expect("benchmark geometry is valid");
            let annihilator = DegreeZeroPlan::new(len, direct.coeff_len(), &wavelet, boundary);

            for kind in &input_kinds {
                let signal = make_signal(len, kind);
                let mut direct_approx = vec![0.0; direct.coeff_len()];
                let mut direct_detail = vec![0.0; direct.coeff_len()];
                let mut direct_scratch = vec![0.0; direct.scratch_len()];
                let mut annihilator_approx = vec![0.0; direct.coeff_len()];
                let mut annihilator_detail = vec![0.0; direct.coeff_len()];
                let mut annihilator_scratch = vec![0.0; annihilator.scratch_len()];
                let mut streaming_approx = vec![0.0; direct.coeff_len()];
                let mut streaming_detail = vec![0.0; direct.coeff_len()];
                let mut adaptive_approx = vec![0.0; direct.coeff_len()];
                let mut adaptive_detail = vec![0.0; direct.coeff_len()];
                let mut adaptive_direct_scratch = vec![0.0; direct.scratch_len()];

                direct.forward_into(
                    &signal,
                    &mut direct_approx,
                    &mut direct_detail,
                    &mut direct_scratch,
                );
                let events = annihilator.forward_into(
                    &signal,
                    &mut annihilator_approx,
                    &mut annihilator_detail,
                    &mut annihilator_scratch,
                );
                let (max_absolute, relative_l2) = errors(
                    &direct_approx,
                    &direct_detail,
                    &annihilator_approx,
                    &annihilator_detail,
                );

                let direct_iterations = calibrate(|| {
                    direct.forward_into(
                        black_box(&signal),
                        &mut direct_approx,
                        &mut direct_detail,
                        &mut direct_scratch,
                    );
                    black_box((&direct_approx, &direct_detail));
                });
                let annihilator_iterations = calibrate(|| {
                    black_box(annihilator.forward_into(
                        black_box(&signal),
                        &mut annihilator_approx,
                        &mut annihilator_detail,
                        &mut annihilator_scratch,
                    ));
                    black_box((&annihilator_approx, &annihilator_detail));
                });
                let streaming_iterations = calibrate(|| {
                    black_box(annihilator.forward_streaming_into(
                        black_box(&signal),
                        &mut streaming_approx,
                        &mut streaming_detail,
                    ));
                    black_box((&streaming_approx, &streaming_detail));
                });
                let adaptive_iterations = calibrate(|| {
                    if prefer_annihilator(black_box(&signal)) {
                        black_box(annihilator.forward_streaming_into(
                            black_box(&signal),
                            &mut adaptive_approx,
                            &mut adaptive_detail,
                        ));
                    } else {
                        direct.forward_into(
                            black_box(&signal),
                            &mut adaptive_approx,
                            &mut adaptive_detail,
                            &mut adaptive_direct_scratch,
                        );
                    }
                    black_box((&adaptive_approx, &adaptive_detail));
                });

                let mut direct_samples = Vec::with_capacity(SAMPLES);
                let mut annihilator_samples = Vec::with_capacity(SAMPLES);
                let mut streaming_samples = Vec::with_capacity(SAMPLES);
                let mut adaptive_samples = Vec::with_capacity(SAMPLES);
                for sample in 0..SAMPLES {
                    let mut run_direct = || {
                        measure(direct_iterations, || {
                            direct.forward_into(
                                black_box(&signal),
                                &mut direct_approx,
                                &mut direct_detail,
                                &mut direct_scratch,
                            );
                            black_box((&direct_approx, &direct_detail));
                        })
                    };
                    let mut run_annihilator = || {
                        measure(annihilator_iterations, || {
                            black_box(annihilator.forward_into(
                                black_box(&signal),
                                &mut annihilator_approx,
                                &mut annihilator_detail,
                                &mut annihilator_scratch,
                            ));
                            black_box((&annihilator_approx, &annihilator_detail));
                        })
                    };
                    let mut run_adaptive = || {
                        measure(adaptive_iterations, || {
                            if prefer_annihilator(black_box(&signal)) {
                                black_box(annihilator.forward_streaming_into(
                                    black_box(&signal),
                                    &mut adaptive_approx,
                                    &mut adaptive_detail,
                                ));
                            } else {
                                direct.forward_into(
                                    black_box(&signal),
                                    &mut adaptive_approx,
                                    &mut adaptive_detail,
                                    &mut adaptive_direct_scratch,
                                );
                            }
                            black_box((&adaptive_approx, &adaptive_detail));
                        })
                    };
                    let mut run_streaming = || {
                        measure(streaming_iterations, || {
                            black_box(annihilator.forward_streaming_into(
                                black_box(&signal),
                                &mut streaming_approx,
                                &mut streaming_detail,
                            ));
                            black_box((&streaming_approx, &streaming_detail));
                        })
                    };
                    match sample % 4 {
                        0 => {
                            direct_samples.push(run_direct());
                            annihilator_samples.push(run_annihilator());
                            streaming_samples.push(run_streaming());
                            adaptive_samples.push(run_adaptive());
                        }
                        1 => {
                            annihilator_samples.push(run_annihilator());
                            streaming_samples.push(run_streaming());
                            adaptive_samples.push(run_adaptive());
                            direct_samples.push(run_direct());
                        }
                        2 => {
                            streaming_samples.push(run_streaming());
                            adaptive_samples.push(run_adaptive());
                            direct_samples.push(run_direct());
                            annihilator_samples.push(run_annihilator());
                        }
                        _ => {
                            adaptive_samples.push(run_adaptive());
                            direct_samples.push(run_direct());
                            annihilator_samples.push(run_annihilator());
                            streaming_samples.push(run_streaming());
                        }
                    }
                }

                let direct_ns = median(&mut direct_samples);
                let annihilator_ns = median(&mut annihilator_samples);
                let streaming_ns = median(&mut streaming_samples);
                let adaptive_ns = median(&mut adaptive_samples);
                println!(
                    "{wavelet_name},{len},{},{events},{:.17e},{:.17e},{:.3e},{max_absolute:.3e},{relative_l2:.3e},{direct_ns:.2},{annihilator_ns:.2},{streaming_ns:.2},{adaptive_ns:.2},{:.3},{:.3},{:.3}",
                    kind.name(),
                    annihilator.low_base,
                    annihilator.high_base,
                    annihilator.factor_error,
                    direct_ns / annihilator_ns,
                    direct_ns / streaming_ns,
                    direct_ns / adaptive_ns,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn factorization_reconstructs_long_analysis_filters() {
        for name in ["db20", "db38", "coif17"] {
            let wavelet = Wavelet::from_name(name).unwrap();
            for filter in [wavelet.dec_lo(), wavelet.dec_hi()] {
                let (_, _, error) = factor_degree_zero(filter);
                assert!(error <= f64::EPSILON, "{name} factor error: {error:e}");
            }
        }
    }

    #[test]
    fn complete_split_matches_planned_transform() {
        for name in ["db20", "db38", "coif17"] {
            let wavelet = Wavelet::from_name(name).unwrap();
            for boundary in [Boundary::Symmetric, Boundary::Periodization] {
                for kind in [InputKind::Runs(64), InputKind::Dense] {
                    let signal = make_signal(4_096, &kind);
                    let mut planner = DwtPlanner::<f64>::new();
                    let direct = planner.plan_dwt(signal.len(), &wavelet, boundary).unwrap();
                    let split =
                        DegreeZeroPlan::new(signal.len(), direct.coeff_len(), &wavelet, boundary);
                    let mut expected_approx = vec![0.0; direct.coeff_len()];
                    let mut expected_detail = vec![0.0; direct.coeff_len()];
                    let mut direct_scratch = vec![0.0; direct.scratch_len()];
                    direct.forward_into(
                        &signal,
                        &mut expected_approx,
                        &mut expected_detail,
                        &mut direct_scratch,
                    );
                    let mut actual_approx = vec![0.0; direct.coeff_len()];
                    let mut actual_detail = vec![0.0; direct.coeff_len()];
                    let mut split_scratch = vec![0.0; split.scratch_len()];
                    split.forward_into(
                        &signal,
                        &mut actual_approx,
                        &mut actual_detail,
                        &mut split_scratch,
                    );
                    let (max_absolute, relative_l2) = errors(
                        &expected_approx,
                        &expected_detail,
                        &actual_approx,
                        &actual_detail,
                    );
                    assert!(
                        max_absolute <= 1e-13,
                        "{name} {boundary:?} {} max absolute error: {max_absolute:e}",
                        kind.name()
                    );
                    assert!(
                        relative_l2 <= 2e-15,
                        "{name} {boundary:?} {} relative L2 error: {relative_l2:e}",
                        kind.name()
                    );

                    actual_approx.fill(0.0);
                    actual_detail.fill(0.0);
                    split.forward_streaming_into(&signal, &mut actual_approx, &mut actual_detail);
                    let (max_absolute, relative_l2) = errors(
                        &expected_approx,
                        &expected_detail,
                        &actual_approx,
                        &actual_detail,
                    );
                    assert!(
                        max_absolute <= 1e-13,
                        "{name} {boundary:?} {} streaming max absolute error: {max_absolute:e}",
                        kind.name()
                    );
                    assert!(
                        relative_l2 <= 2e-15,
                        "{name} {boundary:?} {} streaming relative L2 error: {relative_l2:e}",
                        kind.name()
                    );
                }
            }
        }
    }
}
