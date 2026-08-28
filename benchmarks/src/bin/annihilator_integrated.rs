use std::hint::black_box;
use std::time::{Duration, Instant};

use wavelets::{Boundary, DwtPlanner, Wavelet, WaveletNum};

const SAMPLES: usize = 9;
const TARGET: Duration = Duration::from_millis(3);

#[derive(Clone, Copy)]
enum InputKind {
    Runs(usize),
    Dense,
}

impl InputKind {
    fn name(self) -> String {
        match self {
            Self::Runs(len) => format!("runs-{len}"),
            Self::Dense => "dense".to_owned(),
        }
    }
}

fn signal<T: WaveletNum>(len: usize, kind: InputKind) -> Vec<T> {
    match kind {
        InputKind::Runs(run_len) => (0..len)
            .map(|index| {
                let run = index / run_len;
                T::from_f64(1.0 + (run as f64 * 0.17).sin() + 0.1 * run as f64)
            })
            .collect(),
        InputKind::Dense => {
            let mut state = 0x9e37_79b9_7f4a_7c15_u64;
            (0..len)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    T::from_f64((state as i64) as f64 / i64::MAX as f64)
                })
                .collect()
        }
    }
}

fn calibrate(mut execute: impl FnMut()) -> usize {
    let mut iterations = 1;
    loop {
        let start = Instant::now();
        for _ in 0..iterations {
            execute();
        }
        if start.elapsed() >= TARGET || iterations >= 1 << 24 {
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

fn benchmark<T: WaveletNum>(precision: &str, all_boundaries: bool) {
    let boundaries: &[Boundary] = if all_boundaries {
        &[
            Boundary::Zero,
            Boundary::Constant,
            Boundary::Symmetric,
            Boundary::Reflect,
            Boundary::Periodic,
            Boundary::Smooth,
            Boundary::Antisymmetric,
            Boundary::Antireflect,
            Boundary::Periodization,
        ]
    } else {
        &[Boundary::Symmetric]
    };
    let lengths: &[usize] = if all_boundaries {
        &[4_096]
    } else {
        &[4_096, 262_144]
    };
    let kinds: &[InputKind] = if all_boundaries {
        &[
            InputKind::Runs(64),
            InputKind::Runs(4_096),
            InputKind::Dense,
        ]
    } else {
        &[
            InputKind::Runs(64),
            InputKind::Runs(256),
            InputKind::Runs(4_096),
            InputKind::Dense,
        ]
    };
    for wavelet_name in ["db38", "coif17"] {
        let wavelet = Wavelet::from_name(wavelet_name).unwrap();
        for &boundary in boundaries {
            for &len in lengths {
                let mut planner = DwtPlanner::<T>::new();
                let plan = planner.plan_dwt(len, &wavelet, boundary).unwrap();
                for &kind in kinds {
                    let signal = signal::<T>(len, kind);
                    let mut approx = vec![T::zero(); plan.coeff_len()];
                    let mut detail = vec![T::zero(); plan.coeff_len()];
                    let mut scratch = vec![T::zero(); plan.scratch_len()];
                    plan.forward_into(&signal, &mut approx, &mut detail, &mut scratch);
                    let iterations = calibrate(|| {
                        plan.forward_into(
                            black_box(&signal),
                            &mut approx,
                            &mut detail,
                            &mut scratch,
                        );
                        black_box((&approx, &detail));
                    });
                    let mut samples = Vec::with_capacity(SAMPLES);
                    for _ in 0..SAMPLES {
                        samples.push(measure(iterations, || {
                            plan.forward_into(
                                black_box(&signal),
                                &mut approx,
                                &mut detail,
                                &mut scratch,
                            );
                            black_box((&approx, &detail));
                        }));
                    }
                    println!(
                        "{precision},{wavelet_name},{},{len},{},{:.2}",
                        boundary.as_str(),
                        kind.name(),
                        median(&mut samples)
                    );
                }
            }
        }
    }
}

fn main() {
    let all_boundaries = std::env::args().any(|argument| argument == "--boundaries");
    println!("precision,wavelet,boundary,n,input,time_ns");
    benchmark::<f32>("f32", all_boundaries);
    benchmark::<f64>("f64", all_boundaries);
}
