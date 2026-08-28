use std::hint::black_box;
use std::time::{Duration, Instant};

use wavelets::{Boundary, Dwt, DwtPlanner, Wavelet};
use wavelets_benchmarks::signal;

const SAMPLE_COUNT: usize = 41;
const SAMPLE_TARGET: Duration = Duration::from_millis(5);
const MAX_BATCH_ITERATIONS: usize = 100_000_000;

struct Executor {
    plan: std::sync::Arc<dyn Dwt<f64>>,
    approx: Vec<f64>,
    detail: Vec<f64>,
    scratch: Vec<f64>,
}

impl Executor {
    fn new(len: usize, wavelet: &Wavelet) -> Self {
        let mut planner = DwtPlanner::<f64>::new();
        let plan = planner.plan_dwt(len, wavelet, Boundary::Symmetric).unwrap();
        Self {
            approx: vec![0.0; plan.coeff_len()],
            detail: vec![0.0; plan.coeff_len()],
            scratch: vec![0.0; plan.scratch_len()],
            plan,
        }
    }

    fn execute(&mut self, signal: &[f64]) {
        self.plan.forward_into(
            black_box(signal),
            &mut self.approx,
            &mut self.detail,
            &mut self.scratch,
        );
        black_box((&self.approx, &self.detail));
    }
}

fn direct_equivalent(wavelet: &Wavelet) -> Wavelet {
    Wavelet::from_filters(
        wavelet.dec_lo(),
        wavelet.dec_hi(),
        wavelet.rec_lo(),
        wavelet.rec_hi(),
    )
    .unwrap()
}

fn run_batch(executor: &mut Executor, signal: &[f64], iterations: usize) -> Duration {
    let start = Instant::now();
    for _ in 0..iterations {
        executor.execute(signal);
    }
    start.elapsed()
}

fn calibrate(executor: &mut Executor, signal: &[f64]) -> usize {
    let mut iterations = 1;
    loop {
        let elapsed = run_batch(executor, signal, iterations);
        if elapsed >= SAMPLE_TARGET || iterations == MAX_BATCH_ITERATIONS {
            return iterations;
        }
        let ratio = SAMPLE_TARGET.as_secs_f64() / elapsed.as_secs_f64().max(1.0e-9);
        let growth = ratio.ceil().clamp(2.0, 10.0) as usize;
        iterations = iterations.saturating_mul(growth).min(MAX_BATCH_ITERATIONS);
    }
}

fn median(samples: &mut [f64]) -> f64 {
    samples.sort_unstable_by(f64::total_cmp);
    samples[samples.len() / 2]
}

fn main() {
    println!("wavelet,length,direct_ns,automatic_ns,speedup");
    for wavelet_name in ["db20", "sym20", "db38", "coif17"] {
        let wavelet = Wavelet::from_name(wavelet_name).unwrap();
        let direct_wavelet = direct_equivalent(&wavelet);
        for len in [512, 1_024, 2_048, 4_096, 16_384] {
            let signal = signal::<f64>(len);
            let mut direct = Executor::new(len, &direct_wavelet);
            let mut automatic = Executor::new(len, &wavelet);
            let iterations = calibrate(&mut direct, &signal);
            run_batch(&mut direct, &signal, iterations);
            run_batch(&mut automatic, &signal, iterations);

            let mut direct_samples = Vec::with_capacity(SAMPLE_COUNT);
            let mut automatic_samples = Vec::with_capacity(SAMPLE_COUNT);
            for sample in 0..SAMPLE_COUNT {
                let (direct_elapsed, automatic_elapsed) = if sample.is_multiple_of(2) {
                    (
                        run_batch(&mut direct, &signal, iterations),
                        run_batch(&mut automatic, &signal, iterations),
                    )
                } else {
                    let automatic_elapsed = run_batch(&mut automatic, &signal, iterations);
                    let direct_elapsed = run_batch(&mut direct, &signal, iterations);
                    (direct_elapsed, automatic_elapsed)
                };
                direct_samples.push(direct_elapsed.as_nanos() as f64 / iterations as f64);
                automatic_samples.push(automatic_elapsed.as_nanos() as f64 / iterations as f64);
            }

            let direct_ns = median(&mut direct_samples);
            let automatic_ns = median(&mut automatic_samples);
            println!(
                "{wavelet_name},{len},{direct_ns:.3},{automatic_ns:.3},{:.3}",
                direct_ns / automatic_ns
            );
        }
    }
}
