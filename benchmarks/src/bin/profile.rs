use std::hint::black_box;

use wavelets::{Boundary, Decomposition, Dwt, DwtPlanner, Level, WavedecPlan, Wavelet};

const SIGNAL_LEN: usize = 4_096;
const DEFAULT_ITERATIONS: usize = 1_000;

#[hotpath::measure]
fn single_forward_into(
    plan: &dyn Dwt<f64>,
    signal: &[f64],
    approx: &mut [f64],
    detail: &mut [f64],
    scratch: &mut [f64],
) {
    plan.forward_into(signal, approx, detail, scratch);
}

#[hotpath::measure]
fn single_inverse_into(
    plan: &dyn Dwt<f64>,
    approx: &[f64],
    detail: &[f64],
    reconstructed: &mut [f64],
    scratch: &mut [f64],
) {
    plan.inverse_into(approx, detail, reconstructed, scratch);
}

#[hotpath::measure]
fn multilevel_forward_into(
    plan: &WavedecPlan<f64>,
    signal: &[f64],
    decomposition: &mut Decomposition<f64>,
    scratch: &mut [f64],
) {
    plan.forward_into(signal, decomposition, scratch);
}

#[hotpath::measure]
fn multilevel_inverse_into(
    plan: &WavedecPlan<f64>,
    decomposition: &Decomposition<f64>,
    reconstructed: &mut [f64],
    scratch: &mut [f64],
) {
    plan.inverse_into(decomposition, reconstructed, scratch);
}

#[hotpath::measure]
fn single_forward_allocating(plan: &dyn Dwt<f64>, signal: &[f64]) -> (Vec<f64>, Vec<f64>) {
    plan.forward(signal)
}

#[hotpath::measure]
fn single_inverse_allocating(plan: &dyn Dwt<f64>, approx: &[f64], detail: &[f64]) -> Vec<f64> {
    plan.inverse(approx, detail)
}

#[hotpath::measure]
fn multilevel_forward_allocating(plan: &WavedecPlan<f64>, signal: &[f64]) -> Decomposition<f64> {
    plan.forward(signal)
}

#[hotpath::measure]
fn multilevel_inverse_allocating(
    plan: &WavedecPlan<f64>,
    decomposition: &Decomposition<f64>,
) -> Vec<f64> {
    plan.inverse(decomposition)
}

#[hotpath::main]
fn main() {
    let iterations = std::env::var("WAVELETS_PROFILE_ITERATIONS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_ITERATIONS);
    let signal: Vec<_> = (0..SIGNAL_LEN)
        .map(|index| {
            let x = index as f64;
            (x * 0.013).sin() + 0.25 * (x * 0.071).cos() + (index % 17) as f64 / 17.0
        })
        .collect();
    let wavelet = Wavelet::daubechies(4).expect("db4 is built in");
    let mut planner = DwtPlanner::<f64>::new();
    let single = planner
        .plan_dwt(SIGNAL_LEN, &wavelet, Boundary::Symmetric)
        .expect("profile case is valid");
    let multilevel = planner
        .plan_wavedec(SIGNAL_LEN, &wavelet, Boundary::Symmetric, Level::Max)
        .expect("profile case is valid");

    let mut approx = vec![0.0; single.coeff_len()];
    let mut detail = vec![0.0; single.coeff_len()];
    let mut single_output = vec![0.0; SIGNAL_LEN];
    let mut single_scratch = vec![0.0; single.scratch_len()];
    let mut decomposition = multilevel.allocate_decomposition();
    let mut multilevel_output = vec![0.0; SIGNAL_LEN];
    let mut multilevel_scratch = vec![0.0; multilevel.scratch_len()];

    for _ in 0..iterations {
        single_forward_into(
            single.as_ref(),
            &signal,
            &mut approx,
            &mut detail,
            &mut single_scratch,
        );
        single_inverse_into(
            single.as_ref(),
            &approx,
            &detail,
            &mut single_output,
            &mut single_scratch,
        );
        multilevel_forward_into(
            &multilevel,
            &signal,
            &mut decomposition,
            &mut multilevel_scratch,
        );
        multilevel_inverse_into(
            &multilevel,
            &decomposition,
            &mut multilevel_output,
            &mut multilevel_scratch,
        );

        let (allocated_approx, allocated_detail) =
            single_forward_allocating(single.as_ref(), &signal);
        let allocated_single_output =
            single_inverse_allocating(single.as_ref(), &allocated_approx, &allocated_detail);
        let allocated_decomposition = multilevel_forward_allocating(&multilevel, &signal);
        let allocated_multilevel_output =
            multilevel_inverse_allocating(&multilevel, &allocated_decomposition);

        black_box((
            &single_output,
            &multilevel_output,
            allocated_single_output,
            allocated_multilevel_output,
        ));
    }
}
