use std::hint::black_box;

use wavelets::{Boundary, Decomposition, Dwt, DwtPlanner, Level, WavedecPlan, Wavelet, WaveletNum};

const DEFAULT_SIGNAL_LEN: usize = 4_096;
const DEFAULT_ITERATIONS: usize = 1_000;

#[hotpath::measure]
fn single_forward_into<T: WaveletNum>(
    plan: &dyn Dwt<T>,
    signal: &[T],
    approx: &mut [T],
    detail: &mut [T],
    scratch: &mut [T],
) {
    plan.forward_into(signal, approx, detail, scratch);
}

#[hotpath::measure]
fn single_inverse_into<T: WaveletNum>(
    plan: &dyn Dwt<T>,
    approx: &[T],
    detail: &[T],
    reconstructed: &mut [T],
    scratch: &mut [T],
) {
    plan.inverse_into(approx, detail, reconstructed, scratch);
}

#[hotpath::measure]
fn multilevel_forward_into<T: WaveletNum>(
    plan: &WavedecPlan<T>,
    signal: &[T],
    decomposition: &mut Decomposition<T>,
    scratch: &mut [T],
) {
    plan.forward_into(signal, decomposition, scratch);
}

#[hotpath::measure]
fn multilevel_inverse_into<T: WaveletNum>(
    plan: &WavedecPlan<T>,
    decomposition: &Decomposition<T>,
    reconstructed: &mut [T],
    scratch: &mut [T],
) {
    plan.inverse_into(decomposition, reconstructed, scratch);
}

#[hotpath::measure]
fn single_forward_allocating<T: WaveletNum>(plan: &dyn Dwt<T>, signal: &[T]) -> (Vec<T>, Vec<T>) {
    plan.forward(signal)
}

#[hotpath::measure]
fn single_inverse_allocating<T: WaveletNum>(
    plan: &dyn Dwt<T>,
    approx: &[T],
    detail: &[T],
) -> Vec<T> {
    plan.inverse(approx, detail)
}

#[hotpath::measure]
fn multilevel_forward_allocating<T: WaveletNum>(
    plan: &WavedecPlan<T>,
    signal: &[T],
) -> Decomposition<T> {
    plan.forward(signal)
}

#[hotpath::measure]
fn multilevel_inverse_allocating<T: WaveletNum>(
    plan: &WavedecPlan<T>,
    decomposition: &Decomposition<T>,
) -> Vec<T> {
    plan.inverse(decomposition)
}

#[hotpath::main]
fn main() {
    let iterations = std::env::var("WAVELETS_PROFILE_ITERATIONS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_ITERATIONS);
    let signal_len = std::env::var("WAVELETS_PROFILE_LEN")
        .ok()
        .map(|value| value.parse().expect("profile length must be an integer"))
        .unwrap_or(DEFAULT_SIGNAL_LEN);
    let wavelet_name =
        std::env::var("WAVELETS_PROFILE_WAVELET").unwrap_or_else(|_| "db4".to_owned());
    let boundary_name =
        std::env::var("WAVELETS_PROFILE_BOUNDARY").unwrap_or_else(|_| "symmetric".to_owned());
    let precision =
        std::env::var("WAVELETS_PROFILE_PRECISION").unwrap_or_else(|_| "f64".to_owned());
    let wavelet: Wavelet = wavelet_name
        .parse()
        .expect("profile wavelet must be supported");
    let boundary: Boundary = boundary_name
        .parse()
        .expect("profile boundary must be supported");
    eprintln!(
        "profiling {precision}/{wavelet_name}/{boundary_name}/{signal_len} for {iterations} iterations"
    );

    match precision.as_str() {
        "f32" => profile::<f32>(signal_len, iterations, &wavelet, boundary),
        "f64" => profile::<f64>(signal_len, iterations, &wavelet, boundary),
        _ => panic!("profile precision must be f32 or f64"),
    }
}

fn profile<T: WaveletNum>(
    signal_len: usize,
    iterations: usize,
    wavelet: &Wavelet,
    boundary: Boundary,
) {
    let signal: Vec<_> = (0..signal_len)
        .map(|index| {
            let x = index as f64;
            T::from_f64((x * 0.013).sin() + 0.25 * (x * 0.071).cos() + (index % 17) as f64 / 17.0)
        })
        .collect();
    let mut planner = DwtPlanner::<T>::new();
    let single = planner
        .plan_dwt(signal_len, wavelet, boundary)
        .expect("profile case is valid");
    let multilevel = planner
        .plan_wavedec(signal_len, wavelet, boundary, Level::Max)
        .expect("profile case is valid");

    let mut approx = vec![T::zero(); single.coeff_len()];
    let mut detail = vec![T::zero(); single.coeff_len()];
    let mut single_output = vec![T::zero(); signal_len];
    let mut single_scratch = vec![T::zero(); single.scratch_len()];
    let mut decomposition = multilevel.allocate_decomposition();
    let mut multilevel_output = vec![T::zero(); signal_len];
    let mut multilevel_scratch = vec![T::zero(); multilevel.scratch_len()];

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
