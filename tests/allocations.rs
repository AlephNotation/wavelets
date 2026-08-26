#![cfg(feature = "count-allocations")]

use std::hint::black_box;

use wavelets::{Boundary, DwtPlanner, Level, Wavelet};

#[test]
fn planned_forward_and_inverse_do_not_allocate() {
    let wavelet = Wavelet::daubechies(38).unwrap();
    let mut planner = DwtPlanner::<f64>::new();
    let plan = planner
        .plan_dwt(4096, &wavelet, Boundary::Antireflect)
        .unwrap();
    let signal = vec![1.0; plan.signal_len()];
    let mut approx = vec![0.0; plan.coeff_len()];
    let mut detail = vec![0.0; plan.coeff_len()];
    let mut reconstructed = vec![0.0; plan.signal_len()];
    let mut scratch = vec![0.0; plan.scratch_len()];

    // Initialize the measurement crate's thread-local state before measuring.
    allocation_counter::measure(|| {});
    let allocations = allocation_counter::measure(|| {
        plan.forward_into(
            black_box(&signal),
            black_box(&mut approx),
            black_box(&mut detail),
            black_box(&mut scratch),
        );
        plan.inverse_into(
            black_box(&approx),
            black_box(&detail),
            black_box(&mut reconstructed),
            black_box(&mut scratch),
        );
    });

    assert_eq!(allocations.count_total, 0, "hot path allocated memory");
    assert_eq!(allocations.bytes_total, 0, "hot path allocated bytes");
}

#[test]
fn planned_multilevel_forward_and_inverse_do_not_allocate() {
    let wavelet = Wavelet::daubechies(8).unwrap();
    let mut planner = DwtPlanner::<f64>::new();
    let plan = planner
        .plan_wavedec(4096, &wavelet, Boundary::Antireflect, Level::Max)
        .unwrap();
    let signal = vec![1.0; plan.signal_len()];
    let mut decomposition = plan.allocate_decomposition();
    let mut reconstructed = vec![0.0; plan.signal_len()];
    let mut scratch = vec![0.0; plan.scratch_len()];

    allocation_counter::measure(|| {});
    let allocations = allocation_counter::measure(|| {
        plan.forward_into(
            black_box(&signal),
            black_box(&mut decomposition),
            black_box(&mut scratch),
        );
        plan.inverse_into(
            black_box(&decomposition),
            black_box(&mut reconstructed),
            black_box(&mut scratch),
        );
    });

    assert_eq!(allocations.count_total, 0, "multilevel hot path allocated");
    assert_eq!(allocations.bytes_total, 0, "multilevel hot path allocated");
}
