use std::hint::black_box;
use std::mem::size_of;
use std::time::Duration;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use wavelets::{Boundary, DwtPlanner, Wavelet, WaveletNum};
use wavelets_benchmarks::signal;

fn criterion_config() -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3))
        .sample_size(40)
}

fn allocate_at_page_offset<T: WaveletNum>(len: usize, target: usize) -> (Vec<T>, usize) {
    const PAGE: usize = 4_096;

    let element_size = size_of::<T>();
    assert!(target < PAGE && target.is_multiple_of(element_size));
    let storage = vec![T::zero(); len + PAGE / element_size];
    let base = storage.as_ptr() as usize % PAGE;
    let start = (target + PAGE - base) % PAGE / element_size;
    (storage, start)
}

fn benchmark_axis_forward<T: WaveletNum>(criterion: &mut Criterion, precision: &str) {
    const SIGNAL_LEN: usize = 256;

    let mut group = criterion.benchmark_group(format!("axis_forward/{precision}"));
    group.throughput(Throughput::Elements((SIGNAL_LEN * 256) as u64));

    for order in [4, 38] {
        for (axis, outer, inner) in [("axis0", 1, 256), ("last-axis", 256, 1)] {
            for (layout, offsets) in [("page-aliased", [16, 16, 16]), ("separated", [16, 80, 144])]
            {
                let wavelet = Wavelet::daubechies(order).unwrap();
                let mut planner = DwtPlanner::<T>::new();
                let plan = planner
                    .plan_dwt(SIGNAL_LEN, &wavelet, Boundary::Symmetric)
                    .unwrap();
                let input_len = outer * SIGNAL_LEN * inner;
                let output_len = outer * plan.coeff_len() * inner;
                let (mut input, input_start) = allocate_at_page_offset(input_len, offsets[0]);
                input[input_start..input_start + input_len]
                    .copy_from_slice(&signal::<T>(input_len));
                let (mut approx, approx_start) = allocate_at_page_offset(output_len, offsets[1]);
                let (mut detail, detail_start) = allocate_at_page_offset(output_len, offsets[2]);
                let mut scratch = vec![T::zero(); plan.scratch_len()];

                group.bench_function(
                    format!("db{order}/symmetric/256x256/{axis}/{layout}"),
                    |bencher| {
                        bencher.iter(|| {
                            plan.forward_axis_into(
                                black_box(&input[input_start..input_start + input_len]),
                                outer,
                                inner,
                                &mut approx[approx_start..approx_start + output_len],
                                &mut detail[detail_start..detail_start + output_len],
                                &mut scratch,
                            );
                            black_box(&approx[approx_start..approx_start + output_len]);
                            black_box(&detail[detail_start..detail_start + output_len]);
                        });
                    },
                );
            }
        }
    }
    group.finish();
}

fn benchmark_axis_inverse<T: WaveletNum>(criterion: &mut Criterion, precision: &str) {
    const SIGNAL_LEN: usize = 16;
    const INNER: usize = 64 * 256;

    let mut group = criterion.benchmark_group(format!("axis_inverse/{precision}"));
    group.throughput(Throughput::Elements((SIGNAL_LEN * INNER) as u64));

    for (layout, offsets) in [("page-aliased", [16, 16, 16]), ("separated", [16, 80, 144])] {
        let wavelet = Wavelet::daubechies(38).unwrap();
        let mut planner = DwtPlanner::<T>::new();
        let plan = planner
            .plan_dwt(SIGNAL_LEN, &wavelet, Boundary::Symmetric)
            .unwrap();
        let coefficients = plan.coeff_len() * INNER;
        let output_len = SIGNAL_LEN * INNER;
        let input = signal::<T>(output_len);
        let (mut approx, approx_start) = allocate_at_page_offset(coefficients, offsets[0]);
        let (mut detail, detail_start) = allocate_at_page_offset(coefficients, offsets[1]);
        let (mut out, output_start) = allocate_at_page_offset(output_len, offsets[2]);
        let mut scratch = vec![T::zero(); plan.scratch_len()];

        plan.forward_axis_into(
            &input,
            1,
            INNER,
            &mut approx[approx_start..approx_start + coefficients],
            &mut detail[detail_start..detail_start + coefficients],
            &mut scratch,
        );

        group.bench_function(format!("db38/symmetric/16x64x256/{layout}"), |bencher| {
            bencher.iter(|| {
                plan.inverse_axis_into(
                    black_box(&approx[approx_start..approx_start + coefficients]),
                    black_box(&detail[detail_start..detail_start + coefficients]),
                    1,
                    INNER,
                    &mut out[output_start..output_start + output_len],
                    &mut scratch,
                );
                black_box(&out[output_start..output_start + output_len]);
            });
        });
    }
    group.finish();
}

fn axis(criterion: &mut Criterion) {
    benchmark_axis_forward::<f32>(criterion, "f32");
    benchmark_axis_forward::<f64>(criterion, "f64");
    benchmark_axis_inverse::<f32>(criterion, "f32");
    benchmark_axis_inverse::<f64>(criterion, "f64");
}

criterion_group! {
    name = benches;
    config = criterion_config();
    targets = axis
}
criterion_main!(benches);
