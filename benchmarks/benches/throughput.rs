use std::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use wavelets::{Boundary, DwtPlanner, Level, WaveletNum};
use wavelets_benchmarks::{Case, representative_cases, signal, wavelet};

fn criterion_config() -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2))
        .sample_size(30)
}

fn benchmark_single_level<T: WaveletNum>(criterion: &mut Criterion, precision: &str) {
    let mut group = criterion.benchmark_group(format!("single_level/{precision}"));

    for Case {
        len,
        order,
        boundary_name,
        boundary,
    } in representative_cases()
    {
        let wavelet = wavelet(order);
        let mut planner = DwtPlanner::<T>::new();
        let plan = planner
            .plan_dwt(len, &wavelet, boundary)
            .expect("benchmark case is valid");
        let signal = signal::<T>(len);
        let mut approx = vec![T::zero(); plan.coeff_len()];
        let mut detail = vec![T::zero(); plan.coeff_len()];
        let mut reconstructed = vec![T::zero(); plan.signal_len()];
        let mut scratch = vec![T::zero(); plan.scratch_len()];

        plan.forward_into(&signal, &mut approx, &mut detail, &mut scratch);
        group.throughput(Throughput::Elements(len as u64));

        group.bench_function(
            format!("forward/db{order}/{boundary_name}/{len}"),
            |bencher| {
                bencher.iter(|| {
                    plan.forward_into(black_box(&signal), &mut approx, &mut detail, &mut scratch);
                    black_box((&approx, &detail));
                });
            },
        );
        group.bench_function(
            format!("inverse/db{order}/{boundary_name}/{len}"),
            |bencher| {
                bencher.iter(|| {
                    plan.inverse_into(
                        black_box(&approx),
                        black_box(&detail),
                        &mut reconstructed,
                        &mut scratch,
                    );
                    black_box(&reconstructed);
                });
            },
        );
    }

    group.finish();
}

fn benchmark_multilevel<T: WaveletNum>(criterion: &mut Criterion, precision: &str) {
    const LENGTH: usize = 4_096;
    const BOUNDARIES: [(&str, Boundary); 3] = [
        ("symmetric", Boundary::Symmetric),
        ("antireflect", Boundary::Antireflect),
        ("periodization", Boundary::Periodization),
    ];

    let mut group = criterion.benchmark_group(format!("multilevel/{precision}"));
    for (boundary_name, boundary) in BOUNDARIES {
        let wavelet = wavelet(4);
        let mut planner = DwtPlanner::<T>::new();
        let plan = planner
            .plan_wavedec(LENGTH, &wavelet, boundary, Level::Max)
            .expect("benchmark case is valid");
        let signal = signal::<T>(LENGTH);
        let mut decomposition = plan.allocate_decomposition();
        let mut reconstructed = vec![T::zero(); LENGTH];
        let mut scratch = vec![T::zero(); plan.scratch_len()];

        plan.forward_into(&signal, &mut decomposition, &mut scratch);
        group.throughput(Throughput::Elements(LENGTH as u64));

        group.bench_function(format!("forward/db4/{boundary_name}/{LENGTH}"), |bencher| {
            bencher.iter(|| {
                plan.forward_into(black_box(&signal), &mut decomposition, &mut scratch);
                black_box(decomposition.as_slice());
            });
        });
        group.bench_function(format!("inverse/db4/{boundary_name}/{LENGTH}"), |bencher| {
            bencher.iter(|| {
                plan.inverse_into(black_box(&decomposition), &mut reconstructed, &mut scratch);
                black_box(&reconstructed);
            });
        });
    }

    group.finish();
}

fn benchmark_allocating<T: WaveletNum>(criterion: &mut Criterion, precision: &str) {
    const LENGTH: usize = 4_096;

    let wavelet = wavelet(4);
    let signal = signal::<T>(LENGTH);
    let mut planner = DwtPlanner::<T>::new();
    let single = planner
        .plan_dwt(LENGTH, &wavelet, Boundary::Symmetric)
        .expect("benchmark case is valid");
    let (approx, detail) = single.forward(&signal);
    let multilevel = planner
        .plan_wavedec(LENGTH, &wavelet, Boundary::Symmetric, Level::Max)
        .expect("benchmark case is valid");
    let decomposition = multilevel.forward(&signal);

    let mut group = criterion.benchmark_group(format!("allocating/{precision}"));
    group.throughput(Throughput::Elements(LENGTH as u64));
    group.bench_function("single_forward/db4/symmetric/4096", |bencher| {
        bencher.iter(|| black_box(single.forward(black_box(&signal))));
    });
    group.bench_function("single_inverse/db4/symmetric/4096", |bencher| {
        bencher.iter(|| black_box(single.inverse(black_box(&approx), black_box(&detail))));
    });
    group.bench_function("multilevel_forward/db4/symmetric/4096", |bencher| {
        bencher.iter(|| black_box(multilevel.forward(black_box(&signal))));
    });
    group.bench_function("multilevel_inverse/db4/symmetric/4096", |bencher| {
        bencher.iter(|| black_box(multilevel.inverse(black_box(&decomposition))));
    });
    group.finish();
}

fn throughput(criterion: &mut Criterion) {
    benchmark_single_level::<f32>(criterion, "f32");
    benchmark_single_level::<f64>(criterion, "f64");
    benchmark_multilevel::<f32>(criterion, "f32");
    benchmark_multilevel::<f64>(criterion, "f64");
    benchmark_allocating::<f32>(criterion, "f32");
    benchmark_allocating::<f64>(criterion, "f64");
}

criterion_group! {
    name = benches;
    config = criterion_config();
    targets = throughput
}
criterion_main!(benches);
