use std::hint::black_box;
use std::time::Duration;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use wavelets::{Boundary, DwtPlanner, Level};
use wavelets_benchmarks::wavelet;

fn criterion_config() -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2))
        .sample_size(30)
}

fn planning(criterion: &mut Criterion) {
    const LENGTH: usize = 4_096;

    let wavelet = wavelet(4);
    let mut group = criterion.benchmark_group("planning/f64");

    group.bench_function("single/cold/db4/symmetric/4096", |bencher| {
        bencher.iter_batched(
            DwtPlanner::<f64>::new,
            |mut planner| {
                black_box(
                    planner
                        .plan_dwt(LENGTH, &wavelet, Boundary::Symmetric)
                        .expect("benchmark case is valid"),
                )
            },
            BatchSize::SmallInput,
        );
    });

    let mut planner = DwtPlanner::<f64>::new();
    let live_single = planner
        .plan_dwt(LENGTH, &wavelet, Boundary::Symmetric)
        .expect("benchmark case is valid");
    group.bench_function("single/cache_hit/db4/symmetric/4096", |bencher| {
        bencher.iter(|| {
            black_box(
                planner
                    .plan_dwt(LENGTH, &wavelet, Boundary::Symmetric)
                    .expect("benchmark case is valid"),
            )
        });
    });
    black_box(&live_single);

    group.bench_function("multilevel/cold/db4/symmetric/4096", |bencher| {
        bencher.iter_batched(
            DwtPlanner::<f64>::new,
            |mut planner| {
                black_box(
                    planner
                        .plan_wavedec(LENGTH, &wavelet, Boundary::Symmetric, Level::Max)
                        .expect("benchmark case is valid"),
                )
            },
            BatchSize::SmallInput,
        );
    });

    let live_multilevel = planner
        .plan_wavedec(LENGTH, &wavelet, Boundary::Symmetric, Level::Max)
        .expect("benchmark case is valid");
    group.bench_function("multilevel/cache_hit/db4/symmetric/4096", |bencher| {
        bencher.iter(|| {
            black_box(
                planner
                    .plan_wavedec(LENGTH, &wavelet, Boundary::Symmetric, Level::Max)
                    .expect("benchmark case is valid"),
            )
        });
    });
    black_box(&live_multilevel);

    group.finish();
}

criterion_group! {
    name = benches;
    config = criterion_config();
    targets = planning
}
criterion_main!(benches);
