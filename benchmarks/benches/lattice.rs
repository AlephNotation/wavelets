use std::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use wavelets::{Boundary, DwtPlanner, Wavelet};
use wavelets_benchmarks::signal;

fn criterion_config() -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3))
        .sample_size(40)
}

fn direct_equivalent(wavelet: &Wavelet) -> Wavelet {
    Wavelet::from_filters(
        wavelet.dec_lo(),
        wavelet.dec_hi(),
        wavelet.rec_lo(),
        wavelet.rec_hi(),
    )
    .expect("built-in filters form a valid custom bank")
}

fn benchmark_lattice(criterion: &mut Criterion) {
    const LENGTHS: [usize; 5] = [256, 512, 1_024, 4_096, 16_384];
    let mut group = criterion.benchmark_group("lattice/f64");

    for wavelet_name in ["db20", "sym20", "db38", "coif17"] {
        let wavelet = Wavelet::from_name(wavelet_name).unwrap();
        let direct_wavelet = direct_equivalent(&wavelet);
        for len in LENGTHS {
            let signal = signal::<f64>(len);
            for (executor, planned_wavelet) in
                [("direct", &direct_wavelet), ("automatic", &wavelet)]
            {
                let mut planner = DwtPlanner::<f64>::new();
                let plan = planner
                    .plan_dwt(len, planned_wavelet, Boundary::Symmetric)
                    .unwrap();
                let mut approx = vec![0.0; plan.coeff_len()];
                let mut detail = vec![0.0; plan.coeff_len()];
                let mut scratch = vec![0.0; plan.scratch_len()];

                group.bench_function(format!("{wavelet_name}/{len}/{executor}"), |bencher| {
                    bencher.iter(|| {
                        plan.forward_into(
                            black_box(&signal),
                            &mut approx,
                            &mut detail,
                            &mut scratch,
                        );
                        black_box((&approx, &detail));
                    });
                });
            }
        }
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = criterion_config();
    targets = benchmark_lattice
}
criterion_main!(benches);
