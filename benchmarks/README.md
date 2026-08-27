# Performance tooling

This independent package has two deliberately separate jobs:

- Criterion measures uninstrumented latency and throughput with statistical
  sampling. These are the numbers used for regression tracking and published
  comparisons.
- Hotpath profiles representative execution to explain where time and
  allocations go. Its instrumentation changes the measured program, so its
  timings are diagnostic and are never compared with Criterion results.

The package follows the library's Rust 1.98 minimum. It remains isolated so
benchmark-only dependencies, features, and commands do not become part of the
published crate.

## Criterion

Run every benchmark from the repository root:

```text
cargo bench --manifest-path benchmarks/Cargo.toml
```

Run just the planning suite or a filtered transform case:

```text
cargo bench --manifest-path benchmarks/Cargo.toml --bench planning
cargo bench --manifest-path benchmarks/Cargo.toml --bench throughput -- db4/symmetric/4096
```

The throughput suite covers `f32` and `f64`, forward and inverse transforms,
allocation-free and allocating APIs, representative signal lengths, filter
orders, all boundary modes, and multilevel transforms. Planning has its own
suite so setup cost is not mixed into execution cost.

## Hotpath

The profiling driver uses public APIs with a 4,096-sample `f64` signal, db4,
and symmetric extension. The default run makes 1,000 calls to each measured
allocation-free and allocating operation:

```text
cargo run --release --manifest-path benchmarks/Cargo.toml \
  --bin profile --features hotpath
```

Enable allocation metrics and write a machine-readable report with:

```text
HOTPATH_OUTPUT_FORMAT=json \
HOTPATH_OUTPUT_PATH=benchmarks/reports/hotpath.json \
cargo run --release --manifest-path benchmarks/Cargo.toml \
  --bin profile --features hotpath-alloc
```

Set `WAVELETS_PROFILE_ITERATIONS` when a shorter smoke run or a longer profile
is useful. The `hotpath-cpu` feature is also forwarded for CPU sampling.

Cross-library PyWavelets and GSL runners belong beside these native tools, but
their results must record tool versions, compiler flags, CPU model, operating
system, and enabled ISA. GSL will necessarily be a labeled subset because its
DWT API is periodic, binary64, and power-of-two only.
