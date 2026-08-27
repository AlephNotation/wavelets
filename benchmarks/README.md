# Performance tooling

This independent package has three deliberately separate jobs:

- Criterion measures uninstrumented latency and throughput with statistical
  sampling. These are the numbers used for regression tracking and published
  comparisons.
- Hotpath profiles representative execution to explain where time and
  allocations go. Its instrumentation changes the measured program, so its
  timings are diagnostic and are never compared with Criterion results.
- The cross-library runner measures `wavelets`, PyWavelets, and a compatible
  GSL subset with the same calibrated in-process sampling protocol.

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

## Cross-library comparison

Install the pinned comparison environment, then run the canonical matrix:

```text
python3 -m pip install -r benchmarks/requirements.txt
python3 benchmarks/compare/compare.py
```

Each timed case performs its own warmup and calibrated in-process batches.
Subprocess startup, Rust planning, wavelet construction, and input generation
are outside the timed samples. Rust reports both its allocating and
reusable-buffer APIs; PyWavelets reports its normal allocating API. Raw samples
and complete environment metadata are written to
`benchmarks/reports/comparison.json`.

Use the general sampling and case filters for a short integration run:

```text
python3 benchmarks/compare/compare.py \
  --samples 3 --sample-ms 1 --warmup-batches 1 \
  --filter multilevel/forward/f64/db1/periodization/1024
```

Install GSL and pass `--gsl` to include it. The direct comparison is
deliberately restricted to Haar, `f64`, GSL periodic/`wavelets` periodization,
and power-of-two multilevel transforms. GSL's public 1D API always performs the
complete transform; Haar is the current intersection where
`wavelets::Level::Max` performs the same number of levels. Comparing longer GSL
filters would time different work and is therefore rejected by the harness.

```text
# Debian/Ubuntu: apt-get install libgsl-dev
# macOS:         brew install gsl
python3 benchmarks/compare/compare.py --gsl
```

The displayed `x` values are ratios, not percentages. For example, an `into x`
of `2.00x` means PyWavelets took twice as long as the Rust reusable-buffer API.
`GSL/Rust` is the analogous GSL-to-Rust reusable-buffer ratio.
Reports record installed distribution and module-reported versions, compiler
flags, CPU model, operating system, enabled ISA, batch sizes, checksums, and
every timing sample.
