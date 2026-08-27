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
allocation-free and allocating APIs, representative signal lengths,
Daubechies, Symlet, Coiflet, biorthogonal, and reverse-biorthogonal filters, all
boundary modes, and multilevel transforms. Planning has its own suite so setup
cost is not mixed into execution cost.

## Hotpath

The profiling driver uses public APIs. It defaults to a 4,096-sample `f64`
signal, db4, symmetric extension, and 1,000 calls to each measured
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

Use `WAVELETS_PROFILE_PRECISION` (`f32` or `f64`), `WAVELETS_PROFILE_LEN`,
`WAVELETS_PROFILE_WAVELET`, and `WAVELETS_PROFILE_BOUNDARY` to select a
different structural case. Set `WAVELETS_PROFILE_ITERATIONS` when a shorter
smoke run or a longer profile is useful. The `hotpath-cpu` feature is also
forwarded for CPU sampling.

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

Curated release results are checked into `benchmarks/results/` with the raw
samples that produced each published table. Those reports identify the exact
Git revision and whether tracked source files were dirty during measurement.
GitHub-hosted runners are suitable for integration smoke tests, but their
timings are not published as representative hardware results.

See the [published results](results/README.md) for the current hardware matrix.

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

## Same-interpreter Python comparison

The Python binding comparison imports both `wavelets_rs` and PyWavelets into
the same CPython interpreter and passes the same NumPy inputs to each. This
removes the language-boundary asymmetry from the native comparison above.

Create the development environment and build the extension from the repository
root:

```text
python3 -m venv python/.venv
python/.venv/bin/python -m pip install -r python/requirements-dev.txt
(cd python && .venv/bin/maturin develop --release)
```

Run the complete canonical matrix:

```text
python/.venv/bin/python benchmarks/compare/python_api.py
```

Every case reports two `wavelets-rs` paths. `planned` reuses a plan, matching
the package's intended repeated-transform API. `cold` constructs the wavelet
and plan inside every timed call, providing a conservative bound for an
uncached backend integration. PyWavelets receives a preconstructed
`pywt.Wavelet`, as it would in a tuned application. All paths create their
NumPy outputs inside the timer, and inverse cases pass the same PyWavelets
coefficient arrays to both engines. The harness validates complete outputs
before timing, calibrates each engine independently, rotates their execution
order, and retains every raw sample in
`benchmarks/reports/python-api.json`.
