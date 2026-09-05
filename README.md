# wavelets

`wavelets` is a plan-once, execute-many Rust implementation of the
one-dimensional discrete wavelet transform. It follows PyWavelets' filter,
boundary, and coefficient conventions while providing reusable,
allocation-free SIMD execution.

The crate is pre-release and intentionally focused on 1D transforms. It
currently supports:

- single- and multilevel DWT/IDWT for `f32` and `f64`;
- Haar, `db1..db38`, `sym2..sym20`, `coif1..coif17`, and every PyWavelets
  `bior`/`rbio` filter bank;
- all nine PyWavelets boundary modes and custom filter banks;
- runtime-dispatched SIMD on x86 and aarch64; and
- reusable plans, caller-owned scratch space, and contiguous coefficient
  storage.

Built-in filters and every supported boundary are tested against PyWavelets,
alongside exhaustive reconstruction, orthogonality, and vanishing-moment
checks.

See [Migrating from PyWavelets](MIGRATING_FROM_PYWAVELETS.md) for API mappings
and compatibility details. Complete API documentation is available on
[docs.rs](https://docs.rs/wavelets).

## Quick start

```rust
use wavelets::{Boundary, DwtPlanner, Wavelet};

let signal = [1.0_f64, 2.0, 3.0, 4.0];
let wavelet: Wavelet = "db2".parse()?;
let mut planner = DwtPlanner::new();
let plan = planner.plan_dwt(signal.len(), &wavelet, Boundary::Symmetric)?;

let (approx, detail) = plan.forward(&signal);
let reconstructed = plan.inverse(&approx, &detail);

# Ok::<(), wavelets::WaveletError>(())
```

Use `forward_into` and `inverse_into` with preallocated buffers to remove
allocation from repeated transforms. The `dwt`, `idwt`, `wavedec`, and
`waverec` functions provide allocating convenience APIs for one-off work.

## Cargo features

The default build uses the direct FIR implementation, including SIMD,
boundary, axis-batching, and Haar-specialized kernels.

The opt-in `experimental-kernels` feature adds paraunitary lattice and
annihilator-split analysis backends:

```toml
wavelets = { version = "0.1.0-alpha.12", features = ["experimental-kernels"] }
```

## Performance

This representative same-interpreter benchmark measures a 4,096-sample `f64`
db4 forward DWT with symmetric extension. NumPy output allocation is included;
Rust planning is reused and outside the timer.

| Hardware | SIMD | `wavelets-rs` | PyWavelets | Speedup | 92-case median |
| --- | --- | ---: | ---: | ---: | ---: |
| AMD Ryzen 7 8745HS | AVX-512 | 2.13 µs | 10.55 µs | 4.95x | 4.82x |
| AMD EPYC 7R13 | AVX2/FMA | 4.09 µs | 14.99 µs | 3.67x | 3.96x–4.02x |
| Apple M4 Max | NEON | 2.58 µs | 9.15 µs | 3.55x | 3.56x |

The db4 rows use the direct FIR kernel. The published full-suite revisions may
also use experimental kernels where the wavelet, signal, and hardware qualify.
Every raw sample, environment description, absolute timing, and methodology is
published under [benchmarks/results](benchmarks/results/README.md).

The independent [benchmark package](benchmarks/README.md) contains Criterion,
profiling, cross-library, and same-interpreter runners without adding benchmark
dependencies to the library crate.

## Python evaluation package

The repository contains an unpublished `wavelets-rs` Python package for direct
same-interpreter comparisons with PyWavelets. It supports reusable plans over
contiguous NumPy `float32` and `float64` arrays and releases the GIL during
execution. See [python/README.md](python/README.md) for build and usage
instructions.

## Development

Run the default and experimental test configurations with:

```text
cargo test --all-targets
cargo test --all-targets --features experimental-kernels
```

Fuzz targets and checked-in seed corpora live in the independent `fuzz/`
workspace. Run `cargo +nightly fuzz list` to inspect the available targets.

## License

Licensed under the MIT license.
