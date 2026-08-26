# wavelets

`wavelets` is a plan-once, execute-many Rust implementation of the
one-dimensional discrete wavelet transform. Its boundary names, filter
orientation, and coefficient conventions follow PyWavelets, while reusable
plans provide allocation-free execution.

The crate is pre-release and intentionally narrow. The current correctness
slice includes:

- immutable, cheaply cloned filter banks;
- high-precision, reproducibly generated Haar and `db1..db38` filters;
- all nine PyWavelets boundary modes;
- fixed-length `f32` and `f64` DWT/IDWT plans;
- contiguous multilevel decompositions; and
- exhaustive reconstruction, PyWavelets reference, orthogonality, and
  vanishing-moment tests.

The remaining built-in families, reusable multilevel plans, SIMD kernels, and
benchmarks remain before the first beta.

The implementation invariants and deliberate departures from the initial
sketch are recorded in [DESIGN.md](DESIGN.md).

## Fuzzing

Fuzz targets live in the independent `fuzz/` workspace so nightly-only tooling
does not affect the library's MSRV. Install `cargo-fuzz`, then list or run the
targets with:

```text
cargo fuzz list
cargo fuzz run dwt_roundtrip -- -max_len=2051
cargo fuzz run wavedec_roundtrip -- -max_len=2051
cargo fuzz run custom_filter_bank -- -max_len=2051
```

Named seed corpora are checked in. New public subsystems should add a bounded
target using the shared decoders in `fuzz/src/lib.rs`; CI automatically builds
and smoke-tests every target returned by `cargo fuzz list`.

```rust
use wavelets::{Boundary, DwtPlanner, Wavelet};

let signal = [1.0_f64, 2.0, 3.0, 4.0];
let wavelet = Wavelet::haar();
let mut planner = DwtPlanner::new();
let plan = planner.plan_dwt(signal.len(), &wavelet, Boundary::Symmetric)?;

let (approx, detail) = plan.forward(&signal);
let reconstructed = plan.inverse(&approx, &detail);
assert!(reconstructed
    .iter()
    .zip(signal)
    .all(|(actual, expected)| (actual - expected).abs() < 1e-12));

# Ok::<(), wavelets::WaveletError>(())
```

## License

Licensed under the MIT license.
