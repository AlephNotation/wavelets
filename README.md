# wavelets

`wavelets` is a plan-once, execute-many Rust implementation of the
one-dimensional discrete wavelet transform. Its boundary names, filter
orientation, and coefficient conventions follow PyWavelets, while reusable
plans provide allocation-free execution.

The crate is pre-release and intentionally narrow. The current correctness
slice includes:

- immutable, cheaply cloned filter banks;
- high-precision, reproducibly generated Haar, `db1..db38`, `sym2..sym20`,
  `coif1..coif17`, and all PyWavelets `bior`/`rbio` filters;
- all nine PyWavelets boundary modes;
- PyWavelets-style `dwt`/`idwt` convenience functions and canonical name parsing;
- fixed-length `f32` and `f64` DWT/IDWT plans;
- safe portable SIMD analysis and synthesis with planner-cached runtime dispatch;
- reusable allocation-free multilevel plans and contiguous decompositions; and
- exhaustive reconstruction, PyWavelets reference, orthogonality, and
  vanishing-moment tests.

See [Migrating from PyWavelets](MIGRATING_FROM_PYWAVELETS.md) for the direct
API mapping, repeated-transform pattern, and current compatibility boundary.
The complete public API documentation is available on
[docs.rs](https://docs.rs/wavelets).

## PyWavelets-style API

The allocating facade keeps one-off transforms close to their PyWavelets
equivalents while delegating to the same planned kernels:

```rust
use wavelets::{Boundary, Wavelet, dwt, idwt};

let signal = [1.0_f64, 2.0, 3.0, 4.0];
let wavelet: Wavelet = "db2".parse()?;
let boundary: Boundary = "symmetric".parse()?;
let (approx, detail) = dwt(&signal, &wavelet, boundary)?;
let reconstructed = idwt(&approx, &detail, &wavelet, boundary)?;

# Ok::<(), wavelets::WaveletError>(())
```

Move to `DwtPlanner` when the signal length and transform configuration repeat.

## Python bindings

The repository also contains an unpublished `wavelets-rs` Python package for
same-interpreter evaluation against PyWavelets. It exposes reusable plans over
contiguous NumPy `float32` and `float64` arrays and releases the GIL during
transform execution:

```python
import numpy as np
import wavelets_rs

signal = np.arange(4096, dtype=np.float64)
plan = wavelets_rs.plan_dwt(len(signal), "db4", mode="symmetric")
approx, detail = plan.forward(signal)
reconstructed = plan.inverse(approx, detail)
```

Build and test it with the instructions in [python/README.md](python/README.md).
The Python benchmark reports both reused-plan and cold plan-plus-execute
timings, so an upstream backend discussion does not depend on hidden setup
costs.

## Fuzzing

Fuzz targets live in the independent `fuzz/` workspace so nightly-only tooling
does not affect the library's MSRV. Install `cargo-fuzz`, then list or run the
targets with:

```text
cargo +nightly fuzz list
cargo +nightly fuzz run dwt_roundtrip -- -max_len=2051
cargo +nightly fuzz run wavedec_roundtrip -- -max_len=2051
cargo +nightly fuzz run custom_filter_bank -- -max_len=2051
```

Named seed corpora are checked in. New public subsystems should add a bounded
target using the shared decoders in `fuzz/src/lib.rs`; CI automatically builds
and smoke-tests every target returned by `cargo fuzz list` under its pinned
nightly toolchain.

## Performance

The repository's independent `benchmarks/` package keeps performance tooling
out of the library's dependency graph. Criterion measures uninstrumented
throughput and planning cost; an opt-in Hotpath driver profiles where
representative runs spend time and allocate. A neutral in-process sampling
harness compares both Rust execution APIs with PyWavelets and the genuinely
compatible subset of GSL.

Published Apple M4 Max/NEON results are available with every raw timing sample
and the exact source revision in
[benchmarks/results](benchmarks/results/README.md).

### Python-to-Python API

This comparison imports `wavelets_rs` and PyWavelets into the same CPython
interpreter and passes the same NumPy arrays to both. Output creation and
destruction are timed. `planned` reuses the intended `wavelets-rs` plan;
`cold` also charges canonical wavelet construction and planning to every call.
PyWavelets receives a preconstructed `pywt.Wavelet`.

Median end-to-end times for a 4,096-sample db4 transform with symmetric
extension:

| Precision | Transform | `wavelets-rs` planned | `wavelets-rs` cold | PyWavelets | Py / planned | Py / cold |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| f64 | single forward | 2.79 us | 4.40 us | 9.86 us | 3.54x | 2.24x |
| f64 | single inverse | 2.60 us | 4.16 us | 6.32 us | 2.43x | 1.52x |
| f64 | multilevel forward | 7.52 us | 11.13 us | 31.19 us | 4.15x | 2.80x |
| f64 | multilevel inverse | 6.17 us | 9.67 us | 21.43 us | 3.48x | 2.22x |
| f32 | single forward | 1.60 us | 3.09 us | 9.48 us | 5.91x | 3.07x |
| f32 | single inverse | 1.42 us | 2.96 us | 6.23 us | 4.39x | 2.11x |
| f32 | multilevel forward | 4.52 us | 7.74 us | 30.93 us | 6.84x | 4.00x |
| f32 | multilevel inverse | 3.67 us | 6.90 us | 21.84 us | 5.94x | 3.16x |

The planned binding wins all 70 canonical cases: 1.77x to 8.42x, with a 3.54x
median. The cold path wins 64 of 70 and has a 2.14x median; its six losses are
short f64 single-level transforms of at most 256 samples, where setup dominates.

### Native Rust API

These measurements use the lower-level native runner. Planning and input
generation are outside the timer; Rust reports both allocating and reusable
caller-buffer paths.

| Precision | Transform | Rust allocating | Rust `into` | PyWavelets | PyWavelets / `into` |
| --- | --- | ---: | ---: | ---: | ---: |
| f64 | single forward | 2.57 us | 2.24 us | 9.97 us | 4.45x |
| f64 | single inverse | 2.22 us | 1.94 us | 6.61 us | 3.40x |
| f64 | multilevel forward | 6.07 us | 5.11 us | 33.00 us | 6.46x |
| f64 | multilevel inverse | 4.68 us | 4.12 us | 21.86 us | 5.30x |
| f32 | single forward | 1.42 us | 1.23 us | 9.91 us | 8.04x |
| f32 | single inverse | 1.19 us | 998.62 ns | 6.57 us | 6.57x |
| f32 | multilevel forward | 3.35 us | 3.05 us | 32.47 us | 10.63x |
| f32 | multilevel inverse | 2.44 us | 2.15 us | 22.23 us | 10.33x |

Across the complete 70-case matrix, PyWavelets divided by Rust `into` ranges
from 2.34x to 32.98x, with a 4.65x median. Representative x86_64/AVX2 results
remain pending access to physical hardware; shared-runner timings are
deliberately not presented as authoritative results.

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

Reusable multilevel transforms allocate their coefficient and scratch storage
once:

```rust
use wavelets::{Boundary, DwtPlanner, Level, Wavelet};

let signal = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
let wavelet = Wavelet::haar();
let mut planner = DwtPlanner::new();
let plan = planner.plan_wavedec(
    signal.len(),
    &wavelet,
    Boundary::Symmetric,
    Level::Max,
)?;
let mut decomposition = plan.allocate_decomposition();
let mut reconstructed = vec![0.0; plan.signal_len()];
let mut scratch = vec![0.0; plan.scratch_len()];

plan.forward_into(&signal, &mut decomposition, &mut scratch);
plan.inverse_into(&decomposition, &mut reconstructed, &mut scratch);

# Ok::<(), wavelets::WaveletError>(())
```

## License

Licensed under the MIT license.
