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
- adaptive finite-difference analysis for structured signals with long filters,
  with automatic SIMD fallback for dense inputs;
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
| f64 | single forward | 2.70 us | 5.18 us | 9.78 us | 3.63x | 1.89x |
| f64 | single inverse | 2.60 us | 5.15 us | 6.30 us | 2.43x | 1.22x |
| f64 | multilevel forward | 7.56 us | 16.93 us | 33.01 us | 4.37x | 1.95x |
| f64 | multilevel inverse | 6.39 us | 15.50 us | 21.55 us | 3.37x | 1.39x |
| f32 | single forward | 1.57 us | 3.75 us | 9.50 us | 6.04x | 2.54x |
| f32 | single inverse | 1.44 us | 3.63 us | 6.23 us | 4.33x | 1.72x |
| f32 | multilevel forward | 4.24 us | 13.06 us | 32.77 us | 7.74x | 2.51x |
| f32 | multilevel inverse | 3.77 us | 12.29 us | 22.01 us | 5.84x | 1.79x |

The planned binding wins all 92 canonical cases: 1.80x to 11.13x, with a 3.63x
median. The deliberately conservative cold path wins 69 of 92 and has a 1.36x
median; filter construction and boundary-row compilation dominate its short
and long-filter losses.

For 4,096-sample periodized multilevel Haar, the fused planned path takes 3.13
us forward versus PyWavelets' 22.50 us (7.19x), and 3.81 us inverse versus
17.53 us (4.61x).

Long-filter f64 forward results make the compiled boundary-row benefit visible:

| Wavelet | Length | Boundary | `wavelets-rs` planned | `wavelets-rs` cold | PyWavelets | Py / planned |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| db38 | 16 | symmetric | 511 ns | 13.62 us | 3.50 us | 6.86x |
| db38 | 16 | antireflect | 541 ns | 23.91 us | 4.01 us | 7.41x |
| coif17 | 16 | symmetric | 594 ns | 21.91 us | 5.63 us | 9.48x |
| coif17 | 16 | antireflect | 587 ns | 40.59 us | 6.53 us | 11.13x |
| db38 | 4,096 | symmetric | 19.81 us | 41.30 us | 128.34 us | 6.48x |
| db38 | 4,096 | antireflect | 18.90 us | 46.04 us | 119.00 us | 6.30x |
| coif17 | 4,096 | symmetric | 27.44 us | 63.62 us | 203.67 us | 7.42x |
| coif17 | 4,096 | antireflect | 26.30 us | 71.92 us | 186.62 us | 7.10x |

At 4,096 samples, planned multilevel forward transforms are 5.69x to 6.34x
faster than PyWavelets across these four long-filter boundary cases.

Structured long-filter inputs expose the adaptive finite-difference backend.
These 4,096-sample symmetric forward transforms include output allocation and
exact structure discovery inside every timed call:

| Precision | Wavelet | Input | `wavelets-rs` planned | `wavelets-rs` cold | PyWavelets | Py / planned | Py / cold |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| f64 | db38 | dense control | 19.81 us | 41.30 us | 128.34 us | 6.48x | 3.11x |
| f64 | db38 | runs of 64 | 12.05 us | 33.13 us | 121.94 us | 10.12x | 3.68x |
| f64 | db38 | runs of 256 | 10.36 us | 31.71 us | 128.70 us | 12.43x | 4.06x |
| f64 | db38 | constant | 9.67 us | 30.79 us | 128.89 us | 13.32x | 4.19x |
| f64 | coif17 | dense control | 27.44 us | 63.62 us | 203.67 us | 7.42x | 3.20x |
| f64 | coif17 | runs of 64 | 12.49 us | 47.82 us | 188.26 us | 15.07x | 3.94x |
| f64 | coif17 | runs of 256 | 10.36 us | 46.00 us | 194.87 us | 18.81x | 4.24x |
| f64 | coif17 | constant | 9.91 us | 45.97 us | 206.23 us | 20.82x | 4.49x |
| f32 | coif17 | dense control | 15.13 us | 50.00 us | 89.80 us | 5.93x | 1.80x |
| f32 | coif17 | runs of 64 | 11.79 us | 47.06 us | 88.55 us | 7.51x | 1.88x |
| f32 | coif17 | runs of 256 | 9.60 us | 45.14 us | 88.48 us | 9.21x | 1.96x |
| f32 | coif17 | constant | 8.90 us | 44.50 us | 88.68 us | 9.96x | 1.99x |

Across all 24 symmetric and antireflect structured-suite cases—including the
dense controls—the planned binding wins by 5.93x to 20.82x with a 10.13x
median. The cold path also wins every case, with a 3.16x median. Within the
symmetric results, selecting the adaptive path for a constant input reduces
`wavelets-rs` planned time by 2.05x for f64 db38, 2.77x for f64 coif17, and
1.70x for f32 coif17 relative to each dense control.

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
