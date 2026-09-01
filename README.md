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

Published Apple M4 Max/NEON, AMD Ryzen 7 8745HS/AVX-512, and AMD EPYC
7R13/AVX2 results are available with every raw timing sample and the exact
source revision in
[benchmarks/results](benchmarks/results/README.md).

<a id="python-to-python-api"></a>

### AMD Ryzen 7 8745HS / AVX-512

The x86 report compares both implementations in the same CPython 3.12
interpreter, using the same NumPy inputs and including output creation and
destruction. These are 4,096-sample f64 forward transforms with symmetric
extension:

| Wavelet | Transform | `wavelets-rs` planned | `wavelets-rs` cold | PyWavelets | Py / planned |
| --- | --- | ---: | ---: | ---: | ---: |
| db4 | single level | 2.13 us | 5.64 us | 10.55 us | 4.95x |
| db20 | single level | 5.36 us | 43.22 us | 61.48 us | 11.47x |
| db38 | single level | 9.78 us | 36.46 us | 123.10 us | 12.58x |
| coif17 | single level | 13.47 us | 124.02 us | 179.79 us | 13.34x |
| db38 | multilevel | 30.58 us | 155.66 us | 261.36 us | 8.55x |
| coif17 | multilevel | 45.84 us | 601.09 us | 386.16 us | 8.43x |

The planned binding wins all 92 canonical cases, ranging from 3.21x to 13.88x
with a 4.82x median. Across the 24-case long-filter structured suite it is
5.66x to 13.45x faster, with a 12.72x median.

The paired native diagnostic shows what the AVX-512 paraunitary lattice
executor contributes to supported long f64 analysis filters. At 4,096 samples
it is 1.55x–1.62x faster than the direct-equivalent production executor; at
262,144 samples the gain reaches 2.05x–2.64x. The planner keeps the direct path
for the measured 512- and 1,024-sample transforms and crosses over at 2,048.

Raw reports:
[Python API](benchmarks/results/amd-ryzen-7-8745hs-python-api.json) and
[native lattice crossover](benchmarks/results/amd-ryzen-7-8745hs-lattice.csv).

### AMD EPYC 7R13 / AVX2

The AVX2-only report was replicated in three independent processes on an AWS
`c6a.large`; the CPU exposed AVX2/FMA and no AVX-512. These are the median of
the three process medians for 4,096-sample symmetric f64 forward transforms:

| Wavelet | Transform | `wavelets-rs` reused plan | PyWavelets | Speedup |
| --- | --- | ---: | ---: | ---: |
| db4 | single level | 4.09 us | 14.99 us | 3.67x |
| db20 | single level | 14.90 us | 90.93 us | 6.10x |
| db38 | single level | 28.57 us | 177.61 us | 6.22x |
| coif17 | single level | 39.61 us | 267.89 us | 6.76x |
| db38 | multilevel | 69.45 us | 389.42 us | 5.61x |
| coif17 | multilevel | 101.07 us | 587.96 us | 5.82x |

Reused-plan execution wins all 92 canonical cases in every run, with a
3.96x–4.02x median speedup. The structured long-filter median is
7.21x–7.24x. All three complete reports are published:
[run 1](benchmarks/results/amd-epyc-7r13-avx2-python-api-run-1.json),
[run 2](benchmarks/results/amd-epyc-7r13-avx2-python-api-run-2.json), and
[run 3](benchmarks/results/amd-epyc-7r13-avx2-python-api-run-3.json).

### Apple M4 Max / NEON: Python-to-Python API

This comparison imports `wavelets_rs` and PyWavelets into the same CPython
interpreter and passes the same NumPy arrays to both. Output creation and
destruction are timed. `planned` reuses the intended `wavelets-rs` plan;
`cold` also charges canonical wavelet construction and planning to every call.
PyWavelets receives a preconstructed `pywt.Wavelet`.

Median end-to-end times for a 4,096-sample db4 transform with symmetric
extension:

| Precision | Transform | `wavelets-rs` planned | `wavelets-rs` cold | PyWavelets | Py / planned | Py / cold |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| f64 | single forward | 2.58 us | 4.84 us | 9.15 us | 3.55x | 1.89x |
| f64 | single inverse | 2.48 us | 4.68 us | 5.93 us | 2.39x | 1.27x |
| f64 | multilevel forward | 6.96 us | 15.79 us | 30.61 us | 4.40x | 1.94x |
| f64 | multilevel inverse | 6.13 us | 14.95 us | 20.60 us | 3.36x | 1.38x |
| f32 | single forward | 1.49 us | 3.53 us | 8.96 us | 6.03x | 2.54x |
| f32 | single inverse | 1.39 us | 3.46 us | 5.89 us | 4.25x | 1.71x |
| f32 | multilevel forward | 3.94 us | 12.21 us | 30.34 us | 7.70x | 2.48x |
| f32 | multilevel inverse | 3.59 us | 11.82 us | 21.08 us | 5.87x | 1.78x |

The planned binding wins all 92 canonical cases: 1.77x to 11.70x, with a 3.56x
median. The deliberately conservative cold path wins 68 of 92 and has a 1.38x
median; filter construction and boundary-row compilation dominate its short
and long-filter losses.

For 4,096-sample periodized multilevel Haar, the fused planned path takes 3.04
us forward versus PyWavelets' 21.99 us (7.24x), and 3.43 us inverse versus
16.56 us (4.83x).

Long-filter f64 forward results show the compiled boundary rows together with
the NEON paraunitary lattice backend:

| Wavelet | Length | Boundary | `wavelets-rs` planned | `wavelets-rs` cold | PyWavelets | Py / planned |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| db38 | 16 | symmetric | 510 ns | 13.49 us | 3.34 us | 6.54x |
| db38 | 16 | antireflect | 524 ns | 23.70 us | 3.73 us | 7.12x |
| coif17 | 16 | symmetric | 601 ns | 21.78 us | 5.32 us | 8.86x |
| coif17 | 16 | antireflect | 593 ns | 41.54 us | 6.23 us | 10.49x |
| db38 | 4,096 | symmetric | 10.33 us | 30.21 us | 112.33 us | 10.88x |
| db38 | 4,096 | antireflect | 10.52 us | 36.26 us | 112.52 us | 10.70x |
| coif17 | 4,096 | symmetric | 15.37 us | 50.49 us | 179.56 us | 11.68x |
| coif17 | 4,096 | antireflect | 15.40 us | 60.25 us | 180.06 us | 11.70x |

At 4,096 samples, planned multilevel forward transforms are 7.72x to 8.32x
faster than PyWavelets across these four long-filter boundary cases.

Structured long-filter inputs expose the adaptive finite-difference backend.
These 4,096-sample symmetric forward transforms include output allocation and
exact structure discovery inside every timed call:

| Precision | Wavelet | Input | `wavelets-rs` planned | `wavelets-rs` cold | PyWavelets | Py / planned | Py / cold |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| f64 | db38 | dense control | 10.33 us | 30.21 us | 112.33 us | 10.88x | 3.72x |
| f64 | db38 | runs of 64 | 10.79 us | 31.84 us | 112.74 us | 10.45x | 3.54x |
| f64 | db38 | runs of 256 | 9.05 us | 29.98 us | 113.22 us | 12.51x | 3.78x |
| f64 | db38 | constant | 8.45 us | 29.47 us | 113.75 us | 13.46x | 3.86x |
| f64 | coif17 | dense control | 15.37 us | 50.49 us | 179.56 us | 11.68x | 3.56x |
| f64 | coif17 | runs of 64 | 11.52 us | 47.26 us | 180.31 us | 15.65x | 3.82x |
| f64 | coif17 | runs of 256 | 9.30 us | 45.37 us | 180.12 us | 19.37x | 3.97x |
| f64 | coif17 | constant | 8.59 us | 44.29 us | 180.00 us | 20.95x | 4.06x |
| f32 | coif17 | dense control | 14.71 us | 49.38 us | 84.66 us | 5.75x | 1.71x |
| f32 | coif17 | runs of 64 | 11.36 us | 46.43 us | 84.41 us | 7.43x | 1.82x |
| f32 | coif17 | runs of 256 | 9.22 us | 44.18 us | 84.88 us | 9.21x | 1.92x |
| f32 | coif17 | constant | 8.57 us | 43.62 us | 84.31 us | 9.84x | 1.93x |

Across all 24 symmetric and antireflect structured-suite cases—including the
dense controls—the planned binding wins by 5.75x to 20.95x with an 11.28x
median. The cold path also wins every case, with a 3.15x median. Within the
symmetric results, selecting the adaptive path for a constant input reduces
`wavelets-rs` planned time by 1.22x for f64 db38, 1.79x for f64 coif17, and
1.72x for f32 coif17 relative to each dense control.

### Apple M4 Max / NEON: Native Rust API

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
from 2.34x to 32.98x, with a 4.65x median.

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
