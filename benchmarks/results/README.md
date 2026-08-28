# Published benchmark results

These are end-to-end transform execution measurements, not isolated inner-loop
claims. Input generation happens before every timer; each section below states
its exact planning and output-materialization boundaries. PyWavelets uses its
normal allocating Python API.

## Apple M4 Max / NEON

Measured on macOS 15.6 with an Apple M4 Max using the runtime-selected NEON
backend. Both reports use Rust 1.98's release profile with no additional
`RUSTFLAGS`, Python 3.14.6, NumPy 2.5.2, and PyWavelets distribution 1.9.0
(whose module reports 1.8.0).

### Same-interpreter Python API

This report was generated from clean commit
`a3eda32076a77fb21086564b380a53fc514a77a1`. Both implementations run in the
same CPython interpreter and receive the same NumPy inputs. The `planned` path
reuses a `wavelets-rs` plan. The deliberately conservative `cold` path creates
the canonical wavelet and plan inside every call, while PyWavelets receives a
preconstructed `pywt.Wavelet`. All paths create and destroy their outputs inside
the timed batch.

Signal length 4,096, db4, symmetric extension:

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

The planned binding wins all 92 canonical cases, ranging from 1.77x to 11.70x
with a 3.56x median. The cold path wins 68 of 92, ranging from 0.03x to 4.69x
with a 1.38x median. Its losses expose the intended plan-once tradeoff: filter
construction and boundary-row compilation can dominate execution for short or
long-filter transforms.

For 4,096-sample periodized multilevel Haar, the fused planned path takes 3.04
us forward versus PyWavelets' 21.99 us (7.24x), and 3.43 us inverse versus
16.56 us (4.83x).

#### Long-filter boundary stress

These f64 forward cases exercise compiled sparse edge rows together with the
NEON paraunitary lattice backend on long transform interiors:

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

At length 4,096, planned multilevel forward transforms range from 7.72x to
8.32x faster than PyWavelets across the same wavelets and boundaries.

#### Structured long-filter inputs

The structured suite measures the adaptive finite-difference executor without
giving it privileged inputs or timing boundaries: both engines receive the
same dense NumPy array, discover any structure during the call, allocate their
normal outputs, and destroy those outputs inside the timed batch. These are the
symmetric-extension cases at length 4,096:

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

Across all 24 symmetric and antireflect cases, including dense controls, the
planned binding wins every case by 5.75x to 20.95x with an 11.28x median. The
cold path also wins all 24, ranging from 1.40x to 4.06x with a 3.15x median.
Within the symmetric results, selecting the adaptive path for a constant input
reduces `wavelets-rs` planned time by 1.22x for f64 db38, 1.79x for f64
coif17, and 1.72x for f32 coif17 relative to each dense control.

Complete environment metadata, checksums, batch sizes, and all 6,720 raw
timing samples are in
[apple-m4-max-python-api.json](apple-m4-max-python-api.json).

### Native Rust API

The native report was generated from clean commit
`497327670c6881cec64c4e04913974dbddd83a80` (`wavelets` 0.1.0-alpha.4) and also
included GSL 2.8. Planning and wavelet construction are outside the timer. The
Rust `allocating` path allocates output on every call, while the Rust `into`
path reuses caller-owned output and scratch buffers.

Each number below is the median of 20 calibrated in-process samples targeting
10 ms apiece after three warmup batches. Checksums are validated between Rust's
two APIs and PyWavelets before a report is accepted. The complete 70-case matrix,
environment metadata, batch sizes, checksums, and all 216 sets of raw samples
are in [apple-m4-max-neon.json](apple-m4-max-neon.json).

#### Representative PyWavelets comparison

Signal length 4,096, db4, symmetric extension:

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

Across all 70 canonical cases, PyWavelets divided by Rust `into` ranged from
2.34x to 32.98x, with a median of 4.65x. Small-signal ratios include the normal
fixed cost of crossing PyWavelets' Python API, so the representative table above
is a better guide for sustained transform work.

#### Strictly comparable GSL subset

GSL's public 1D API always computes a complete transform. The shared semantic
subset is therefore full-depth `f64` Haar with periodic extension, corresponding
to `wavelets` db1 with periodization. No other GSL cases are compared.

| Length | Direction | Rust `into` | GSL | GSL / Rust |
| ---: | --- | ---: | ---: | ---: |
| 1,024 | forward | 479.22 ns | 2.34 us | 4.89x |
| 1,024 | inverse | 535.92 ns | 2.21 us | 4.13x |
| 4,096 | forward | 1.73 us | 9.25 us | 5.36x |
| 4,096 | inverse | 1.95 us | 9.16 us | 4.71x |
| 16,384 | forward | 8.76 us | 47.54 us | 5.43x |
| 16,384 | inverse | 7.51 us | 43.03 us | 5.73x |

## x86_64 / AVX2

Representative physical-hardware results are pending. GitHub-hosted runners
exercise the benchmark harness but are virtualized, shared, and intentionally
excluded from the published performance matrix.

## Reproduction

From the repository root, install the pinned Python requirements and GSL, then
run:

```text
python3 -m pip install -r benchmarks/requirements.txt
python3 benchmarks/compare/compare.py \
  --gsl \
  --output benchmarks/reports/comparison.json
```

Omit `--gsl` if GSL is unavailable. The benchmark runner records the source
revision and dirty state automatically.

Build the Python extension and reproduce the same-interpreter report with:

```text
python3 -m venv python/.venv
python/.venv/bin/python -m pip install -r python/requirements-dev.txt
(cd python && .venv/bin/maturin develop --release)
python/.venv/bin/python benchmarks/compare/python_api.py \
  --output benchmarks/reports/python-api.json
```
