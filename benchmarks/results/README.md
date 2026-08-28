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
`90545e854d1aca37b2bfcfff54a10f5fe24d0baa`. Both implementations run in the
same CPython interpreter and receive the same NumPy inputs. The `planned` path
reuses a `wavelets-rs` plan. The deliberately conservative `cold` path creates
the canonical wavelet and plan inside every call, while PyWavelets receives a
preconstructed `pywt.Wavelet`. All paths create and destroy their outputs inside
the timed batch.

Signal length 4,096, db4, symmetric extension:

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

The planned binding wins all 92 canonical cases, ranging from 1.80x to 11.13x
with a 3.63x median. The cold path wins 69 of 92, ranging from 0.03x to 4.73x
with a 1.36x median. Its losses expose the intended plan-once tradeoff: filter
construction and boundary-row compilation can dominate execution for short or
long-filter transforms.

For 4,096-sample periodized multilevel Haar, the fused planned path takes 3.13
us forward versus PyWavelets' 22.50 us (7.19x), and 3.81 us inverse versus
17.53 us (4.61x).

#### Long-filter boundary stress

These f64 forward cases exercise the filters and boundaries most affected by
compiled sparse edge rows:

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

At length 4,096, planned multilevel forward transforms range from 5.69x to
6.34x faster than PyWavelets across the same wavelets and boundaries.

#### Structured long-filter inputs

The structured suite measures the adaptive finite-difference executor without
giving it privileged inputs or timing boundaries: both engines receive the
same dense NumPy array, discover any structure during the call, allocate their
normal outputs, and destroy those outputs inside the timed batch. These are the
symmetric-extension cases at length 4,096:

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

Across all 24 symmetric and antireflect cases, including dense controls, the
planned binding wins every case by 5.93x to 20.82x with a 10.13x median. The
cold path also wins all 24, ranging from 1.46x to 4.49x with a 3.16x median.
Within the symmetric results, selecting the adaptive path for a constant input
reduces `wavelets-rs` planned time by 2.05x for f64 db38, 2.77x for f64
coif17, and 1.70x for f32 coif17 relative to each dense control.

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
