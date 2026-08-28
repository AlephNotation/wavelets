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
`0fe0934152dc1b169ef8fa3d020c28ce79b8d483`. Both implementations run in the
same CPython interpreter and receive the same NumPy inputs. The `planned` path
reuses a `wavelets-rs` plan. The deliberately conservative `cold` path creates
the canonical wavelet and plan inside every call, while PyWavelets receives a
preconstructed `pywt.Wavelet`. All paths create and destroy their outputs inside
the timed batch.

Signal length 4,096, db4, symmetric extension:

| Precision | Transform | `wavelets-rs` planned | `wavelets-rs` cold | PyWavelets | Py / planned | Py / cold |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| f64 | single forward | 2.85 us | 5.40 us | 10.37 us | 3.63x | 1.92x |
| f64 | single inverse | 2.67 us | 4.99 us | 6.44 us | 2.41x | 1.29x |
| f64 | multilevel forward | 7.18 us | 16.16 us | 31.57 us | 4.40x | 1.95x |
| f64 | multilevel inverse | 6.45 us | 15.61 us | 21.83 us | 3.38x | 1.40x |
| f32 | single forward | 1.61 us | 3.81 us | 9.77 us | 6.06x | 2.56x |
| f32 | single inverse | 1.51 us | 3.76 us | 6.51 us | 4.30x | 1.73x |
| f32 | multilevel forward | 4.05 us | 12.55 us | 31.48 us | 7.77x | 2.51x |
| f32 | multilevel inverse | 3.78 us | 12.41 us | 22.72 us | 6.02x | 1.83x |

The planned binding wins all 92 canonical cases, ranging from 1.78x to 10.77x
with a 3.63x median. The cold path wins 69 of 92, ranging from 0.03x to 4.50x
with a 1.40x median. Its losses expose the intended plan-once tradeoff: filter
construction and boundary-row compilation can dominate execution for short or
long-filter transforms.

#### Long-filter boundary stress

These f64 forward cases exercise the filters and boundaries most affected by
compiled sparse edge rows:

| Wavelet | Length | Boundary | `wavelets-rs` planned | `wavelets-rs` cold | PyWavelets | Py / planned |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| db38 | 16 | symmetric | 574 ns | 14.05 us | 3.74 us | 6.51x |
| db38 | 16 | antireflect | 559 ns | 23.88 us | 4.16 us | 7.44x |
| coif17 | 16 | symmetric | 637 ns | 22.11 us | 5.88 us | 9.23x |
| coif17 | 16 | antireflect | 622 ns | 40.82 us | 6.70 us | 10.77x |
| db38 | 4,096 | symmetric | 19.84 us | 42.03 us | 129.29 us | 6.52x |
| db38 | 4,096 | antireflect | 19.14 us | 45.25 us | 122.39 us | 6.39x |
| coif17 | 4,096 | symmetric | 27.21 us | 65.25 us | 205.93 us | 7.57x |
| coif17 | 4,096 | antireflect | 27.77 us | 74.20 us | 206.92 us | 7.45x |

At length 4,096, planned multilevel forward transforms range from 5.85x to
6.63x faster than PyWavelets across the same wavelets and boundaries. Complete
environment metadata, checksums, batch sizes, and all 5,520 raw timing samples
are in
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
