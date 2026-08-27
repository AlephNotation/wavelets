# Published benchmark results

These are end-to-end transform execution measurements, not isolated inner-loop
claims. Planning, wavelet construction, and input generation happen before the
timer. The Rust `allocating` path allocates its output on every call; the Rust
`into` path reuses caller-owned output and scratch buffers. PyWavelets uses its
normal allocating Python API.

## Apple M4 Max / NEON

Measured on macOS 15.6 with an Apple M4 Max using the runtime-selected NEON
backend. The source was clean commit `156fd8e3bcaf98959a5a7acf64aaf74aa70f7afd`
(`wavelets` 0.1.0-alpha.4), compiled with Rust 1.98's release profile and no
additional `RUSTFLAGS`. The comparison used Python 3.14.6, NumPy 2.5.2,
PyWavelets distribution 1.9.0 (whose module reports 1.8.0), and GSL 2.8.

Each number below is the median of 20 calibrated in-process samples targeting
10 ms apiece after three warmup batches. Checksums are validated between Rust's
two APIs and PyWavelets before a report is accepted. The complete 70-case matrix,
environment metadata, batch sizes, checksums, and all 216 sets of raw samples
are in [apple-m4-max-neon.json](apple-m4-max-neon.json).

### Representative PyWavelets comparison

Signal length 4,096, db4, symmetric extension:

| Precision | Transform | Rust allocating | Rust `into` | PyWavelets | PyWavelets / `into` |
| --- | --- | ---: | ---: | ---: | ---: |
| f64 | single forward | 3.09 us | 2.77 us | 9.73 us | 3.52x |
| f64 | single inverse | 2.51 us | 2.19 us | 6.57 us | 3.01x |
| f64 | multilevel forward | 7.10 us | 6.44 us | 30.38 us | 4.72x |
| f64 | multilevel inverse | 5.06 us | 4.49 us | 21.26 us | 4.73x |
| f32 | single forward | 1.63 us | 1.52 us | 9.44 us | 6.21x |
| f32 | single inverse | 1.33 us | 1.10 us | 6.10 us | 5.55x |
| f32 | multilevel forward | 3.68 us | 3.56 us | 30.45 us | 8.55x |
| f32 | multilevel inverse | 2.70 us | 2.36 us | 21.73 us | 9.20x |

Across all 70 canonical cases, PyWavelets divided by Rust `into` ranged from
1.84x to 29.87x, with a median of 3.63x. Small-signal ratios include the normal
fixed cost of crossing PyWavelets' Python API, so the representative table above
is a better guide for sustained transform work.

### Strictly comparable GSL subset

GSL's public 1D API always computes a complete transform. The shared semantic
subset is therefore full-depth `f64` Haar with periodic extension, corresponding
to `wavelets` db1 with periodization. No other GSL cases are compared.

| Length | Direction | Rust `into` | GSL | GSL / Rust |
| ---: | --- | ---: | ---: | ---: |
| 1,024 | forward | 536.97 ns | 2.36 us | 4.39x |
| 1,024 | inverse | 743.22 ns | 2.26 us | 3.03x |
| 4,096 | forward | 1.94 us | 8.97 us | 4.63x |
| 4,096 | inverse | 2.82 us | 8.57 us | 3.04x |
| 16,384 | forward | 8.98 us | 47.70 us | 5.31x |
| 16,384 | inverse | 11.11 us | 43.63 us | 3.93x |

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
