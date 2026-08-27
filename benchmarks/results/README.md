# Published benchmark results

These are end-to-end transform execution measurements, not isolated inner-loop
claims. Planning, wavelet construction, and input generation happen before the
timer. The Rust `allocating` path allocates its output on every call; the Rust
`into` path reuses caller-owned output and scratch buffers. PyWavelets uses its
normal allocating Python API.

## Apple M4 Max / NEON

Measured on macOS 15.6 with an Apple M4 Max using the runtime-selected NEON
backend. The source was clean commit `2c965ba5874d7036610505bb29e9ba23a78202a5`
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
| f64 | single forward | 2.57 us | 2.22 us | 10.01 us | 4.51x |
| f64 | single inverse | 2.15 us | 1.92 us | 6.39 us | 3.33x |
| f64 | multilevel forward | 5.89 us | 4.99 us | 31.53 us | 6.32x |
| f64 | multilevel inverse | 4.49 us | 3.95 us | 21.78 us | 5.51x |
| f32 | single forward | 1.41 us | 1.22 us | 10.00 us | 8.21x |
| f32 | single inverse | 1.16 us | 989.41 ns | 6.48 us | 6.55x |
| f32 | multilevel forward | 3.27 us | 2.97 us | 31.25 us | 10.52x |
| f32 | multilevel inverse | 2.43 us | 2.10 us | 22.21 us | 10.59x |

Across all 70 canonical cases, PyWavelets divided by Rust `into` ranged from
1.96x to 32.95x, with a median of 4.73x. Small-signal ratios include the normal
fixed cost of crossing PyWavelets' Python API, so the representative table above
is a better guide for sustained transform work.

### Strictly comparable GSL subset

GSL's public 1D API always computes a complete transform. The shared semantic
subset is therefore full-depth `f64` Haar with periodic extension, corresponding
to `wavelets` db1 with periodization. No other GSL cases are compared.

| Length | Direction | Rust `into` | GSL | GSL / Rust |
| ---: | --- | ---: | ---: | ---: |
| 1,024 | forward | 476.57 ns | 2.27 us | 4.76x |
| 1,024 | inverse | 619.61 ns | 2.20 us | 3.55x |
| 4,096 | forward | 1.69 us | 9.16 us | 5.43x |
| 4,096 | inverse | 2.26 us | 8.84 us | 3.91x |
| 16,384 | forward | 8.50 us | 45.49 us | 5.36x |
| 16,384 | inverse | 8.88 us | 42.16 us | 4.75x |

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
