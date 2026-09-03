# Published benchmark results

These are end-to-end transform execution measurements, not isolated inner-loop
claims. Input generation happens before every timer; each section below states
its exact planning and output-materialization boundaries. PyWavelets uses its
normal allocating Python API.

The newest expanded report uses the default direct-FIR configuration. Older
reports predate the feature split and include what is now the opt-in
`experimental-kernels` configuration where a lattice or annihilator executor
qualified. Each section identifies its configuration explicitly.

## Apple M4 Max / NEON

Measured on macOS 15.6 with an Apple M4 Max using the runtime-selected NEON
backend. The reports use Rust 1.98's release profile with no additional
`RUSTFLAGS`, Python 3.14.6, NumPy 2.5.2, and PyWavelets distribution 1.9.0
(whose module reports 1.8.0).

### Expanded common-workload matrix (default direct FIR)

This same-interpreter report was generated from clean commit
`2576e7e3a82d9f560511ef6ff501035a0fa6fd53`. It expands the canonical suite
from 92 to 132 cases and uses the default direct-FIR kernels. Haar, db2, and db4
are measured in both directions over short-to-long signals; the existing
filter, boundary, odd-length, long-filter, and multilevel sweeps remain.

| Workload | Cases | Reused-plan range | Reused-plan median | Plan + execute range | Plan + execute median | Plan + execute wins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| f64 Haar/db2/db4, lengths 16–16,384 | 36 | 2.02–4.19x | 3.43x | 0.51–2.31x | 1.00x | 18/36 |
| f32 Haar/db2/db4, lengths 64–4,096 | 18 | 3.81–5.89x | 4.44x | 0.61–2.41x | 1.11x | 10/18 |
| Common filters, lengths at most 256 | 30 | 3.41–4.71x | 3.88x | 0.51–1.15x | 0.86x | 6/30 |
| Common filters, lengths at least 1,024 | 24 | 2.02–5.89x | 3.21x | 0.83–2.41x | 1.56x | 22/24 |
| Complete canonical matrix | 132 | 1.79–10.00x | 3.54x | 0.02–4.69x | 1.18x | 81/132 |

The reusable-plan path wins all 54 common-filter cases and all 132 cases in the
complete canonical matrix. Rebuilding the plan for every call is usually
counterproductive below 1,024 samples, which is why the crate exposes reusable
plans rather than hiding planning inside execution.

The previously disclosed losses for 16-sample f64 db38 and coif17 antireflect
analysis now use a planner-selected materialized planar representation. They
take 1.11 us versus PyWavelets' 3.94 us (3.56x) and 1.53 us versus 6.55 us
(4.28x), respectively. The planner retains the existing direct representation
when the transform has enough interior work that materialization would not pay
for itself.

Complete environment metadata, checksums, batch sizes, and all 7,920 raw timing
samples are in
[apple-m4-max-python-api-representative.json](apple-m4-max-python-api-representative.json).

### Same-interpreter Python API

This report was generated from clean commit
`a3eda32076a77fb21086564b380a53fc514a77a1`. Both implementations run in the
same CPython interpreter and receive the same NumPy inputs. The `reused plan`
measurement reuses a `wavelets-rs` plan. The deliberately conservative
`plan + execute` measurement creates the canonical wavelet and plan inside
every call, while PyWavelets receives a preconstructed `pywt.Wavelet`. All paths
create and destroy their outputs inside the timed batch.

Signal length 4,096, db4, symmetric extension:

| Precision | Transform | `wavelets-rs` reused plan | `wavelets-rs` plan + execute | PyWavelets | Py / reused plan | Py / plan + execute |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| f64 | single forward | 2.58 us | 4.84 us | 9.15 us | 3.55x | 1.89x |
| f64 | single inverse | 2.48 us | 4.68 us | 5.93 us | 2.39x | 1.27x |
| f64 | multilevel forward | 6.96 us | 15.79 us | 30.61 us | 4.40x | 1.94x |
| f64 | multilevel inverse | 6.13 us | 14.95 us | 20.60 us | 3.36x | 1.38x |
| f32 | single forward | 1.49 us | 3.53 us | 8.96 us | 6.03x | 2.54x |
| f32 | single inverse | 1.39 us | 3.46 us | 5.89 us | 4.25x | 1.71x |
| f32 | multilevel forward | 3.94 us | 12.21 us | 30.34 us | 7.70x | 2.48x |
| f32 | multilevel inverse | 3.59 us | 11.82 us | 21.08 us | 5.87x | 1.78x |

The reused-plan binding wins all 92 canonical cases, ranging from 1.77x to
11.70x with a 3.56x median. The plan-plus-execute path wins 68 of 92, ranging
from 0.03x to 4.69x with a 1.38x median. Its losses expose the intended plan-once
tradeoff: filter construction and boundary-row compilation can dominate
execution for short or long-filter transforms.

For 4,096-sample periodized multilevel Haar, the fused reused-plan path takes
3.04 us forward versus PyWavelets' 21.99 us (7.24x), and 3.43 us inverse versus
16.56 us (4.83x).

#### Long-filter boundary stress

These f64 forward cases exercise compiled sparse edge rows together with the
NEON paraunitary lattice backend on long transform interiors:

| Wavelet | Length | Boundary | `wavelets-rs` reused plan | `wavelets-rs` plan + execute | PyWavelets | Py / reused plan |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| db38 | 16 | symmetric | 510 ns | 13.49 us | 3.34 us | 6.54x |
| db38 | 16 | antireflect | 524 ns | 23.70 us | 3.73 us | 7.12x |
| coif17 | 16 | symmetric | 601 ns | 21.78 us | 5.32 us | 8.86x |
| coif17 | 16 | antireflect | 593 ns | 41.54 us | 6.23 us | 10.49x |
| db38 | 4,096 | symmetric | 10.33 us | 30.21 us | 112.33 us | 10.88x |
| db38 | 4,096 | antireflect | 10.52 us | 36.26 us | 112.52 us | 10.70x |
| coif17 | 4,096 | symmetric | 15.37 us | 50.49 us | 179.56 us | 11.68x |
| coif17 | 4,096 | antireflect | 15.40 us | 60.25 us | 180.06 us | 11.70x |

At length 4,096, reused-plan multilevel forward transforms range from 7.72x to
8.32x faster than PyWavelets across the same wavelets and boundaries.

#### Structured long-filter inputs

The structured suite measures the adaptive finite-difference executor without
giving it privileged inputs or timing boundaries: both engines receive the
same dense NumPy array, discover any structure during the call, allocate their
normal outputs, and destroy those outputs inside the timed batch. These are the
symmetric-extension cases at length 4,096:

| Precision | Wavelet | Input | `wavelets-rs` reused plan | `wavelets-rs` plan + execute | PyWavelets | Py / reused plan | Py / plan + execute |
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
reused-plan binding wins every case by 5.75x to 20.95x with an 11.28x median.
The plan-plus-execute path also wins all 24, ranging from 1.40x to 4.06x with a
3.15x median. Within the symmetric results, selecting the adaptive path for a
constant input reduces `wavelets-rs` reused-plan time by 1.22x for f64 db38,
1.79x for f64 coif17, and 1.72x for f32 coif17 relative to each dense control.

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

## AMD Ryzen 7 8745HS / AVX-512

Measured on physical x86_64 hardware running Ubuntu with the runtime-selected
AVX-512 backend. The reports use Rust 1.98's release profile with no additional
`RUSTFLAGS`, CPython 3.12.3, NumPy 2.5.2, and PyWavelets distribution 1.9.0
(whose module reports 1.8.0).

### Same-interpreter Python API

This report was generated from clean commit
`f93c35049bdf20cccfbfdebfe20f02b2523b16db` with the same timing boundaries as
the Apple report. Representative 4,096-sample symmetric-extension results are:

| Precision | Wavelet | Transform | `wavelets-rs` reused plan | `wavelets-rs` plan + execute | PyWavelets | Py / reused plan | Py / plan + execute |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| f64 | db4 | single forward | 2.13 us | 5.64 us | 10.55 us | 4.95x | 1.87x |
| f64 | db20 | single forward | 5.36 us | 43.22 us | 61.48 us | 11.47x | 1.42x |
| f64 | db38 | single forward | 9.78 us | 36.46 us | 123.10 us | 12.58x | 3.38x |
| f64 | coif17 | single forward | 13.47 us | 124.02 us | 179.79 us | 13.34x | 1.45x |
| f64 | db38 | multilevel forward | 30.58 us | 155.66 us | 261.36 us | 8.55x | 1.68x |
| f64 | coif17 | multilevel forward | 45.84 us | 601.09 us | 386.16 us | 8.43x | 0.64x |

The reused-plan binding wins all 92 canonical cases, ranging from 3.21x to
13.88x with a 4.82x median. The plan-plus-execute path wins 64 of 92; its 1.29x
median includes wavelet construction and planning on every call.

Across the 24 structured long-filter cases, including dense controls and both
boundary modes, reused-plan execution ranges from 5.66x to 13.45x faster than
PyWavelets with a 12.72x median. Plan-plus-execute ranges from 1.32x to 3.38x
faster with a 2.25x median. Complete metadata, checksums, calibrated batch sizes,
and all 6,720 timing samples are in
[amd-ryzen-7-8745hs-python-api.json](amd-ryzen-7-8745hs-python-api.json).

### Native lattice crossover

This paired diagnostic alternates the built-in automatic plan with the normal
production executor for an equivalent custom filter bank, which has no
generated lattice factor. Planning and allocation are outside the timer. The
custom-filter control can still perform its normal adaptive structure check,
so this measures the complete executor choice rather than an isolated
instruction-level kernel.

| Wavelet | Length | Direct-equivalent | Automatic | Speedup |
| --- | ---: | ---: | ---: | ---: |
| db20 | 4,096 | 7.21 us | 4.66 us | 1.55x |
| sym20 | 4,096 | 7.21 us | 4.67 us | 1.55x |
| db38 | 4,096 | 14.45 us | 8.96 us | 1.61x |
| coif17 | 4,096 | 20.52 us | 12.66 us | 1.62x |
| db20 | 262,144 | 430.65 us | 207.52 us | 2.08x |
| sym20 | 262,144 | 428.50 us | 209.28 us | 2.05x |
| db38 | 262,144 | 811.92 us | 329.60 us | 2.46x |
| coif17 | 262,144 | 1.09 ms | 414.16 us | 2.64x |

The planner retains the direct executor through the measured 1,024-sample
cases, then crosses over at 2,048 samples with a 1.23x–1.26x gain. The complete
32-case matrix from 512 through 1,048,576 samples is in
[amd-ryzen-7-8745hs-lattice.csv](amd-ryzen-7-8745hs-lattice.csv). AVX2-only
machines use the direct, butterfly, and adaptive annihilator executors because
the x86 lattice backend is currently selected only when AVX-512 is available.

## AMD EPYC 7R13 / AVX2

Measured on an AWS `c6a.large` instance backed by an AMD EPYC 7R13 (Milan).
The guest exposed AVX2 and FMA but no AVX-512. Each benchmark process was
pinned to one vCPU. The reports use Ubuntu 24.04, Rust 1.98's release profile
with no additional `RUSTFLAGS`, CPython 3.12.3, NumPy 2.5.2, and PyWavelets
distribution 1.9.0 (whose module reports 1.8.0).

Three independent same-interpreter runs were generated from clean commit
`1fbba8f2eedf41dca688772faada04209b84ec40`. Each run contains 20 calibrated
samples per engine after three warmup batches. The table reports the median of
the three process medians for 4,096-sample symmetric-extension transforms:

| Precision | Wavelet | Transform | `wavelets-rs` reused plan | PyWavelets | Py / Rust |
| --- | --- | --- | ---: | ---: | ---: |
| f64 | db4 | single forward | 4.09 us | 14.99 us | 3.67x |
| f64 | db20 | single forward | 14.90 us | 90.93 us | 6.10x |
| f64 | db38 | single forward | 28.57 us | 177.61 us | 6.22x |
| f64 | coif17 | single forward | 39.61 us | 267.89 us | 6.76x |
| f64 | db38 | multilevel forward | 69.45 us | 389.42 us | 5.61x |
| f64 | coif17 | multilevel forward | 101.07 us | 587.96 us | 5.82x |

Reused-plan execution wins all 92 canonical cases in every run. The canonical
median speedup ranges from 3.96x to 4.02x across the three processes. Across
the 24 structured long-filter cases, the median ranges from 7.21x to 7.24x and
the complete observed range is 4.47x–11.57x.

The plan-plus-execute canonical median ranges from 1.21x to 1.33x; its
structured-suite median ranges from 2.11x to 2.28x. Long-filter planning is
more process-layout-sensitive than execution: several individual planning
medians changed by roughly 2x between the first and later processes while
reused execution and PyWavelets remained stable. All three reports are
published rather than selecting the most favorable process:

- [run 1](amd-epyc-7r13-avx2-python-api-run-1.json)
- [run 2](amd-epyc-7r13-avx2-python-api-run-2.json)
- [run 3](amd-epyc-7r13-avx2-python-api-run-3.json)

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
