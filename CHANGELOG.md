# Changelog

All notable changes to `wavelets` are documented here.

## Unreleased

- Audited and documented the public API before the first beta.
- Added `Display` for `Wavelet` and mutable contiguous coefficient access with
  `Decomposition::as_mut_slice`.
- Added reproducible source-revision metadata to cross-library benchmark
  reports.
- Published the canonical Apple M4 Max/NEON comparison with all raw samples.
- Batched independent analysis vectors and removed zero-initialized first FMAs
  from the analysis and synthesis kernels.
- Reused shared and overlapping coefficient vectors in periodized SIMD
  reconstruction.
- Made the profiling driver configurable across precision, wavelet, boundary,
  and signal length.
- Added safe PyO3/NumPy bindings for reusable single-level and multilevel
  `f32`/`f64` plans.
- Added a same-interpreter Python benchmark that reports both reused-plan and
  plan-plus-execute performance against PyWavelets with raw samples.
- Published the Apple M4 Max Python-to-Python comparison with both planning
  boundaries and all raw timing samples.
- Compiled antireflect extension into constant-time affine sample rules,
  removing repeated boundary walks for long filters and multilevel transforms.
- Composed boundary extension and analysis filters into contiguous sparse edge
  rows, coalescing repeated input samples while keeping low/high execution
  fused.
- Added an algebraically detected two-tap butterfly backend for analysis and
  synthesis, including matching custom filter banks, with safe SIMD dispatch.
- Fused pairs of complete two-tap multilevel butterflies so intermediate
  approximations stay in registers, with direction-specific planner costing
  and smaller scratch layouts for fully fused cascades.
- Added a scratch-free annihilator-split analysis backend for structured signals
  with long filters. Plans factor both analysis channels into constant bases and
  finite-difference corrections, then use a precision-aware event cost model to
  select it or fall back to the existing SIMD kernel on every execution.
- Added reproducibly generated, conditioned paraunitary lattice factors and
  safe NEON and AVX-512 `f64` analysis backends for selected long orthogonal
  filters. The planner keeps compiled boundary rows and automatically retains
  the direct kernel below the measured crossover.
- Added and published a 24-case same-interpreter structured-input benchmark
  suite with dense controls, two run densities, two boundary modes, and both
  supported precisions for the qualifying long filters.
- Published physical AMD Ryzen 7 8745HS/AVX-512 results with the complete
  112-case same-interpreter report and native lattice crossover matrix.
- Published three independent AWS AMD EPYC 7R13/AVX2 same-interpreter reports,
  including every raw sample and cross-process stability results.

## 0.1.0-alpha.4 - 2026-08-27

- Added all canonical PyWavelets biorthogonal and reverse-biorthogonal filter
  banks from independently generated high-precision coefficients.
- Expanded the reference, reconstruction, fuzz, and benchmark matrices to the
  complete built-in family set.

## 0.1.0-alpha.3 - 2026-08-27

- Added independently generated `coif1..coif17` filter banks.

## 0.1.0-alpha.2 - 2026-08-27

- Added independently generated `sym2..sym20` filter banks.

## 0.1.0-alpha.1 - 2026-08-26

- Published the initial scalar and safe-SIMD DWT/IDWT implementation, all nine
  boundary modes, reusable single-level and multilevel plans, reference tests,
  fuzz targets, and benchmark tooling.
