# Changelog

All notable changes to `wavelets` are documented here.

## Unreleased

- Made direct FIR the default execution model and placed the paraunitary
  lattice and annihilator-split analysis backends behind the opt-in
  `experimental-kernels` Cargo feature.

## 0.1.0-alpha.10 - 2026-09-01

- Preserved the stored two-tap FIR evaluation order in the optimized Haar
  analysis kernel, restoring bit-for-bit agreement with direct convolution.
- Folded cyclically equivalent synthesis taps while planning periodized
  transforms whose coefficient bands are shorter than a filter phase. This
  removes repeated hot-path work and improves `f32` numerical stability for
  short signals with long filters.

## 0.1.0-alpha.9 - 2026-09-01

- Restored the documented scalar fallback on architectures without an x86 or
  Arm SIMD backend, including WebAssembly, by making fused-axis and packed-row
  selection follow the target's available executor set.
- Added a WebAssembly compile check to CI so portable fallback support cannot
  silently regress.

## 0.1.0-alpha.8 - 2026-09-01

- Added `Dwt::axis_scratch_len` so callers can plan the exact reusable scratch
  required by a tensor geometry while keeping axis execution allocation-free.
- Added cost-selected fused forward-axis kernels that reuse loaded samples
  across neighboring outputs on AVX-512, AVX2, and NEON, including a dedicated
  AVX2 `f64` crossover.
- Added packed contiguous-row analysis so last-axis transforms can batch
  independent rows through the same SIMD axis executor without transposing.
- Measured and encoded separate fusion and row-packing crossovers by precision
  and ISA, with Criterion coverage around the AVX2 `f64` thresholds.
- Made lattice planning respect fearless_simd's normalized dispatch target so
  builds with AVX-512 multiversioning disabled select a genuinely AVX2 plan.

## 0.1.0-alpha.7 - 2026-09-01

- Added ordered boundary sample programs for scalar and contiguous-axis
  transforms, preserving filter-tap accumulation order while retaining
  coalesced sparse edge rows for batched SIMD axis execution.
- Added a stability regression test for short `f32` signals transformed by
  long filters with smooth extension.

## 0.1.0-alpha.6 - 2026-09-01

- Batched neighboring output pairs during long-filter axis synthesis so loaded
  coefficient vectors are reused across outputs, eliminating the severe
  cache-aliasing regression observed in leading-axis `db38` reconstruction.
- Added page-offset regression benchmarks for allocation-free axis synthesis.

## 0.1.0-alpha.5 - 2026-09-01

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
- Added allocation-free batched axis transforms over contiguous tensors, with
  safe SIMD execution across independent inner-axis lanes and compiled boundary
  handling for all nine extension modes.

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
