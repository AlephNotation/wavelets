# Changelog

All notable changes to `wavelets` are documented here.

## Unreleased

- Audited and documented the public API before the first beta.
- Added `Display` for `Wavelet` and mutable contiguous coefficient access with
  `Decomposition::as_mut_slice`.
- Added reproducible source-revision metadata to cross-library benchmark
  reports.
- Published the canonical Apple M4 Max/NEON comparison with all raw samples.

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
