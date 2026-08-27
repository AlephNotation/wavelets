# Migrating from PyWavelets

`wavelets` preserves PyWavelets' filter orientation, boundary semantics, and
coefficient ordering for its implemented one-dimensional, real-valued subset.
The API is Rust-native rather than Python source-compatible.

## Direct mapping

| PyWavelets | `wavelets` |
| --- | --- |
| `pywt.Wavelet("db4")` | `"db4".parse::<Wavelet>()?` |
| `pywt.Wavelet("sym4")` | `Wavelet::symlet(4)?` |
| `pywt.Wavelet("coif4")` | `Wavelet::coiflet(4)?` |
| `pywt.Wavelet("bior4.4")` | `Wavelet::biorthogonal(4, 4)?` |
| `pywt.Wavelet("rbio4.4")` | `Wavelet::reverse_biorthogonal(4, 4)?` |
| mode `"symmetric"` | `"symmetric".parse::<Boundary>()?` or `Boundary::Symmetric` |
| `pywt.dwt(x, wavelet, mode)` | `dwt(&x, &wavelet, boundary)?` |
| `pywt.idwt(c_a, c_d, wavelet, mode)` | `idwt(&c_a, &c_d, &wavelet, boundary)?` |
| `pywt.wavedec(x, wavelet, mode, level=None)` | `wavedec(&x, &wavelet, boundary, Level::Max)?` |
| `pywt.wavedec(..., level=n)` | `wavedec(..., Level::Exact(n))?` |
| `[cA_L, cD_L, ..., cD_1]` | `decomposition.bands()` |
| `pywt.waverec(coeffs, wavelet, mode)` | `waverec(&decomposition)?` |
| `pywt.dwt_max_level(len, filter_len)` | `dwt_max_level(len, filter_len)` |

```rust
use wavelets::{Boundary, Level, Wavelet, dwt, idwt, wavedec, waverec};

let signal = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
let wavelet: Wavelet = "db2".parse()?;
let boundary: Boundary = "symmetric".parse()?;

let (c_a, c_d) = dwt(&signal, &wavelet, boundary)?;
let single_level = idwt(&c_a, &c_d, &wavelet, boundary)?;

let decomposition = wavedec(&signal, &wavelet, boundary, Level::Max)?;
let pywavelets_order: Vec<&[f64]> = decomposition.bands().collect();
let multilevel = waverec(&decomposition)?;

# let _ = (single_level, pywavelets_order, multilevel);
# Ok::<(), wavelets::WaveletError>(())
```

## Odd-length reconstruction

Standalone PyWavelets `idwt` reconstructs the canonical even length implied by
its coefficient arrays. `wavelets::idwt` does the same, so transforming an
odd-length signal and immediately reconstructing it produces one additional
boundary-derived sample.

A `DwtPlanner` plan knows the original input length and its `inverse` method
returns exactly that length. `Decomposition` retains the same information for
`waverec`.

## Current boundary

The compatible subset currently covers one-dimensional `f32` and `f64`
transforms, Haar, `db1..db38`, `sym2..sym20`, `coif1..coif17`, all 15 `bior`
pairs and their `rbio` reverses, custom filter banks, and all nine extension
modes. Complex values, multidimensional axes, omitted single-level coefficient
bands, and PyWavelets' other transform families are not implemented yet.

For repeated transforms, replace the allocating facade with `DwtPlanner` or
`WavedecPlan`. Planning fixes the signal length, wavelet, boundary mode, buffer
layout, and SIMD dispatch once; the `_into` execution methods then allocate
nothing.
