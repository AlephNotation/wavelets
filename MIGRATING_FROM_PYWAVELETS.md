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
| `wavelet.name` | `wavelet.name()` or `wavelet.to_string()` |
| `wavelet.filter_bank` | `(wavelet.dec_lo(), wavelet.dec_hi(), wavelet.rec_lo(), wavelet.rec_hi())` |
| mode `"symmetric"` | `"symmetric".parse::<Boundary>()?` or `Boundary::Symmetric` |
| `pywt.dwt(x, wavelet, mode)` | `dwt(&x, &wavelet, boundary)?` |
| `pywt.idwt(c_a, c_d, wavelet, mode)` | `idwt(&c_a, &c_d, &wavelet, boundary)?` |
| `pywt.dwt(x, wavelet, mode, axis=k)` | `plan.forward_axis_into(...)` over a flattened contiguous tensor |
| `pywt.idwt(c_a, c_d, wavelet, mode, axis=k)` | `plan.inverse_axis_into(...)` over flattened contiguous tensors |
| `pywt.wavedec(x, wavelet, mode, level=None)` | `wavedec(&x, &wavelet, boundary, Level::Max)?` |
| `pywt.wavedec(..., level=n)` | `wavedec(..., Level::Exact(n))?` |
| `[cA_L, cD_L, ..., cD_1]` | `decomposition.bands()` |
| `coeffs[0]` | `decomposition.approx()` |
| detail `cD_n` | `decomposition.detail(n)` |
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

## Repeated transforms

Python callers normally repeat `pywt.dwt` or `pywt.wavedec`. In Rust, move
configuration-dependent work out of the loop:

```rust
use wavelets::{Boundary, DwtPlanner, Wavelet};

let wavelet = Wavelet::daubechies(4)?;
let mut planner = DwtPlanner::<f64>::new();
let plan = planner.plan_dwt(4096, &wavelet, Boundary::Symmetric)?;

let signal = vec![0.0; plan.signal_len()];
let mut c_a = vec![0.0; plan.coeff_len()];
let mut c_d = vec![0.0; plan.coeff_len()];
let mut scratch = vec![0.0; plan.scratch_len()];

plan.forward_into(&signal, &mut c_a, &mut c_d, &mut scratch);
# Ok::<(), wavelets::WaveletError>(())
```

Planning and buffer allocation are outside the hot path. Plans are immutable,
`Send + Sync`, and cheaply shared through the `Arc` returned by the planner.
Concurrent calls must provide separate mutable output and scratch buffers.

The same pattern applies to `plan_wavedec`: allocate one `Decomposition` with
`allocate_decomposition`, allocate `scratch_len()` samples once, and reuse both.

## Tensor axes

A single-level `Dwt` plan can execute along any axis of a row-major contiguous
tensor without transposing it. Pass the tensor as a flat slice and describe its
shape around the transformed axis as `[outer, axis, inner]`:

- `axis` is the extent used to create the plan;
- `outer` is the product of dimensions before the transformed axis; and
- `inner` is the product of dimensions after it.

Use `axis_scratch_len(outer, inner)` to size reusable scratch, then call
`forward_axis_into` or `inverse_axis_into`. Input and output buffers remain flat;
the coefficient-axis extent changes from `signal_len()` to `coeff_len()`.

These methods execute one-dimensional transforms across a tensor axis. The
crate does not yet provide ndarray container integration or high-level
separable transforms corresponding to `dwt2`, `wavedec2`, `dwtn`, and their
inverse operations; callers that need those operations must compose axis plans.

## Errors and buffer contracts

Rust construction and planning failures return `WaveletError`. Once a plan has
been constructed, wrong buffer lengths represent a caller programming error and
the `_into` methods panic. Use `signal_len`, `coeff_len`, and `scratch_len` from
the plan instead of reproducing the sizing formulas.

`Decomposition::detail` uses one-based mathematical levels: `detail(1)` is
`cD_1`, while `bands()` iterates in PyWavelets list order, `cA_L, cD_L, ...,
cD_1`. `as_slice()` and `as_mut_slice()` expose that same contiguous physical
order.

## Current boundary

The compatible subset currently covers one-dimensional `f32` and `f64`
transforms, single-level execution along any axis of a row-major contiguous
tensor, Haar, `db1..db38`, `sym2..sym20`, `coif1..coif17`, all 15 `bior` pairs
and their `rbio` reverses, custom filter banks, and all nine extension modes.
Complex values, high-level multidimensional transform orchestration, omitted
single-level coefficient bands, borrowed decomposition construction, and
PyWavelets' other transform families are not implemented yet.

For repeated transforms, replace the allocating facade with `DwtPlanner` or
`WavedecPlan`. Planning fixes the signal length, wavelet, boundary mode, buffer
layout, and SIMD dispatch once; the `_into` execution methods then allocate
nothing.
