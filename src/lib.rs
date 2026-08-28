//! Plan-once, allocation-free one-dimensional discrete wavelet transforms.
//!
//! `wavelets` follows the same broad execution model as `rustfft`: construct a
//! fixed-size plan once, then reuse it without allocating in the hot path.
//! Boundary mode names, filter orientation, and coefficient ordering match
//! PyWavelets for the supported real-valued 1D DWT subset.
//!
//! # One-off transforms
//!
//! The allocating convenience functions are the shortest path for a transform
//! performed once:
//!
//! ```
//! use wavelets::{Boundary, Wavelet, dwt, idwt};
//!
//! let signal = [1.0_f64, 2.0, 3.0, 4.0];
//! let wavelet: Wavelet = "db2".parse()?;
//! let (approx, detail) = dwt(&signal, &wavelet, Boundary::Symmetric)?;
//! let reconstructed = idwt(&approx, &detail, &wavelet, Boundary::Symmetric)?;
//!
//! assert_eq!(reconstructed.len(), signal.len());
//! # Ok::<(), wavelets::WaveletError>(())
//! ```
//!
//! Standalone [`idwt`] follows PyWavelets and reconstructs the canonical even
//! length implied by its coefficient bands. For an odd original length, use a
//! fixed-length plan or [`waverec`] to retain the exact length.
//!
//! # Reusable plans
//!
//! [`DwtPlanner`] performs all reusable setup. The `_into` methods then execute
//! without allocating:
//!
//! ```
//! use wavelets::{Boundary, DwtPlanner, Wavelet};
//!
//! let signal = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
//! let wavelet = Wavelet::daubechies(2)?;
//! let mut planner = DwtPlanner::<f32>::new();
//! let plan = planner.plan_dwt(signal.len(), &wavelet, Boundary::Symmetric)?;
//!
//! let mut approx = vec![0.0; plan.coeff_len()];
//! let mut detail = vec![0.0; plan.coeff_len()];
//! let mut reconstructed = vec![0.0; plan.signal_len()];
//! let mut scratch = vec![0.0; plan.scratch_len()];
//!
//! plan.forward_into(&signal, &mut approx, &mut detail, &mut scratch);
//! plan.inverse_into(&approx, &detail, &mut reconstructed, &mut scratch);
//! assert_eq!(reconstructed.len(), signal.len());
//! # Ok::<(), wavelets::WaveletError>(())
//! ```
//!
//! [`DwtPlanner::plan_wavedec`] provides the same model for multilevel
//! decomposition. Its [`Decomposition`] stores `cA_L, cD_L, ..., cD_1` in one
//! contiguous allocation and exposes detail bands through one-based levels.
//!
//! # Errors and execution contracts
//!
//! Wavelet construction and planning return [`WaveletError`] for invalid input.
//! Once a plan exists, incorrect input or output buffer lengths are programming
//! errors and the execution methods panic. Allocate buffers from the plan's
//! sizing methods to make those requirements explicit.
//!
//! Only [`f32`] and [`f64`] are supported. [`WaveletNum`] is sealed so that the
//! set of numeric types can grow without exposing internal SIMD requirements.

#![deny(unsafe_code)]
#![warn(missing_docs)]

mod boundary;
mod coefficients;
mod decomposition;
mod error;
mod lattice;
#[cfg(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64"))]
mod lattice_coefficients;
mod num;
mod plan;
mod simd;
mod transform;
mod wavelet;

pub use boundary::Boundary;
pub use decomposition::{Decomposition, Level, WavedecPlan, dwt_max_level, wavedec, waverec};
pub use error::WaveletError;
pub use num::WaveletNum;
pub use plan::{Dwt, DwtPlanner};
pub use transform::{dwt, idwt};
pub use wavelet::{Wavelet, WaveletFamily};
