//! Plan-once, allocation-free one-dimensional discrete wavelet transforms.
//!
//! `wavelets` follows the same broad execution model as `rustfft`: construct a
//! fixed-size plan once, then reuse it without allocating in the hot path.
//! Boundary mode names and coefficient conventions match PyWavelets.

#![deny(unsafe_code)]
#![warn(missing_docs)]

mod boundary;
mod coefficients;
mod decomposition;
mod error;
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
