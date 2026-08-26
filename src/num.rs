use std::fmt::Debug;
use std::ops::{Add, AddAssign, Mul, Sub};

mod private {
    pub trait Sealed {}

    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// Numeric types supported by wavelet transform plans.
///
/// This trait is sealed and is currently implemented only for [`f32`] and
/// [`f64`].
pub trait WaveletNum:
    private::Sealed
    + Copy
    + Debug
    + Send
    + Sync
    + 'static
    + Add<Output = Self>
    + AddAssign
    + Mul<Output = Self>
    + Sub<Output = Self>
{
    /// Additive identity.
    fn zero() -> Self;

    /// Converts a filter coefficient into this numeric type.
    fn from_f64(value: f64) -> Self;
}

impl WaveletNum for f32 {
    #[inline]
    fn zero() -> Self {
        0.0
    }

    #[inline]
    fn from_f64(value: f64) -> Self {
        value as Self
    }
}

impl WaveletNum for f64 {
    #[inline]
    fn zero() -> Self {
        0.0
    }

    #[inline]
    fn from_f64(value: f64) -> Self {
        value
    }
}
