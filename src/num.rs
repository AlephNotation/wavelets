use std::fmt::Debug;
use std::ops::{Add, AddAssign, Mul, Sub};

mod private {
    use fearless_simd::{Level, dispatch};

    pub trait Sealed {}

    pub trait SimdSynthesis: Sealed + Sized {
        fn inverse_linear(
            level: Level,
            rec_lo: &[Self],
            rec_hi: &[Self],
            approx: &[Self],
            detail: &[Self],
            out: &mut [Self],
        ) -> usize;
    }

    impl Sealed for f32 {}
    impl Sealed for f64 {}

    impl SimdSynthesis for f32 {
        #[inline]
        fn inverse_linear(
            level: Level,
            rec_lo: &[Self],
            rec_hi: &[Self],
            approx: &[Self],
            detail: &[Self],
            out: &mut [Self],
        ) -> usize {
            dispatch!(level, simd => crate::simd::inverse_linear_f32(
                simd, rec_lo, rec_hi, approx, detail, out
            ))
        }
    }

    impl SimdSynthesis for f64 {
        #[inline]
        fn inverse_linear(
            level: Level,
            rec_lo: &[Self],
            rec_hi: &[Self],
            approx: &[Self],
            detail: &[Self],
            out: &mut [Self],
        ) -> usize {
            dispatch!(level, simd => crate::simd::inverse_linear_f64(
                simd, rec_lo, rec_hi, approx, detail, out
            ))
        }
    }
}

/// Numeric types supported by wavelet transform plans.
///
/// This trait is sealed and is currently implemented only for [`f32`] and
/// [`f64`].
pub trait WaveletNum:
    private::Sealed
    + private::SimdSynthesis
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

#[inline]
pub(crate) fn inverse_linear_simd<T: WaveletNum>(
    level: fearless_simd::Level,
    rec_lo: &[T],
    rec_hi: &[T],
    approx: &[T],
    detail: &[T],
    out: &mut [T],
) -> usize {
    if level.is_fallback() {
        0
    } else {
        <T as private::SimdSynthesis>::inverse_linear(level, rec_lo, rec_hi, approx, detail, out)
    }
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
