use std::fmt::Debug;
use std::ops::{Add, AddAssign, Mul, Sub};

mod private {
    use fearless_simd::{Level, dispatch};

    pub trait Sealed {}

    pub trait SimdKernels: Sealed + Sized {
        #[cfg(feature = "experimental-kernels")]
        fn is_finite(value: Self) -> bool;

        fn mul_add(value: Self, multiplier: Self, accumulator: Self) -> Self;

        fn forward_axis(
            level: Level,
            analysis: crate::simd::AxisAnalysis<'_, Self>,
            approx: &mut [Self],
            detail: &mut [Self],
        ) -> usize;

        fn forward_axis_fused4(
            level: Level,
            analysis: crate::simd::AxisAnalysis<'_, Self>,
            approx: &mut [Self],
            detail: &mut [Self],
        ) -> usize {
            let _ = level;
            let _ = analysis;
            let _ = approx;
            let _ = detail;
            0
        }

        fn forward_axis_fused8(
            level: Level,
            analysis: crate::simd::AxisAnalysis<'_, Self>,
            approx: &mut [Self],
            detail: &mut [Self],
        ) -> usize {
            let _ = level;
            let _ = analysis;
            let _ = approx;
            let _ = detail;
            0
        }

        fn inverse_axis(
            level: Level,
            synthesis: crate::simd::AxisSynthesis<'_, Self>,
            out: &mut [Self],
        ) -> usize;

        fn inverse_axis_batched(
            level: Level,
            synthesis: crate::simd::AxisSynthesis<'_, Self>,
            out: &mut [Self],
        ) -> usize;

        fn forward_interior(
            level: Level,
            interior: crate::simd::AnalysisInterior<'_, Self>,
            approx: &mut [Self],
            detail: &mut [Self],
        ) -> usize;

        fn forward_planar(
            level: Level,
            analysis: crate::simd::PlanarAnalysis<'_, Self>,
            approx: &mut [Self],
            detail: &mut [Self],
        ) -> usize;

        fn forward_butterfly(
            level: Level,
            analysis: crate::simd::ButterflyAnalysis<'_, Self>,
            approx: &mut [Self],
            detail: &mut [Self],
        ) -> usize;

        fn forward_butterfly_pair(
            level: Level,
            analysis: crate::simd::ButterflyPairAnalysis<'_, Self>,
            approx: &mut [Self],
            first_detail: &mut [Self],
            second_detail: &mut [Self],
        ) -> usize;

        #[cfg(feature = "experimental-kernels")]
        fn forward_lattice(
            level: Level,
            analysis: crate::simd::LatticeAnalysis<'_, Self>,
            approx: &mut [Self],
            detail: &mut [Self],
        ) -> usize;

        fn inverse_periodized(
            level: Level,
            interior: crate::simd::PeriodizedInterior<'_, Self>,
            out: &mut [Self],
        ) -> usize;

        fn inverse_linear(
            level: Level,
            synthesis: crate::simd::LinearSynthesis<'_, Self>,
            out: &mut [Self],
        ) -> usize;

        fn inverse_butterfly(
            level: Level,
            synthesis: crate::simd::ButterflySynthesis<'_, Self>,
            out: &mut [Self],
        ) -> usize;

        fn inverse_butterfly_pair(
            level: Level,
            synthesis: crate::simd::ButterflyPairSynthesis<'_, Self>,
            out: &mut [Self],
        ) -> usize;
    }

    impl Sealed for f32 {}
    impl Sealed for f64 {}

    macro_rules! impl_simd_kernels {
        ($type:ty) => {
            impl SimdKernels for $type {
                #[inline]
                #[cfg(feature = "experimental-kernels")]
                fn is_finite(value: Self) -> bool {
                    value.is_finite()
                }

                #[inline]
                fn mul_add(value: Self, multiplier: Self, accumulator: Self) -> Self {
                    value.mul_add(multiplier, accumulator)
                }

                #[inline]
                fn forward_axis(
                    level: Level,
                    analysis: crate::simd::AxisAnalysis<'_, Self>,
                    approx: &mut [Self],
                    detail: &mut [Self],
                ) -> usize {
                    dispatch!(level, simd => crate::simd::forward_axis(
                        simd, analysis, approx, detail
                    ))
                }

                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                #[inline]
                fn forward_axis_fused4(
                    level: Level,
                    analysis: crate::simd::AxisAnalysis<'_, Self>,
                    approx: &mut [Self],
                    detail: &mut [Self],
                ) -> usize {
                    crate::simd::axis_fusion::forward4(level, analysis, approx, detail)
                }

                #[cfg(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64"))]
                #[inline]
                fn forward_axis_fused8(
                    level: Level,
                    analysis: crate::simd::AxisAnalysis<'_, Self>,
                    approx: &mut [Self],
                    detail: &mut [Self],
                ) -> usize {
                    crate::simd::axis_fusion::forward8(level, analysis, approx, detail)
                }

                #[inline]
                fn inverse_axis(
                    level: Level,
                    synthesis: crate::simd::AxisSynthesis<'_, Self>,
                    out: &mut [Self],
                ) -> usize {
                    dispatch!(level, simd => crate::simd::inverse_axis(simd, synthesis, out))
                }

                #[inline]
                fn inverse_axis_batched(
                    level: Level,
                    synthesis: crate::simd::AxisSynthesis<'_, Self>,
                    out: &mut [Self],
                ) -> usize {
                    dispatch!(level, simd => crate::simd::inverse_axis_batched(simd, synthesis, out))
                }

                #[inline]
                fn forward_interior(
                    level: Level,
                    interior: crate::simd::AnalysisInterior<'_, Self>,
                    approx: &mut [Self],
                    detail: &mut [Self],
                ) -> usize {
                    dispatch!(level, simd => crate::simd::forward_interior(
                        simd, interior, approx, detail
                    ))
                }

                #[inline]
                fn forward_planar(
                    level: Level,
                    analysis: crate::simd::PlanarAnalysis<'_, Self>,
                    approx: &mut [Self],
                    detail: &mut [Self],
                ) -> usize {
                    dispatch!(level, simd => crate::simd::forward_planar(
                        simd, analysis, approx, detail
                    ))
                }

                #[inline]
                fn forward_butterfly(
                    level: Level,
                    analysis: crate::simd::ButterflyAnalysis<'_, Self>,
                    approx: &mut [Self],
                    detail: &mut [Self],
                ) -> usize {
                    dispatch!(level, simd => crate::simd::forward_butterfly(
                        simd, analysis, approx, detail
                    ))
                }

                #[inline]
                fn forward_butterfly_pair(
                    level: Level,
                    analysis: crate::simd::ButterflyPairAnalysis<'_, Self>,
                    approx: &mut [Self],
                    first_detail: &mut [Self],
                    second_detail: &mut [Self],
                ) -> usize {
                    dispatch!(level, simd => crate::simd::forward_butterfly_pair(
                        simd, analysis, approx, first_detail, second_detail
                    ))
                }

                #[inline]
                #[cfg(feature = "experimental-kernels")]
                fn forward_lattice(
                    level: Level,
                    analysis: crate::simd::LatticeAnalysis<'_, Self>,
                    approx: &mut [Self],
                    detail: &mut [Self],
                ) -> usize {
                    dispatch!(level, simd => crate::simd::forward_lattice(
                        simd, analysis, approx, detail
                    ))
                }

                #[inline]
                fn inverse_periodized(
                    level: Level,
                    interior: crate::simd::PeriodizedInterior<'_, Self>,
                    out: &mut [Self],
                ) -> usize {
                    dispatch!(level, simd => crate::simd::inverse_periodized(
                        simd, interior, out
                    ))
                }

                #[inline]
                fn inverse_linear(
                    level: Level,
                    synthesis: crate::simd::LinearSynthesis<'_, Self>,
                    out: &mut [Self],
                ) -> usize {
                    dispatch!(level, simd => crate::simd::inverse_linear(simd, synthesis, out))
                }

                #[inline]
                fn inverse_butterfly(
                    level: Level,
                    synthesis: crate::simd::ButterflySynthesis<'_, Self>,
                    out: &mut [Self],
                ) -> usize {
                    dispatch!(level, simd => crate::simd::inverse_butterfly(simd, synthesis, out))
                }

                #[inline]
                fn inverse_butterfly_pair(
                    level: Level,
                    synthesis: crate::simd::ButterflyPairSynthesis<'_, Self>,
                    out: &mut [Self],
                ) -> usize {
                    dispatch!(level, simd => crate::simd::inverse_butterfly_pair(
                        simd, synthesis, out
                    ))
                }
            }
        };
    }

    impl_simd_kernels!(f32);
    impl_simd_kernels!(f64);
}

/// Numeric types supported by wavelet transform plans.
///
/// This trait is sealed and is currently implemented only for [`f32`] and
/// [`f64`].
pub trait WaveletNum:
    private::Sealed
    + private::SimdKernels
    + Copy
    + Debug
    + Send
    + Sync
    + 'static
    + Add<Output = Self>
    + AddAssign
    + Mul<Output = Self>
    + PartialEq
    + Sub<Output = Self>
{
    /// Additive identity.
    fn zero() -> Self;

    /// Converts a filter coefficient into this numeric type.
    fn from_f64(value: f64) -> Self;
}

#[inline]
pub(crate) fn mul_add<T: WaveletNum>(value: T, multiplier: T, accumulator: T) -> T {
    <T as private::SimdKernels>::mul_add(value, multiplier, accumulator)
}

#[inline]
#[cfg(feature = "experimental-kernels")]
pub(crate) fn is_finite<T: WaveletNum>(value: T) -> bool {
    <T as private::SimdKernels>::is_finite(value)
}

#[inline]
pub(crate) fn forward_axis_simd<T: WaveletNum>(
    level: fearless_simd::Level,
    analysis: crate::simd::AxisAnalysis<'_, T>,
    approx: &mut [T],
    detail: &mut [T],
) -> usize {
    if level.is_fallback() {
        0
    } else {
        <T as private::SimdKernels>::forward_axis(level, analysis, approx, detail)
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
pub(crate) fn forward_axis_fused4_simd<T: WaveletNum>(
    level: fearless_simd::Level,
    analysis: crate::simd::AxisAnalysis<'_, T>,
    approx: &mut [T],
    detail: &mut [T],
) -> usize {
    if level.is_fallback() {
        0
    } else {
        <T as private::SimdKernels>::forward_axis_fused4(level, analysis, approx, detail)
    }
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64"))]
#[inline]
pub(crate) fn forward_axis_fused8_simd<T: WaveletNum>(
    level: fearless_simd::Level,
    analysis: crate::simd::AxisAnalysis<'_, T>,
    approx: &mut [T],
    detail: &mut [T],
) -> usize {
    if level.is_fallback() {
        0
    } else {
        <T as private::SimdKernels>::forward_axis_fused8(level, analysis, approx, detail)
    }
}

#[inline]
pub(crate) fn inverse_axis_simd<T: WaveletNum>(
    level: fearless_simd::Level,
    synthesis: crate::simd::AxisSynthesis<'_, T>,
    out: &mut [T],
) -> usize {
    if level.is_fallback() {
        0
    } else {
        <T as private::SimdKernels>::inverse_axis(level, synthesis, out)
    }
}

#[inline]
pub(crate) fn inverse_axis_batched_simd<T: WaveletNum>(
    level: fearless_simd::Level,
    synthesis: crate::simd::AxisSynthesis<'_, T>,
    out: &mut [T],
) -> usize {
    if level.is_fallback() {
        0
    } else {
        <T as private::SimdKernels>::inverse_axis_batched(level, synthesis, out)
    }
}

#[inline]
pub(crate) fn forward_interior_simd<T: WaveletNum>(
    level: fearless_simd::Level,
    interior: crate::simd::AnalysisInterior<'_, T>,
    approx: &mut [T],
    detail: &mut [T],
) -> usize {
    if level.is_fallback() {
        0
    } else {
        <T as private::SimdKernels>::forward_interior(level, interior, approx, detail)
    }
}

#[inline]
pub(crate) fn forward_planar_simd<T: WaveletNum>(
    level: fearless_simd::Level,
    analysis: crate::simd::PlanarAnalysis<'_, T>,
    approx: &mut [T],
    detail: &mut [T],
) -> usize {
    if level.is_fallback() {
        0
    } else {
        <T as private::SimdKernels>::forward_planar(level, analysis, approx, detail)
    }
}

#[inline]
pub(crate) fn forward_butterfly_simd<T: WaveletNum>(
    level: fearless_simd::Level,
    analysis: crate::simd::ButterflyAnalysis<'_, T>,
    approx: &mut [T],
    detail: &mut [T],
) -> usize {
    if level.is_fallback() {
        0
    } else {
        <T as private::SimdKernels>::forward_butterfly(level, analysis, approx, detail)
    }
}

#[inline]
pub(crate) fn forward_butterfly_pair_simd<T: WaveletNum>(
    level: fearless_simd::Level,
    analysis: crate::simd::ButterflyPairAnalysis<'_, T>,
    approx: &mut [T],
    first_detail: &mut [T],
    second_detail: &mut [T],
) -> usize {
    if level.is_fallback() {
        0
    } else {
        <T as private::SimdKernels>::forward_butterfly_pair(
            level,
            analysis,
            approx,
            first_detail,
            second_detail,
        )
    }
}

#[inline]
#[cfg(feature = "experimental-kernels")]
pub(crate) fn forward_lattice_simd<T: WaveletNum>(
    level: fearless_simd::Level,
    analysis: crate::simd::LatticeAnalysis<'_, T>,
    approx: &mut [T],
    detail: &mut [T],
) -> usize {
    if level.is_fallback() {
        0
    } else {
        <T as private::SimdKernels>::forward_lattice(level, analysis, approx, detail)
    }
}

#[inline]
pub(crate) fn inverse_linear_simd<T: WaveletNum>(
    level: fearless_simd::Level,
    synthesis: crate::simd::LinearSynthesis<'_, T>,
    out: &mut [T],
) -> usize {
    if level.is_fallback() {
        0
    } else {
        <T as private::SimdKernels>::inverse_linear(level, synthesis, out)
    }
}

#[inline]
pub(crate) fn inverse_butterfly_simd<T: WaveletNum>(
    level: fearless_simd::Level,
    synthesis: crate::simd::ButterflySynthesis<'_, T>,
    out: &mut [T],
) -> usize {
    if level.is_fallback() {
        0
    } else {
        <T as private::SimdKernels>::inverse_butterfly(level, synthesis, out)
    }
}

#[inline]
pub(crate) fn inverse_butterfly_pair_simd<T: WaveletNum>(
    level: fearless_simd::Level,
    synthesis: crate::simd::ButterflyPairSynthesis<'_, T>,
    out: &mut [T],
) -> usize {
    if level.is_fallback() {
        0
    } else {
        <T as private::SimdKernels>::inverse_butterfly_pair(level, synthesis, out)
    }
}

#[inline]
pub(crate) fn inverse_periodized_simd<T: WaveletNum>(
    level: fearless_simd::Level,
    interior: crate::simd::PeriodizedInterior<'_, T>,
    out: &mut [T],
) -> usize {
    if level.is_fallback() {
        0
    } else {
        <T as private::SimdKernels>::inverse_periodized(level, interior, out)
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
