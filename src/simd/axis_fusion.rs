use fearless_simd::{Level, Simd};

use super::{AxisAnalysis, SimdSample, forward_axis_fused};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
pub(crate) fn forward4<T: SimdSample<fearless_simd::Avx2>>(
    level: Level,
    analysis: AxisAnalysis<'_, T>,
    approx: &mut [T],
    detail: &mut [T],
) -> usize {
    let Some(simd) = level.__dispatch_target().as_avx2() else {
        return 0;
    };
    simd.vectorize(|| forward_axis_fused::<_, _, 4>(simd, analysis, approx, detail))
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
pub(crate) fn forward8<T: SimdSample<fearless_simd::Avx512>>(
    level: Level,
    analysis: AxisAnalysis<'_, T>,
    approx: &mut [T],
    detail: &mut [T],
) -> usize {
    let Some(simd) = level.__dispatch_target().as_avx512() else {
        return 0;
    };
    simd.vectorize(|| forward_axis_fused::<_, _, 8>(simd, analysis, approx, detail))
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) fn forward8<T: SimdSample<fearless_simd::Neon>>(
    level: Level,
    analysis: AxisAnalysis<'_, T>,
    approx: &mut [T],
    detail: &mut [T],
) -> usize {
    let Some(simd) = level.__dispatch_target().as_neon() else {
        return 0;
    };
    simd.vectorize(|| forward_axis_fused::<_, _, 8>(simd, analysis, approx, detail))
}
