use fearless_simd::{Simd, SimdFloatElement, prelude::*};

mod analysis;
mod axis;
#[cfg(feature = "experimental-kernels")]
mod lattice;
mod synthesis;

#[cfg(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64"))]
pub(crate) mod axis_fusion;

pub(crate) use self::analysis::{
    AnalysisInterior, ButterflyAnalysis, ButterflyPairAnalysis, PlanarAnalysis, forward_butterfly,
    forward_butterfly_pair, forward_interior, forward_planar,
};
#[cfg(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64"))]
pub(crate) use self::axis::forward_axis_fused;
pub(crate) use self::axis::{
    AxisAnalysis, AxisSynthesis, forward_axis, inverse_axis, inverse_axis_batched,
};
#[cfg(feature = "experimental-kernels")]
pub(crate) use self::lattice::{LatticeAnalysis, MIN_LATTICE_OUTPUTS, forward_lattice};
pub(crate) use self::synthesis::{
    ButterflyPairSynthesis, ButterflySynthesis, LinearSynthesis, PeriodizedInterior,
    inverse_butterfly, inverse_butterfly_pair, inverse_linear, inverse_periodized,
};

pub(crate) trait SimdSample<S: Simd>: SimdFloatElement {
    type Vector: SimdFloat<S, Element = Self>;
}

impl<S: Simd> SimdSample<S> for f32 {
    type Vector = S::f32s;
}

impl<S: Simd> SimdSample<S> for f64 {
    type Vector = S::f64s;
}

#[cfg(test)]
mod tests;
