use std::sync::Arc;

use fearless_simd::Level as SimdLevel;

mod analysis;
#[cfg(feature = "experimental-kernels")]
mod annihilator;
mod axis;
mod synthesis;

pub(crate) use self::analysis::EdgeTerm;
use self::analysis::{
    AnalysisBackends, AnalysisKernel, AnalysisPlan, analysis_butterfly, analyze_edges,
    analyze_interior, build_analysis, coefficient_len,
};
#[cfg(feature = "experimental-kernels")]
use self::analysis::{lattice_preempts_annihilator, lattice_simd_supported};
#[cfg(feature = "experimental-kernels")]
use self::annihilator::AnnihilatorFilter;
use self::axis::{AxisGeometry, AxisPlan};
use self::synthesis::{
    PeriodizedSynthesis, extend_polyphase, inverse_butterfly, inverse_linear, inverse_periodized,
    periodized_phases_are_swapped, synthesis_butterfly,
};
#[cfg(feature = "experimental-kernels")]
use crate::lattice::LatticeFilter;
#[cfg(feature = "experimental-kernels")]
use crate::num::forward_lattice_simd;
use crate::num::{forward_butterfly_simd, forward_interior_simd};
#[cfg(feature = "experimental-kernels")]
use crate::simd::LatticeAnalysis;
use crate::simd::{AnalysisInterior, ButterflyAnalysis};
use crate::{Boundary, Wavelet, WaveletError, WaveletNum};

/// A reusable, fixed-length one-level DWT/IDWT plan.
///
/// Buffer-size mistakes are programming errors and cause the `_into` methods
/// to panic. Use the plan's sizing methods to prepare buffers once.
///
/// Plans are returned behind [`Arc`] by [`crate::DwtPlanner::plan_dwt`], so cloning a
/// plan handle is cheap and the same immutable plan can be shared between
/// threads. Each concurrent execution must use distinct output and scratch
/// buffers.
pub trait Dwt<T: WaveletNum>: Send + Sync {
    /// Returns the input and reconstructed signal length fixed by this plan.
    fn signal_len(&self) -> usize;

    /// Returns the required length of each output coefficient band.
    fn coeff_len(&self) -> usize;

    /// Returns the minimum scratch-buffer length.
    fn scratch_len(&self) -> usize;

    /// Returns the scratch-buffer length required for an axis transform.
    ///
    /// This may exceed [`Self::scratch_len`] when the tensor geometry allows
    /// independent contiguous signals to be packed into SIMD lanes. Callers
    /// that reuse an axis geometry can allocate this buffer once.
    fn axis_scratch_len(&self, _outer: usize, _inner: usize) -> usize {
        self.scratch_len()
    }

    /// Allocates and computes `(approximation, detail)` coefficients.
    ///
    /// # Panics
    ///
    /// Panics when `signal.len()` differs from [`Self::signal_len`].
    fn forward(&self, signal: &[T]) -> (Vec<T>, Vec<T>) {
        let mut approx = vec![T::zero(); self.coeff_len()];
        let mut detail = vec![T::zero(); self.coeff_len()];
        let mut scratch = vec![T::zero(); self.scratch_len()];
        self.forward_into(signal, &mut approx, &mut detail, &mut scratch);
        (approx, detail)
    }

    /// Allocates and reconstructs a signal of [`Self::signal_len`] samples.
    ///
    /// # Panics
    ///
    /// Panics unless both coefficient bands have exactly [`Self::coeff_len`]
    /// samples.
    fn inverse(&self, approx: &[T], detail: &[T]) -> Vec<T> {
        let mut out = vec![T::zero(); self.signal_len()];
        let mut scratch = vec![T::zero(); self.scratch_len()];
        self.inverse_into(approx, detail, &mut out, &mut scratch);
        out
    }

    /// Computes one decomposition level without allocating.
    ///
    /// `signal`, `approx`, and `detail` must have exactly the lengths reported
    /// by [`Self::signal_len`] and [`Self::coeff_len`]. `scratch` may be longer
    /// than [`Self::scratch_len`] but not shorter.
    ///
    /// # Panics
    ///
    /// Panics when any buffer violates those length requirements.
    fn forward_into(&self, signal: &[T], approx: &mut [T], detail: &mut [T], scratch: &mut [T]);

    /// Reconstructs the plan's original signal length without allocating.
    ///
    /// `approx`, `detail`, and `out` must have exactly the lengths reported by
    /// [`Self::coeff_len`] and [`Self::signal_len`]. `scratch` may be longer
    /// than [`Self::scratch_len`] but not shorter.
    ///
    /// # Panics
    ///
    /// Panics when any buffer violates those length requirements.
    fn inverse_into(&self, approx: &[T], detail: &[T], out: &mut [T], scratch: &mut [T]);

    /// Computes one decomposition level over an axis of a contiguous tensor.
    ///
    /// The flat buffers are interpreted as `[outer, axis, inner]`, where the
    /// planned signal length is the input axis extent and [`Self::coeff_len`]
    /// is the output axis extent. This layout covers every axis of a
    /// row-major contiguous tensor. `scratch` must contain at least
    /// [`Self::axis_scratch_len`] elements for this geometry.
    fn forward_axis_into(
        &self,
        signal: &[T],
        outer: usize,
        inner: usize,
        approx: &mut [T],
        detail: &mut [T],
        scratch: &mut [T],
    );

    /// Reconstructs an axis of a contiguous tensor.
    ///
    /// The coefficient buffers are interpreted as `[outer, coeff, inner]`
    /// and `out` as `[outer, signal, inner]`.
    fn inverse_axis_into(
        &self,
        approx: &[T],
        detail: &[T],
        outer: usize,
        inner: usize,
        out: &mut [T],
        scratch: &mut [T],
    );
}

pub(crate) fn validate_plan(len: usize, boundary: Boundary) -> Result<(), WaveletError> {
    if len == 0 {
        return Err(WaveletError::EmptySignal);
    }
    if len == 1 && matches!(boundary, Boundary::Reflect | Boundary::Antireflect) {
        return Err(WaveletError::BoundaryRequiresLongerSignal {
            len,
            minimum: 2,
            boundary: boundary.as_str(),
        });
    }
    Ok(())
}

pub(crate) fn create_dwt_plan<T: WaveletNum>(
    len: usize,
    wavelet: &Wavelet,
    boundary: Boundary,
    simd_level: SimdLevel,
) -> Result<PlannedDwt<T>, WaveletError> {
    validate_plan(len, boundary)?;
    let filters = PreparedFilterBank::new(wavelet, boundary == Boundary::Periodization);
    Ok(PlannedDwt::new(len, boundary, filters, simd_level))
}

#[derive(Clone, Copy, Debug)]
struct Butterfly<T> {
    low_scale: T,
    high_scale: T,
}

#[derive(Clone, Debug)]
pub(crate) struct PreparedFilterBank<T> {
    data: Arc<[T]>,
    filter_len: usize,
    analysis_butterfly: Option<Butterfly<T>>,
    #[cfg(feature = "experimental-kernels")]
    analysis_annihilator: Option<Arc<AnnihilatorFilter<T>>>,
    #[cfg(feature = "experimental-kernels")]
    analysis_lattice: Option<Arc<LatticeFilter<T>>>,
    synthesis_butterfly: Option<Butterfly<T>>,
}

impl<T: WaveletNum> PreparedFilterBank<T> {
    pub(crate) fn new(wavelet: &Wavelet, periodized: bool) -> Self {
        let filter_len = wavelet.filter_len();
        let mut data = Vec::with_capacity(4 * filter_len);
        data.extend(wavelet.dec_lo().iter().copied().map(T::from_f64));
        data.extend(wavelet.dec_hi().iter().copied().map(T::from_f64));
        #[cfg(feature = "experimental-kernels")]
        let analysis_annihilator =
            AnnihilatorFilter::new(&data[..filter_len], &data[filter_len..2 * filter_len])
                .map(Arc::new);
        #[cfg(all(
            feature = "experimental-kernels",
            any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64")
        ))]
        let analysis_lattice = (!periodized)
            .then(|| LatticeFilter::new(wavelet))
            .flatten()
            .map(Arc::new);
        #[cfg(all(
            feature = "experimental-kernels",
            not(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64"))
        ))]
        let analysis_lattice = None;

        let rec_lo_start = data.len();
        extend_polyphase(&mut data, wavelet.rec_lo());
        if periodized && periodized_phases_are_swapped(filter_len) {
            data[rec_lo_start..].rotate_left(filter_len / 2);
        }

        let rec_hi_start = data.len();
        extend_polyphase(&mut data, wavelet.rec_hi());
        if periodized && periodized_phases_are_swapped(filter_len) {
            data[rec_hi_start..].rotate_left(filter_len / 2);
        }

        Self {
            data: data.into(),
            filter_len,
            analysis_butterfly: analysis_butterfly(wavelet).map(|butterfly| Butterfly {
                low_scale: T::from_f64(butterfly.low_scale),
                high_scale: T::from_f64(butterfly.high_scale),
            }),
            #[cfg(feature = "experimental-kernels")]
            analysis_annihilator,
            #[cfg(feature = "experimental-kernels")]
            analysis_lattice,
            synthesis_butterfly: synthesis_butterfly(wavelet).map(|butterfly| Butterfly {
                low_scale: T::from_f64(butterfly.low_scale),
                high_scale: T::from_f64(butterfly.high_scale),
            }),
        }
    }

    fn analysis(&self) -> (&[T], &[T]) {
        let (dec_lo, remaining) = self.data.split_at(self.filter_len);
        let (dec_hi, _) = remaining.split_at(self.filter_len);
        (dec_lo, dec_hi)
    }

    fn synthesis(&self) -> (&[T], &[T]) {
        let synthesis = &self.data[2 * self.filter_len..];
        synthesis.split_at(self.filter_len)
    }
}

#[derive(Clone, Copy)]
struct F64Butterfly {
    low_scale: f64,
    high_scale: f64,
}

#[derive(Debug)]
pub(crate) struct PlannedDwt<T> {
    signal_len: usize,
    coeff_len: usize,
    filters: PreparedFilterBank<T>,
    analysis: AnalysisPlan<T>,
    periodized_synthesis: Option<PeriodizedSynthesis<T>>,
    axis: AxisPlan,
    simd_level: SimdLevel,
}

impl<T: WaveletNum> PlannedDwt<T> {
    pub(crate) fn new(
        signal_len: usize,
        boundary: Boundary,
        filters: PreparedFilterBank<T>,
        simd_level: SimdLevel,
    ) -> Self {
        let filter_len = filters.filter_len;
        let coeff_len = coefficient_len(signal_len, filter_len, boundary);
        let periodized_synthesis = (boundary == Boundary::Periodization).then(|| {
            let (rec_lo, rec_hi) = filters.synthesis();
            PeriodizedSynthesis::new(signal_len, coeff_len, rec_lo, rec_hi)
        });
        let axis = AxisPlan::new(signal_len, filter_len, periodized_synthesis.is_some());
        let (dec_lo, dec_hi) = filters.analysis();
        #[cfg(feature = "experimental-kernels")]
        let lattice = lattice_simd_supported(simd_level)
            .then(|| filters.analysis_lattice.clone())
            .flatten();
        #[cfg(feature = "experimental-kernels")]
        let annihilator = if lattice.is_some() && lattice_preempts_annihilator(simd_level) {
            None
        } else {
            filters.analysis_annihilator.clone()
        };
        let analysis = build_analysis(
            signal_len,
            coeff_len,
            dec_lo,
            dec_hi,
            AnalysisBackends {
                butterfly: filters.analysis_butterfly,
                #[cfg(feature = "experimental-kernels")]
                annihilator,
                #[cfg(feature = "experimental-kernels")]
                lattice,
            },
            boundary,
        );
        Self {
            signal_len,
            coeff_len,
            filters,
            analysis,
            periodized_synthesis,
            axis,
            simd_level,
        }
    }

    pub(crate) fn full_butterfly_analysis(&self) -> Option<(T, T)> {
        if self.signal_len != 2 * self.coeff_len || self.analysis.prefix_len != 0 {
            return None;
        }
        let interior = self.analysis.interior.as_ref()?;
        if interior.first_newest != 1 || interior.output_len != self.coeff_len {
            return None;
        }
        match &interior.kernel {
            AnalysisKernel::Butterfly {
                low_scale,
                high_scale,
            } => Some((*low_scale, *high_scale)),
            AnalysisKernel::Direct => None,
            #[cfg(feature = "experimental-kernels")]
            AnalysisKernel::Lattice(_) => None,
        }
    }

    pub(crate) fn full_butterfly_synthesis(&self) -> Option<(T, T)> {
        if self.signal_len != 2 * self.coeff_len {
            return None;
        }
        self.filters
            .synthesis_butterfly
            .map(|butterfly| (butterfly.low_scale, butterfly.high_scale))
    }

    fn synthesis_filters(&self) -> (&[T], &[T]) {
        let (rec_lo, rec_hi) = self.filters.synthesis();
        self.periodized_synthesis
            .as_ref()
            .map_or((rec_lo, rec_hi), |layout| layout.filters(rec_lo, rec_hi))
    }
}

impl<T: WaveletNum> Dwt<T> for PlannedDwt<T> {
    fn signal_len(&self) -> usize {
        self.signal_len
    }

    fn coeff_len(&self) -> usize {
        self.coeff_len
    }

    fn scratch_len(&self) -> usize {
        0
    }

    fn axis_scratch_len(&self, outer: usize, inner: usize) -> usize {
        self.axis.scratch_len(self, AxisGeometry::new(outer, inner))
    }

    fn forward_into(&self, signal: &[T], approx: &mut [T], detail: &mut [T], scratch: &mut [T]) {
        assert_eq!(signal.len(), self.signal_len, "incorrect signal length");
        assert_eq!(
            approx.len(),
            self.coeff_len,
            "incorrect approximation length"
        );
        assert_eq!(detail.len(), self.coeff_len, "incorrect detail length");
        assert!(
            scratch.len() >= self.scratch_len(),
            "scratch buffer is too small"
        );

        #[cfg(feature = "experimental-kernels")]
        {
            if let Some(annihilator) = &self.analysis.annihilator
                && annihilator.should_execute(signal)
            {
                annihilator.forward_into(signal, approx, detail);
                return;
            }
        }

        let (dec_lo, dec_hi) = self.filters.analysis();
        let prefix_len = self.analysis.prefix_len;
        analyze_edges(
            signal,
            &self.analysis.edges,
            0,
            dec_lo,
            dec_hi,
            &mut approx[..prefix_len],
            &mut detail[..prefix_len],
        );

        let mut suffix_start = prefix_len;
        if let Some(interior) = &self.analysis.interior {
            let interior_end = prefix_len + interior.output_len;
            let interior_approx = &mut approx[prefix_len..interior_end];
            let interior_detail = &mut detail[prefix_len..interior_end];
            let vectorized = match &interior.kernel {
                AnalysisKernel::Direct => forward_interior_simd(
                    self.simd_level,
                    AnalysisInterior {
                        dec_lo,
                        dec_hi,
                        signal,
                        first_newest: interior.first_newest,
                    },
                    interior_approx,
                    interior_detail,
                ),
                AnalysisKernel::Butterfly {
                    low_scale,
                    high_scale,
                } => forward_butterfly_simd(
                    self.simd_level,
                    ButterflyAnalysis {
                        signal,
                        first_newest: interior.first_newest,
                        low_scale: *low_scale,
                        high_scale: *high_scale,
                    },
                    interior_approx,
                    interior_detail,
                ),
                #[cfg(feature = "experimental-kernels")]
                AnalysisKernel::Lattice(filter) => forward_lattice_simd(
                    self.simd_level,
                    LatticeAnalysis {
                        signal,
                        first_pair: (interior.first_newest - 1) / 2,
                        sections: &filter.sections,
                        scale: filter.scale,
                    },
                    interior_approx,
                    interior_detail,
                ),
            };

            for output in vectorized..interior.output_len {
                let newest = interior.first_newest + 2 * output;
                let (low, high) = match &interior.kernel {
                    AnalysisKernel::Direct => analyze_interior(signal, newest, dec_lo, dec_hi),
                    #[cfg(feature = "experimental-kernels")]
                    AnalysisKernel::Lattice(_) => analyze_interior(signal, newest, dec_lo, dec_hi),
                    AnalysisKernel::Butterfly {
                        low_scale,
                        high_scale,
                    } => {
                        let earlier = signal[newest - 1];
                        let later = signal[newest];
                        (
                            later * *low_scale + earlier * *low_scale,
                            later * (T::zero() - *high_scale) + earlier * *high_scale,
                        )
                    }
                };
                interior_approx[output] = low;
                interior_detail[output] = high;
            }
            suffix_start = interior_end;
        }

        analyze_edges(
            signal,
            &self.analysis.edges,
            prefix_len,
            dec_lo,
            dec_hi,
            &mut approx[suffix_start..],
            &mut detail[suffix_start..],
        );
    }

    fn inverse_into(&self, approx: &[T], detail: &[T], out: &mut [T], scratch: &mut [T]) {
        assert_eq!(
            approx.len(),
            self.coeff_len,
            "incorrect approximation length"
        );
        assert_eq!(detail.len(), self.coeff_len, "incorrect detail length");
        assert_eq!(out.len(), self.signal_len, "incorrect output length");
        assert!(
            scratch.len() >= self.scratch_len(),
            "scratch buffer is too small"
        );

        if let Some(butterfly) = self.filters.synthesis_butterfly {
            inverse_butterfly(self.simd_level, butterfly, approx, detail, out);
        } else if let Some(layout) = &self.periodized_synthesis {
            inverse_periodized(self, layout, approx, detail, out);
        } else {
            inverse_linear(self, approx, detail, out);
        }
    }

    fn forward_axis_into(
        &self,
        signal: &[T],
        outer: usize,
        inner: usize,
        approx: &mut [T],
        detail: &mut [T],
        scratch: &mut [T],
    ) {
        self.axis.forward_into(
            self,
            signal,
            AxisGeometry::new(outer, inner),
            approx,
            detail,
            scratch,
        );
    }

    fn inverse_axis_into(
        &self,
        approx: &[T],
        detail: &[T],
        outer: usize,
        inner: usize,
        out: &mut [T],
        scratch: &mut [T],
    ) {
        self.axis.inverse_into(
            self,
            approx,
            detail,
            AxisGeometry::new(outer, inner),
            out,
            scratch,
        );
    }
}
#[cfg(test)]
mod tests;
