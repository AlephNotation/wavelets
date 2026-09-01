use std::collections::HashMap;
use std::marker::PhantomData;
use std::mem::size_of;
use std::sync::{Arc, Weak};

use fearless_simd::Level as SimdLevel;

use crate::decomposition::{Level, WavedecPlan, resolve_levels};
use crate::lattice::LatticeFilter;
use crate::num::{
    forward_axis_simd, forward_butterfly_simd, forward_interior_simd, forward_lattice_simd,
    inverse_axis_batched_simd, inverse_axis_simd, inverse_butterfly_simd, inverse_linear_simd,
    inverse_periodized_simd, is_finite, mul_add,
};
use crate::simd::{
    AnalysisInterior, AxisAnalysis, AxisSynthesis, ButterflyAnalysis, ButterflySynthesis,
    LatticeAnalysis, LinearSynthesis, MIN_LATTICE_OUTPUTS, PeriodizedInterior,
};
use crate::{Boundary, Wavelet, WaveletError, WaveletNum};

/// A reusable, fixed-length one-level DWT/IDWT plan.
///
/// Buffer-size mistakes are programming errors and cause the `_into` methods
/// to panic. Use the plan's sizing methods to prepare buffers once.
///
/// Plans are returned behind [`Arc`] by [`DwtPlanner::plan_dwt`], so cloning a
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
    /// row-major contiguous tensor without transposing it.
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

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct PlanKey {
    signal_len: usize,
    wavelet_id: u64,
    boundary: Boundary,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct MultilevelPlanKey {
    signal_len: usize,
    wavelet_id: u64,
    boundary: Boundary,
    levels: usize,
}

/// Creates and caches fixed-length discrete wavelet transform plans.
///
/// The planner detects the best available safe SIMD backend once. Repeated
/// requests using the same live [`Wavelet`] and transform configuration share
/// the cached plan.
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
/// use wavelets::{Boundary, DwtPlanner, Wavelet};
///
/// let wavelet = Wavelet::haar();
/// let mut planner = DwtPlanner::<f64>::new();
/// let first = planner.plan_dwt(128, &wavelet, Boundary::Periodization)?;
/// let second = planner.plan_dwt(128, &wavelet, Boundary::Periodization)?;
/// assert!(Arc::ptr_eq(&first, &second));
/// # Ok::<(), wavelets::WaveletError>(())
/// ```
pub struct DwtPlanner<T: WaveletNum> {
    cache: HashMap<PlanKey, Weak<dyn Dwt<T>>>,
    multilevel_cache: HashMap<MultilevelPlanKey, Weak<WavedecPlan<T>>>,
    simd_level: SimdLevel,
    marker: PhantomData<T>,
}

impl<T: WaveletNum> DwtPlanner<T> {
    /// Constructs an empty planner.
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            multilevel_cache: HashMap::new(),
            simd_level: SimdLevel::new(),
            marker: PhantomData,
        }
    }

    /// Plans a one-level transform for signals of exactly `len` samples.
    ///
    /// Planning validates the boundary/length combination and prepares the
    /// edge-extension, polyphase, and applicable structured-signal filter
    /// layouts. Repeated identical requests reuse the same live plan.
    ///
    /// # Errors
    ///
    /// Returns [`WaveletError::EmptySignal`] for `len == 0`, or
    /// [`WaveletError::BoundaryRequiresLongerSignal`] when the selected
    /// extension mode is undefined for `len`.
    pub fn plan_dwt(
        &mut self,
        len: usize,
        wavelet: &Wavelet,
        boundary: Boundary,
    ) -> Result<Arc<dyn Dwt<T>>, WaveletError> {
        let key = PlanKey {
            signal_len: len,
            wavelet_id: wavelet.id(),
            boundary,
        };
        if let Some(plan) = self.cache.get(&key).and_then(Weak::upgrade) {
            return Ok(plan);
        }

        let plan: Arc<dyn Dwt<T>> =
            Arc::new(create_dwt_plan(len, wavelet, boundary, self.simd_level)?);
        self.cache.insert(key, Arc::downgrade(&plan));
        Ok(plan)
    }

    /// Plans a multilevel transform for signals of exactly `len` samples.
    ///
    /// Every single-level plan, band offset, and scratch region is prepared up
    /// front. Repeated requests resolving to the same number of levels reuse
    /// the same live plan.
    ///
    /// # Errors
    ///
    /// Returns [`WaveletError::EmptySignal`] for `len == 0`,
    /// [`WaveletError::InvalidLevel`] when an exact level exceeds the maximum,
    /// or a boundary/length planning error at an intermediate level.
    pub fn plan_wavedec(
        &mut self,
        len: usize,
        wavelet: &Wavelet,
        boundary: Boundary,
        level: Level,
    ) -> Result<Arc<WavedecPlan<T>>, WaveletError> {
        let levels = resolve_levels(len, wavelet.filter_len(), level)?;
        let key = MultilevelPlanKey {
            signal_len: len,
            wavelet_id: wavelet.id(),
            boundary,
            levels,
        };
        if let Some(plan) = self.multilevel_cache.get(&key).and_then(Weak::upgrade) {
            return Ok(plan);
        }

        let filters = PreparedFilterBank::new(wavelet, boundary == Boundary::Periodization);
        let plan = Arc::new(WavedecPlan::new(
            len,
            wavelet,
            boundary,
            levels,
            filters,
            self.simd_level,
        )?);
        self.multilevel_cache.insert(key, Arc::downgrade(&plan));
        Ok(plan)
    }
}

impl<T: WaveletNum> Default for DwtPlanner<T> {
    fn default() -> Self {
        Self::new()
    }
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

#[derive(Debug)]
struct InteriorAnalysis<T> {
    first_newest: usize,
    output_len: usize,
    kernel: AnalysisKernel<T>,
}

#[derive(Clone, Debug)]
enum AnalysisKernel<T> {
    Direct,
    Butterfly { low_scale: T, high_scale: T },
    Lattice(Arc<LatticeFilter<T>>),
}

#[derive(Clone, Copy, Debug)]
struct Butterfly<T> {
    low_scale: T,
    high_scale: T,
}

#[derive(Debug)]
struct AnalysisPlan<T> {
    edges: EdgePlan<T>,
    prefix_len: usize,
    interior: Option<InteriorAnalysis<T>>,
    annihilator: Option<AnnihilatorAnalysis<T>>,
}

struct AnalysisBackends<T> {
    butterfly: Option<Butterfly<T>>,
    annihilator: Option<Arc<AnnihilatorFilter<T>>>,
    lattice: Option<Arc<LatticeFilter<T>>>,
}

// The scan costs more than the existing SIMD kernel below this support on the
// measured NEON backend. Keeping the cutoff algebraic lets equivalent custom
// banks qualify without coupling execution to built-in wavelet names.
const MIN_ANNIHILATOR_FILTER_LEN_F64: usize = 76;
const MIN_ANNIHILATOR_FILTER_LEN_F32: usize = 102;
const ANNIHILATOR_BASE_FILTER_LEN: usize = 64;
const ANNIHILATOR_EVENT_COST_SCALE_F64: usize = 6;
const ANNIHILATOR_EVENT_COST_SCALE_F32: usize = 12;

#[derive(Debug)]
struct AnnihilatorFilter<T> {
    low_base: T,
    high_base: T,
    low_correction: Box<[T]>,
    high_correction: Box<[T]>,
}

impl<T: WaveletNum> AnnihilatorFilter<T> {
    fn new(dec_lo: &[T], dec_hi: &[T]) -> Option<Self> {
        let minimum_filter_len = if size_of::<T>() == size_of::<f32>() {
            MIN_ANNIHILATOR_FILTER_LEN_F32
        } else {
            MIN_ANNIHILATOR_FILTER_LEN_F64
        };
        if dec_lo.len() < minimum_filter_len {
            return None;
        }
        let (low_base, low_correction) = factor_degree_zero(dec_lo);
        let (high_base, high_correction) = factor_degree_zero(dec_hi);
        Some(Self {
            low_base,
            high_base,
            low_correction: low_correction.into_boxed_slice(),
            high_correction: high_correction.into_boxed_slice(),
        })
    }
}

#[derive(Debug)]
struct AnnihilatorAnalysis<T> {
    filter: Arc<AnnihilatorFilter<T>>,
    boundary: Boundary,
    phase: isize,
    first_extended_index: isize,
    extension_len: usize,
    maximum_events: usize,
}

impl<T: WaveletNum> AnnihilatorAnalysis<T> {
    fn new(
        signal_len: usize,
        coeff_len: usize,
        boundary: Boundary,
        filter: Arc<AnnihilatorFilter<T>>,
    ) -> Self {
        let filter_len = filter.low_correction.len() + 1;
        let phase = if boundary == Boundary::Periodization {
            (filter_len / 2) as isize
        } else {
            1
        };
        let first_extended_index = phase - (filter_len - 1) as isize;
        let extension_len = 2 * coeff_len - 2 + filter_len;
        // A correction event touches about half the filter in both bands. This
        // conservative M4-derived budget admits the measured db38/coif17 win
        // regions while rejecting db20-like marginal cases entirely above.
        // f32 direct SIMD processes twice as many samples per vector while
        // scalar correction scattering does not, so each event must be
        // charged twice as heavily as f64.
        let event_cost_scale = if size_of::<T>() == size_of::<f32>() {
            ANNIHILATOR_EVENT_COST_SCALE_F32
        } else {
            ANNIHILATOR_EVENT_COST_SCALE_F64
        };
        let maximum_events = ((signal_len as u128
            * filter_len.saturating_sub(ANNIHILATOR_BASE_FILTER_LEN) as u128)
            / (event_cost_scale * filter_len) as u128) as usize;
        Self {
            filter,
            boundary,
            phase,
            first_extended_index,
            extension_len,
            maximum_events,
        }
    }

    fn should_execute(&self, signal: &[T]) -> bool {
        let mut events = 0;
        let mut previous = signal[0];
        if !is_finite(previous) {
            return false;
        }
        for &current in &signal[1..] {
            if !is_finite(current) {
                return false;
            }
            let amplitude = current - previous;
            if !is_finite(amplitude) {
                return false;
            }
            if amplitude != T::zero() {
                events += 1;
                if events > self.maximum_events {
                    return false;
                }
            }
            previous = current;
        }

        // The finite extension can introduce additional jumps even when the
        // original signal is sparse. Count only the two O(filter_len) halos;
        // the interior transitions were counted above.
        if self.first_extended_index < 0 {
            let mut previous = extended_sample(signal, self.first_extended_index, self.boundary);
            if !is_finite(previous) {
                return false;
            }
            for index in self.first_extended_index + 1..=0 {
                let current = extended_sample(signal, index, self.boundary);
                let amplitude = current - previous;
                if !is_finite(current) || !is_finite(amplitude) {
                    return false;
                }
                if amplitude != T::zero() {
                    events += 1;
                    if events > self.maximum_events {
                        return false;
                    }
                }
                previous = current;
            }
        }

        let final_extended_index = self.first_extended_index + self.extension_len as isize - 1;
        let final_signal_index = signal.len() as isize - 1;
        if final_extended_index > final_signal_index {
            let mut previous = signal[signal.len() - 1];
            for index in signal.len() as isize..=final_extended_index {
                let current = extended_sample(signal, index, self.boundary);
                let amplitude = current - previous;
                if !is_finite(current) || !is_finite(amplitude) {
                    return false;
                }
                if amplitude != T::zero() {
                    events += 1;
                    if events > self.maximum_events {
                        return false;
                    }
                }
                previous = current;
            }
        }
        true
    }

    fn forward_into(&self, signal: &[T], approx: &mut [T], detail: &mut [T]) {
        for coefficient in 0..approx.len() {
            let base_index = self.first_extended_index + 2 * coefficient as isize;
            let sample = extended_sample(signal, base_index, self.boundary);
            approx[coefficient] = self.filter.low_base * sample;
            detail[coefficient] = self.filter.high_base * sample;
        }

        let mut previous = extended_sample(signal, self.first_extended_index, self.boundary);
        for offset in 1..self.extension_len {
            let event = self.first_extended_index + offset as isize;
            let current = extended_sample(signal, event, self.boundary);
            let amplitude = current - previous;
            if amplitude != T::zero() {
                self.scatter_event(event, amplitude, approx, detail);
            }
            previous = current;
        }
    }

    #[inline]
    fn scatter_event(&self, event: isize, amplitude: T, approx: &mut [T], detail: &mut [T]) {
        let first_tap = (self.phase - event).rem_euclid(2) as usize;
        for tap in (first_tap..self.filter.low_correction.len()).step_by(2) {
            let output_offset = event + tap as isize - self.phase;
            if output_offset < 0 {
                continue;
            }
            let coefficient = output_offset as usize / 2;
            if coefficient >= approx.len() {
                break;
            }
            approx[coefficient] = mul_add(
                amplitude,
                self.filter.low_correction[tap],
                approx[coefficient],
            );
            detail[coefficient] = mul_add(
                amplitude,
                self.filter.high_correction[tap],
                detail[coefficient],
            );
        }
    }
}

fn factor_degree_zero<T: WaveletNum>(filter: &[T]) -> (T, Vec<T>) {
    debug_assert!(filter.len() >= 2);
    let mut base = T::zero();
    for &tap in filter {
        base += tap;
    }
    let mut running = T::zero();
    let correction = filter[..filter.len() - 1]
        .iter()
        .map(|&tap| {
            running += tap;
            running
        })
        .collect();
    (base, correction)
}

#[derive(Debug)]
pub(crate) struct EdgePlan<T> {
    // Each row is the filter-composed finite-boundary transform for one
    // approximation/detail coefficient pair. Both channels share every input
    // load, and repeated references to the same finite sample are coalesced.
    pub(crate) row_offsets: Box<[usize]>,
    pub(crate) terms: Box<[EdgeTerm<T>]>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct EdgeTerm<T> {
    pub(crate) input: usize,
    pub(crate) low: T,
    pub(crate) high: T,
}

#[derive(Clone, Copy, Debug)]
struct PeriodizedSynthesis {
    initial_coefficient: usize,
    simd_start: usize,
    simd_available: usize,
    phases_are_swapped: bool,
}

// Below 24 taps, reduced coefficient loads do not repay the batched kernel's
// extra bookkeeping on the supported SIMD backends.
const MIN_BATCHED_AXIS_HALF_FILTER_LEN: usize = 12;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AxisSynthesisKernel {
    Direct,
    Batched,
}

impl AxisSynthesisKernel {
    fn select(signal_len: usize, filter_len: usize, periodized: bool) -> Self {
        if !periodized
            && filter_len / 2 >= MIN_BATCHED_AXIS_HALF_FILTER_LEN
            && signal_len.div_ceil(2) >= 2
        {
            Self::Batched
        } else {
            Self::Direct
        }
    }
}

impl PeriodizedSynthesis {
    fn new(signal_len: usize, coeff_len: usize, filter_len: usize) -> Self {
        let half_filter_len = filter_len / 2;
        let shift = half_filter_len - 1;
        let phases_are_swapped = periodized_phases_are_swapped(filter_len);
        let initial_coefficient = (shift / 2) % coeff_len;
        let complete_pairs = signal_len / 2;
        let (simd_start, simd_available) = if coeff_len >= half_filter_len {
            let start = half_filter_len - 1 - initial_coefficient;
            let final_coefficient = coeff_len - usize::from(phases_are_swapped);
            let available = final_coefficient
                .saturating_sub(half_filter_len - 1)
                .min(complete_pairs.saturating_sub(start));
            (start, available)
        } else {
            (0, 0)
        };
        Self {
            initial_coefficient,
            simd_start,
            simd_available,
            phases_are_swapped,
        }
    }
}

fn periodized_phases_are_swapped(filter_len: usize) -> bool {
    !(filter_len / 2 - 1).is_multiple_of(2)
}

#[derive(Clone, Debug)]
pub(crate) struct PreparedFilterBank<T> {
    data: Arc<[T]>,
    filter_len: usize,
    analysis_butterfly: Option<Butterfly<T>>,
    analysis_annihilator: Option<Arc<AnnihilatorFilter<T>>>,
    analysis_lattice: Option<Arc<LatticeFilter<T>>>,
    synthesis_butterfly: Option<Butterfly<T>>,
}

impl<T: WaveletNum> PreparedFilterBank<T> {
    pub(crate) fn new(wavelet: &Wavelet, periodized: bool) -> Self {
        let filter_len = wavelet.filter_len();
        let mut data = Vec::with_capacity(4 * filter_len);
        data.extend(wavelet.dec_lo().iter().copied().map(T::from_f64));
        data.extend(wavelet.dec_hi().iter().copied().map(T::from_f64));
        let analysis_annihilator =
            AnnihilatorFilter::new(&data[..filter_len], &data[filter_len..2 * filter_len])
                .map(Arc::new);
        #[cfg(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64"))]
        let analysis_lattice = (!periodized)
            .then(|| LatticeFilter::new(wavelet))
            .flatten()
            .map(Arc::new);
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64")))]
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
            analysis_annihilator,
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

fn analysis_butterfly(wavelet: &Wavelet) -> Option<F64Butterfly> {
    // Detect the matrix factorization itself rather than a built-in name so
    // equivalent caller-supplied banks select the same kernel.
    let [low_first, low_second] = wavelet.dec_lo() else {
        return None;
    };
    let [high_first, high_second] = wavelet.dec_hi() else {
        return None;
    };
    (*low_first == *low_second && *high_first == -*high_second).then_some(F64Butterfly {
        low_scale: *low_first,
        high_scale: *high_second,
    })
}

fn synthesis_butterfly(wavelet: &Wavelet) -> Option<F64Butterfly> {
    // Synthesis may qualify independently from analysis for a custom bank.
    let [low_first, low_second] = wavelet.rec_lo() else {
        return None;
    };
    let [high_first, high_second] = wavelet.rec_hi() else {
        return None;
    };
    (*low_first == *low_second && *high_first == -*high_second).then_some(F64Butterfly {
        low_scale: *low_first,
        high_scale: *high_first,
    })
}

#[derive(Debug)]
pub(crate) struct PlannedDwt<T> {
    signal_len: usize,
    coeff_len: usize,
    filters: PreparedFilterBank<T>,
    analysis: AnalysisPlan<T>,
    periodized_synthesis: Option<PeriodizedSynthesis>,
    axis_synthesis_kernel: AxisSynthesisKernel,
    simd_level: SimdLevel,
}

fn lattice_simd_supported(level: SimdLevel) -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        !level.is_fallback()
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        level.as_avx512().is_some()
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64")))]
    {
        let _ = level;
        false
    }
}

fn lattice_preempts_annihilator(level: SimdLevel) -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        // On AVX-512, the lattice is faster even than the annihilator's
        // zero-event endpoint. Scanning for structure can therefore never
        // select a better executor once this backend is available.
        level.as_avx512().is_some()
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        let _ = level;
        false
    }
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
        let periodized_synthesis = (boundary == Boundary::Periodization)
            .then(|| PeriodizedSynthesis::new(signal_len, coeff_len, filter_len));
        let axis_synthesis_kernel =
            AxisSynthesisKernel::select(signal_len, filter_len, periodized_synthesis.is_some());
        let (dec_lo, dec_hi) = filters.analysis();
        let lattice = lattice_simd_supported(simd_level)
            .then(|| filters.analysis_lattice.clone())
            .flatten();
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
                annihilator,
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
            axis_synthesis_kernel,
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
            AnalysisKernel::Direct | AnalysisKernel::Lattice(_) => None,
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

    fn inverse_linear(&self, approx: &[T], detail: &[T], out: &mut [T]) {
        let (rec_lo, rec_hi) = self.filters.synthesis();
        let half_filter_len = rec_lo.len() / 2;
        let (even_lo, odd_lo) = rec_lo.split_at(half_filter_len);
        let (even_hi, odd_hi) = rec_hi.split_at(half_filter_len);

        let vectorized_pairs = inverse_linear_simd(
            self.simd_level,
            LinearSynthesis {
                rec_lo,
                rec_hi,
                approx,
                detail,
            },
            out,
        );

        // Cropping the full convolution by `filter_len - 2` makes each output
        // pair consume the same reversed coefficient window. Fusing both
        // polyphase dots keeps that window hot and loads it only once.
        for (tail_coefficient, samples) in out[2 * vectorized_pairs..].chunks_mut(2).enumerate() {
            let coefficient = vectorized_pairs + tail_coefficient;
            let coefficient_end = coefficient + half_filter_len;
            let approx = &approx[coefficient..coefficient_end];
            let detail = &detail[coefficient..coefficient_end];
            let (first, second) = synthesis_pair(even_lo, even_hi, odd_lo, odd_hi, approx, detail);
            samples[0] = first;
            if samples.len() == 2 {
                samples[1] = second;
            }
        }
    }

    fn inverse_periodized(
        &self,
        layout: PeriodizedSynthesis,
        approx: &[T],
        detail: &[T],
        out: &mut [T],
    ) {
        let (rec_lo, rec_hi) = self.filters.synthesis();
        let half_filter_len = rec_lo.len() / 2;
        let (first_lo, second_lo) = rec_lo.split_at(half_filter_len);
        let (first_hi, second_hi) = rec_hi.split_at(half_filter_len);

        let (scalar_prefix, remainder) = out[..2 * layout.simd_start].as_chunks_mut::<2>();
        debug_assert!(remainder.is_empty());
        for (pair, samples) in scalar_prefix.iter_mut().enumerate() {
            let first_coefficient = (layout.initial_coefficient + pair) % self.coeff_len;
            let (first, second) = synthesis_periodized_pair(
                (first_lo, first_hi),
                (second_lo, second_hi),
                approx,
                detail,
                first_coefficient,
                layout.phases_are_swapped,
            );
            samples[0] = first;
            samples[1] = second;
        }

        let vectorized = inverse_periodized_simd(
            self.simd_level,
            PeriodizedInterior {
                first_lo,
                first_hi,
                second_lo,
                second_hi,
                approx,
                detail,
                first_coefficient: half_filter_len - 1,
                second_offset: usize::from(layout.phases_are_swapped),
            },
            &mut out[2 * layout.simd_start..2 * (layout.simd_start + layout.simd_available)],
        );

        let scalar_start = layout.simd_start + vectorized;
        for (tail_pair, samples) in out[2 * scalar_start..].chunks_mut(2).enumerate() {
            let pair = scalar_start + tail_pair;
            let first_coefficient = (layout.initial_coefficient + pair) % self.coeff_len;
            let (first, second) = synthesis_periodized_pair(
                (first_lo, first_hi),
                (second_lo, second_hi),
                approx,
                detail,
                first_coefficient,
                layout.phases_are_swapped,
            );
            samples[0] = first;
            if samples.len() == 2 {
                samples[1] = second;
            }
        }
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

        if let Some(annihilator) = &self.analysis.annihilator
            && annihilator.should_execute(signal)
        {
            annihilator.forward_into(signal, approx, detail);
            return;
        }

        let (dec_lo, dec_hi) = self.filters.analysis();
        let prefix_len = self.analysis.prefix_len;
        analyze_edges(
            signal,
            &self.analysis.edges,
            0,
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
                    AnalysisKernel::Direct | AnalysisKernel::Lattice(_) => {
                        analyze_interior(signal, newest, dec_lo, dec_hi)
                    }
                    AnalysisKernel::Butterfly {
                        low_scale,
                        high_scale,
                    } => {
                        let earlier = signal[newest - 1];
                        let later = signal[newest];
                        (
                            (earlier + later) * *low_scale,
                            (earlier - later) * *high_scale,
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
        } else if let Some(layout) = self.periodized_synthesis {
            self.inverse_periodized(layout, approx, detail, out);
        } else {
            self.inverse_linear(approx, detail, out);
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
        assert_eq!(
            signal.len(),
            axis_buffer_len(outer, self.signal_len, inner),
            "incorrect axis input length"
        );
        let output_len = axis_buffer_len(outer, self.coeff_len, inner);
        assert_eq!(
            approx.len(),
            output_len,
            "incorrect axis approximation length"
        );
        assert_eq!(detail.len(), output_len, "incorrect axis detail length");
        assert!(
            scratch.len() >= self.scratch_len(),
            "scratch buffer is too small"
        );

        if inner == 1 {
            for outer_index in 0..outer {
                let signal_start = outer_index * self.signal_len;
                let output_start = outer_index * self.coeff_len;
                self.forward_into(
                    &signal[signal_start..signal_start + self.signal_len],
                    &mut approx[output_start..output_start + self.coeff_len],
                    &mut detail[output_start..output_start + self.coeff_len],
                    scratch,
                );
            }
            return;
        }

        let (dec_lo, dec_hi) = self.filters.analysis();
        let (interior_first_newest, interior_len) =
            self.analysis.interior.as_ref().map_or((0, 0), |interior| {
                (interior.first_newest, interior.output_len)
            });
        let vectorized = forward_axis_simd(
            self.simd_level,
            AxisAnalysis {
                signal,
                dec_lo,
                dec_hi,
                edge_row_offsets: &self.analysis.edges.row_offsets,
                edge_terms: &self.analysis.edges.terms,
                signal_len: self.signal_len,
                coeff_len: self.coeff_len,
                outer,
                inner,
                prefix_len: self.analysis.prefix_len,
                interior_first_newest,
                interior_len,
            },
            approx,
            detail,
        );
        analyze_axis_tail(self, signal, outer, inner, vectorized, approx, detail);
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
        let coefficient_len = axis_buffer_len(outer, self.coeff_len, inner);
        assert_eq!(
            approx.len(),
            coefficient_len,
            "incorrect axis approximation length"
        );
        assert_eq!(
            detail.len(),
            coefficient_len,
            "incorrect axis detail length"
        );
        assert_eq!(
            out.len(),
            axis_buffer_len(outer, self.signal_len, inner),
            "incorrect axis output length"
        );
        assert!(
            scratch.len() >= self.scratch_len(),
            "scratch buffer is too small"
        );

        if inner == 1 {
            for outer_index in 0..outer {
                let coeff_start = outer_index * self.coeff_len;
                let output_start = outer_index * self.signal_len;
                self.inverse_into(
                    &approx[coeff_start..coeff_start + self.coeff_len],
                    &detail[coeff_start..coeff_start + self.coeff_len],
                    &mut out[output_start..output_start + self.signal_len],
                    scratch,
                );
            }
            return;
        }

        let (rec_lo, rec_hi) = self.filters.synthesis();
        let synthesis = AxisSynthesis {
            approx,
            detail,
            rec_lo,
            rec_hi,
            signal_len: self.signal_len,
            coeff_len: self.coeff_len,
            outer,
            inner,
            periodized_initial: self
                .periodized_synthesis
                .map(|layout| layout.initial_coefficient),
            periodized_phases_are_swapped: self
                .periodized_synthesis
                .is_some_and(|layout| layout.phases_are_swapped),
        };
        let vectorized = match self.axis_synthesis_kernel {
            AxisSynthesisKernel::Direct => inverse_axis_simd(self.simd_level, synthesis, out),
            AxisSynthesisKernel::Batched => {
                inverse_axis_batched_simd(self.simd_level, synthesis, out)
            }
        };
        synthesize_axis_tail(self, approx, detail, outer, inner, vectorized, out);
    }
}

fn axis_buffer_len(outer: usize, axis: usize, inner: usize) -> usize {
    outer
        .checked_mul(axis)
        .and_then(|value| value.checked_mul(inner))
        .expect("axis buffer length overflow")
}

fn analyze_axis_tail<T: WaveletNum>(
    plan: &PlannedDwt<T>,
    signal: &[T],
    outer: usize,
    inner: usize,
    first_lane: usize,
    approx: &mut [T],
    detail: &mut [T],
) {
    let (dec_lo, dec_hi) = plan.filters.analysis();
    let prefix_len = plan.analysis.prefix_len;
    let interior_len = plan
        .analysis
        .interior
        .as_ref()
        .map_or(0, |interior| interior.output_len);
    let interior_end = prefix_len + interior_len;

    for outer_index in 0..outer {
        let signal_start = outer_index * plan.signal_len * inner;
        let output_start = outer_index * plan.coeff_len * inner;
        for lane in first_lane..inner {
            for output in 0..prefix_len {
                let (low, high) = analyze_axis_edge_scalar(
                    &signal[signal_start..],
                    inner,
                    lane,
                    &plan.analysis.edges,
                    output,
                );
                approx[output_start + output * inner + lane] = low;
                detail[output_start + output * inner + lane] = high;
            }

            if let Some(interior) = &plan.analysis.interior {
                for interior_output in 0..interior.output_len {
                    let newest = interior.first_newest + 2 * interior_output;
                    let mut low = T::zero();
                    let mut high = T::zero();
                    for tap in 0..dec_lo.len() {
                        let sample = signal[signal_start + (newest - tap) * inner + lane];
                        low = mul_add(sample, dec_lo[tap], low);
                        high = mul_add(sample, dec_hi[tap], high);
                    }
                    let output = prefix_len + interior_output;
                    approx[output_start + output * inner + lane] = low;
                    detail[output_start + output * inner + lane] = high;
                }
            }

            for output in interior_end..plan.coeff_len {
                let edge_row = prefix_len + output - interior_end;
                let (low, high) = analyze_axis_edge_scalar(
                    &signal[signal_start..],
                    inner,
                    lane,
                    &plan.analysis.edges,
                    edge_row,
                );
                approx[output_start + output * inner + lane] = low;
                detail[output_start + output * inner + lane] = high;
            }
        }
    }
}

fn analyze_axis_edge_scalar<T: WaveletNum>(
    signal: &[T],
    inner: usize,
    lane: usize,
    edges: &EdgePlan<T>,
    row: usize,
) -> (T, T) {
    let mut low = T::zero();
    let mut high = T::zero();
    for term in &edges.terms[edges.row_offsets[row]..edges.row_offsets[row + 1]] {
        let sample = signal[term.input * inner + lane];
        low = mul_add(sample, term.low, low);
        high = mul_add(sample, term.high, high);
    }
    (low, high)
}

fn synthesize_axis_tail<T: WaveletNum>(
    plan: &PlannedDwt<T>,
    approx: &[T],
    detail: &[T],
    outer: usize,
    inner: usize,
    first_lane: usize,
    out: &mut [T],
) {
    let (rec_lo, rec_hi) = plan.filters.synthesis();
    let half_filter_len = rec_lo.len() / 2;
    let (first_lo, second_lo) = rec_lo.split_at(half_filter_len);
    let (first_hi, second_hi) = rec_hi.split_at(half_filter_len);
    let output_pairs = plan.signal_len.div_ceil(2);

    for outer_index in 0..outer {
        let coeff_start = outer_index * plan.coeff_len * inner;
        let output_start = outer_index * plan.signal_len * inner;
        for lane in first_lane..inner {
            for pair in 0..output_pairs {
                let first_coefficient = plan
                    .periodized_synthesis
                    .map_or(pair + half_filter_len - 1, |layout| {
                        (layout.initial_coefficient + pair) % plan.coeff_len
                    });
                let second_coefficient = if plan
                    .periodized_synthesis
                    .is_some_and(|layout| layout.phases_are_swapped)
                {
                    increment_wrapping(first_coefficient, plan.coeff_len)
                } else {
                    first_coefficient
                };
                let mut first = T::zero();
                let mut second = T::zero();
                for tap in 0..half_filter_len {
                    let first_index = if plan.periodized_synthesis.is_some() {
                        (first_coefficient + plan.coeff_len - tap % plan.coeff_len) % plan.coeff_len
                    } else {
                        first_coefficient - tap
                    };
                    let second_index = if plan.periodized_synthesis.is_some() {
                        (second_coefficient + plan.coeff_len - tap % plan.coeff_len)
                            % plan.coeff_len
                    } else {
                        second_coefficient - tap
                    };
                    let first_offset = coeff_start + first_index * inner + lane;
                    let second_offset = coeff_start + second_index * inner + lane;
                    first = mul_add(approx[first_offset], first_lo[tap], first);
                    first = mul_add(detail[first_offset], first_hi[tap], first);
                    second = mul_add(approx[second_offset], second_lo[tap], second);
                    second = mul_add(detail[second_offset], second_hi[tap], second);
                }
                let first_output = output_start + 2 * pair * inner + lane;
                out[first_output] = first;
                if 2 * pair + 1 < plan.signal_len {
                    out[first_output + inner] = second;
                }
            }
        }
    }
}

fn inverse_butterfly<T: WaveletNum>(
    simd_level: SimdLevel,
    butterfly: Butterfly<T>,
    approx: &[T],
    detail: &[T],
    out: &mut [T],
) {
    let vectorized_pairs = inverse_butterfly_simd(
        simd_level,
        ButterflySynthesis {
            approx,
            detail,
            low_scale: butterfly.low_scale,
            high_scale: butterfly.high_scale,
        },
        out,
    );

    for (pair, samples) in out[2 * vectorized_pairs..].chunks_mut(2).enumerate() {
        let coefficient = vectorized_pairs + pair;
        let low = approx[coefficient] * butterfly.low_scale;
        let high = detail[coefficient] * butterfly.high_scale;
        samples[0] = low + high;
        if samples.len() == 2 {
            samples[1] = low - high;
        }
    }
}

fn extend_polyphase<T: WaveletNum>(out: &mut Vec<T>, filter: &[f64]) {
    debug_assert!(filter.len().is_multiple_of(2));
    out.extend(
        filter
            .iter()
            .step_by(2)
            .chain(filter.iter().skip(1).step_by(2))
            .copied()
            .map(T::from_f64),
    );
}

#[inline(always)]
fn synthesis_pair<T: WaveletNum>(
    even_lo: &[T],
    even_hi: &[T],
    odd_lo: &[T],
    odd_hi: &[T],
    approx: &[T],
    detail: &[T],
) -> (T, T) {
    let mut even = T::zero();
    let mut odd = T::zero();
    for ((((even_low, even_high), (odd_low, odd_high)), approximation), detail) in even_lo
        .iter()
        .zip(even_hi)
        .zip(odd_lo.iter().zip(odd_hi))
        .zip(approx.iter().rev())
        .zip(detail.iter().rev())
    {
        even += *even_low * *approximation;
        even += *even_high * *detail;
        odd += *odd_low * *approximation;
        odd += *odd_high * *detail;
    }
    (even, odd)
}

#[inline(always)]
fn synthesis_pair_windows<T: WaveletNum>(
    (first_lo, first_hi): (&[T], &[T]),
    (second_lo, second_hi): (&[T], &[T]),
    (first_approx, first_detail): (&[T], &[T]),
    (second_approx, second_detail): (&[T], &[T]),
) -> (T, T) {
    let mut first = T::zero();
    let mut second = T::zero();
    for (
        ((first_low, first_high), (second_low, second_high)),
        ((first_approx, first_detail), (second_approx, second_detail)),
    ) in first_lo
        .iter()
        .zip(first_hi)
        .zip(second_lo.iter().zip(second_hi))
        .zip(
            first_approx
                .iter()
                .rev()
                .zip(first_detail.iter().rev())
                .zip(second_approx.iter().rev().zip(second_detail.iter().rev())),
        )
    {
        first += *first_low * *first_approx;
        first += *first_high * *first_detail;
        second += *second_low * *second_approx;
        second += *second_high * *second_detail;
    }
    (first, second)
}

#[inline(always)]
fn synthesis_pair_cyclic<T: WaveletNum>(
    (first_lo, first_hi): (&[T], &[T]),
    (second_lo, second_hi): (&[T], &[T]),
    approx: &[T],
    detail: &[T],
    first_coefficient: usize,
    second_coefficient: usize,
) -> (T, T) {
    let mut first = T::zero();
    let mut second = T::zero();
    let mut first_coefficient = first_coefficient;
    let mut second_coefficient = second_coefficient;
    for (((first_low, first_high), second_low), second_high) in
        first_lo.iter().zip(first_hi).zip(second_lo).zip(second_hi)
    {
        first += *first_low * approx[first_coefficient];
        first += *first_high * detail[first_coefficient];
        second += *second_low * approx[second_coefficient];
        second += *second_high * detail[second_coefficient];
        first_coefficient = decrement_wrapping(first_coefficient, approx.len());
        second_coefficient = decrement_wrapping(second_coefficient, approx.len());
    }
    (first, second)
}

#[inline(always)]
fn synthesis_periodized_pair<T: WaveletNum>(
    first_filters: (&[T], &[T]),
    second_filters: (&[T], &[T]),
    approx: &[T],
    detail: &[T],
    first_coefficient: usize,
    phases_are_swapped: bool,
) -> (T, T) {
    let second_coefficient = if phases_are_swapped {
        increment_wrapping(first_coefficient, approx.len())
    } else {
        first_coefficient
    };
    let filter_len = first_filters.0.len();

    if first_coefficient + 1 >= filter_len && second_coefficient + 1 >= filter_len {
        let first_start = first_coefficient + 1 - filter_len;
        let second_start = second_coefficient + 1 - filter_len;
        synthesis_pair_windows(
            first_filters,
            second_filters,
            (
                &approx[first_start..=first_coefficient],
                &detail[first_start..=first_coefficient],
            ),
            (
                &approx[second_start..=second_coefficient],
                &detail[second_start..=second_coefficient],
            ),
        )
    } else {
        synthesis_pair_cyclic(
            first_filters,
            second_filters,
            approx,
            detail,
            first_coefficient,
            second_coefficient,
        )
    }
}

#[inline]
fn increment_wrapping(value: usize, len: usize) -> usize {
    if value + 1 == len { 0 } else { value + 1 }
}

#[inline]
fn decrement_wrapping(value: usize, len: usize) -> usize {
    if value == 0 { len - 1 } else { value - 1 }
}

pub(crate) fn coefficient_len(signal_len: usize, filter_len: usize, boundary: Boundary) -> usize {
    if boundary == Boundary::Periodization {
        signal_len.div_ceil(2)
    } else {
        (signal_len + filter_len - 1) / 2
    }
}

fn build_analysis<T: WaveletNum>(
    signal_len: usize,
    coeff_len: usize,
    dec_lo: &[T],
    dec_hi: &[T],
    backends: AnalysisBackends<T>,
    boundary: Boundary,
) -> AnalysisPlan<T> {
    debug_assert_eq!(dec_lo.len(), dec_hi.len());
    let filter_len = dec_lo.len();
    let phase = if boundary == Boundary::Periodization {
        filter_len / 2
    } else {
        1
    };
    let is_interior = |coefficient: usize| {
        let newest = (2 * coefficient + phase) as isize;
        let oldest = newest - (filter_len - 1) as isize;
        oldest >= 0 && newest < signal_len as isize
    };
    let interior_start = (0..coeff_len).find(|&coefficient| is_interior(coefficient));
    let interior_end = interior_start.map_or(coeff_len, |start| {
        start
            + (start..coeff_len)
                .take_while(|&coefficient| is_interior(coefficient))
                .count()
    });
    debug_assert!((interior_end..coeff_len).all(|coefficient| !is_interior(coefficient)));

    let prefix_end = interior_start.unwrap_or(coeff_len);
    let edge_count = prefix_end + coeff_len - interior_end;
    let mut row_offsets = Vec::with_capacity(edge_count + 1);
    let mut terms = Vec::<EdgeTerm<T>>::with_capacity(edge_count * filter_len);
    // A dense position map avoids hashing when it is no larger than the raw
    // edge-rule grid. Large signals retain an O(filter_len) sparse planner.
    let dense_position_limit = edge_count.saturating_mul(filter_len);
    let mut dense_positions =
        (signal_len <= dense_position_limit).then(|| vec![usize::MAX; signal_len]);
    let mut sparse_positions = dense_positions
        .is_none()
        .then(|| HashMap::<usize, usize>::with_capacity(3 * filter_len));
    row_offsets.push(0);
    for coefficient in (0..prefix_end).chain(interior_end..coeff_len) {
        if let Some(positions) = &mut sparse_positions {
            positions.clear();
        }
        let row_start = terms.len();
        let newest = (2 * coefficient + phase) as isize;
        for tap in 0..filter_len {
            for_each_extension_term(
                newest - tap as isize,
                signal_len,
                boundary,
                |input, weight| {
                    let low = dec_lo[tap] * weight;
                    let high = dec_hi[tap] * weight;
                    let position = if let Some(positions) = &mut dense_positions {
                        let position = positions[input];
                        if position != usize::MAX && position >= row_start {
                            Some(position)
                        } else {
                            positions[input] = terms.len();
                            None
                        }
                    } else {
                        let positions = sparse_positions
                            .as_mut()
                            .expect("one edge position map is always available");
                        if let Some(&position) = positions.get(&input) {
                            Some(position)
                        } else {
                            positions.insert(input, terms.len());
                            None
                        }
                    };
                    if let Some(position) = position {
                        terms[position].low += low;
                        terms[position].high += high;
                    } else {
                        terms.push(EdgeTerm { input, low, high });
                    }
                },
            );
        }
        row_offsets.push(terms.len());
    }

    AnalysisPlan {
        edges: EdgePlan {
            row_offsets: row_offsets.into_boxed_slice(),
            terms: terms.into_boxed_slice(),
        },
        prefix_len: prefix_end,
        interior: interior_start.map(|start| InteriorAnalysis {
            first_newest: 2 * start + phase,
            output_len: interior_end - start,
            kernel: backends.butterfly.map_or_else(
                || {
                    if boundary == Boundary::Periodization
                        || interior_end - start < MIN_LATTICE_OUTPUTS
                    {
                        AnalysisKernel::Direct
                    } else {
                        backends
                            .lattice
                            .clone()
                            .map_or(AnalysisKernel::Direct, AnalysisKernel::Lattice)
                    }
                },
                |butterfly| AnalysisKernel::Butterfly {
                    low_scale: butterfly.low_scale,
                    high_scale: butterfly.high_scale,
                },
            ),
        }),
        annihilator: backends
            .annihilator
            .map(|filter| AnnihilatorAnalysis::new(signal_len, coeff_len, boundary, filter)),
    }
}

#[inline]
fn analyze_interior<T: WaveletNum>(
    signal: &[T],
    newest: usize,
    dec_lo: &[T],
    dec_hi: &[T],
) -> (T, T) {
    let mut low = T::zero();
    let mut high = T::zero();
    for tap in 0..dec_lo.len() {
        let sample = signal[newest - tap];
        low += dec_lo[tap] * sample;
        high += dec_hi[tap] * sample;
    }
    (low, high)
}

#[inline]
fn analyze_edges<T: WaveletNum>(
    signal: &[T],
    edges: &EdgePlan<T>,
    first_row: usize,
    approx: &mut [T],
    detail: &mut [T],
) {
    debug_assert_eq!(approx.len(), detail.len());
    debug_assert!(first_row + approx.len() < edges.row_offsets.len());
    for (row, (approximation, detail)) in
        (first_row..).zip(approx.iter_mut().zip(detail.iter_mut()))
    {
        let mut low = T::zero();
        let mut high = T::zero();
        for term in &edges.terms[edges.row_offsets[row]..edges.row_offsets[row + 1]] {
            let sample = signal[term.input];
            low += term.low * sample;
            high += term.high * sample;
        }
        *approximation = low;
        *detail = high;
    }
}

fn for_each_extension_term<T: WaveletNum>(
    index: isize,
    signal_len: usize,
    boundary: Boundary,
    mut visit: impl FnMut(usize, T),
) {
    // Every supported extension is a linear map from the finite signal to one
    // requested sample. Planning composes these terms with both analysis
    // filters; execution never needs to know which boundary mode produced it.
    let weight = |value: f64| T::from_f64(value);
    if (0..signal_len as isize).contains(&index) {
        visit(index as usize, weight(1.0));
        return;
    }

    match boundary {
        Boundary::Zero => {}
        Boundary::Constant => {
            visit(if index < 0 { 0 } else { signal_len - 1 }, weight(1.0));
        }
        Boundary::Periodic => {
            visit(index.rem_euclid(signal_len as isize) as usize, weight(1.0));
        }
        Boundary::Periodization => {
            let periodic_len = signal_len + signal_len % 2;
            let wrapped = index.rem_euclid(periodic_len as isize) as usize;
            visit(wrapped.min(signal_len - 1), weight(1.0));
        }
        Boundary::Symmetric => {
            let period = 2 * signal_len;
            let wrapped = index.rem_euclid(period as isize) as usize;
            let reflected = if wrapped < signal_len {
                wrapped
            } else {
                period - 1 - wrapped
            };
            visit(reflected, weight(1.0));
        }
        Boundary::Antisymmetric => {
            let period = 2 * signal_len;
            let wrapped = index.rem_euclid(period as isize) as usize;
            if wrapped < signal_len {
                visit(wrapped, weight(1.0));
            } else {
                visit(period - 1 - wrapped, weight(-1.0));
            }
        }
        Boundary::Reflect => {
            let span = signal_len - 1;
            let period = 2 * span;
            let wrapped = index.rem_euclid(period as isize) as usize;
            let reflected = if wrapped < signal_len {
                wrapped
            } else {
                period - wrapped
            };
            visit(reflected, weight(1.0));
        }
        Boundary::Smooth => {
            if signal_len == 1 {
                visit(0, weight(1.0));
            } else if index < 0 {
                let distance = (-index) as f64;
                visit(0, weight(1.0 + distance));
                visit(1, weight(-distance));
            } else {
                let distance = (index - (signal_len - 1) as isize) as f64;
                visit(signal_len - 1, weight(1.0 + distance));
                visit(signal_len - 2, weight(-distance));
            }
        }
        Boundary::Antireflect => for_each_antireflect_term(index, signal_len, visit),
    }
}

#[inline]
fn extended_sample<T: WaveletNum>(signal: &[T], index: isize, boundary: Boundary) -> T {
    if (0..signal.len() as isize).contains(&index) {
        return signal[index as usize];
    }
    if boundary == Boundary::Smooth {
        if signal.len() == 1 {
            return signal[0];
        }
        if index < 0 {
            let distance = T::from_f64((-index) as f64);
            return signal[0] + (signal[0] - signal[1]) * distance;
        }
        let last = signal.len() - 1;
        let distance = T::from_f64((index - last as isize) as f64);
        return signal[last] + (signal[last] - signal[last - 1]) * distance;
    }
    let mut sample = T::zero();
    for_each_extension_term(index, signal.len(), boundary, |input, weight| {
        sample = mul_add(signal[input], weight, sample);
    });
    sample
}

fn for_each_antireflect_term<T: WaveletNum>(
    index: isize,
    signal_len: usize,
    mut visit: impl FnMut(usize, T),
) {
    debug_assert!(signal_len >= 2);
    debug_assert!(!(0..signal_len as isize).contains(&index));

    let last = signal_len - 1;
    let distance = if index < 0 {
        (-index) as usize
    } else {
        index as usize - last
    };
    let segment = (distance - 1) / last;
    let offset = (distance - 1) % last + 1;
    let weight = |value: isize| T::from_f64(value as f64);

    if index < 0 {
        if segment == 0 {
            visit(0, weight(2));
            visit(offset, weight(-1));
        } else if segment.is_multiple_of(2) {
            visit(0, weight(segment as isize + 2));
            visit(last, weight(-(segment as isize)));
            visit(offset, weight(-1));
        } else {
            visit(0, weight(segment as isize + 1));
            visit(last, weight(-(segment as isize) - 1));
            visit(last - offset, weight(1));
        }
    } else if segment == 0 {
        visit(last, weight(2));
        visit(last - offset, weight(-1));
    } else if segment.is_multiple_of(2) {
        visit(0, weight(-(segment as isize)));
        visit(last, weight(segment as isize + 2));
        visit(last - offset, weight(-1));
    } else {
        visit(0, weight(-(segment as isize) - 1));
        visit(last, weight(segment as isize + 1));
        visit(offset, weight(1));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planner_reuses_live_plans() {
        let mut planner = DwtPlanner::<f64>::new();
        let wavelet = Wavelet::haar();
        let first = planner.plan_dwt(8, &wavelet, Boundary::Symmetric).unwrap();
        let second = planner.plan_dwt(8, &wavelet, Boundary::Symmetric).unwrap();
        assert!(Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn axis_synthesis_kernel_selection_follows_transform_geometry() {
        assert_eq!(
            AxisSynthesisKernel::select(16, 8, false),
            AxisSynthesisKernel::Direct
        );
        assert_eq!(
            AxisSynthesisKernel::select(2, 76, false),
            AxisSynthesisKernel::Direct
        );
        assert_eq!(
            AxisSynthesisKernel::select(16, 76, true),
            AxisSynthesisKernel::Direct
        );
        assert_eq!(
            AxisSynthesisKernel::select(16, 24, false),
            AxisSynthesisKernel::Batched
        );
    }

    #[test]
    fn axis_execution_matches_independent_signal_execution() {
        let boundaries = [
            Boundary::Zero,
            Boundary::Constant,
            Boundary::Symmetric,
            Boundary::Reflect,
            Boundary::Periodic,
            Boundary::Smooth,
            Boundary::Antisymmetric,
            Boundary::Antireflect,
            Boundary::Periodization,
        ];
        let wavelets = [Wavelet::haar(), Wavelet::daubechies(4).unwrap()];

        for signal_len in [2, 7, 32] {
            for wavelet in &wavelets {
                for &boundary in &boundaries {
                    let plan =
                        create_dwt_plan::<f64>(signal_len, wavelet, boundary, SimdLevel::new())
                            .unwrap();
                    let outer = 2;
                    let inner = 5;
                    let signal: Vec<_> = (0..outer * signal_len * inner)
                        .map(|index| {
                            let centered = (index * 37 + 11) % 101;
                            (centered as f64 - 50.0) / 17.0
                        })
                        .collect();
                    let mut actual_approx = vec![0.0; outer * plan.coeff_len * inner];
                    let mut actual_detail = actual_approx.clone();
                    plan.forward_axis_into(
                        &signal,
                        outer,
                        inner,
                        &mut actual_approx,
                        &mut actual_detail,
                        &mut [],
                    );

                    let mut expected_approx = actual_approx.clone();
                    let mut expected_detail = actual_detail.clone();
                    for outer_index in 0..outer {
                        for lane in 0..inner {
                            let row: Vec<_> = (0..signal_len)
                                .map(|sample| {
                                    signal[(outer_index * signal_len + sample) * inner + lane]
                                })
                                .collect();
                            let (approx, detail) = plan.forward(&row);
                            for coefficient in 0..plan.coeff_len {
                                let output =
                                    (outer_index * plan.coeff_len + coefficient) * inner + lane;
                                expected_approx[output] = approx[coefficient];
                                expected_detail[output] = detail[coefficient];
                            }
                        }
                    }
                    assert_slices_close(&actual_approx, &expected_approx, 2.0e-13);
                    assert_slices_close(&actual_detail, &expected_detail, 2.0e-13);

                    let mut actual_output = vec![0.0; signal.len()];
                    plan.inverse_axis_into(
                        &actual_approx,
                        &actual_detail,
                        outer,
                        inner,
                        &mut actual_output,
                        &mut [],
                    );
                    let mut expected_output = actual_output.clone();
                    for outer_index in 0..outer {
                        for lane in 0..inner {
                            let approx: Vec<_> = (0..plan.coeff_len)
                                .map(|coefficient| {
                                    actual_approx[(outer_index * plan.coeff_len + coefficient)
                                        * inner
                                        + lane]
                                })
                                .collect();
                            let detail: Vec<_> = (0..plan.coeff_len)
                                .map(|coefficient| {
                                    actual_detail[(outer_index * plan.coeff_len + coefficient)
                                        * inner
                                        + lane]
                                })
                                .collect();
                            let row = plan.inverse(&approx, &detail);
                            for sample in 0..signal_len {
                                expected_output
                                    [(outer_index * signal_len + sample) * inner + lane] =
                                    row[sample];
                            }
                        }
                    }
                    assert_slices_close(&actual_output, &expected_output, 2.0e-13);
                }
            }
        }
    }

    #[test]
    fn batched_axis_inverse_matches_independent_signal_execution() {
        let wavelet = Wavelet::daubechies(38).unwrap();
        for boundary in [Boundary::Symmetric, Boundary::Periodization] {
            let plan = create_dwt_plan::<f64>(16, &wavelet, boundary, SimdLevel::new()).unwrap();
            let outer = 2;
            let inner = 257;
            let coefficient_count = outer * plan.coeff_len * inner;
            let approx: Vec<_> = (0..coefficient_count)
                .map(|index| ((index * 37 + 11) % 251) as f64 / 37.0 - 3.0)
                .collect();
            let detail: Vec<_> = (0..coefficient_count)
                .map(|index| ((index * 41 + 17) % 241) as f64 / 41.0 - 2.5)
                .collect();
            let mut actual = vec![0.0; outer * plan.signal_len * inner];
            plan.inverse_axis_into(&approx, &detail, outer, inner, &mut actual, &mut []);

            let mut expected = actual.clone();
            for outer_index in 0..outer {
                for lane in 0..inner {
                    let approx_row: Vec<_> = (0..plan.coeff_len)
                        .map(|coefficient| {
                            approx[(outer_index * plan.coeff_len + coefficient) * inner + lane]
                        })
                        .collect();
                    let detail_row: Vec<_> = (0..plan.coeff_len)
                        .map(|coefficient| {
                            detail[(outer_index * plan.coeff_len + coefficient) * inner + lane]
                        })
                        .collect();
                    let row = plan.inverse(&approx_row, &detail_row);
                    for sample in 0..plan.signal_len {
                        expected[(outer_index * plan.signal_len + sample) * inner + lane] =
                            row[sample];
                    }
                }
            }
            assert_slices_close(&actual, &expected, 2.0e-13);
        }
    }

    fn assert_slices_close(actual: &[f64], expected: &[f64], tolerance: f64) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            let error = (actual - expected).abs();
            assert!(
                error <= tolerance,
                "value {index}: actual={actual:.17e}, expected={expected:.17e}, error={error:.3e}"
            );
        }
    }

    #[test]
    fn planner_reuses_equivalent_live_multilevel_plans() {
        let mut planner = DwtPlanner::<f64>::new();
        let wavelet = Wavelet::haar();
        let first = planner
            .plan_wavedec(16, &wavelet, Boundary::Symmetric, Level::Max)
            .unwrap();
        let second = planner
            .plan_wavedec(16, &wavelet, Boundary::Symmetric, Level::Exact(4))
            .unwrap();
        assert!(Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn reflect_rejects_a_single_sample() {
        let mut planner = DwtPlanner::<f64>::new();
        let error = match planner.plan_dwt(1, &Wavelet::haar(), Boundary::Reflect) {
            Ok(_) => panic!("length-one reflect plan unexpectedly succeeded"),
            Err(error) => error,
        };
        assert!(matches!(
            error,
            WaveletError::BoundaryRequiresLongerSignal { .. }
        ));
    }

    #[test]
    fn antireflect_terms_cover_repeated_reflections() {
        let signal = [1.0_f64, 3.0, 6.0];
        let expected = [
            -29.0, -27.0, -24.0, -21.0, -19.0, -17.0, -14.0, -11.0, -9.0, -7.0, -4.0, -1.0, 1.0,
            3.0, 6.0, 9.0, 11.0, 13.0, 16.0, 19.0, 21.0, 23.0, 26.0, 29.0, 31.0, 33.0, 36.0,
        ];

        let actual: Vec<_> = (-12..=14)
            .map(|index| extended_sample(&signal, index, Boundary::Antireflect))
            .collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn two_sample_antireflect_is_linear_extrapolation() {
        let signal = [-3.0_f64, 5.0];
        for index in -128..=128 {
            let actual = extended_sample(&signal, index, Boundary::Antireflect);
            let expected = signal[0] + index as f64 * (signal[1] - signal[0]);
            assert_eq!(actual, expected, "extended sample {index}");
        }
    }

    #[test]
    fn butterfly_selection_depends_on_filter_algebra() {
        let wavelet =
            Wavelet::from_filters(&[0.5, 0.5], &[-0.25, 0.25], &[0.75, 0.75], &[0.125, -0.125])
                .unwrap();
        let plan =
            create_dwt_plan::<f64>(128, &wavelet, Boundary::Symmetric, SimdLevel::new()).unwrap();

        assert!(matches!(
            plan.analysis
                .interior
                .as_ref()
                .map(|interior| &interior.kernel),
            Some(AnalysisKernel::Butterfly {
                low_scale: 0.5,
                high_scale: 0.25,
            })
        ));
        assert!(matches!(
            plan.filters.synthesis_butterfly,
            Some(Butterfly {
                low_scale: 0.75,
                high_scale: 0.125,
            })
        ));
    }

    #[test]
    fn compiled_edge_rows_coalesce_repeated_inputs() {
        let wavelet = Wavelet::coiflet(17).unwrap();
        let plan =
            create_dwt_plan::<f64>(16, &wavelet, Boundary::Antireflect, SimdLevel::new()).unwrap();

        for offsets in plan.analysis.edges.row_offsets.windows(2) {
            let row = &plan.analysis.edges.terms[offsets[0]..offsets[1]];
            assert!(row.len() <= plan.signal_len);
            for (position, term) in row.iter().enumerate() {
                assert!(
                    row[..position]
                        .iter()
                        .all(|earlier| earlier.input != term.input),
                    "edge row contains input {} more than once",
                    term.input
                );
            }
        }
    }

    #[test]
    fn annihilator_selection_depends_on_filter_support() {
        let db20 = equivalent_custom_wavelet(&Wavelet::daubechies(20).unwrap());
        let db38 = equivalent_custom_wavelet(&Wavelet::daubechies(38).unwrap());
        let coif17 = equivalent_custom_wavelet(&Wavelet::coiflet(17).unwrap());
        let short =
            create_dwt_plan::<f64>(4_096, &db20, Boundary::Symmetric, SimdLevel::new()).unwrap();
        let long =
            create_dwt_plan::<f64>(4_096, &db38, Boundary::Symmetric, SimdLevel::new()).unwrap();

        assert!(short.analysis.annihilator.is_none());
        assert!(long.analysis.annihilator.is_some());

        let f32_db38 =
            create_dwt_plan::<f32>(4_096, &db38, Boundary::Symmetric, SimdLevel::new()).unwrap();
        let f32_coif17 =
            create_dwt_plan::<f32>(4_096, &coif17, Boundary::Symmetric, SimdLevel::new()).unwrap();
        assert!(f32_db38.analysis.annihilator.is_none());
        assert!(f32_coif17.analysis.annihilator.is_some());
    }

    #[test]
    fn dense_signal_rejects_annihilator_execution() {
        let wavelet = equivalent_custom_wavelet(&Wavelet::daubechies(38).unwrap());
        let plan =
            create_dwt_plan::<f64>(4_096, &wavelet, Boundary::Symmetric, SimdLevel::new()).unwrap();
        let signal: Vec<_> = (0..plan.signal_len)
            .map(|index| (index as f64 * 0.173).sin())
            .collect();

        assert!(
            !plan
                .analysis
                .annihilator
                .as_ref()
                .unwrap()
                .should_execute(&signal)
        );
    }

    #[test]
    fn non_finite_or_overflowing_differences_use_direct_execution() {
        let wavelet = equivalent_custom_wavelet(&Wavelet::daubechies(38).unwrap());
        let plan =
            create_dwt_plan::<f64>(4_096, &wavelet, Boundary::Symmetric, SimdLevel::new()).unwrap();
        let annihilator = plan.analysis.annihilator.as_ref().unwrap();

        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut signal = vec![1.0; plan.signal_len];
            signal[128] = value;
            assert!(!annihilator.should_execute(&signal));
        }
        let mut overflowing = vec![f64::MAX; plan.signal_len];
        overflowing[128..].fill(-f64::MAX);
        assert!(!annihilator.should_execute(&overflowing));
    }

    #[test]
    fn annihilator_matches_direct_kernel_for_every_boundary() {
        let boundaries = [
            Boundary::Zero,
            Boundary::Constant,
            Boundary::Symmetric,
            Boundary::Reflect,
            Boundary::Periodic,
            Boundary::Smooth,
            Boundary::Antisymmetric,
            Boundary::Antireflect,
            Boundary::Periodization,
        ];
        for wavelet in [
            Wavelet::daubechies(38).unwrap(),
            Wavelet::coiflet(17).unwrap(),
        ] {
            for boundary in boundaries {
                let f64_signal: Vec<_> = (0..4_096)
                    .map(|index| {
                        let run = index / 64;
                        1.0 + (run as f64 * 0.17).sin() + 0.1 * run as f64
                    })
                    .collect();
                assert_annihilator_matches_direct(
                    &wavelet,
                    boundary,
                    f64_signal,
                    |actual: f64, expected: f64| {
                        (actual - expected).abs() <= 2e-12_f64.max(2e-15 * expected.abs())
                    },
                );

                if wavelet.filter_len() >= MIN_ANNIHILATOR_FILTER_LEN_F32 {
                    let f32_signal: Vec<_> = (0..4_096)
                        .map(|index| {
                            let run = index / 64;
                            (1.0 + (run as f64 * 0.17).sin() + 0.1 * run as f64) as f32
                        })
                        .collect();
                    assert_annihilator_matches_direct(
                        &wavelet,
                        boundary,
                        f32_signal,
                        |actual: f32, expected: f32| {
                            (actual - expected).abs() <= 2e-4_f32.max(2e-6 * expected.abs())
                        },
                    );
                }
            }
        }
    }

    #[test]
    fn lattice_matches_direct_kernel_for_every_supported_boundary() {
        let boundaries = [
            Boundary::Zero,
            Boundary::Constant,
            Boundary::Symmetric,
            Boundary::Reflect,
            Boundary::Periodic,
            Boundary::Smooth,
            Boundary::Antisymmetric,
            Boundary::Antireflect,
        ];
        for wavelet in [
            Wavelet::daubechies(20).unwrap(),
            Wavelet::symlet(20).unwrap(),
            Wavelet::daubechies(38).unwrap(),
            Wavelet::coiflet(17).unwrap(),
        ] {
            for boundary in boundaries {
                let signal: Vec<_> = (0..4_096)
                    .map(|index| {
                        let index = index as f64;
                        (index * 0.173).sin() + 0.25 * (index * 0.037).cos()
                    })
                    .collect();
                assert_lattice_matches_direct(&wavelet, boundary, signal, 4.0e-13);
            }
        }
    }

    #[test]
    fn lattice_remains_finite_over_wide_dynamic_range() {
        for wavelet in [
            Wavelet::daubechies(38).unwrap(),
            Wavelet::coiflet(17).unwrap(),
        ] {
            let signal: Vec<_> = (0_usize..4_096)
                .map(|index| {
                    let exponent = ((index * 811) % 1_801) as i32 - 900;
                    let mantissa = 1.0 + ((index * 37) % 997) as f64 / 997.0;
                    let sign = if index.is_multiple_of(2) { 1.0 } else { -1.0 };
                    sign * mantissa * 2.0_f64.powi(exponent)
                })
                .collect();
            assert_lattice_matches_direct(&wavelet, Boundary::Symmetric, signal, 2.0e-12);
        }
    }

    #[test]
    fn avx512_lattice_preempts_the_dominated_structure_scan() {
        let level = SimdLevel::new();
        if !lattice_preempts_annihilator(level) {
            return;
        }
        let plan = create_dwt_plan::<f64>(
            4_096,
            &Wavelet::daubechies(38).unwrap(),
            Boundary::Symmetric,
            level,
        )
        .unwrap();
        assert!(plan.analysis.annihilator.is_none());
        assert!(matches!(
            plan.analysis
                .interior
                .as_ref()
                .map(|interior| &interior.kernel),
            Some(AnalysisKernel::Lattice(_))
        ));
    }

    fn assert_lattice_matches_direct(
        wavelet: &Wavelet,
        boundary: Boundary,
        signal: Vec<f64>,
        relative_tolerance: f64,
    ) {
        let mut accelerated =
            create_dwt_plan::<f64>(signal.len(), wavelet, boundary, SimdLevel::new()).unwrap();
        accelerated.analysis.annihilator = None;
        if lattice_simd_supported(SimdLevel::new()) {
            assert!(matches!(
                accelerated
                    .analysis
                    .interior
                    .as_ref()
                    .map(|interior| &interior.kernel),
                Some(AnalysisKernel::Lattice(_))
            ));
        }

        let mut direct =
            create_dwt_plan::<f64>(signal.len(), wavelet, boundary, SimdLevel::new()).unwrap();
        direct.analysis.annihilator = None;
        direct.analysis.interior.as_mut().unwrap().kernel = AnalysisKernel::Direct;

        let mut actual_approx = vec![0.0; accelerated.coeff_len];
        let mut actual_detail = vec![0.0; accelerated.coeff_len];
        accelerated.forward_into(&signal, &mut actual_approx, &mut actual_detail, &mut []);
        let mut expected_approx = vec![0.0; direct.coeff_len];
        let mut expected_detail = vec![0.0; direct.coeff_len];
        direct.forward_into(&signal, &mut expected_approx, &mut expected_detail, &mut []);

        let scale = expected_approx
            .iter()
            .chain(&expected_detail)
            .copied()
            .map(f64::abs)
            .fold(1.0, f64::max);
        let mut maximum_error = 0.0_f64;
        for (&actual, &expected) in actual_approx
            .iter()
            .chain(&actual_detail)
            .zip(expected_approx.iter().chain(&expected_detail))
        {
            assert!(actual.is_finite());
            maximum_error = maximum_error.max((actual - expected).abs());
        }
        assert!(
            maximum_error <= relative_tolerance * scale,
            "{} {boundary:?} maximum relative error {:.3e} exceeds {relative_tolerance:.3e}",
            wavelet.name(),
            maximum_error / scale,
        );
    }

    fn assert_annihilator_matches_direct<T: WaveletNum>(
        wavelet: &Wavelet,
        boundary: Boundary,
        signal: Vec<T>,
        close: impl Fn(T, T) -> bool,
    ) {
        let wavelet = equivalent_custom_wavelet(wavelet);
        let accelerated =
            create_dwt_plan::<T>(signal.len(), &wavelet, boundary, SimdLevel::new()).unwrap();
        let annihilator = accelerated.analysis.annihilator.as_ref().unwrap();
        let mut direct =
            create_dwt_plan::<T>(signal.len(), &wavelet, boundary, SimdLevel::new()).unwrap();
        direct.analysis.annihilator = None;

        let mut actual_approx = vec![T::zero(); accelerated.coeff_len];
        let mut actual_detail = vec![T::zero(); accelerated.coeff_len];
        annihilator.forward_into(&signal, &mut actual_approx, &mut actual_detail);
        let mut expected_approx = vec![T::zero(); direct.coeff_len];
        let mut expected_detail = vec![T::zero(); direct.coeff_len];
        direct.forward_into(&signal, &mut expected_approx, &mut expected_detail, &mut []);

        for (coefficient, (&actual, &expected)) in actual_approx
            .iter()
            .chain(&actual_detail)
            .zip(expected_approx.iter().chain(&expected_detail))
            .enumerate()
        {
            assert!(
                close(actual, expected),
                "{boundary:?} coefficient {coefficient}: {actual:?} != {expected:?}"
            );
        }
    }

    fn equivalent_custom_wavelet(wavelet: &Wavelet) -> Wavelet {
        Wavelet::from_filters(
            wavelet.dec_lo(),
            wavelet.dec_hi(),
            wavelet.rec_lo(),
            wavelet.rec_hi(),
        )
        .unwrap()
    }

    fn extended_sample(signal: &[f64], index: isize, boundary: Boundary) -> f64 {
        let mut value = 0.0;
        for_each_extension_term::<f64>(index, signal.len(), boundary, |input, weight| {
            value += signal[input] * weight;
        });
        value
    }
}
